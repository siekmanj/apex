import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.distributions import kl_divergence

import numpy as np

from copy import deepcopy

class Buffer:
  def __init__(self, discount=0.99):
    self.discount = discount

    self.clear()

  def __len__(self):
    return len(self.states)

  def clear(self):
    self.states     = []
    self.actions    = []
    self.rewards    = []
    self.values     = []
    self.returns    = []
    self.advantages = []

    self.size = 0

    self.traj_idx = [0]

  def push(self, state, action, reward, value, done=False):
    self.states  += [state]
    self.actions += [action]
    self.rewards += [reward]
    self.values  += [value]

    self.size += 1

  def end_trajectory(self, terminal_value=0):

    self.traj_idx += [self.size]

    rewards = self.rewards[self.traj_idx[-2]:self.traj_idx[-1]]
    returns = []

    R = terminal_value
    for r in reversed(rewards):
      R = self.discount * R + r
      returns += [R]
    returns.reverse()

    self.returns += returns
    self.buffer_ready = False

  def _finish_buffer(self):
    self.states  = torch.Tensor(self.states)
    self.actions = torch.Tensor(self.actions)
    self.rewards = torch.Tensor(self.rewards)
    self.returns = torch.Tensor(self.returns)
    self.values  = torch.Tensor(self.values)
        
    a = self.returns - self.values
    a = (a - a.mean()) / (a.std() + 1e-4)
    self.advantages = a
    self.advantage_calculated = True

  def sample(self, batch_size=64, recurrent=False):
    if not self.buffer_ready:
      self._finish_buffer()

    if recurrent:
      raise NotImplementedError
    else:
      idxs = np.random.randint(0, self.size-1, size=batch_size)
      #print(np.max(idxs), len(self.returns), len(self.states))

      states     = self.states[idxs]
      actions    = self.actions[idxs] 
      returns    = self.returns[idxs]
      advantages = self.advantages[idxs]

      return states, actions, returns, advantages
      
class PPO:
  def __init__(self, actor, critic, args, buff, env_fn):
    self.actor = actor
    self.critic = critic

    self.old_actor = deepcopy(actor)
    self.old_critic = deepcopy(critic)
    self.buffer = buff
    self.env = env_fn()

    self.actor_optim  = optim.Adam(actor.parameters(), lr=args.a_lr, eps=args.a_eps)
    self.critic_optim = optim.Adam(critic.parameters(), lr=args.c_lr, eps=args.c_eps)

    self.recurrent = False
    self.entropy_coeff = args.entropy_coeff
    self.grad_clip = args.grad_clip

  def collect_experience(self, n, batches, max_steps=400):
    env = self.env
    with torch.no_grad():

      num_steps = 0
      while num_steps < n:

        state = torch.Tensor(env.reset())

        traj_len, done = 0, False
        while not done and traj_len < max_steps:
          action = self.actor(state, deterministic=False).numpy()
          value  = self.critic(state).numpy()

          next_state, reward, done, _ = env.step(action)

          self.buffer.push(state.numpy(), action, reward, value)

          state = torch.Tensor(next_state)
          traj_len += 1
        
        num_steps += traj_len
        value = self.critic(state)
        self.buffer.end_trajectory(terminal_value=(not done) * value.numpy())

      return [self.buffer.sample(batch_size=64, recurrent=self.recurrent) for _ in range(batches)], num_steps

  def update_policy(self, epochs, batch_size):

    batches, timesteps = self.collect_experience(1000, epochs)
    for i, batch in enumerate(batches):
      states, actions, returns, advantages = batch

      with torch.no_grad():
        old_pdf       = self.old_actor.pdf(states)
        old_log_probs = old_pdf.log_prob(actions).sum(-1, keepdim=True)

      pdf       = self.actor.pdf(states)
      log_probs = pdf.log_prob(actions).sum(-1, keepdim=True)

      ratio = (log_probs - old_log_probs).exp()

      cpi_loss = ratio * advantages
      clip_loss = ratio.clamp(0.8, 1.2) * advantages
      actor_loss = -torch.min(cpi_loss, clip_loss).mean()

      values = self.critic(states)
      critic_loss = F.mse_loss(values, returns)

      entropy_penalty = -self.entropy_coeff * pdf.entropy().mean()

      self.actor_optim.zero_grad()
      (actor_loss + entropy_penalty).backward()
      torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
      self.actor_optim.step()

      self.critic_optim.zero_grad()
      critic_loss.backward()
      torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
      self.critic_optim.step()

      kl_div = kl_divergence(pdf, old_pdf).mean() 
      print("epoch {:2d}) critic loss {:9.5f} | actor_loss {:9.5f} | entropy loss {:5.3f} | KL {:5.3f} | log_prob {:5.4f}".format(i, critic_loss.item(), actor_loss.item(), entropy_penalty, kl_div, log_probs.mean()))
      if kl_div > 0.02:
        print("Max KL reached, aborting update")
        break

    self.buffer.clear()
    return timesteps

def eval_policy(policy, env, max_steps=400):
  sum_reward = 0
  for _ in range(10):
    state = torch.Tensor(env.reset())

    done = False
    traj_len = 0
    while not done and traj_len < max_steps:
      action = policy(state, deterministic=True).numpy()
      next_state, reward, done, _ = env.step(action)
      state = torch.Tensor(next_state)
      sum_reward += reward
      traj_len += 1
  return sum_reward / 10

def run_experiment(args):
  from policies import FF_Stochastic_Actor, FF_V

  from rrl import env_factory

  env_fn = env_factory(args.env_name)
  
  eval_env = env_fn()

  state_dim = env_fn().observation_space.shape[0]
  action_dim = env_fn().action_space.shape[0]

  actor = FF_Stochastic_Actor(state_dim, action_dim, env_name=args.env_name)
  critic = FF_V(state_dim)

  algo = PPO(actor, critic, args, Buffer(discount=args.discount), env_fn)

  steps = 0
  while steps < 10000: # Prenormalize
    pass

  while steps < args.timesteps:
    algo.update_policy(args.epochs, args.batch_size)

    with torch.no_grad():
      print("Eval reward: {}".format(eval_policy(actor, eval_env)))

  exit(1)


