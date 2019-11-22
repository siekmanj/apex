import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.distributions import kl_divergence
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

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
    self.buffer_ready = False

  def push(self, state, action, reward, value, done=False):
    #print("Storing {} {} {} {}".format(state.shape, action.shape, reward.shape, value.shape))

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
    self.buffer_ready = True

  def sample(self, batch_size=64, recurrent=False):
    if not self.buffer_ready:
      self._finish_buffer()

    if recurrent:
      raise NotImplementedError
    else:
      #print("Creating new random indices!")
      random_indices = SubsetRandomSampler(range(self.size))
      sampler = BatchSampler(random_indices, batch_size, drop_last=True)

      for i, idxs in enumerate(sampler):
        #print("Yielding batch {} of {}".format(i, len(sampler)))

        states     = self.states[idxs]
        actions    = self.actions[idxs] 
        returns    = self.returns[idxs]
        advantages = self.advantages[idxs]

        yield states, actions, returns, advantages
      
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

  #@ray.remote
  def collect_experience(self, n, batches, max_steps=400):
    env = self.env
    with torch.no_grad():

      num_steps = 0
      while num_steps < n:

        state = torch.Tensor(env.reset())

        traj_len, done = 0, False
        while not done and traj_len < max_steps:
          norm_state = self.actor.normalize_state(state)
          action = self.actor(norm_state, deterministic=False).numpy()
          value  = self.critic(norm_state).numpy()

          next_state, reward, done, _ = env.step(action)

          reward = np.array([reward])

          self.buffer.push(state.numpy(), action, reward, value)

          state = torch.Tensor(next_state)
          traj_len += 1
        
        num_steps += traj_len
        value = self.critic(state)
        self.buffer.end_trajectory(terminal_value=(not done) * value.numpy())

      return num_steps

  def update_policy(self, epochs, batch_size):
    kl     = []
    a_loss = []
    c_loss = []
    ratios = []
    ent    = []
    adv    = []
    a_p_n  = []
    c_p_n  = []

    timesteps = self.collect_experience(2000, epochs)
    self.old_actor.load_state_dict(self.actor.state_dict())  # WAY faster than deepcopy

    for epoch in range(epochs):
      for i, batch in enumerate(self.buffer.sample(batch_size)):
        states, actions, returns, advantages = batch

        """
        with torch.no_grad():
          states = self.actor.normalize_state(states, update=False)
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
        #critic_loss = 0.5 * (returns - values).pow(2).mean()

        entropy_penalty = -self.entropy_coeff * pdf.entropy().mean()

        self.actor_optim.zero_grad()
        (actor_loss + entropy_penalty).backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optim.step()
        """
        obs_batch = states
        action_batch = actions
        advantage_batch = advantages
        return_batch = returns
        # TODO, move this outside loop?
        with torch.no_grad():
          obs_batch = self.actor.normalize_state(obs_batch, update=False)
          old_pdf = self.old_actor.pdf(obs_batch)
          old_log_probs = old_pdf.log_prob(action_batch).sum(-1, keepdim=True)

        values = self.critic(obs_batch)
        pdf = self.actor.pdf(obs_batch)
        
        log_probs = pdf.log_prob(action_batch).sum(-1, keepdim=True)
        
        ratio = (log_probs - old_log_probs).exp()

        cpi_loss = ratio * advantage_batch
        clip_loss = ratio.clamp(0.8, 1.2) * advantage_batch
        actor_loss = -torch.min(cpi_loss, clip_loss).mean()

        if not torch.isfinite(actor_loss) or actor_loss.item() > 100:
          print("NON FINITE ACTR LOSS {}, old logs {}, logs {}, adv {}".format(actor_loss, old_log_probs, log_probs, advantage_batch))
          print("ratio {}, cpi {}, clip {}".format(ratio, cpi_loss, clip_loss))
          exit(1)

        critic_loss = 0.5 * (return_batch - values).pow(2).mean()

        entropy_penalty = -self.entropy_coeff * pdf.entropy().mean()

        # TODO: add ability to optimize critic and actor seperately, with different learning rates

        self.actor_optim.zero_grad()
        (actor_loss + entropy_penalty).backward()

        # Clip the gradient norm to prevent "unlucky" minibatches from 
        # causing pathalogical updates
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()

        # Clip the gradient norm to prevent "unlucky" minibatches from 
        # causing pathalogical updates
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optim.step()


        actor_norm = np.sqrt(np.mean(np.square(np.concatenate([p.grad.data.numpy().flatten() for p in self.actor.parameters() if p.grad is not None]))))
        critic_norm = np.sqrt(np.mean(np.square(np.concatenate([p.grad.data.numpy().flatten() for p in self.critic.parameters() if p.grad is not None]))))


        print("BATCH {} RATIO {} CLOSS {} ALOSS {} ENTROPY {}".format(i, ratio.mean(), critic_loss.mean(), actor_loss.mean(), pdf.entropy().mean()))

        with torch.no_grad():

          a_loss += [actor_loss.item()]
          c_loss += [critic_loss.item()]
          ratios += [ratio.mean().numpy()]
          ent    += [pdf.entropy().mean().numpy()]
          adv    += [advantages.mean().numpy()]
          a_p_n  += [actor_norm]
          c_p_n  += [critic_norm]

      kl_div = kl_divergence(pdf, old_pdf).mean() 
      kl     += [kl_div.detach().numpy()]
      if kl_div > 0.1:
        print("Max KL reached, aborting update")
        break

    self.buffer.clear()
    return timesteps, np.mean(kl), np.mean(a_loss), np.mean(c_loss), np.mean(ratios), np.mean(ent), np.mean(adv), np.mean(a_p_n), np.mean(c_p_n)

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

  from rrl import env_factory, create_logger

  import locale, os
  locale.setlocale(locale.LC_ALL, '')

  env_fn = env_factory(args.env_name)
  
  eval_env = env_fn()

  state_dim = env_fn().observation_space.shape[0]
  action_dim = env_fn().action_space.shape[0]

  actor = FF_Stochastic_Actor(state_dim, action_dim, env_name=args.env_name)
  critic = FF_V(state_dim)

  algo = PPO(actor, critic, args, Buffer(discount=args.discount), env_fn)

  steps = 0
  while steps < args.prenormalize_steps: # Prenormalize
    state = torch.Tensor(eval_env.reset())
    done = False
    traj_len = 0
    while not done and traj_len < 400:
      state = actor.normalize_state(state)
      action = actor(state, deterministic=False).numpy()
      next_state, reward, done, _ = eval_env.step(action)
      state = torch.Tensor(next_state)
      traj_len += 1
      steps += 1
    print("Prenormalizing {:5d}/{:5d}".format(steps, args.prenormalize_steps), end="\r")
  print()

  # create a tensorboard logging object
  logger = create_logger(args)

  if args.save_actor is None:
    args.save_actor = os.path.join(logger.dir, 'actor.pt')


  i = 0
  steps = 0
  while steps < args.timesteps:
    timesteps, kl, a_loss, c_loss, ratio, entropy, adv, a_norm, c_norm = algo.update_policy(args.epochs, args.batch_size)
    steps += timesteps
    with torch.no_grad():
      iter_eval = eval_policy(actor, eval_env)
    print("iter {:2d}) return {:5.1f} | critic {:9.5f} | actor {:9.5f} | entropy {:5.3f} | KL {:5.3f} | r {:5.4f} | {:n} of {:n}".format(i, iter_eval, c_loss, a_loss, entropy, kl, ratio, steps, int(args.timesteps)))
    logger.add_scalar(args.env_name + '/ppo/iteration_return', iter_eval, i)
    logger.add_scalar(args.env_name + '/ppo/timestep_return', iter_eval, steps)
    logger.add_scalar(args.env_name + '/ppo/critic_loss', c_loss, i)
    logger.add_scalar(args.env_name + '/ppo/actor_loss', a_loss, i)
    logger.add_scalar(args.env_name + '/ppo/entropy', entropy, i)
    logger.add_scalar(args.env_name + '/ppo/actor_param_norm', a_norm, i)
    logger.add_scalar(args.env_name + '/ppo/critic_param_norm', c_norm, i)
    logger.add_scalar(args.env_name + '/ppo/kl', kl, i)

    i += 1

  exit(1)


