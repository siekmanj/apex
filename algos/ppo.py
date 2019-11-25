"""Proximal Policy Optimization (clip objective)."""
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import kl_divergence

from time import time

import numpy as np
import os

import ray

from policies.critic import Critic, FF_V
from policies.actor import Actor, FF_Stochastic_Actor

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

    self.ep_returns = [] # for logging
    self.ep_lens = [] # for logging

    self.size = 0

    self.traj_idx = [0]
    self.buffer_ready = False

  def push(self, state, action, reward, value, done=False):
    self.states  += [state]
    self.actions += [action]
    self.rewards += [reward]
    self.values  += [value]

    self.size += 1

  def end_trajectory(self, terminal_value=0):
    if terminal_value is None:
        terminal_value = np.zeros(shape=(1,))

    self.traj_idx += [self.size]
    rewards = self.rewards[self.traj_idx[-2]:self.traj_idx[-1]]

    returns = []

    R = terminal_value.squeeze(0).copy()
    for reward in reversed(rewards):
        R = self.discount * R + reward
        returns.insert(0, R)

    self.returns += returns

    self.ep_returns += [np.sum(rewards)]
    self.ep_lens    += [len(rewards)]

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
      random_indices = SubsetRandomSampler(range(self.size))
      sampler = BatchSampler(random_indices, batch_size, drop_last=True)

      for i, idxs in enumerate(sampler):
        #print(np.max(idxs))
        #print(self.states.size())

        #idxs = torch.LongTensor(idxs)
        #print(self.returns)
        #print(idxs.size())
        #if True:
        #  wtf = self.states[[0, 1]]

        states     = self.states[idxs]
        actions    = self.actions[idxs] 
        returns    = self.returns[idxs]
        advantages = self.advantages[idxs]

        yield states, actions, returns, advantages

@ray.remote
class PPO_Worker:
  def __init__(self, policy, critic, env_fn, gamma):
    self.actor = deepcopy(policy)
    self.critic = deepcopy(critic)
    self.env = env_fn()
    self.gamma = gamma

  def update_policy(self, new_actor_params, new_critic_params, input_norm=None):
    for p, new_p in zip(self.actor.parameters(), new_actor_params):
      p.data.copy_(new_p)

    for p, new_p in zip(self.critic.parameters(), new_critic_params):
      p.data.copy_(new_p)

    if input_norm is not None:
      self.actor.welford_state_mean, self.actor.welford_state_mean_diff, self.actor.welford_state_n = input_norm

  def collect_experience(self, max_traj_len, min_steps):

    num_steps = 0
    memory = Buffer(self.gamma)
    while num_steps < min_steps:
      state = torch.Tensor(self.env.reset())

      done = False
      value = 0
      traj_len = 0

      while not done and traj_len < max_traj_len:
          state = torch.Tensor(state)
          norm_state = self.actor.normalize_state(state, update=False)
          action = self.actor(norm_state, False)
          value = self.critic(norm_state)
          next_state, reward, done, _ = self.env.step(action.numpy())

          reward = np.array([reward])

          memory.push(state.numpy(), action.numpy(), reward, value.numpy())

          state = next_state

          traj_len += 1
          num_steps += 1

      value = self.critic(torch.Tensor(state))
      memory.end_trajectory(terminal_value=(not done) * value.numpy())

    return memory

class PPO:
    def __init__(self, actor, critic, env_fn, discount=0.99, entropy_coeff=0.0, a_lr=1e-4, c_lr=1e-4, eps=1e-5, grad_clip=0.05, workers=4):

      self.actor = actor
      self.old_actor = deepcopy(actor)
      self.critic = critic

      self.actor_optim = optim.Adam(self.actor.parameters(), lr=a_lr, eps=eps)
      self.critic_optim = optim.Adam(self.critic.parameters(), lr=c_lr, eps=eps)

      self.env_fn = env_fn
      self.discount = discount
      self.entropy_coeff = entropy_coeff
      self.grad_clip = grad_clip

      if not ray.is_initialized():
        if False: #redis_addr is not None:
          ray.init(redis_address=redis_addr)
        else:
          ray.init()

      self.workers = [PPO_Worker.remote(actor, critic, env_fn, discount) for _ in range(workers)]

    def update_policy(self, states, actions, returns, advantages):
      with torch.no_grad():
        states = self.actor.normalize_state(states, update=False)
        old_pdf = self.old_actor.pdf(states)
        old_log_probs = old_pdf.log_prob(actions).sum(-1, keepdim=True)

      values = self.critic(states)
      pdf = self.actor.pdf(states)
      
      log_probs = pdf.log_prob(actions).sum(-1, keepdim=True)
      
      ratio = (log_probs - old_log_probs).exp()

      cpi_loss = ratio * advantages
      clip_loss = ratio.clamp(0.8, 1.2) * advantages
      actor_loss = -torch.min(cpi_loss, clip_loss).mean()

      critic_loss = 0.5 * (returns - values).pow(2).mean()

      entropy_penalty = -self.entropy_coeff * pdf.entropy().mean()

      self.actor_optim.zero_grad()
      self.critic_optim.zero_grad()

      (actor_loss + entropy_penalty).backward()
      critic_loss.backward()

      torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
      torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
      self.actor_optim.step()
      self.critic_optim.step()

      return kl_divergence(pdf, old_pdf).mean().detach().numpy()

    def merge_buffers(self, buffers):
      memory = Buffer()

      for b in buffers:
        offset = len(memory)

        memory.states  += b.states
        memory.actions += b.actions
        memory.rewards += b.rewards
        memory.values  += b.values
        memory.returns += b.returns

        memory.ep_returns += b.ep_returns
        memory.ep_lens    += b.ep_lens

        memory.traj_idx += [offset + i for i in b.traj_idx[1:]]
        memory.size     += b.size
      return memory

    def do_iteration(self, num_steps, max_traj_len, epochs, kl_thresh=0.02, verbose=True):
      self.old_actor.load_state_dict(self.actor.state_dict())

      start = time()
      actor_param_id  = ray.put(list(self.actor.parameters()))
      critic_param_id = ray.put(list(self.critic.parameters()))
      norm_id = ray.put([self.actor.welford_state_mean, self.actor.welford_state_mean_diff, self.actor.welford_state_n])
      if verbose:
        print("\tDEBUG: {:5.4f}s to place policy params in shared memory.".format(time() - start))

      steps = num_steps // len(self.workers)

      start = time()
      for w in self.workers:
        w.update_policy.remote(actor_param_id, critic_param_id, input_norm=norm_id)
      if verbose:
        print("\tDEBUG: {:5.4f}s to copy policy params to workers.".format(time() - start))

      start = time()
      buffers = ray.get([w.collect_experience.remote(max_traj_len, steps) for w in self.workers])
      memory = self.merge_buffers(buffers)

      total_steps = len(memory)
      elapsed = time() - start
      if verbose:
        print("\tDEBUG: {:3.2f}s to collect {:6n} timesteps | {:3.2}k/s.".format(elapsed, total_steps, (total_steps/1000)/elapsed))

      start = time()
      kls = []
      for _ in range(epochs):
        for batch in memory.sample():
          states, actions, returns, advantages = batch
          
          kl = self.update_policy(states, actions, returns, advantages)
          kls += [kl]
          if kl > kl_thresh:
              print("Max kl reached, stopping optimization early.")
              break
      if verbose:
        print("\tDEBUG: {:3.2f}s to update policy.".format(time() - start))
      return np.mean(kls), len(memory)

def eval_policy(policy, env, update_normalizer, deterministic, min_timesteps=2000, max_traj_len=400, verbose=True):
  with torch.no_grad():
    steps = 0
    ep_returns = []
    while steps < min_timesteps: # Prenormalize
      state = torch.Tensor(env.reset())
      done = False
      traj_len = 0
      ep_return = 0

      while not done and traj_len < max_traj_len:
        state = policy.normalize_state(state, update=update_normalizer)
        action = policy(state, deterministic=deterministic)
        next_state, reward, done, _ = env.step(action.numpy())
        state = torch.Tensor(next_state)
        ep_return += reward
        traj_len += 1
        steps += 1
        if verbose:
          print("Evaluating {:5d}/{:5d}".format(steps, min_timesteps), end="\r")
      ep_returns += [ep_return]

  print()
  return np.mean(ep_returns)
  

def run_experiment(args):
    torch.set_num_threads(1) # see: https://github.com/pytorch/pytorch/issues/13757

    from rrl import env_factory, create_logger
    import locale, os
    locale.setlocale(locale.LC_ALL, '')

    # wrapper function for creating parallelized envs
    env_fn = env_factory(args.env_name, state_est=args.state_est, mirror=False)
    obs_dim = env_fn().observation_space.shape[0]
    action_dim = env_fn().action_space.shape[0]

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    policy = FF_Stochastic_Actor(obs_dim, action_dim, env_name=args.env_name, fixed_std=torch.ones(action_dim)*np.exp(-2))
    critic = FF_V(obs_dim)

    env = env_fn()
    eval_policy(policy, env, True, False, min_timesteps=args.prenormalize_steps, max_traj_len=args.traj_len)

    policy.train(0)
    critic.train(0)

    algo = PPO(policy, critic, env_fn, workers=args.workers)

    # create a tensorboard logging object
    logger = create_logger(args)

    if args.save_actor is None:
      args.save_actor = os.path.join(logger.dir, 'actor.pt')

    print()
    print("Proximal Policy Optimization:")
    print("\tseed:               {}".format(args.seed))
    print("\tenv:                {}".format(args.env_name))
    print("\ttimesteps:          {}".format(args.timesteps))
    print("\tprenormalize steps: {}".format(args.prenormalize_steps))
    print("\ttraj_len:           {}".format(args.traj_len))
    print("\tdiscount:           {}".format(args.discount))
    print("\tactor_lr:           {}".format(args.a_lr))
    print("\tcritic_lr:          {}".format(args.c_lr))
    print("\tadam eps:           {}".format(args.eps))
    print("\tentropy coeff:      {}".format(args.entropy_coeff))
    print("\tgrad clip:          {}".format(args.grad_clip))
    print("\tbatch size:         {}".format(args.batch_size))
    print("\tepochs:             {}".format(args.epochs))
    print("\tworkers:            {}".format(args.workers))
    print()

    itr = 0
    timesteps = 0
    best_reward = None
    while timesteps < args.timesteps:
      kl, steps = algo.do_iteration(args.num_steps, args.traj_len, args.epochs)
      eval_reward = eval_policy(algo.actor, env, False, True, min_timesteps=args.traj_len*3, max_traj_len=args.traj_len, verbose=False)

      timesteps += steps
      print("iter {:4d} | return: {:5.2f} | KL {:5.4f} | timesteps {:n}".format(itr, eval_reward, kl, timesteps))

      if best_reward is None or eval_reward > best_reward:
        torch.save(algo.actor, args.save_actor)
        print("\t(best policy so far! saving to {})".format(args.save_actor))

      logger.add_scalar(args.env_name + '/kl', kl, itr)
      logger.add_scalar(args.env_name + '/return', eval_reward, itr)
      itr += 1
