import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from copy import deepcopy

class Buffer:
  def __init__(self, discount=0.99):
    self.states  = []
    self.actions = []
    self.rewards = []
    self.values  = []
    self.returns = []

    self.discount = discount

    self.size = 0

    self.traj_idx = [0]
 
  def __len__(self):
    return len(self.states)

  def push(self, state, action, reward, value, done=False):
    self.states  += [state]
    self.actions += [action]
    self.rewards += [reward]
    self.values  += [value]

    self.size += 1

  def end_trajectory(self, terminal_value=None):
    if terminal_value is None:
      R = 0
    else:
      R = terminal_value

    self.traj_idx += [self.size-1]

    rewards = self.rewards[self.traj_idx[-2]:self.traj_idx[-1]]
    returns = []

    for r in reversed(rewards):
      R = self.discount * R + r
      returns += [R]
    returns.reverse()

    self.returns += returns

  def sample(self, batch_size=64, recurrent=False):
    if recurrent:
      raise NotImplementedError
    else:
      idxs = np.random.randint(0, self.size-1, size=batch_size)
      print(np.max(idxs), len(self.returns), len(self.states))
      states  = [self.states[i]  for i in idxs]
      actions = [self.actions[i] for i in idxs]
      rewards = [self.rewards[i] for i in idxs]
      values  = [self.values[i]  for i in idxs]
      returns = [self.returns[i] for i in idxs]
      return states, actions, rewards, values, returns
    
      
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

    batches, timesteps = self.collect_experience(500, epochs)
    for batch in batches:
      states, actions, rewards, values, returns = batch

    return timesteps

def run_experiment(args):
  from policies import FF_Stochastic_Actor, FF_V

  from rrl import env_factory

  env_fn = env_factory(args.env_name)

  state_dim = env_fn().observation_space.shape[0]
  action_dim = env_fn().action_space.shape[0]

  actor = FF_Stochastic_Actor(state_dim, action_dim, env_name=args.env_name, learn_std=True)
  critic = FF_V(state_dim)

  algo = PPO(actor, critic, args, Buffer(discount=args.discount), env_fn)

  steps = 0
  while steps < args.timesteps:
    algo.update_policy(args.epochs, args.batch_size)
  exit(1)


