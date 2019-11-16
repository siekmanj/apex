import torch
import torch.nn as nn

import numpy as np


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
    self.reward  += [reward]
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

    print(indices)

    for r in reversed(rewards):
      R = self.discount * R + r
      returns += [R]

    returns = returns.reverse()

  def sample(self, batch_size=64):
    pass
    
      
class PPO:
  def __init__(self, actor, critic, args, buff):
    self.actor = actor
    self.critic = critic
    self.buffer = buff

  def collect_experience(self, env, n, max_steps=400):
    with torch.no_grad():

      num_steps, done = 0, False
      while num_steps < n:

        state = torch.Tensor(env.reset())

        traj_len = 0
        while not done and traj_len < max_steps:
          action = self.actor(state)
          value  = self.critic(state)

          next_state, reward, done, _ = env.step(action.numpy())

          self.buffer.push(state.numpy(), action.numpy(), reward, value.numpy())

          state = torch.Tensor(next_state)
          traj_len += 1
        
        num_steps += traj_len
        value = self.critic(state)

