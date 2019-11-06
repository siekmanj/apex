import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from rl.algos.dpg import ReplayBuffer

class TD3():
  def __init__(self, actor, q1, q2, a_lr, c_lr, discount=0.99, tau=0.001, center_reward=False):
    if actor.is_recurrent or critic.is_recurrent:
      self.recurrent = True
    else:
      self.recurrent = False

    self.behavioral_actor  = actor
    self.behavioral_q1 = q1
    self.behavioral_q2 = q2

    self.target_actor = copy.deepcopy(actor)
    self.target_q1 = copy.deepcopy(q1)
    self.target_q2 = copy.deepcopy(q2)

    self.soft_update(1.0)

    self.actor_optimizer  = torch.optim.Adam(self.behavioral_actor.parameters(), lr=a_lr)
    self.q1_optimizer = torch.optim.Adam(self.behavioral_q1.parameters(), lr=c_lr, weight_decay=1e-2)
    self.q2_optimizer = torch.optim.Adam(self.behavioral_q2.parameters(), lr=c_lr, weight_decay=1e-2)

    self.discount   = discount
    self.tau        = tau
    self.center_reward = center_reward

  def soft_update(self, tau):
    for param, target_param in zip(self.behavioral_q1.parameters(), self.target_q1.parameters()):
      target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    for param, target_param in zip(self.behavioral_q2.parameters(), self.target_q2.parameters()):
      target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    for param, target_param in zip(self.behavioral_actor.parameters(), self.target_actor.parameters()):
      target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

  def update_policy(self, replay_buffer, batch_size=256, traj_len=1000, grad_clip=None):
    states, actions, next_states, rewards, not_dones, steps = replay_buffer.sample(batch_size, sample_trajectories=self.recurrent, max_len=traj_len)

    with torch.no_grad():
      noise = (torch.randn_like(actions) * self.policy_noise).clamp(-1, 1)
      next_action = (self.target_actor(next_states) + noise).clamp(-1, 1)


    target_q1 = rewards + (not_dones * self.discount * self.target_q1(next_states, self.target_actor(next_states))).detach()
    target_q2 = rewards + (not_dones * self.discount * self.target_q2(next_states, self.target_actor(next_states))).detach()

    current_q1 = self.behavioral_q1(states, actions)
    current_q2 = self.behavioral_q2(states, actions)

    critic_loss = F.mse_loss(current_q1, target_q1) + F.mse_loss(current_q2, target_q2)

    self.q1_optimizer.zero_grad()
    self.q2_optimizer.zero_grad()

    critic_loss.backward()

    self.q1_optimizer.step()
    self.q2_optimizer.step()

    if grad_clip is not None:
      torch.nn.utils.clip_grad_norm_(self.behavioral_critic.parameters(), grad_clip)

    self.critic_optimizer.step()

    actor_loss = -self.behavioral_q1(states, self.behavioral_actor(states)).mean()

    self.actor_optimizer.zero_grad()
    actor_loss.backward()

    if grad_clip is not None:
      torch.nn.utils.clip_grad_norm_(self.behavioral_actor.parameters(), grad_clip)

    self.actor_optimizer.step()
    
    self.soft_update(self.tau)

    return critic_loss.item(), steps
