import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from algos.dpg import eval_policy, collect_experience

from algos.dpg import ReplayBuffer

class TD3():
  def __init__(self, actor, q, a_lr, c_lr, discount=0.99, tau=0.001, center_reward=False, policy_noise=0.2, update_freq=2, noise_clip=0.5, normalize=False):
    if actor.is_recurrent:
      self.recurrent = True
    else:
      self.recurrent = False

    self.behavioral_actor  = actor
    self.behavioral_q = q

    self.target_actor = copy.deepcopy(actor)
    self.target_q = copy.deepcopy(q)

    self.soft_update(1.0)

    self.actor_optimizer  = torch.optim.Adam(self.behavioral_actor.parameters(), lr=a_lr)
    self.q_optimizer = torch.optim.Adam(self.behavioral_q.parameters(), lr=c_lr, weight_decay=1e-2)

    self.discount   = discount
    self.tau        = tau
    self.center_reward = center_reward
    self.update_every = update_freq

    self.policy_noise = policy_noise

    self.normalize = normalize

    self.n = 0

  def soft_update(self, tau):
    for param, target_param in zip(self.behavioral_q.parameters(), self.target_q.parameters()):
      target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    for param, target_param in zip(self.behavioral_actor.parameters(), self.target_actor.parameters()):
      target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

  def update_policy(self, replay_buffer, batch_size=256, traj_len=1000, grad_clip=None, noise_clip=0.2):
    self.n += 1

    states, actions, next_states, rewards, not_dones, steps = replay_buffer.sample(batch_size, sample_trajectories=self.recurrent, max_len=traj_len)

    with torch.no_grad():
      if self.normalize:
        states      = self.behavioral_actor.normalize_state(states, update=False)
        next_states = self.behavioral_actor.normalize_state(next_states, update=False)

      noise        = (torch.randn_like(actions) * self.policy_noise).clamp(-noise_clip, noise_clip)
      next_actions = (self.target_actor(next_states) + noise)

      target_q1, target_q2 = self.target_q(next_states, next_actions)

      target_q = rewards + not_dones * self.discount * torch.min(target_q1, target_q2)

    current_q1, current_q2 = self.behavioral_q(states, actions)

    critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

    self.q_optimizer.zero_grad()

    critic_loss.backward()

    self.q_optimizer.step()

    if self.n % self.update_every == 0:
      actor_loss = -self.behavioral_q.Q1(states, self.behavioral_actor(states)).mean()

      self.actor_optimizer.zero_grad()
      actor_loss.backward()

      self.actor_optimizer.step()
      
      self.soft_update(self.tau)

      return critic_loss.item(), -actor_loss.item(), steps
    else:
      return critic_loss.item(), 0, steps


def run_experiment(args):
  from time import time

  from rrl import env_factory, create_logger
  from policies.critic import LSTM_Q, TD3Critic
  from policies.actor import FF_Actor, LSTM_Actor

  import locale, os
  locale.setlocale(locale.LC_ALL, '')

  # wrapper function for creating parallelized envs
  env = env_factory(args.env_name)()
  eval_env = env_factory(args.env_name)()

  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if hasattr(env, 'seed'):
    env.seed(args.seed)

  obs_space = env.observation_space.shape[0]
  act_space = env.action_space.shape[0]

  if args.recurrent:
    actor = LSTM_Actor(obs_space, act_space, hidden_size=args.hidden_size, env_name=args.env_name, hidden_layers=args.layers)
    Q1 = LSTM_Q(obs_space, act_space, hidden_size=args.hidden_size, env_name=args.env_name, hidden_layers=args.layers)
    Q2 = LSTM_Q(obs_space, act_space, hidden_size=args.hidden_size, env_name=args.env_name, hidden_layers=args.layers)
  else:
    actor = FF_Actor(obs_space, act_space, hidden_size=args.hidden_size, env_name=args.env_name, hidden_layers=args.layers)
    Q = TD3Critic(obs_space, act_space, 256, 256)

  algo = TD3(actor, Q, args.a_lr, args.c_lr,
             discount=args.discount, 
             tau=args.tau, 
             center_reward=args.center_reward, 
             policy_noise=args.policy_noise, 
             update_freq=args.update_every, 
             noise_clip=args.noise_clip,
             normalize=args.normalize)

  replay_buff = ReplayBuffer(obs_space, act_space, args.timesteps)

  if algo.recurrent:
    print("Recurrent Twin-Delayed Deep Deterministic Policy Gradient:")
  else:
    print("Twin-Delayed Deep Deterministic Policy Gradient:")

  print(args)
  print("\tenv:            {}".format(args.env_name))
  print("\tseed:           {}".format(args.seed))
  print("\ttimesteps:      {:n}".format(args.timesteps))
  print("\tactor_lr:       {}".format(args.a_lr))
  print("\tcritic_lr:      {}".format(args.c_lr))
  print("\tdiscount:       {}".format(args.discount))
  print("\ttau:            {}".format(args.tau))
  print("\tnorm reward:    {}".format(args.center_reward))
  print("\tnorm states:    {}".format(args.normalize))
  print("\tbatch_size:     {}".format(args.batch_size))
  print("\twarmup period:  {:n}".format(args.start_timesteps))
  print()

  iter = 0
  episode_reward = 0
  episode_timesteps = 0

  # create a tensorboard logging object
  logger = create_logger(args)

  if args.save_actor is None:
    args.save_actor = os.path.join(logger.dir, 'actor.pt')

  #if args.save_critic is None:
  #  args.save_critic = os.path.join(logger.dir, 'critic.pt')

  # Keep track of some statistics for each episode
  training_start = time()
  episode_start = time()
  episode_loss = 0
  actor_loss = 0
  update_steps = 0
  best_reward = None

  # Fill replay buffer, update policy until n timesteps have passed
  timesteps = 0
  state = env.reset().astype(np.float32)
  while timesteps < args.timesteps:
    buffer_ready = (algo.recurrent and iter > args.batch_size) or (not algo.recurrent and replay_buff.size > args.batch_size)
    warmup = timesteps < args.start_timesteps

    state, r, done = collect_experience(algo.behavioral_actor, env, replay_buff, state, episode_timesteps,
                                        max_len=args.traj_len,
                                        random_action=warmup,
                                        noise=args.expl_noise, 
                                        do_trajectory=algo.recurrent,
                                        normalize=algo.normalize)

    episode_reward += r
    episode_timesteps += 1
    timesteps += 1

    # Update the policy once our replay buffer is big enough
    if buffer_ready and done and not warmup:
      update_steps = 0

      if algo.recurrent:
        num_updates = 1
      else:
        num_updates = episode_timesteps

      for _ in range(num_updates):
        u_loss, a_loss, u_steps = algo.update_policy(replay_buff, args.batch_size, traj_len=args.traj_len)
        episode_loss += u_loss / num_updates
        actor_loss += a_loss / num_updates
        update_steps += u_steps

    if done:
      episode_elapsed = (time() - episode_start)
      episode_secs_per_sample = episode_elapsed / episode_timesteps
      logger.add_scalar(args.env_name + ' episode length', episode_timesteps, iter)
      logger.add_scalar(args.env_name + ' episode reward', episode_reward, iter)
      logger.add_scalar(args.env_name + ' critic loss', episode_loss, iter)
      if actor_loss > 0:
        logger.add_scalar(args.env_name + ' actor loss', actor_loss, iter)

      completion = 1 - float(timesteps) / args.timesteps
      avg_sample_r = (time() - training_start)/timesteps
      secs_remaining = avg_sample_r * args.timesteps * completion
      hrs_remaining = int(secs_remaining//(60*60))
      min_remaining = int(secs_remaining - hrs_remaining*60*60)//60

      if iter % args.eval_every == 0 and iter != 0:
        eval_reward = eval_policy(algo.behavioral_actor, eval_env, max_traj_len=args.traj_len)
        logger.add_scalar(args.env_name + ' eval episode', eval_reward, iter)
        logger.add_scalar(args.env_name + ' eval timestep', eval_reward, timesteps)

        print("evaluation after {:4d} episodes | return: {:7.3f} | timesteps {:9n}{:100s}".format(iter, eval_reward, timesteps, ''))

        if best_reward is None or eval_reward > best_reward:
          torch.save(algo.behavioral_actor, args.save_actor)
          #torch.save(algo.behavioral_critic, args.save_critic)
          best_reward = eval_reward
          print("\t(best policy so far! saving to {})".format(args.save_actor))

    try:
      print("episode {:5d} | episode timestep {:5d}/{:5d} | return {:5.1f} | update timesteps: {:7n} | {:3.1f}s/1k samples | approx. {:3d}h {:02d}m remain\t\t\t\t".format(
        iter, 
        episode_timesteps, 
        args.traj_len, 
        episode_reward, 
        update_steps, 
        1000*episode_secs_per_sample, 
        hrs_remaining, 
        min_remaining), end='\r')

    except NameError:
      pass

    if done:
      if hasattr(algo.behavioral_actor, 'init_hidden_state'):
        algo.behavioral_actor.init_hidden_state()

      episode_start, episode_reward, episode_timesteps, episode_loss = time(), 0, 0, 0
      iter += 1
