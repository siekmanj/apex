import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from torch.nn.utils.rnn import pad_sequence

from algos.dpg import ReplayBuffer

class SAC():
  def __init__(self, actor, critic, a_lr, c_lr, discount=0.99, alpha=1):
    self.actor  = actor
    self.critic = critic

    self.target_actor  = copy.deepcopy(actor)
    self.target_critic = copy.deepcopy(critic)

    self.actor_optim  = torch.optim.Adam(self.actor.parameters(), lr=a_lr)
    self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=c_lr)

  def update(self, buff, batch_size=64):
    pass
    


def run_experiment(args):
  from time import time

  from util.env import env_factory
  from util.log import create_logger

  from policies.critic import FF_Q, LSTM_Q
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
    actor = LSTM_Actor(obs_space, act_space, env_name=args.env_name)
    critic = LSTM_Q(obs_space, act_space, env_name=args.env_name)
  else:
    actor = FF_Actor(obs_space, act_space, env_name=args.env_name)
    critic = FF_Q(obs_space, act_space, env_name=args.env_name)

  algo = SAC(actor, critic, args.a_lr, args.c_lr, discount=args.discount)

  replay_buff = ReplayBuffer(obs_space, act_space, args.timesteps)

  if algo.recurrent:
    print("Recurrent Soft Actor Critic:")
  else:
    print("Soft Actor Critic:")
  print("\tenv:            {}".format(args.env_name))
  print("\tseed:           {}".format(args.seed))
  print("\ttimesteps:      {:n}".format(args.timesteps))
  print("\tactor_lr:       {}".format(args.a_lr))
  print("\tcritic_lr:      {}".format(args.c_lr))
  print("\tdiscount:       {}".format(args.discount))
  print("\ttau:            {}".format(args.tau))
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

  if args.save_critic is None:
    args.save_critic = os.path.join(logger.dir, 'critic.pt')

  # Keep track of some statistics for each episode
  training_start = time()
  episode_start = time()
  episode_loss = 0
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
      if not algo.recurrent:
        num_updates = episode_timesteps * args.updates
      else:
        num_updates = args.updates
      for _ in range(num_updates):
        u_loss, u_steps = algo.update_policy(replay_buff, args.batch_size, traj_len=args.traj_len)
        episode_loss += u_loss / num_updates
        update_steps += u_steps

    if done:
      episode_elapsed = (time() - episode_start)
      episode_secs_per_sample = episode_elapsed / episode_timesteps
      logger.add_scalar(args.env_name + '/episode_length', episode_timesteps, iter)
      logger.add_scalar(args.env_name + '/episode_return', episode_reward, iter)
      logger.add_scalar(args.env_name + '/critic loss', episode_loss, iter)

      completion = 1 - float(timesteps) / args.timesteps
      avg_sample_r = (time() - training_start)/timesteps
      secs_remaining = avg_sample_r * args.timesteps * completion
      hrs_remaining = int(secs_remaining//(60*60))
      min_remaining = int(secs_remaining - hrs_remaining*60*60)//60

      if iter % args.eval_every == 0 and iter != 0:
        eval_reward = eval_policy(algo.behavioral_actor, eval_env, max_traj_len=args.traj_len)
        logger.add_scalar(args.env_name + '/return', eval_reward, iter)

        print("evaluation after {:4d} episodes | return: {:7.3f} | timesteps {:9n}{:100s}".format(iter, eval_reward, timesteps, ''))

        if best_reward is None or eval_reward > best_reward:
          torch.save(algo.behavioral_actor, args.save_actor)
          torch.save(algo.behavioral_critic, args.save_critic)
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
