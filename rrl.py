import os

import torch
import hashlib
from collections import OrderedDict

from util.env import env_factory, eval_policy

class color:
 BOLD   = '\033[1m\033[48m'
 END    = '\033[0m'
 ORANGE = '\033[38;5;202m'
 BLACK  = '\033[38;5;240m'


def print_logo(subtitle="", option=2):
  pass

if __name__ == "__main__":
  import sys, argparse, time, os
  parser = argparse.ArgumentParser()
  parser.add_argument("--nolog",      action='store_true')

  print_logo(subtitle="Maintained by Oregon State University's Dynamic Robotics Lab")

  if len(sys.argv) < 2:
    print("Usage: python apex.py [algorithm name]", sys.argv)

  elif sys.argv[1] == 'ars':
    """
      Utility for running Augmented Random Search.

    """
    from algos.ars import run_experiment
    sys.argv.remove(sys.argv[1])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hidden_size",          default=32, type=int)                 # neurons in hidden layer
    parser.add_argument("--timesteps",    "-t",   default=1e8, type=float)                # timesteps to run experiment ofr
    parser.add_argument("--load_model",   "-l",   default=None, type=str)               # load a model from a saved file.
    parser.add_argument('--std',          "-sd",  default=0.0075, type=float)           # the standard deviation of the parameter noise vectors
    parser.add_argument("--deltas",       "-d",   default=64, type=int)                 # number of parameter noise vectors to use
    parser.add_argument("--lr",           "-lr",  default=0.01, type=float)             # the learning rate used to update policy
    parser.add_argument("--reward_shift", "-rs",  default=1, type=float)                # the reward shift (to counter Gym's alive_bonus)
    parser.add_argument("--traj_len",     "-tl",  default=1000, type=int)               # max trajectory length for environment
    parser.add_argument("--algo",         "-a",   default='v1', type=str)               # whether to use ars v1 or v2
    parser.add_argument("--normalize"     '-n',   action='store_true')                  # normalize states online
    parser.add_argument("--recurrent",    "-r",   action='store_true')                  # whether to use a recurrent policy
    parser.add_argument("--logdir",               default="./logs/ars/", type=str)
    parser.add_argument("--seed",     "-s",       default=0, type=int)
    parser.add_argument("--env_name", "-e",       default="Hopper-v3")
    parser.add_argument("--average_every",        default=10, type=int)
    parser.add_argument("--save_model",   "-m",   default=None, type=str)               # where to save the trained model to
    parser.add_argument("--redis",                default=None)
    args = parser.parse_args()
    run_experiment(args)

  elif sys.argv[1] == 'ddpg' or sys.argv[1] == 'rdpg':

    if sys.argv[1] == 'ddpg':
      recurrent = False
    if sys.argv[1] == 'rdpg':
      recurrent = True

    sys.argv.remove(sys.argv[1])
    """
      Utility for running Recurrent/Deep Deterministic Policy Gradients.
    """
    from algos.dpg import run_experiment
    parser.add_argument("--timesteps",       "-t",  default=1e6,   type=float)      # number of timesteps in replay buffer
    parser.add_argument("--start_timesteps",        default=1e4,   type=int)      # number of timesteps to generate random actions for
    parser.add_argument("--load_actor",             default=None,  type=str)      # load an actor from a .pt file
    parser.add_argument("--load_critic",            default=None,  type=str)      # load a critic from a .pt file
    parser.add_argument('--discount',               default=0.99,  type=float)    # the discount factor
    parser.add_argument('--expl_noise',             default=0.2,   type=float)    # random noise used for exploration
    parser.add_argument('--tau',                    default=0.01, type=float)     # update factor for target networks
    parser.add_argument("--a_lr",           "-alr", default=1e-5,  type=float)    # adam learning rate for critic
    parser.add_argument("--c_lr",           "-clr", default=1e-4,  type=float)    # adam learning rate for actor
    parser.add_argument("--traj_len",       "-tl",  default=1000,  type=int)      # max trajectory length for environment
    parser.add_argument("--center_reward",  "-r",   action='store_true')          # normalize rewards to a normal distribution
    parser.add_argument("--normc_init",             default=True,  type=bool)     # using col norm to init weights
    parser.add_argument("--normalize"       '-n',   action='store_true')          # normalize states online
    parser.add_argument("--batch_size",             default=64,    type=int)      # batch size for policy update
    parser.add_argument("--updates",                default=1,    type=int)       # (if recurrent) number of times to update policy per episode
    parser.add_argument("--eval_every",             default=100,   type=int)      # how often to evaluate the trained policy
    parser.add_argument("--save_actor",             default=None, type=str)
    parser.add_argument("--save_critic",            default=None, type=str)

    if not recurrent:
      parser.add_argument("--logdir",                 default="./logs/ddpg/", type=str)
    else:
      parser.add_argument("--logdir",                 default="./logs/rdpg/", type=str)

    parser.add_argument("--seed",     "-s",   default=0, type=int)
    parser.add_argument("--env_name", "-e",   default="Hopper-v3")
    args = parser.parse_args()

    args.recurrent = recurrent

    run_experiment(args)

  elif sys.argv[1] == 'td3' or sys.argv[1] == 'rtd3':

    if sys.argv[1] == 'td3':
      recurrent = False
    if sys.argv[1] == 'rtd3':
      recurrent = True

    sys.argv.remove(sys.argv[1])
    """
      Utility for running Twin-Delayed Deep Deterministic policy gradients.

    """
    from algos.td3 import run_experiment
    parser.add_argument("--timesteps",       "-t",  default=1e6,   type=float)    # number of timesteps in replay buffer
    parser.add_argument("--start_timesteps",        default=1e4,   type=float)      # number of timesteps to generate random actions for
    parser.add_argument("--load_actor",             default=None,  type=str)      # load an actor from a .pt file
    parser.add_argument("--load_critic1",           default=None,  type=str)      # load a critic from a .pt file
    parser.add_argument("--load_critic2",           default=None,  type=str)      # load a critic from a .pt file
    parser.add_argument('--discount',               default=0.99,  type=float)    # the discount factor
    parser.add_argument('--expl_noise',             default=0.1,   type=float)    # random noise used for exploration
    parser.add_argument('--max_action',             default=1.0,   type=float)    # 
    parser.add_argument('--policy_noise',           default=0.2,   type=float)    # 
    parser.add_argument('--noise_clip',             default=0.5,   type=float)    # 
    parser.add_argument('--tau',                    default=0.005, type=float)    # update factor for target networks
    parser.add_argument("--a_lr",           "-alr", default=3e-4,  type=float)    # adam learning rate for critic
    parser.add_argument("--c_lr",           "-clr", default=3e-4,  type=float)    # adam learning rate for actor
    parser.add_argument("--traj_len",       "-tl",  default=1000,  type=int)      # max trajectory length for environment
    parser.add_argument("--center_reward",  "-r",   action='store_true')          # normalize rewards to a normal distribution
    parser.add_argument("--normc_init",             default=True,  type=bool)     # using col norm to init weights
    parser.add_argument("--normalize",              action='store_true')          # normalize states online
    parser.add_argument("--batch_size",             default=256,    type=int)     # batch size for policy update
    parser.add_argument("--updates",                default=1,    type=int)       # (if recurrent) number of times to update policy per episode
    parser.add_argument("--update_every",           default=2,    type=int)       # how many episodes to skip before updating
    parser.add_argument("--eval_every",             default=100,   type=int)      # how often to evaluate the trained policy
    parser.add_argument("--save_actor",             default=None, type=str)
    parser.add_argument("--save_critics",           default=None, type=str)

    if not recurrent:
      parser.add_argument("--logdir",                 default="./logs/td3/", type=str)
    else:
      parser.add_argument("--logdir",                 default="./logs/rtd3/", type=str)

    parser.add_argument("--seed",     "-s",   default=0, type=int)
    parser.add_argument("--env_name", "-e",   default="Hopper-v3")
    args = parser.parse_args()
    args.recurrent = recurrent

    run_experiment(args)

  elif sys.argv[1] == 'ppo':
    sys.argv.remove(sys.argv[1])
    """
      Utility for running Proximal Policy Optimization.

    """
    from algos.ppo import run_experiment
    parser.add_argument("--seed",                   default=0,             type=int)      # number of timesteps to run experiment for
    parser.add_argument("--timesteps",       "-t",  default=1e6,           type=float)      # number of timesteps to run experiment for
    parser.add_argument("--env_name",               default='Cassie-v0',   type=str)
    parser.add_argument("--traj_len",       "-tl",  default=400,           type=int)      # max trajectory length for environment
    parser.add_argument("--prenormalize_steps",     default=10000,         type=int)      
    parser.add_argument("--num_steps",              default=5000,          type=int)      
    parser.add_argument("--recurrent",              action='store_true')

    parser.add_argument('--discount',               default=0.99,          type=float)    # the discount factor
    parser.add_argument('--std',                    default=0.13,          type=float)    # the fixed exploration std
    parser.add_argument("--a_lr",           "-alr", default=1e-4,          type=float)    # adam learning rate for actor
    parser.add_argument("--c_lr",           "-clr", default=1e-4,          type=float)    # adam learning rate for critic
    parser.add_argument("--eps",            "-ep",  default=1e-5,          type=float)    # adam eps
    parser.add_argument("--kl",                     default=0.02,          type=float)    # kl abort threshold
    parser.add_argument("--entropy_coeff",          default=0.0,           type=float)
    parser.add_argument("--grad_clip",              default=0.05,          type=float)
    parser.add_argument("--batch_size",             default=64,            type=int)      # batch size for policy update
    parser.add_argument("--epochs",                 default=3,             type=int)      # number of updates per iter

    parser.add_argument("--save_actor",             default=None,          type=str)
    parser.add_argument("--save_critic",            default=None,          type=str)
    parser.add_argument("--logdir",                 default="./logs/ppo/", type=str)
    parser.add_argument("--workers",                default=4,             type=int)
    parser.add_argument("--redis",                  default=None,          type=str)
    args = parser.parse_args()

    run_experiment(args)

  elif sys.argv[1] == 'eval':
    sys.argv.remove(sys.argv[1])

    parser.add_argument("--policy", default="./trained_models/ddpg/ddpg_actor.pt", type=str)
    parser.add_argument("--env_name", default=None, type=str)
    parser.add_argument("--traj_len", default=400, type=int)
    args = parser.parse_args()

    policy = torch.load(args.policy)

    eval_policy(policy, env_name=args.env_name, max_traj_len=args.traj_len)

  elif sys.argv[1] == 'cassie':
    sys.argv.remove(sys.argv[1])
    from cassie.udp import run_udp

    parser.add_argument("--policy", default='logs/ppo/Cassie-nodelta-stateest-clockbased/bcbc77-seed0/actor.pt', type=str)
    args = parser.parse_args()

    run_udp(args)
  else:
    print("Invalid algorithm '{}'".format(sys.argv[1]))
