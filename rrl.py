import gym
import torch
import hashlib, os
from collections import OrderedDict

class color:
 BOLD   = '\033[1m\033[48m'
 END    = '\033[0m'
 ORANGE = '\033[38;5;202m'
 BLACK  = '\033[38;5;240m'


def print_logo(subtitle="", option=2):
  pass

def env_factory(path, state_est=True, mirror=False, **kwargs):
    from functools import partial

    """
    Returns an *uninstantiated* environment constructor.

    Since environments containing cpointers (e.g. Mujoco envs) can't be serialized, 
    this allows us to pass their constructors to Ray remote functions instead 
    (since the gym registry isn't shared across ray subprocesses we can't simply 
    pass gym.make() either)

    Note: env.unwrapped.spec is never set, if that matters for some reason.
    """
    if path in ['Cassie-v0', 'CassieMimic-v0', 'CassieRandomDynamics-v0']:
      from cassie import CassieEnv, CassieTSEnv, CassieIKEnv, CassieEnv_nodelta, CassieEnv_rand_dyn, CassieEnv_speed_dfreq

      if path == 'Cassie-v0':
        env_fn = partial(CassieEnv, "walking", clock_based=True, state_est=False)
      elif path == 'CassieRandomDynamics-v0':
        env_fn = partial(CassieEnv_rand_dyn, "walking", clock_based=True, state_est=False)
      elif path == 'CassieRandomDynamics-v0':
        env_fn = partial(CassieEnv_rand_dyn, "walking", clock_based=True, state_est=False)

      """
      if mirror:
          from rl.envs.wrappers import SymmetricEnv
          if state_est:
              # with state estimator
              env_fn = partial(SymmetricEnv, env_fn, mirrored_obs=[0.1, 1, 2, 3, 4, -10, -11, 12, 13, 14, -5, -6, 7, 8, 9, 15, 16, 17, 18, 19, 20, -26, -27, 28, 29, 30, -21, -22, 23, 24, 25, 31, 32, 33, 37, 38, 39, 34, 35, 36, 43, 44, 45, 40, 41, 42, 46, 47, 48], mirrored_act=[-5, -6, 7, 8, 9, -0.1, -1, 2, 3, 4])
          else:
              # without state estimator
              env_fn = partial(SymmetricEnv, env_fn, mirrored_obs=[0.1, 1, 2, 3, 4, 5, -13, -14, 15, 16, 17,
                                              18, 19, -6, -7, 8, 9, 10, 11, 12, 20, 21, 22, 23, 24, 25, -33,
                                              -34, 35, 36, 37, 38, 39, -26, -27, 28, 29, 30, 31, 32, 40, 41, 42],
                                              mirrored_act = [-5, -6, 7, 8, 9, -0.1, -1, 2, 3, 4])

      """
      return env_fn

    spec = gym.envs.registry.spec(path)
    _kwargs = spec._kwargs.copy()
    _kwargs.update(kwargs)

    try:
      if callable(spec._entry_point):
        cls = spec._entry_point(**_kwargs)
      else:
        cls = gym.envs.registration.load(spec._entry_point)
    except AttributeError:
      if callable(spec.entry_point):
        cls = spec.entry_point(**_kwargs)
      else:
        cls = gym.envs.registration.load(spec.entry_point)

    return partial(cls, **_kwargs)

def create_logger(args):
  from torch.utils.tensorboard import SummaryWriter
  """Use hyperparms to set a directory to output diagnostic files."""

  arg_dict = args.__dict__
  assert "seed" in arg_dict, \
    "You must provide a 'seed' key in your command line arguments"
  assert "logdir" in arg_dict, \
    "You must provide a 'logdir' key in your command line arguments."
  assert "env_name" in arg_dict, \
    "You must provide a 'env_name' key in your command line arguments."

  # sort the keys so the same hyperparameters will always have the same hash
  arg_dict = OrderedDict(sorted(arg_dict.items(), key=lambda t: t[0]))

  # remove seed so it doesn't get hashed, store value for filename
  # same for logging directory
  seed = str(arg_dict.pop("seed"))
  logdir = str(arg_dict.pop('logdir'))
  env_name = str(arg_dict.pop('env_name'))

  # get a unique hash for the hyperparameter settings, truncated at 10 chars
  arg_hash   = hashlib.md5(str(arg_dict).encode('ascii')).hexdigest()[0:6] + '-seed' + seed
  logdir     = os.path.join(logdir, env_name)
  output_dir = os.path.join(logdir, arg_hash)

  # create a directory with the hyperparm hash as its name, if it doesn't
  # already exist.
  os.makedirs(output_dir, exist_ok=True)

  # Create a file with all the hyperparam settings in plaintext
  info_path = os.path.join(output_dir, "experiment.info")
  file = open(info_path, 'w')
  for key, val in arg_dict.items():
      file.write("%s: %s" % (key, val))
      file.write('\n')

  logger = SummaryWriter(output_dir, flush_secs=0.1)
  print("Logging to " + color.BOLD + color.ORANGE + str(output_dir) + color.END)

  logger.dir = output_dir
  return logger

def eval_policy(policy, max_traj_len=1000, visualize=True, env_name=None):

  if env_name is None:
    env = env_factory(policy.env_name)()
  else:
    env = env_factory(env_name)()

  while True:
    state = env.reset()
    done = False
    timesteps = 0
    eval_reward = 0
    while not done and timesteps < max_traj_len:

      if hasattr(env, 'simrate'):
        start = time.time()
      
      action = policy.forward(torch.Tensor(state)).detach().numpy()
      state, reward, done, _ = env.step(action)
      if visualize:
        env.render()
      eval_reward += reward
      timesteps += 1

      if hasattr(env, 'simrate'):
        # assume 30hz (hack)
        end = time.time()
        delaytime = max(0, 1000 / 30000 - (end-start))
        time.sleep(delaytime)

    print("Eval reward: ", eval_reward)

if __name__ == "__main__":
  import sys, argparse, time, os
  parser = argparse.ArgumentParser()

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
    parser.add_argument("--timesteps",    "-t",   default=1e8, type=int)                # timesteps to run experiment ofr
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
    parser.add_argument("--hidden_size",            default=32,   type=int)       # neurons in hidden layers
    parser.add_argument("--layers",                 default=2,     type=int)      # number of hidden layres
    parser.add_argument("--timesteps",       "-t",  default=1e6,   type=int)      # number of timesteps in replay buffer
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
    parser.add_argument("--hidden_size",            default=256,   type=int)      # neurons in hidden layers
    parser.add_argument("--layers",                 default=2,     type=int)      # number of hidden layres
    parser.add_argument("--timesteps",       "-t",  default=1e6,   type=int)      # number of timesteps in replay buffer
    parser.add_argument("--start_timesteps",        default=1e4,   type=int)      # number of timesteps to generate random actions for
    parser.add_argument("--load_actor",             default=None,  type=str)      # load an actor from a .pt file
    parser.add_argument("--load_critic1",           default=None,  type=str)      # load a critic from a .pt file
    parser.add_argument("--load_critic2",           default=None,  type=str)      # load a critic from a .pt file
    parser.add_argument('--discount',               default=0.99,  type=float)    # the discount factor
    parser.add_argument('--expl_noise',             default=0.1,   type=float)    # random noise used for exploration
    parser.add_argument('--policy_noise',           default=0.2,   type=float)    # random noise used for exploration
    parser.add_argument('--noise_clip',             default=0.5,   type=float)    # random noise used for exploration
    parser.add_argument('--tau',                    default=0.005, type=float)    # update factor for target networks
    parser.add_argument("--a_lr",           "-alr", default=3e-4,  type=float)    # adam learning rate for critic
    parser.add_argument("--c_lr",           "-clr", default=3e-4,  type=float)    # adam learning rate for actor
    parser.add_argument("--traj_len",       "-tl",  default=1000,  type=int)      # max trajectory length for environment
    parser.add_argument("--center_reward",  "-r",   action='store_true')          # normalize rewards to a normal distribution
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
    raise NotImplementedError

  elif sys.argv[1] == 'eval':
    sys.argv.remove(sys.argv[1])

    parser.add_argument("--policy", default="./trained_models/ddpg/ddpg_actor.pt", type=str)
    args = parser.parse_args()

    policy = torch.load(args.policy)

    eval_policy(policy)
  else:
    print("Invalid algorithm '{}'".format(sys.argv[1]))