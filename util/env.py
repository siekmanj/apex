import time 
import torch
import gym

def env_factory(path, state_est=False, mirror=False, speed=None, clock_based=False, **kwargs):
    from functools import partial

    """
    Returns an *uninstantiated* environment constructor.

    Since environments containing cpointers (e.g. Mujoco envs) can't be serialized, 
    this allows us to pass their constructors to Ray remote functions instead 
    (since the gym registry isn't shared across ray subprocesses we can't simply 
    pass gym.make() either)

    Note: env.unwrapped.spec is never set, if that matters for some reason.
    """
    if 'cassie' in path.lower():
      from cassie import CassieEnv_v2
      path = path.lower()

      if 'random_dynamics' in path or 'dynamics_random' in path or 'randomdynamics' in path or 'dynamicsrandom' in path:
        dynamics_randomization = True
      else:
        dynamics_randomization = False
      
      if 'nodelta' in path or 'no_delta' in path:
        no_delta = True
      else:
        no_delta = False
      
      if 'stateest' in path or 'state_est' in path:
        state_est = True
      else:
        state_est = False

      if 'clock_based' in path or 'clockbased' in path:
        clock_based = True
      else:
        clock_based = False

      print("Created cassie env with arguments:")
      print("\tdynamics randomization: {}".format(dynamics_randomization))
      print("\tstate estimation:       {}".format(state_est))
      print("\tno delta:               {}".format(no_delta))
      print("\tclock based:            {}".format(clock_based))
      return partial(CassieEnv_v2, 'walking', clock_based=clock_based, state_est=state_est, no_delta=no_delta, dynamics_randomization=dynamics_randomization)

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

def eval_policy(policy, max_traj_len=1000, visualize=True, env_name=None):

  if env_name is None:
    env = env_factory(policy.env_name)()
  else:
    env = env_factory(env_name)()

  while True:
    env.dynamics_randomization = False
    state = env.reset()
    done = False
    timesteps = 0
    eval_reward = 0
    if hasattr(policy, 'init_hidden_state'):
      policy.init_hidden_state()

    while not done and timesteps < max_traj_len:

      if hasattr(env, 'simrate'):
        start = time.time()
      
      state = policy.normalize_state(state, update=False)
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
