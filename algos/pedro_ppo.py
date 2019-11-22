"""Proximal Policy Optimization (clip objective)."""
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import kl_divergence

import time

import numpy as np
import os

import ray

PEDRO = False

from policies.critic import Critic, FF_V
from policies.actor import Actor, FF_Stochastic_Actor

class PPOBuffer:
    """
    A buffer for storing trajectory data and calculating returns for the policy
    and critic updates.

    This container is intentionally not optimized w.r.t. to memory allocation
    speed because such allocation is almost never a bottleneck for policy 
    gradient. 
    
    On the other hand, experience buffers are a frequent source of
    off-by-one errors and other bugs in policy gradient implementations, so
    this code is optimized for clarity and readability, at the expense of being
    (very) marginally slower than some other implementations. 

    (Premature optimization is the root of all evil).
    """
    def __init__(self, gamma=0.99, lam=0.95, use_gae=False):
        self.states  = []
        self.actions = []
        self.rewards = []
        self.values  = []
        self.returns = []

        self.ep_returns = [] # for logging
        self.ep_lens    = []

        self.gamma, self.lam = gamma, lam

        self.ptr, self.path_idx = 0, 0
    
    def __len__(self):
        return len(self.states)

    def storage_size(self):
        return len(self.states)

    def store(self, state, action, reward, value):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        self.states  += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values  += [value]

        self.ptr += 1
    
    def finish_path(self, last_val=None):
        if last_val is None:
            last_val = np.zeros(shape=(1,))

        path = slice(self.path_idx, self.ptr)
        rewards = self.rewards[path]

        returns = []

        R = last_val.squeeze(0).copy() # Avoid copy?
        for reward in reversed(rewards):
            R = self.gamma * R + reward
            returns.insert(0, R) # TODO: self.returns.insert(self.path_idx, R) ? 
                                 # also technically O(k^2), may be worth just reversing list
                                 # BUG? This is adding copies of R by reference (?)

        self.returns += returns

        self.ep_returns += [np.sum(rewards)]
        self.ep_lens    += [len(rewards)]

        self.path_idx = self.ptr
    
    def get(self):
        return(
            self.states,
            self.actions,
            self.returns,
            self.values
        )

class PPO:
    def __init__(self, 
                 args=None,
                 gamma=None, 
                 lam=None, 
                 lr=None, 
                 eps=None,
                 entropy_coeff=None,
                 clip=None,
                 epochs=None,
                 minibatch_size=None,
                 num_steps=None):

        self.env_name = args['env_name']

        self.gamma         = args['gamma']
        self.lam           = args['lam']
        self.lr            = args['lr']
        self.eps           = args['eps']
        self.entropy_coeff = args['entropy_coeff']
        self.clip          = args['clip']
        self.minibatch_size    = args['minibatch_size']
        self.epochs        = args['epochs']
        self.num_steps     = args['num_steps']
        self.max_traj_len  = args['max_traj_len']

        self.name = args['policy_name']
        self.use_gae = args['use_gae']
        self.n_proc = args['num_procs']

        self.grad_clip = args['max_grad_norm']

        self.max_return = 0

        self.total_steps = 0
        self.highest_reward = -1

        if args['redis_address'] is not None:
            ray.init(redis_address=args['redis_address'])
        else:
            ray.init()

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--n_itr", type=int, default=10000,
                            help="Number of iterations of the learning algorithm")
        
        parser.add_argument("--lr", type=float, default=3e-4,
                            help="Adam learning rate")

        parser.add_argument("--eps", type=float, default=1e-5,
                            help="Adam epsilon (for numerical stability)")
        
        parser.add_argument("--lam", type=float, default=0.95,
                            help="Generalized advantage estimate discount")

        parser.add_argument("--gamma", type=float, default=0.99,
                            help="MDP discount")
        
        parser.add_argument("--entropy_coeff", type=float, default=0.0,
                            help="Coefficient for entropy regularization")

        parser.add_argument("--clip", type=float, default=0.2,
                            help="Clipping parameter for PPO surrogate loss")

        parser.add_argument("--minibatch_size", type=int, default=64,
                            help="Batch size for PPO updates")

        parser.add_argument("--epochs", type=int, default=10,
                            help="Number of optimization epochs per PPO update")

        parser.add_argument("--num_steps", type=int, default=5096,
                            help="Number of sampled timesteps per gradient estimate")

        parser.add_argument("--use_gae", type=bool, default=True,
                            help="Whether or not to calculate returns using Generalized Advantage Estimation")

        parser.add_argument("--num_procs", type=int, default=1,
                            help="Number of threads to train on")

        parser.add_argument("--max_grad_norm", type=float, default=0.5,
                            help="Value to clip gradients at.")

        parser.add_argument("--max_traj_len", type=int, default=1000,
                            help="Max episode horizon")

    def save(self, policy):

        save_path = os.path.join("./trained_models", "ppo")

        try:
            os.makedirs(save_path)
        except OSError:
            pass

        filetype = ".pt" # pytorch model
        torch.save(policy, os.path.join("./trained_models", self.name + filetype))

    @torch.no_grad()
    def sample(self, env_fn, policy, critic, min_steps, max_traj_len, deterministic=False):
        """
        Sample at least min_steps number of total timesteps, truncating 
        trajectories only if they exceed max_traj_len number of timesteps
        """
        env = env_fn()

        memory = PPOBuffer(self.gamma, self.lam)

        num_steps = 0
        while num_steps < min_steps:
            state = torch.Tensor(env.reset())

            done = False
            value = 0
            traj_len = 0

            while not done and traj_len < max_traj_len:
                state = torch.Tensor(state)
                norm_state = policy.normalize_state(state, update=False)
                action = policy(norm_state, deterministic)
                value = critic(norm_state)
                next_state, reward, done, _ = env.step(action.numpy())

                reward = np.array([reward])

                memory.store(state.numpy(), action.numpy(), reward, value.numpy())

                state = next_state

                traj_len += 1
                num_steps += 1

            value = critic(torch.Tensor(state))
            memory.finish_path(last_val=(not done) * value.numpy())

        return memory

    def train(self,
              env_fn,
              policy,
              policy_copy,
              critic,
              n_itr,
              logger=None):

        old_policy = policy_copy

        optimizer = optim.Adam(policy.parameters(), lr=self.lr, eps=self.eps)
        critic_optimizer = optim.Adam(critic.parameters(), lr=self.lr, eps=self.eps)

        start_time = time.time()

        for itr in range(n_itr):
            print("********** Iteration {} ************".format(itr))

            sample_start = time.time()
            batch = self.sample(env_fn, policy, critic, self.num_steps, self.max_traj_len)

            print("time elapsed: {:.2f} s".format(time.time() - start_time))
            print("sample time elapsed: {:.2f} s".format(time.time() - sample_start))

            observations, actions, returns, values = map(torch.Tensor, batch.get())

            print("RETURNS SIZE: {}, VALUES SIZE {}".format(returns.size(), values.size()))
            advantages = returns - values

            print("NUM ADVANTAGES:", len(advantages), advantages.numel(), advantages.size())
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)

            minibatch_size = self.minibatch_size or advantages.numel()

            print("timesteps in batch: %i" % advantages.numel())
            self.total_steps += advantages.numel()

            old_policy.load_state_dict(policy.state_dict())  # WAY faster than deepcopy

            optimizer_start = time.time()
            
            for _ in range(self.epochs):
                losses = []
                sampler = BatchSampler(
                    SubsetRandomSampler(range(advantages.numel())),
                    minibatch_size,
                    drop_last=True
                )

                for indices in sampler:
                    indices = torch.LongTensor(indices)

                    obs_batch = observations[indices]
                    action_batch = actions[indices]

                    return_batch = returns[indices]
                    advantage_batch = advantages[indices]

                    # TODO, move this outside loop?
                    with torch.no_grad():
                      obs_batch = policy.normalize_state(obs_batch, update=False)
                      old_pdf = old_policy.pdf(obs_batch)
                      old_log_probs = old_pdf.log_prob(action_batch).sum(-1, keepdim=True)

                    values = critic(obs_batch)
                    pdf = policy.pdf(obs_batch)

                    
                    log_probs = pdf.log_prob(action_batch).sum(-1, keepdim=True)
                    
                    ratio = (log_probs - old_log_probs).exp()

                    cpi_loss = ratio * advantage_batch
                    clip_loss = ratio.clamp(1.0 - self.clip, 1.0 + self.clip) * advantage_batch
                    actor_loss = -torch.min(cpi_loss, clip_loss).mean()

                    if not torch.isfinite(actor_loss) or actor_loss.item() > 100:
                      print("NON FINITE ACTR LOSS {}, old logs {}, logs {}, adv {}".format(actor_loss, old_log_probs, log_probs, advantage_batch))
                      print("ratio {}, cpi {}, clip {}".format(ratio, cpi_loss, clip_loss))
                      exit(1)

                    critic_loss = 0.5 * (return_batch - values).pow(2).mean()

                    entropy_penalty = -self.entropy_coeff * pdf.entropy().mean()

                    # TODO: add ability to optimize critic and actor seperately, with different learning rates

                    optimizer.zero_grad()
                    (actor_loss + entropy_penalty).backward()

                    # Clip the gradient norm to prevent "unlucky" minibatches from 
                    # causing pathalogical updates
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), self.grad_clip)
                    optimizer.step()

                    critic_optimizer.zero_grad()
                    critic_loss.backward()

                    # Clip the gradient norm to prevent "unlucky" minibatches from 
                    # causing pathalogical updates
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), self.grad_clip)
                    critic_optimizer.step()


                    losses.append([actor_loss.item(),
                                   pdf.entropy().mean().item(),
                                   critic_loss.item(),
                                   ratio.mean().item(),
                                   advantage_batch.mean().item()])

                # TODO: add verbosity arguments to suppress this
                print(' '.join(["%g"%x for x in np.mean(losses, axis=0)]))

                # Early stopping 
                if kl_divergence(pdf, old_pdf).mean() > 0.02:
                    print("Max kl reached, stopping optimization early.")
                    break

            print("optimizer time elapsed: {:.2f} s".format(time.time() - optimizer_start))        


            if logger is not None:
                evaluate_start = time.time()
                test = self.sample(env_fn, policy, critic, 800, self.max_traj_len, deterministic=True)
                print("evaluate time elapsed: {:.2f} s".format(time.time() - evaluate_start))
                avg_eval_reward = np.mean(test.ep_returns)

                pdf     = policy.pdf(observations)
                old_pdf = old_policy.pdf(observations)

                entropy = pdf.entropy().mean().item()
                kl = kl_divergence(pdf, old_pdf).mean().item()

                #print(losses)
                print("EVAL ", avg_eval_reward)
                logger.add_scalar("Cassie-v0/ppo/iteration_return", avg_eval_reward, itr)
                logger.add_scalar("Cassie-v0/ppo/kl", kl, itr)
                logger.add_scalar("Cassie-v0/ppo/entropy", entropy, itr)

            # TODO: add option for how often to save model
            if self.highest_reward < avg_eval_reward:
                self.highest_reward = avg_eval_reward
                self.save(policy)

def run_experiment(args):
    torch.set_num_threads(1) # see: https://github.com/pytorch/pytorch/issues/13757

    from rrl import env_factory, create_logger

    # wrapper function for creating parallelized envs
    env_fn = env_factory(args.env_name, state_est=args.state_est, mirror=args.mirror)
    obs_dim = env_fn().observation_space.shape[0]
    action_dim = env_fn().action_space.shape[0]

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    policy = FF_Stochastic_Actor(obs_dim, action_dim, env_name=args.env_name, fixed_std=torch.ones(action_dim)*np.exp(-2))
    policy_copy = FF_Stochastic_Actor(obs_dim, action_dim, env_name=args.env_name, fixed_std=torch.ones(action_dim)*np.exp(-2))
    critic = FF_V(obs_dim)

    steps = 0
    env = env_fn()
    while steps < args.input_norm_steps: # Prenormalize
      state = torch.Tensor(env.reset())
      done = False
      traj_len = 0
      while not done and traj_len < 400:
        state = policy.normalize_state(state)
        action = policy(state, deterministic=False)
        action += torch.randn(action.size())
        next_state, reward, done, _ = env.step(action.numpy())
        state = torch.Tensor(next_state)
        traj_len += 1
        steps += 1
      print("Prenormalizing {:5d}/{:5d}".format(steps, args.input_norm_steps), end="\r")
    print()

    policy.train(0)
    policy_copy.train(0)
    critic.train(0)

    print("obs_dim: {}, action_dim: {}".format(obs_dim, action_dim))

    if args.mirror:
        algo = MirrorPPO(args=vars(args))
    else:
        algo = PPO(args=vars(args))

    # create a tensorboard logging object
    logger = create_logger(args)

    print()
    print("Synchronous Distributed Proximal Policy Optimization:")
    print("\tenv:            {}".format(args.env_name))
    print("\tmax traj len:   {}".format(args.max_traj_len))
    print("\tseed:           {}".format(args.seed))
    print("\tmirror:         {}".format(args.mirror))
    print("\tnum procs:      {}".format(args.num_procs))
    print("\tlr:             {}".format(args.lr))
    print("\teps:            {}".format(args.eps))
    print("\tlam:            {}".format(args.lam))
    print("\tgamma:          {}".format(args.gamma))
    print("\tentropy coeff:  {}".format(args.entropy_coeff))
    print("\tclip:           {}".format(args.clip))
    print("\tminibatch size: {}".format(args.minibatch_size))
    print("\tepochs:         {}".format(args.epochs))
    print("\tnum steps:      {}".format(args.num_steps))
    print("\tuse gae:        {}".format(args.use_gae))
    print("\tmax grad norm:  {}".format(args.max_grad_norm))
    print("\tmax traj len:   {}".format(args.max_traj_len))
    print()

    algo.train(env_fn, policy, policy_copy, critic, args.n_itr, logger=logger)
