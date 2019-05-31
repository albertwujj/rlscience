import argparse
from datetime import datetime
import numpy as np
import torch
import gym
from tqdm import tqdm

from rainbow.vec_buffer import PrioritizedReplayBuffer, VecNStepTransitioner

from rainbow.rainbow_algo import RainbowQLearn
from rainbow.rainbow_policy import Policy
from rainbow import rainbow_args as args
from env_util import atari_wrap, deepmind_wrap, channel_major_env, benchmark_env, pt_vec_envs

# short run for debugging
debugging = True
if debugging:
    args.n_step = 2
    args.learn_start=100
    args.evaluation_interval = 100
    args.evaluation_size = 10
    args.evaluation_episodes = 2
    args.T_max = int(50e2)

# Setup
device = torch.device('cuda')
torch.backends.cudnn.enabled = args.enable_cudnn
if args.seed is not None:
    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(1, 10000))
    torch.cuda.manual_seed(np.random.randint(1, 10000))

# Environment
env_name='FrostbiteNoFrameskip-v4'
def create_env(rank):
    env = gym.make(env_name)
    env = atari_wrap(env)
    env = deepmind_wrap(env)
    env = channel_major_env(env)
    env = benchmark_env(env, log_dir='env_benchmarks',rank=rank)
    return env

thunk_envs = [lambda: create_env(i) for i in range(args.num_envs)]
envs = pt_vec_envs(thunk_envs, device, args.gamma, 4)

# printing function
out_path = f'outs/rainbow_{env_name}'
f = open(out_path, "w")
def printout(w):
    f.write(w + '\n')
    print(w)

# Simple ISO 8601 timestamped logger
def log(s):
    printout('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)

# print hyperparams
printout(' ' * 26 + 'Options')
for k, v in vars(args).items():
    printout(' ' * 26 + k + ': ' + str(v))

# Agent
nn_args = {'history_length': args.history_length,
           'hidden_size': args.hidden_size, 'noisy_std': args.noisy_std}
policy = Policy(nn_args, envs.action_space.n, args.atoms, args.V_min, args.V_max, device,
                args.model)
qlearn = RainbowQLearn(policy, args.lr, args.adam_eps, args.batch_size, args.n_step, args.gamma)
PER = PrioritizedReplayBuffer(args.memory_capacity, 1)
nstep_trans_tracker = VecNStepTransitioner(args.num_envs, args.n_step, args.gamma)

priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)

T = 0
state = envs.reset()

beta = 1
for T in tqdm(range(args.T_max)):

    if T % args.replay_frequency == 0:
        policy.reset_noise()  # Draw a new set of noisy weights

    action = policy.act(state)  # Choose an action greedily (with noisy weights)
    next_state, reward, done, _ = envs.step(action)  # Step

    nstep_trans = nstep_trans_tracker.next(state, action, reward, done)
    PER.add(nstep_trans)

    # Train and test
    if T >= args.learn_start:
        transes_idx = PER.sample(args.batch_size, beta)
        transes, idx = transes_idx[:-1], transes_idx[-1]
        if T % args.replay_frequency == 0:
            for _ in range(args.num_envs):
                loss = qlearn.update(*transes)  # Train with n-step distributional double-Q learning
                PER.update_priorities(idx, loss)

        # Update target network
        if T % args.target_update == 0:
            policy.update_target_net()

    state = next_state