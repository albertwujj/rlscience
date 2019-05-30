import argparse
from datetime import datetime
import numpy as np
import torch
import gym
from tqdm import tqdm

from rainbow.rainbow_algo import RainbowQLearn
from rainbow.rainbow_policy import Policy
from rainbow.rainbow_buffer import ReplayMemory
from rainbow.rainbow_test import test
from rainbow import rainbow_args as args


from env_util import atari_wrap, deepmind_wrap, channel_major_env, benchmark_env, pt_vec_envs

# Setup
device = torch.device('cuda')
torch.backends.cudnn.enabled = args.enable_cudnn
if args.seed is not None:
    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(1, 10000))
    torch.cuda.manual_seed(np.random.randint(1, 10000))

# Environment
env_name='PongNoFrameskip-v0'
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
policy = Policy(nn_args, envs.action_space.n, args.atoms, args.V_min, args.V_max, args.device,
                args.model)
qlearn = RainbowQLearn(policy, args.lr, args.adam_eps, args.batch_size, args.multi_step, args.gamma)
mem = ReplayMemory(args, args.memory_capacity)
priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)

# Construct validation memory
val_mem = ReplayMemory(args, args.evaluation_size)
T = 0
state = envs.reset()
while T < args.evaluation_size:
    next_state, _, done, _ = envs.step(torch.from_numpy(np.random.randint(0, envs.action_space.n, size=(args.num_envs,1))))
    val_mem.append(state, [None]*args.num_envs, [None]*args.num_envs, done, torch.zeros(84, 84, dtype=torch.uint8))
    state = next_state
    T += 1

if args.evaluate:
    policy.eval()  # Set DQN (online network) to evaluation mode
    avg_reward, avg_Q = test(args, 0, policy, val_mem, evaluate=True)  # Test
    printout('Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
else:
    state = envs.reset()
    for T in tqdm(range(args.T_max)):

        if T % args.replay_frequency == 0:
            policy.reset_noise()  # Draw a new set of noisy weights

        action = policy.act(state)  # Choose an action greedily (with noisy weights)
        next_state, reward, done, _ = envs.step(action)  # Step

        mem.append(state, action, reward, done, next_state)  # Append transition to memory

        # Train and test
        if T >= args.learn_start:
            mem.priority_weight = min(mem.priority_weight + priority_weight_increase,
                                      1)  # Anneal importance sampling weight β to 1

            if T % args.replay_frequency == 0:
                qlearn.update(mem)  # Train with n-step distributional double-Q learning

            if T % args.evaluation_interval == 0:
                policy.eval()  # Set DQN (online network) to evaluation mode
                avg_reward, avg_Q = test(args, T, policy, val_mem)  # Test
                log('T = ' + str(T) + ' / ' + str(args.T_max) + ' | Avg. reward: ' + str(
                    avg_reward) + ' | Avg. Q: ' + str(avg_Q))
                policy.train()  # Set DQN (online network) back to training mode

            # Update target network
            if T % args.target_update == 0:
                policy.update_target_net()

        state = next_state