seed = 1957
env_name='PongNoFrameskip-v4'
T_max = int(50e6)
max_episode_length = int(108e3)
history_length = 4
hidden_size = 512
noisy_std = 0.1
atoms = 51
V_min = -10
V_max = 10
model = None
memory_capacity=int(1e6)
replay_frequency = 4
priority_exponent = 0.5
priority_weight= 0.4
multi_step = 3
gamma = 0.99
target_update = 32e3
lr = 6.25e-5
adam_eps = 1.5e-4
batch_size = 32
learn_start = int(80e3)
evaluate = False
evaluation_interval = int(1e5)
evaluation_episodes = 10
evaluation_size = 500
render = False
enable_cudnn=True

num_envs = 2
