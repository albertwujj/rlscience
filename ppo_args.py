gamma=.99

clip_param = 0.1
ppo_epoch = 10
num_mb = 4
vloss_coef = .5
entropy_coef = 0
lr = 2.5e-4
adam_eps = 1e-5
use_gae=True
gae_lambda=.95
time_limits=True

num_envs = 8
num_steps = 128
num_processes = num_envs
total_steps = 1e6

gail_epoch = 5
gail_batchsize = 128