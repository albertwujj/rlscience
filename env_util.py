import os
from baselines import bench
from baselines.common.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, TimeLimit, EpisodicLifeEnv, \
    FireResetEnv, WarpFrame, ScaledFloatFrame, ClipRewardEnv, FrameStack
from a2c_ppo_acktr.envs import TimeLimitMask, TransposeImage, VecNormalize, VecPyTorch, VecPyTorchFrameStack
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv


# from ikostrikov's pt baselines and OpenAI baselines Env wrappers

def atari_wrap(env, max_episode_steps=None):
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        env = TimeLimitMask(env)
    return env


def deepmind_wrap(atari_env, episode_life=True, clip_rewards=True, frame_stack=False, scale=False):
    """ matching deepmind papers
    """
    if episode_life:
        env = EpisodicLifeEnv(atari_env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env


def channel_major_env(env):
    # swap channel dim to first (which Pytorch requires)
    obs_shape = env.observation_space.shape
    if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
        env = TransposeImage(env, op=[2, 0, 1])
    return env

def benchmark_env(env, log_dir, rank, allow_early_resets=True):
    if log_dir is not None:
        env = bench.Monitor(
            env,

            os.path.join(log_dir, str(rank)),
            allow_early_resets=allow_early_resets)
    return env

def pt_vec_envs(thunk_envs,
                device,
                gamma,
                num_frame_stack,
                ):

    # make pytorch vec envs
    if len(thunk_envs) > 1:
        thunk_envs = ShmemVecEnv(thunk_envs, context='fork')
    else:
        thunk_envs = DummyVecEnv(thunk_envs)

    if len(thunk_envs.observation_space.shape) == 1:
        if gamma is None:
            thunk_envs = VecNormalize(thunk_envs, ret=False)
        else:
            thunk_envs = VecNormalize(thunk_envs, gamma=gamma)

    thunk_envs = VecPyTorch(thunk_envs, device)
    thunk_envs = VecPyTorchFrameStack(thunk_envs, num_frame_stack, device)


    return thunk_envs