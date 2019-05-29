from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr import algo, envs as envslib, storage, utils
from a2c_ppo_acktr import arguments
import multiprocessing as mp
import numpy as np
import os
from collections import deque
import torch
from timeit import default_timer as timer
import ppo_args
device = torch.device("cuda:0")
import gail_util


def pg(envs, printout, use_gail=False):
    if use_gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail_util.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            '/home/paperspace/repos/pytorch-a2c-ppo-acktr-gail/gail_experts', "trajs_reacher.pt")

        gail_train_loader = torch.utils.data.DataLoader(
            gail_util.ExpertDataset(
                file_name, num_trajectories=4, subsample_step=4),
            batch_size=ppo_args.gail_batchsize,
            shuffle=True,
            drop_last=True)

    actor_critic = Policy(envs.observation_space.shape, envs.action_space)
    actor_critic.to(device)
    agent = algo.PPO(actor_critic=actor_critic, clip_param=ppo_args.clip_param, ppo_epoch=ppo_args.ppo_epoch, num_mini_batch=ppo_args.num_mb,
                     value_loss_coef=ppo_args.vloss_coef, entropy_coef=ppo_args.entropy_coef, lr=ppo_args.lr, eps=ppo_args.adam_eps, max_grad_norm=.5)


    rollouts = storage.RolloutStorage(ppo_args.num_steps, ppo_args.num_processes, envs.observation_space.shape, envs.action_space,
                                      actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    num_updates = int(ppo_args.total_steps) // ppo_args.num_steps // ppo_args.num_processes

    episode_rewards=deque(maxlen=10)
    scores = np.zeros((ppo_args.num_envs, 1))
    final_scores = np.zeros((ppo_args.num_envs, 1))
    start = timer()
    for j in range(num_updates):

        utils.update_linear_schedule(
            agent.optimizer, j, num_updates,
            ppo_args.lr)

        for step in range(ppo_args.num_steps):
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.ones_like(masks)
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        if use_gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = ppo_args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(ppo_args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], ppo_args.gamma,
                    rollouts.masks[step])

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, ppo_args.use_gae, ppo_args.gamma,
                                 ppo_args.gae_lambda, ppo_args.time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        save_path = 'saved_models'
        save_interval = 100
        # save for every interval-th update or for the last epoch
        if (j % save_interval == 0
                or j == num_updates - 1):

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, "ppo" + env_name + ".pt"))


        log_interval = 10
        if j % log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * ppo_args.num_processes * ppo_args.num_steps
            end = timer()

            printout(
                f'Updates {j}, num timesteps {total_num_steps}, FPS { int(total_num_steps / (end - start))} \n '
                f'Last {len(episode_rewards)} training episodes: mean/median reward {np.mean(episode_rewards):.1f}/{ np.median(episode_rewards):.1f}, '
                f'min/max reward {np.min(episode_rewards):.1f}/{np.max(episode_rewards):.1f}')
                #f'entropy: {dist_entropy:.3f}, action loss: {action_loss:.3f}')

env_name = 'ReacherBulletEnv-v0'
use_gail = False
prefix_gail = 'gail_' if use_gail else ''
out_path = f'outs/{prefix_gail}ppo_{env_name}'
f = open(out_path, "w")
def printout(w):
    f.write(w + '\n')
    print(w)

envs = envslib.make_vec_envs(env_name, 1957, ppo_args.num_envs, ppo_args.gamma, 'envlog', device, False)
pg(envs, printout, use_gail=use_gail)