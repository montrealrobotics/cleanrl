# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import safety_gymnasium
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from comet_ml import Experiment

import hydra
import os

def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


def train(cfg):
    batch_size = int(cfg.ppo.num_envs * cfg.ppo.num_steps)
    minibatch_size = int(batch_size // cfg.ppo.num_minibatches)
    # fmt: on

    run_name = f"{cfg.ppo.env_id}__{cfg.ppo.exp_name}__{cfg.ppo.seed}__{int(time.time())}"

    experiment = Experiment(
        api_key="FlhfmY238jUlHpcRzzuIw3j2t",
        project_name="risk-aware-exploration",
        workspace="hbutsuak95",
    )


    # TRY NOT TO MODIFY: seeding
    random.seed(cfg.ppo.seed)
    np.random.seed(cfg.ppo.seed)
    torch.manual_seed(cfg.ppo.seed)
    torch.backends.cudnn.deterministic = cfg.ppo.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.ppo.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(cfg.env_id, i, cfg.ppo.capture_video, run_name, cfg.ppo.gamma) for i in range(cfg.ppo.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.ppo.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((cfg.ppo.num_steps, cfg.ppo.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((cfg.ppo.num_steps, cfg.ppo.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((cfg.ppo.num_steps, cfg.ppo.num_envs)).to(device)
    rewards = torch.zeros((cfg.ppo.num_steps, cfg.ppo.num_envs)).to(device)
    dones = torch.zeros((cfg.ppo.num_steps, cfg.ppo.num_envs)).to(device)
    values = torch.zeros((cfg.ppo.num_steps, cfg.ppo.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=cfg.ppo.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(cfg.ppo.num_envs).to(device)
    num_updates = cfg.ppo.total_timesteps // batch_size
    cum_cost, ep_cost = np.array([0.]), np.array([0.])

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if cfg.ppo.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * cfg.ppo.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, cfg.ppo.num_steps):
            global_step += 1 * cfg.ppo.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            if not done:
                cost = torch.Tensor(infos["cost"]).to(device).view(-1)
                ep_cost += infos["cost"]; cum_cost += infos["cost"]
            else:
                cost = torch.Tensor(np.array([infos["final_info"][0]["cost"]])).to(device).view(-1)
                ep_cost += np.array([infos["final_info"][0]["cost"]]); cum_cost += np.array([infos["final_info"][0]["cost"]])

            # Only print when at least 1 env is done
            if "final_info" not in infos:
                continue

            for info in infos["final_info"]:
                # Skip the envs that are not done
                if info is None:
                    continue
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}, episode_cost={ep_cost}")
                experiment.log_metric("charts/episodic_return", info["episode"]["r"], global_step)
                experiment.log_metric("charts/episodic_length", info["episode"]["l"], global_step)
                experiment.log_metric("charts/episodic_cost", ep_cost[0], global_step)
                experiment.log_metric("charts/cummulative_cost", cum_cost[0], global_step)
                ep_cost = np.array([0.])

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(cfg.ppo.num_steps)):
                if t == cfg.ppo.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + cfg.ppo.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + cfg.ppo.gamma * cfg.ppo.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(cfg.ppo.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > cfg.ppo.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if cfg.ppo.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.ppo.clip_coef, 1 + cfg.ppo.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if cfg.ppo.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -cfg.ppo.clip_coef,
                        cfg.ppo.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ppo.ent_coef * entropy_loss + v_loss * cfg.ppo.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.ppo.max_grad_norm)
                optimizer.step()

            if cfg.ppo.target_kl != "None":
                if approx_kl > cfg.ppo.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        experiment.log_metric("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        experiment.log_metric("losses/value_loss", v_loss.item(), global_step)
        experiment.log_metric("losses/policy_loss", pg_loss.item(), global_step)
        experiment.log_metric("losses/entropy", entropy_loss.item(), global_step)
        experiment.log_metric("losses/old_approx_kl", old_approx_kl.item(), global_step)
        experiment.log_metric("losses/approx_kl", approx_kl.item(), global_step)
        experiment.log_metric("losses/clipfrac", np.mean(clipfracs), global_step)
        experiment.log_metric("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        experiment.log_metric("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
    return 1 

if __name__ == "__main__":
    train()

