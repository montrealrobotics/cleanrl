# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import safety_gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from utils import *


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video:
            env = gym.make(args.env_id, render_mode="rgb_array", early_termination=args.early_termination, term_cost=args.term_cost, failure_penalty=args.failure_penalty, reward_goal=args.reward_goal, reward_distance=args.reward_distance)
        else:
            env = gym.make(args.env_id, early_termination=args.early_termination, term_cost=args.term_cost, failure_penalty=args.failure_penalty, reward_goal=args.reward_goal, reward_distance=args.reward_distance)
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
    def __init__(self, envs, risk_size=0):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod()+risk_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod()+risk_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x, risk=None):
        if risk is not None:
            x = torch.cat([x, risk], axis=-1)
        return self.critic(x)

    def get_action_and_value(self, x, risk=None, action=None):
        if risk is not None:
            x = torch.cat([x, risk], axis=-1)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    # if args.track:
    import wandb

    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        sync_tensorboard=True,
        config=vars(args),
        name=run_name,
        monitor_gym=True,
        save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"


    risk_size = args.quantile_num if args.use_risk else 0


    agent = Agent(envs, risk_size).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)


    if args.use_risk:
        risk_input_size = np.array(envs.single_observation_space.shape).prod() if args.risk_input == "state" else np.array(envs.single_observation_space.shape).prod() + np.prod(envs.single_action_space.shape)
        risk_model = BayesRiskEst(risk_input_size, risk_size=risk_size)

        if os.path.exists(args.risk_model_path):
            risk_model.load_state_dict(torch.load(args.risk_model_path, map_location=device))
            print("Risk model loaded successfully")

        risk_model.to(device)

        if args.fine_tune_risk:
            ## Intializing replay buffer 
            risk_rb = ReplayBuffer()
            ## Initializing optimizer 
            opt_risk = optim.Adam(risk_model.parameters(), lr=args.risk_lr, eps=1e-5)
            ## Risk loss function 
            risk_criterion = nn.NLLLoss().to(device)

    ## Dimensions 
    action_dim = envs.single_action_space.shape[0]
    obs_dim = envs.single_observation_space.shape[0]


    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    actions_null = torch.zeros((1, action_dim)).to(device)
    ep_trans = {"obs": [None]*args.num_envs, "actions": [None]*args.num_envs}
    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                risk = risk_model(torch.cat([next_obs, actions_null], axis=-1)) if args.use_risk else None
                action, logprob, _, value = agent.get_action_and_value(next_obs, risk)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            obs_ = next_obs
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            if args.use_risk and args.fine_tune_risk:
                ## Storing episodic data to dump later
                for i in range(args.num_envs):
                    obs_i, next_obs_i, actions_i = obs_[i].unsqueeze(0), next_obs[i].unsqueeze(0), action[i].unsqueeze(0)
                    obs_i = torch.cat([obs_i, actions_i], axis=-1)
                    next_obs_i = torch.cat([next_obs_i, actions_null], axis=-1)
                    # print(obs_i.size(), next_obs_i.size(), actions_i.size())
                    ep_trans["obs"][i] = obs_i if ep_trans["obs"][i] is None else torch.cat([ep_trans["obs"][i], obs_i], axis=0) 
                    # print(ep_trans["obs"][i].size(), obs_i.size())
                    ep_trans["obs"][i] = torch.cat([ep_trans["obs"][i], next_obs_i], axis=0) 


            ## Train Risk model 
            if args.use_risk and args.fine_tune_risk:
                risk_loss = 0
                if (global_step >= args.start_risk_update) and (global_step % args.risk_update_period == 0):
                    for _ in range(args.num_risk_epochs):
                        risk_data = risk_rb.sample(args.num_risk_samples)
                        risk_loss += train_risk(args, risk_model, risk_data, risk_criterion, opt_risk, args.num_update_risk, device)
                    writer.add_scalar("risk/risk_loss", risk_loss / args.num_risk_epochs, global_step)


            # Only print when at least 1 env is done
            if "final_info" not in infos:
                continue

            for i, info in enumerate(infos["final_info"]):
                # Skip the envs that are not done
                if info is None:
                    continue
                ep_len = info["episode"]["l"]


                if args.use_risk and args.fine_tune_risk:
                    ep_risks = np.array(list(reversed(range(int(ep_len))))) if info["cost"] > 0 else np.array([int(ep_len)]*int(ep_len))
                    ep_risks = torch.repeat_interleave(torch.Tensor(ep_risks), 2).to(device).unsqueeze(1)
                    risk_rb.add(ep_trans["obs"][i], ep_risks, ep_risks)
                    ep_trans["obs"][i] = None

                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_obs_risk = risk_model(torch.cat([next_obs, torch.zeros((next_obs.size()[0], action_dim)).to(device)], axis=-1)) if args.use_risk else None 
            next_value = agent.get_value(next_obs, next_obs_risk).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        if args.use_risk:
            with torch.no_grad():
                b_risks = risk_model(torch.cat([b_obs, b_actions], axis=-1))

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                if args.use_risk:
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_risks[mb_inds], b_actions[mb_inds])
                else:
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], None, b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
