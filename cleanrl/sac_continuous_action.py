# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import safety_gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


import risk_utils as utils

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "risk_aware_exploration"
    """the wandb's project name"""
    wandb_entity: str = "kaustubh_umontreal"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the task"""
    early_termination: bool = True 
    """the environment id of the task"""
    reward_goal: float = 10.0
    """the environment id of the task"""
    reward_distance: float = 0.0
    """the environment id of the task"""
    failure_penalty: float = 0.0
    """the environment id of the task"""
    term_cost: int = 1
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""

    risk_lr: float = 1e-5
    """the environment id of the task"""
    risk_model_path: str = "None"
    """the environment id of the task"""
    risk_batch_size: int = 5000
    """the environment id of the task"""
    num_risk_epochs: int = 10 
    """the environment id of the task"""
    fine_tune_risk: str = "None"
    """the environment id of the task"""
    quantile_size: int = 4
    """the environment id of the task"""
    quantile_num: int = 10
    """the environment id of the task"""
    use_risk: bool = False
    """the environment id of the task"""
    num_risk_epochs: int = 10
    """the environment id of the task"""
    risk_update_period: int = 10000
    """the environment id of the task"""
    start_risk_update: int = 250000
    """the environment id of the task"""
    risk_data_size: int = 10000



def make_env(cfg, seed, idx, capture_video, run_name):
    def thunk():
        if "safety" in cfg.env_id:
            if capture_video and idx == 0:
                env = gym.make(cfg.env_id, render_mode="rgb_array", early_termination=cfg.early_termination, term_cost=cfg.term_cost, failure_penalty=cfg.failure_penalty, reward_goal=cfg.reward_goal, reward_distance=cfg.reward_distance)
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            else:
                env = gym.make(cfg.env_id, early_termination=cfg.early_termination, term_cost=cfg.term_cost, failure_penalty=cfg.failure_penalty, reward_goal=cfg.reward_goal, reward_distance=cfg.reward_distance)
        else:
            env = gym.make(cfg.env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env, risk_size=0):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256) if risk_size == 0 else nn.Linear(320, 256)
        self.fc3 = nn.Linear(256, 1)

        if risk_size > 0:
            self.fc_risk = nn.Linear(risk_size, 64)

    def forward(self, x, a, risk=None):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        if risk is not None:
            risk = F.relu(self.fc_risk(risk))
            x = torch.cat([x, risk], axis=-1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env, risk_size=0):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256) if risk_size == 0 else nn.Linear(320, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))

        if risk_size > 0:
            self.fc_risk = nn.Linear(risk_size, 64)
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x, risk):
        x = F.relu(self.fc1(x))
        if risk is not None:
            x = torch.cat([x, F.relu(self.fc_risk(risk))], axis=-1)
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x, risk=None):
        mean, log_std = self(x, risk)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
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
    envs = gym.vector.SyncVectorEnv([make_env(args, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])
    
    risk_size = args.quantile_num if args.use_risk else 0
    actor = Actor(envs, risk_size).to(device)
    qf1 = SoftQNetwork(envs, risk_size).to(device)
    qf2 = SoftQNetwork(envs, risk_size).to(device)
    qf1_target = SoftQNetwork(envs, risk_size).to(device)
    qf2_target = SoftQNetwork(envs, risk_size).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )

    if args.use_risk:
        risk_bins = np.array([i*args.quantile_size for i in range(args.quantile_num+1)])

        risk_model = utils.BayesRiskEst(np.array(envs.single_observation_space.shape).prod(), risk_size=args.quantile_num)
        risk_model.to(device)
        if args.fine_tune_risk:
            risk_rb = utils.ReplayBuffer()
            risk_criterion = nn.NLLLoss()
            opt_risk = optim.Adam(risk_model.parameters(), lr=args.risk_lr, eps=1e-8)
        

    start_time = time.time()

    avg_ep_goal, total_goal, total_cost, avg_ep_cost = [], 0, 0, []

    f_obs = None
    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            risk = risk_model(torch.Tensor(obs).to(device)) if args.use_risk else None
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device), risk)
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if args.use_risk and args.fine_tune_risk:
            f_obs = torch.Tensor(next_obs).reshape(1, -1) if f_obs is None else torch.cat([f_obs, torch.Tensor(next_obs).reshape(1, -1)], axis=0)
            # print(f_obs.size())
            ### Updating the risk model parameters
            if global_step >= args.start_risk_update and global_step % args.risk_update_period == 0:
                risk_data = risk_rb.sample(args.risk_data_size)
                risk_loss = utils.train_risk(args, risk_model, risk_data, risk_criterion, opt_risk, args.num_risk_epochs, device)
                

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                try:
                    avg_ep_cost.append(info["cum_cost"])
                    avg_ep_goal.append(info["cum_goal_met"])
                    total_cost += info["cum_cost"]
                    total_goal += info["cum_goal_met"]
                    writer.add_scalar("goals/Avg Ep Goal", np.mean(avg_ep_goal[-30:]), global_step)
                    writer.add_scalar("cost/Avg Ep Cost", np.mean(avg_ep_cost[-30:]), global_step)
                except:
                    pass

                if args.use_risk and args.fine_tune_risk:
                    f_dist_to_fail = range(info["episode"]["l"][0]) if terminations else [100]*info["episode"]["l"][0]
                    f_risks = torch.Tensor(np.apply_along_axis(lambda x: np.histogram(x, bins=risk_bins)[0], 1, np.expand_dims(f_dist_to_fail, 1)))
                    f_dist_to_fail = torch.Tensor(f_dist_to_fail)
                    print()
                    risk_rb.add(f_obs, f_risks, f_dist_to_fail)
                    f_obs = None

                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_risk = risk_model(data.next_observations) if args.use_risk else None
                risk = risk_model(data.observations) if args.use_risk else None
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations, next_risk)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions, next_risk)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions, next_risk)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions, risk).view(-1)
            qf2_a_values = qf2(data.observations, data.actions, risk).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations, risk)
                    qf1_pi = qf1(data.observations, pi, risk)
                    qf2_pi = qf2(data.observations, pi, risk)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations, risk)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    envs.close()
    writer.close()
