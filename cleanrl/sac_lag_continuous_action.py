# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from utils import *

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
    track: str = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "conservatism_in_rl"
    """the wandb's project name"""
    wandb_entity: str = "kaustubh95"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
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
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""

    # Reset arguments
    replay_ratio: int = 1
    """the ratio of updates to data in the replay buffer"""
    use_resets: str = "False"
    """whether to use resets"""
    reset_interval: int = 200000
    """the interval at which to reset the environment"""

    # Safety arguments
    failure_penalty: float = 0.0
    """Penalty applied when the agent terminates"""
    lambda_lr: float = 1e-3
    """Learning rate for the Lagrangian multiplier"""
    cost_limit: float = 0.05
    """Maximum acceptable cost (failure rate)"""
    safety_alpha: float = 0.2
    """Entropy regularization coefficient for the safety critic."""
    pid_kp: float = 0.1
    """Proportional gain for the PID controller"""
    pid_ki: float = 0.00001
    """Integral gain for the PID controller"""
    pid_kd: float = 0.00001
    """Derivative gain for the PID controller"""
    lambda_init: float = 1.0
    """Initial value for the Lagrangian multiplier"""

    ## Overestimation / Underestimation
    value_evaluation_period: int = 100000
    """Evaluate Q-values every x steps  """
    use_cdq: bool = True
    """Whether to use CDQ or not"""
    use_entropy_critic: bool = False
    """Whether to use entropy critic or not"""
    independent_q_reward: str = "False"
    """Whether to train reward Q-networks independently"""
    independent_q_safety: str = "False"
    """Whether to train safety Q-networks independently"""
    


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SafetyQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)  # Output is the expected cost
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5

class PIDLambdaController:
    def __init__(self, kp=0.1, ki=0.01, kd=0.001, setpoint=0.0, min_lambda=0.0, max_lambda=float('inf'), lambda_init=0.0):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain 
        self.kd = kd  # Derivative gain
        self.setpoint = setpoint  # Target cost limit
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        
        # Internal state
        self.prev_error = 0.0
        self.integral = 0.0
        self.lambda_value = lambda_init
        
    def update(self, current_cost):
        # print(f"current_cost: {current_cost}")
        # Calculate error
        error = current_cost - self.setpoint
        
        # Update integral term
        self.integral += error
        
        # Calculate derivative term
        derivative = error - self.prev_error
        self.prev_error = error
        
        # print(f"error: {error}")
        # print(f"derivative: {derivative}")
        # print(f"integral: {self.integral}")
        # PID formula
        lambda_value = (self.kp * error + 
                       self.ki * self.integral + 
                       self.kd * derivative)
        
        # Clamp lambda between min and max values
        self.lambda_value = self.lambda_value + max(self.min_lambda, min(self.max_lambda, lambda_value))
        
        return self.lambda_value

class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
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

    args.independent_q_reward = args.independent_q_reward == "True"
    args.independent_q_safety = args.independent_q_safety == "True"

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    args.use_resets = args.use_resets == "True"
    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Safety components
    safety_qf1 = SafetyQNetwork(envs).to(device)
    safety_qf2 = SafetyQNetwork(envs).to(device)
    safety_qf1_target = SafetyQNetwork(envs).to(device)
    safety_qf2_target = SafetyQNetwork(envs).to(device)
    safety_qf1_target.load_state_dict(safety_qf1.state_dict())
    safety_qf2_target.load_state_dict(safety_qf2.state_dict())
    safety_q_optimizer = optim.Adam(list(safety_qf1.parameters()) + list(safety_qf2.parameters()), lr=args.q_lr)

    lambda_controller = PIDLambdaController(kp=args.pid_kp, ki=args.pid_ki, kd=args.pid_kd, setpoint=args.cost_limit, min_lambda=0.0, max_lambda=float('inf'), lambda_init=args.lambda_init)
    lambda_value = args.lambda_init
    # # Lagrangian dual variable
    # log_lambda = torch.zeros(1, requires_grad=True, device=device)
    # lambda_optimizer = optim.Adam([log_lambda], lr=args.lambda_lr)
    # lambda_value = log_lambda.exp().item()

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
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    total_failures = 0  
    episode_failures = []
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Modify the reward based on termination
        costs = terminations.astype(float)  # Cost is 1 if terminated, 0 otherwise
        rewards = np.where(terminations, args.failure_penalty, rewards)

        total_failures += np.sum(terminations)
        
        # TRY NOT TO MODIFY: record rewards for plotting purposes'
        if "final_info" in infos:
            for info in infos["final_info"]:
                episode_failures.append(np.sum(terminations))
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}, episode_failures={np.mean(episode_failures[-10:])}, lambda_value={lambda_value}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                writer.add_scalar("charts/total_failures", total_failures, global_step)
                writer.add_scalar("charts/episode_failures", np.sum(terminations), global_step)
                writer.add_scalar("Lagrange/lambda", lambda_value, global_step)
                if False:
                    lambda_value = lambda_controller.update(np.mean(episode_failures[-10:])) 
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)


        if global_step % args.value_evaluation_period == 0:
            # Evaluate reward Q-values
            evaluate_value_estimates(global_step, writer, args, envs.envs[0], actor, qf1, qf2, device, safety_qf1, safety_qf2, safety_mode="both", num_episodes=100, max_steps=1000)
        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        if args.use_resets and global_step % args.reset_interval == 0:
            actor = Actor(envs).to(device)
            qf1 = SoftQNetwork(envs).to(device)
            qf2 = SoftQNetwork(envs).to(device)
            qf1_target = SoftQNetwork(envs).to(device)
            qf2_target = SoftQNetwork(envs).to(device)
            qf1_target.load_state_dict(qf1.state_dict())
            qf2_target.load_state_dict(qf2.state_dict())
            q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
            actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)
            safety_qf1 = SafetyQNetwork(envs).to(device)
            safety_qf2 = SafetyQNetwork(envs).to(device)
            safety_qf1_target = SafetyQNetwork(envs).to(device)
            safety_qf2_target = SafetyQNetwork(envs).to(device)
            safety_qf1_target.load_state_dict(safety_qf1.state_dict())
            safety_qf2_target.load_state_dict(safety_qf2.state_dict())
            safety_q_optimizer = optim.Adam(list(safety_qf1.parameters()) + list(safety_qf2.parameters()), lr=args.q_lr)

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            for _ in range(args.replay_ratio):  ## Update as many times as the replay ratio
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                    qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                    qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

                    # Safety critic target
                    safety_qf1_next_target = safety_qf1_target(data.next_observations, next_state_actions)
                    safety_qf2_next_target = safety_qf2_target(data.next_observations, next_state_actions)
                    min_safety_qf_next_target = torch.max(safety_qf1_next_target, safety_qf2_next_target) #- args.safety_alpha * next_state_log_pi  # Consider safety_alpha

                    next_cost_value = data.dones.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_safety_qf_next_target).view(-1)

                qf1_a_values = qf1(data.observations, data.actions).view(-1)
                qf2_a_values = qf2(data.observations, data.actions).view(-1)
                if args.independent_q_reward:
                    next_q1_values = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * qf1_next_target.view(-1)
                    next_q2_values = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * qf2_next_target.view(-1)
                    qf1_loss = F.mse_loss(qf1_a_values, next_q1_values)
                    qf2_loss = F.mse_loss(qf2_a_values, next_q2_values)
                else:
                    qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                    qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                # Safety critic loss
                safety_qf1_a_values = safety_qf1(data.observations, data.actions).view(-1)
                safety_qf2_a_values = safety_qf2(data.observations, data.actions).view(-1)

                if args.independent_q_safety:
                    next_cost1_value = data.dones.flatten() + (1 - data.dones.flatten()) * args.gamma * safety_qf1_next_target.view(-1)
                    next_cost2_value = data.dones.flatten() + (1 - data.dones.flatten()) * args.gamma * safety_qf2_next_target.view(-1)
                    safety_qf1_loss = F.mse_loss(safety_qf1_a_values, next_cost1_value)
                    safety_qf2_loss = F.mse_loss(safety_qf2_a_values, next_cost2_value)
                else:
                    safety_qf1_loss = F.mse_loss(safety_qf1_a_values, next_cost_value)
                    safety_qf2_loss = F.mse_loss(safety_qf2_a_values, next_cost_value)
                safety_q_loss = safety_qf1_loss + safety_qf2_loss

                # Optimize the Q networks
                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                safety_q_optimizer.zero_grad()
                safety_q_loss.backward()
                safety_q_optimizer.step()

                if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                    for _ in range(
                        args.policy_frequency
                    ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                        pi, log_pi, _ = actor.get_action(data.observations)
                        qf1_pi = qf1(data.observations, pi)
                        qf2_pi = qf2(data.observations, pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi)
                        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                        # Safety component in actor loss
                        safety_qf1_pi = safety_qf1(data.observations, pi)
                        safety_qf2_pi = safety_qf2(data.observations, pi)
                        min_safety_qf_pi = torch.max(safety_qf1_pi, safety_qf2_pi)

                        actor_loss += lambda_value * min_safety_qf_pi.mean()  # Lagrangian term

                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        actor_optimizer.step()

                        if args.autotune:
                            with torch.no_grad():
                                _, log_pi, _ = actor.get_action(data.observations)
                            alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                            a_optimizer.zero_grad()
                            alpha_loss.backward()
                            a_optimizer.step()
                            alpha = log_alpha.exp().item()

                # Update the target networks
                if global_step % args.target_network_frequency == 0:
                    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                    # Update safety critic target networks
                    for param, target_param in zip(safety_qf1.parameters(), safety_qf1_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(safety_qf2.parameters(), safety_qf2_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                # Lagrangian dual variable update
                # with torch.no_grad():
                # cost = safety_qf1_a_values.mean()  # Estimate the cost
                # lambda_loss = -log_lambda * (cost.detach() - args.cost_limit)

                # lambda_optimizer.zero_grad()
                # lambda_loss.backward()
                # lambda_optimizer.step()
                # lambda_value = log_lambda.exp().item()
                # lambda_value = max(0, lambda_value)  # Enforce non-negativity
            # print(f"episode_failures: {np.mean(episode_failures[-10:])}")
            
            # print(f"lambda_value: {lambda_value}")

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                writer.add_scalar("losses/lambda", lambda_value, global_step)
                writer.add_scalar("losses/safety_qf1_loss", safety_qf1_loss.item(), global_step)
                writer.add_scalar("losses/safety_qf2_loss", safety_qf2_loss.item(), global_step)
                writer.add_scalar("losses/safety_q_loss", safety_q_loss.item() / 2.0, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    envs.close()
    writer.close()
