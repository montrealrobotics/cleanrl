# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass, field
from typing import List

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
import pickle as pkl

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
    track: bool = True
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
    autotune: str = "True"
    """automatic tuning of the entropy coefficient"""
    replay_ratio: float = 1.0
    """Number of updates per environment step. If > 1, performs multiple updates per step."""
    reset_interval: int = 200000
    """How often (in environment steps) to perform parameter resets"""
    reset_layers: List[str] = field(default_factory=lambda: ["fc1", "fc2", "fc3"])
    """Names of layers to reset (default is last layer of each network)"""
    reset_critic: bool = True
    """Whether to reset critic networks"""
    reset_actor: bool = True
    """Whether to reset actor network"""
    use_resets: str = "False"
    """Whether to use resets at all"""
    value_evaluation_period: int = 50000
    
    ## Experimental Arguments
    use_entropy_critic: str = "True"
    """Whether to use entropy term in the critic loss"""
    use_cdq: str = "True"
    """Whether to use clipped double Q-learning"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = MuJoCoStateWrapper(env)
        env.action_space.seed(seed)
        return env

    return thunk



class MuJoCoStateWrapper(gym.Env):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.saved_state = None

    def save_state(self):
        """Save the current state of the environment."""
        self.saved_state = {
            "qpos": self.env.data.qpos.copy(),
            "qvel": self.env.data.qvel.copy(),
            "cinert": self.env.data.cinert.copy(),
            "cvel": self.env.data.cvel.copy(),
            "qfrc_actuator": self.env.data.qfrc_actuator.copy(),
            "cfrc_ext": self.env.data.cfrc_ext.copy(),
        }

    def reset(self, use_saved_state=False, **kwargs):
        if use_saved_state and self.saved_state is not None:
            # Reset to the saved state
            self.env.set_state(self.saved_state["qpos"], self.saved_state["qvel"])
            self.env.data.cinert[:] = self.saved_state["cinert"]
            self.env.data.cvel[:] = self.saved_state["cvel"]
            self.env.data.qfrc_actuator[:] = self.saved_state["qfrc_actuator"]
            self.env.data.cfrc_ext[:] = self.saved_state["cfrc_ext"]
            observation = self.env.unwrapped._get_obs()
            info = {}
        else:
            observation, info = self.env.reset(**kwargs)
        return observation, info

    def step(self, action):
        return self.env.step(action)

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)

def initialize_networks(env, device):
    # Initialize the actor and critics
    actor = Actor(env).to(device)
    qf1 = SoftQNetwork(env).to(device)
    qf2 = SoftQNetwork(env).to(device)

    # Initialize target networks 
    qf1_target = SoftQNetwork(env).to(device)
    qf2_target = SoftQNetwork(env).to(device)

    # Copy parameters from critics to targets
    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
        target_param.data.copy_(param.data)
    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
        target_param.data.copy_(param.data)

    # Set target networks to eval mode
    qf1_target.eval() 
    qf2_target.eval()

    # Initialize optimizers
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    return actor, qf1, qf2, qf1_target, qf2_target, q_optimizer, actor_optimizer


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.layer_init_states = {
            'fc1': self.fc1.state_dict(),
            'fc2': self.fc2.state_dict(),
            'fc3': self.fc3.state_dict()
        }

    def reset_layers(self, layer_names):
        """Reset specified layers to their initial states"""
        for name in layer_names:
            if name in self.layer_init_states:
                getattr(self, name).load_state_dict(self.layer_init_states[name])

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        
        # Store initial states for reset
        self.layer_init_states = {
            'fc1': self.fc1.state_dict(),
            'fc2': self.fc2.state_dict(),
            'fc_mean': self.fc_mean.state_dict(),
            'fc_logstd': self.fc_logstd.state_dict()
        }
        
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def reset_layers(self, layer_names):
        layer_names = layer_names + ["fc_mean", "fc_logstd"]
        """Reset specified layers to their initial states"""
        for name in layer_names:
            if name in self.layer_init_states:
                getattr(self, name).load_state_dict(self.layer_init_states[name])

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
            settings={"_service_wait": 60}
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    args.use_resets = True if args.use_resets == "True" else False
    args.autotune = True if args.autotune == "True" else False

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor, qf1, qf2, qf1_target, qf2_target, q_optimizer, actor_optimizer = initialize_networks(envs, device)

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
    total_terminations = 0  # Initialize counter
    
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Track terminations
        total_terminations += sum(terminations)
            
        if global_step % args.value_evaluation_period == 0:
            # Evaluate reward Q-values
            results = evaluate_value_estimates(
                args, envs.envs[0], actor, qf1, qf2, device
            )
            analyze_estimation(writer, global_step, results)
            with open(os.path.join(wandb.run.dir, f"evaluation_results_{global_step}.pkl"), "wb") as f: 
                pkl.dump(results, f)
            
            wandb.save(os.path.join(wandb.run.dir, f"evaluation_results_{global_step}.pkl"))

            writer.add_scalar("ValueEstimation/mean_q_error", results["mean_q_error"], global_step)
            writer.add_scalar("ValueEstimation/mean_q_values", results["mean_q_values"], global_step)
            writer.add_scalar("ValueEstimation/mean_mc_values", results["mean_mc_values"], global_step)
            # writer.add_scalar("ValueEstimation/mean_q_error", mean_q_error, global_step)
            # writer.add_scalar("ValueEstimation/mean_q_values", mean_q_values, global_step)
            # writer.add_scalar("ValueEstimation/mean_mc_values", mean_mc_values, global_step)
            # writer.add_scalar("ValueEstimation/mean_q_mean_error", mean_q_mean_error, global_step)
            # writer.add_scalar("ValueEstimation/var_error_corr", var_error_corr, global_step)
            # writer.add_scalar("ValueEstimation/mean_var", mean_var, global_step)


            # writer.add_scalar("ValueRel/mean_q_error", mean_q_error / mean_mc_values, global_step)
            # writer.add_scalar("ValueRel/mean_q_values", mean_q_values / mean_mc_values, global_step)
            # writer.add_scalar("ValueRel/mean_mc_values", mean_mc_values / mean_mc_values, global_step)
            # writer.add_scalar("ValueRel/mean_q_mean_error", mean_q_mean_error / mean_mc_values, global_step)
            # writer.add_scalar("ValueRel/mean_var", mean_var / mean_mc_values, global_step)
            # writer.add_scalar("ValueRel/var_error_corr", var_error_corr, global_step)
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                writer.add_scalar("charts/total_terminations", total_terminations, global_step)
                writer.add_scalar("charts/episodic_terminations", np.mean(terminations), global_step)
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
            # Check if it's time to reset parameters
            if global_step % args.reset_interval == 0 and args.use_resets:
                # Reinitialize networks
                actor, qf1, qf2, qf1_target, qf2_target, q_optimizer, actor_optimizer = initialize_networks(envs, device)
                print(f"Reinitialized networks at step {global_step}")
                writer.add_scalar("charts/parameter_resets", global_step, global_step)
            
            # Calculate number of updates to perform this step
            num_updates = int(args.replay_ratio)
            # Add remaining fractional updates probabilistically    
            if random.random() < (args.replay_ratio - int(args.replay_ratio)):
                num_updates += 1
                
            for _ in range(num_updates):
                data = rb.sample(args.batch_size)
                # Q-function update
                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                    qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                    qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                    if args.use_cdq == "True":
                        if args.use_entropy_critic == "True":
                            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                        else:
                            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                    else:
                        if args.use_entropy_critic == "True":
                            min_qf_next_target = qf1_next_target - alpha * next_state_log_pi
                        else:
                            min_qf_next_target = qf1_next_target

                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

                qf1_a_values = qf1(data.observations, data.actions).view(-1)
                qf2_a_values = qf2(data.observations, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + int(args.use_cdq == "True") * qf2_loss

                # optimize the model
                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                    for _ in range(args.policy_frequency):
                        pi, log_pi, _ = actor.get_action(data.observations)
                        qf1_pi = qf1(data.observations, pi)
                        qf2_pi = qf2(data.observations, pi)
                        if args.use_cdq == "True":
                            min_qf_pi = torch.min(qf1_pi, qf2_pi)
                        else:
                            min_qf_pi = qf1_pi
                        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

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
