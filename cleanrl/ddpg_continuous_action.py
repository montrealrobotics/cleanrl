# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gym, safety_gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="HalfCheetah-v4",
        help="the id of the environment")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--exploration-noise", type=float, default=0.1,
        help="the scale of exploration noise")
    parser.add_argument("--learning-starts", type=int, default=25e3,
        help="timestep to start learning")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    
    ## Arguments related to risk model 
    parser.add_argument("--use-risk", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use risk model or not ")
    parser.add_argument("--risk-actor", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use risk model in the actor or not ")
    parser.add_argument("--risk-critic", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Use risk model in the critic or not ")
    parser.add_argument("--risk-model-path", type=str, default="./pretrained/no_termination/risk_ancient_sweep_4.pt",
        help="the id of the environment")
    parser.add_argument("--binary-risk", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Use risk model in the critic or not ")  
    parser.add_argument("--model-type", type=str, default="mlp",
        help="specify the NN to use for the risk model")
 
    args = parser.parse_args()
    # fmt: on
    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env, use_risk=False):
        super().__init__()
        self.use_risk = use_risk
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        if self.use_risk:
            self.fc2 = nn.Linear(268, 256)
        else:
            self.fc2 = nn.Linear(256, 256)
        
        self.fc3 = nn.Linear(256, 1)


        if self.use_risk:
            self.risk_encoder = nn.Sequential(
                layer_init(nn.Linear(2, 12)),
                nn.Tanh())
            

    def forward(self, x, a, risk=None):
        if self.use_risk:
            risk = self.risk_encoder(risk)
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        if self.use_risk:
            x = F.relu(self.fc2(torch.cat([x, risk], axis=1)))
        else:
            x = F.relu(self.fc2(x))     
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env, use_risk=False):
        super().__init__()
        self.use_risk = use_risk
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        if self.use_risk:
            self.fc2 = nn.Linear(268, 256)
        else:
            self.fc2 = nn.Linear(256, 256)
        
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )
        if self.use_risk:
            self.risk_encoder = nn.Sequential(
                layer_init(nn.Linear(2, 12)),
                nn.Tanh())

    def forward(self, x, risk=None):
        if self.use_risk:
            risk = self.risk_encoder(risk)
        x = F.relu(self.fc1(x))
        if self.use_risk:
            x = F.relu(self.fc2(torch.cat([x, risk], axis=1)))
        else:
            x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias



class RiskEst(nn.Module):
    def __init__(self, obs_size=64, fc1_size=128, fc2_size=128, fc3_size=128, fc4_size=128, out_size=2):
        super().__init__()
        self.obs_size = obs_size
        self.fc1 = nn.Linear(obs_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.fc4 = nn.Linear(fc3_size, fc4_size)
        self.out = nn.Linear(fc4_size, out_size)

        ## Batch Norm layers
        self.bnorm1 = nn.BatchNorm1d(fc1_size)
        self.bnorm2 = nn.BatchNorm1d(fc2_size)
        self.bnorm3 = nn.BatchNorm1d(fc3_size)
        self.bnorm4 = nn.BatchNorm1d(fc4_size)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.dropout(self.fc3(x)))
        x = self.relu(self.dropout(self.fc4(x)))
        out = self.softmax(self.out(x))
        return out

class BayesRiskEst(nn.Module):
    def __init__(self, obs_size=64, fc1_size=128, fc2_size=128, fc3_size=128, fc4_size=128, out_size=2):
        super().__init__()
        self.obs_size = obs_size
        self.fc1 = nn.Linear(obs_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.fc4 = nn.Linear(fc3_size, fc4_size)
        self.out = nn.Linear(fc4_size, out_size)

        ## Batch Norm layers
        self.bnorm1 = nn.BatchNorm1d(fc1_size)
        self.bnorm2 = nn.BatchNorm1d(fc2_size)
        self.bnorm3 = nn.BatchNorm1d(fc3_size)
        self.bnorm4 = nn.BatchNorm1d(fc4_size)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.2)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.bnorm1(self.relu(self.fc1(x)))
        x = self.bnorm2(self.relu(self.fc2(x)))
        x = self.bnorm3(self.relu(self.dropout(self.fc3(x))))
        x = self.bnorm4(self.relu(self.dropout(self.fc4(x))))
        out = self.logsoftmax(self.out(x))
        return out




if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    args = parse_args()
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
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    if args.model_type == "bayesian":
        risk_model_class = BayesRiskEst 
    else:
        risk_model_class = RiskEst

    print(envs.single_observation_space.shape)
    if args.use_risk:
        if os.path.exists(args.risk_model_path):
            risk_model = risk_model_class(obs_size=np.array(envs.single_observation_space.shape).prod())
            risk_model.load_state_dict(torch.load(args.risk_model_path, map_location=device))
            risk_model.to(device)
            risk_model.eval()
        else:
            raise("No model in the path specified!!")
        
    
        
    actor = Actor(env=envs, use_risk=args.use_risk).to(device)    
    qf1 = QNetwork(envs, use_risk=args.use_risk).to(device)
    qf1_target = QNetwork(envs, use_risk=args.use_risk).to(device)
    target_actor = Actor(envs, use_risk=args.use_risk).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()
    cum_cost, ep_cost, ep_risk_cost_int, cum_risk_cost_int, ep_risk, cum_risk = 0, 0, 0, 0, 0, 0

    all_costs = torch.zeros((args.total_timesteps, args.num_envs)).to(device)
    all_risks = torch.zeros((args.total_timesteps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset()
    cost = 0 
    last_step = 0
    risk = None
    for global_step in range(args.total_timesteps):
        with torch.no_grad():
            if args.use_risk:
                risk = risk_model(torch.Tensor(obs).to(device))
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device), risk)
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminateds, truncateds, infos = envs.step(actions)
        done = np.logical_or(terminateds, truncateds)

        all_costs[global_step] = cost
        if risk is not None:
            all_risks[global_step] = risk

        if not done:
            cost = torch.Tensor(infos["cost"]).to(device).view(-1)
        else:
            cost = torch.Tensor(np.array([infos["final_info"][0]["cost"]])).to(device).view(-1)

        infos["risk"] = risk

        # TRY NOT TO MODIFY: record rewards for plotting purposes

        if "final_info" in infos:
            for info in infos["final_info"]:

                ep_cost = torch.sum(all_costs[last_step:global_step]).item()
                cum_cost += ep_cost

                print(f"global_step={global_step}, episodic_return={info['episode']['r']}, episodic_cost={ep_cost}")

                if args.use_risk:
                    ep_risk = torch.sum(all_risks[last_step:global_step]).item()
                    cum_risk += ep_risk

                    risk_cost_int = torch.logical_and(all_costs[last_step:global_step], all_risks[last_step:global_step])
                    ep_risk_cost_int = torch.sum(risk_cost_int).item()
                    cum_risk_cost_int += ep_risk_cost_int


                    writer.add_scalar("charts/episodic_risk", ep_risk, global_step)
                    writer.add_scalar("charts/cummulative_risk", cum_risk, global_step)
                    writer.add_scalar("charts/episodic_risk_&&_cost", ep_risk_cost_int, global_step)
                    writer.add_scalar("charts/cummulative_risk_&&_cost", cum_risk_cost_int, global_step)
        
                    print(f"global_step={global_step}, ep_Risk_cost_int={ep_risk_cost_int}, cum_Risk_cost_int={cum_risk_cost_int}")
                    print(f"global_step={global_step}, episodic_risk={ep_risk}, cum_risks={cum_risk}, cum_costs={cum_cost}")


                print("-----------------------------------------------------------------------------------------------------------------")
                print()
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                writer.add_scalar("charts/episodic_cost", ep_cost, global_step)
                writer.add_scalar("charts/cummulative_cost", cum_cost, global_step)

                last_step = global_step
                # print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                # writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                # writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(truncateds):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminateds, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                if args.use_risk:
                    next_risks = risk_model(data.next_observations)
                else:
                    next_risks = None
                next_state_actions = target_actor(data.next_observations, next_risks)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions, next_risks)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (qf1_next_target).view(-1)
           
                if args.use_risk:
                    risks = risk_model(data.observations)
                else:
                    risks = None
            
            qf1_a_values = qf1(data.observations, data.actions, risks).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

            # optimize the model
            q_optimizer.zero_grad()
            qf1_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(data.observations, actor(data.observations, risks), risks).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
