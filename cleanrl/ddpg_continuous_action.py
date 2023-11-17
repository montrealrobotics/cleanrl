# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import panda_gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer
from stable_baselines3 import HerReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.vec_env import DummyVecEnv

from src.models.risk_models import *
from src.datasets.risk_datasets import *
from src.utils import * 

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
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument("--upload-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to upload the saved model to huggingface")
    parser.add_argument("--hf-entity", type=str, default="",
        help="the user or org name of the model repository from the Hugging Face Hub")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="HalfCheetah-v4",
        help="the id of the environment")
    parser.add_argument("--collect-data", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="store data while trianing")
    parser.add_argument("--storage-path", type=str, default="./data/ddpg/",
        help="the storage path for the data collected")
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
    # parser.add_argument("--weight", type=float, default=1.0,
    #     help="weight for the risk model")
    
    ## Arguments related to risk model 
    parser.add_argument("--use-risk", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Use risk model or not ")
    parser.add_argument("--risk-actor", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use risk model in the actor or not ")
    parser.add_argument("--risk-critic", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Use risk model in the critic or not ")
    parser.add_argument("--risk-model-path", type=str, default="None",
        help="the id of the environment")
    parser.add_argument("--binary-risk", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Use risk model in the critic or not ")
    parser.add_argument("--model-type", type=str, default="bayesian",
        help="specify the NN to use for the risk model")
    parser.add_argument("--risk-bnorm", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--risk-type", type=str, default="quantile",
        help="whether the risk is binary or continuous")
    parser.add_argument("--fear-radius", type=int, default=5,
        help="fear radius for training the risk model")
    parser.add_argument("--num-risk-datapoints", type=int, default=1000,
        help="fear radius for training the risk model")
    parser.add_argument("--risk-update-period", type=int, default=20000,
        help="how frequently to update the risk model")
    parser.add_argument("--num-risk-epochs", type=int, default=10,
        help="number of sgd steps to update the risk model")
    parser.add_argument("--num-update-risk", type=int, default=10,
        help="number of sgd steps to update the risk model")
    parser.add_argument("--risk-lr", type=float, default=1e-7,
        help="the learning rate of the optimizer")
    parser.add_argument("--risk-batch-size", type=int, default=100,
        help="number of epochs to update the risk model")
    parser.add_argument("--fine-tune-risk", type=str, default="None",
        help="fine tune risk by which method")
    parser.add_argument("--finetune-risk-online", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--start-risk-update", type=int, default=1000,
        help="number of epochs to update the risk model") 
    parser.add_argument("--num-risk-samples", type=int, default=20000,
        help="number of epochs to update the risk model") 
    parser.add_argument("--rb-type", type=str, default="simple",
        help="which type of replay buffer to use for ")
    parser.add_argument("--freeze-risk-layers", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--weight", type=float, default=1.0, 
        help="weight for the 1 class in BCE loss")
    parser.add_argument("--quantile-size", type=int, default=2, help="size of the risk quantile ")
    parser.add_argument("--quantile-num", type=int, default=5, help="number of quantiles to make")

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


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        print(env.observation_space)
        self.obs_fc = nn.Linear(np.array(env.observation_space["observation"].shape).prod(), 64)
        self.ag_fc = nn.Linear(np.array(env.observation_space["achieved_goal"].shape).prod(), 12)
        self.dg_fc = nn.Linear(np.array(env.observation_space["desired_goal"].shape).prod(), 12)

        self.fc1 = nn.Linear(88 + np.prod(env.action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a, risk=None):
        obs, ag, dg = self.obs_fc(x["observation"].float()), self.ag_fc(x["achieved_goal"].float()), self.dg_fc(x["desired_goal"].float())
        x = torch.cat([obs, ag, dg], 1)
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class QRiskNetwork(nn.Module):
    def __init__(self, env, risk_size=2):
        super().__init__()
        print(env.observation_space)
        self.obs_fc = nn.Linear(np.array(env.observation_space["observation"].shape).prod(), 64)
        self.ag_fc = nn.Linear(np.array(env.observation_space["achieved_goal"].shape).prod(), 12)
        self.dg_fc = nn.Linear(np.array(env.observation_space["desired_goal"].shape).prod(), 12)
        self.risk_fc = nn.Linear(risk_size, 12)

        self.fc1 = nn.Linear(100 + np.prod(env.action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a, risk):
        obs, ag, dg, risk = self.obs_fc(x["observation"].float()), self.ag_fc(x["achieved_goal"].float()), self.dg_fc(x["desired_goal"].float()), self.risk_fc(risk.float())
        x = torch.cat([obs, ag, dg, risk], 1)
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.obs_fc = nn.Linear(np.array(env.observation_space["observation"].shape).prod(), 64)
        self.ag_fc = nn.Linear(np.array(env.observation_space["achieved_goal"].shape).prod(), 12)
        self.dg_fc = nn.Linear(np.array(env.observation_space["desired_goal"].shape).prod(), 12)
        self.fc1 = nn.Linear(88, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x, risk=None):
        obs, ag, dg = self.obs_fc(x["observation"].float()), self.ag_fc(x["achieved_goal"].float()), self.dg_fc(x["desired_goal"].float())
        x = torch.cat([obs, ag, dg], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias

class RiskActor(nn.Module):
    def __init__(self, env, risk_size=2):
        super().__init__()
        self.obs_fc = nn.Linear(np.array(env.observation_space["observation"].shape).prod(), 64)
        self.ag_fc = nn.Linear(np.array(env.observation_space["achieved_goal"].shape).prod(), 12)
        self.dg_fc = nn.Linear(np.array(env.observation_space["desired_goal"].shape).prod(), 12)
        self.risk_fc = nn.Linear(risk_size, 12)
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x, risk):
        obs, ag, dg, risk = self.obs_fc(x["observation"].float()), self.ag_fc(x["achieved_goal"].float()), self.dg_fc(x["desired_goal"].float()), self.risk_fc(risk.float())
        x = torch.cat([obs, ag, dg, risk], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias



def convert_dict_to_tensor(data, device):
    new_data = {}
    for key in data.keys():
        new_data[key] = torch.Tensor(data[key]).to(device).double()
    return new_data

if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    args = parse_args()
    # run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    # if args.track:
    import wandb

    run = wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        sync_tensorboard=True,
        config=vars(args),
        monitor_gym=True,
        save_code=True,
    )
    run_name = run.name
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    risk_bins = np.array([i*args.quantile_size for i in range(args.quantile_num)])
    args.use_risk = False if args.risk_model_path == "None" else True 
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = DummyVecEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])

    # envs = gym.wrappers.RecordEpisodeStatistics(gym.make(args.env_id)) #SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.action_space, gym.spaces.Box), "only continuous action space is supported"

    risk_model_class = {"bayesian": {"continuous": BayesRiskEstCont, "binary": BayesRiskEst, "quantile": BayesRiskEst}, 
                        "mlp": {"continuous": RiskEst, "binary": RiskEst}} 

    risk_size_dict = {"continuous": 1, "binary": 2, "quantile": args.quantile_num}
    risk_size = risk_size_dict[args.risk_type]
    if args.fine_tune_risk != "None":
        if args.rb_type == "balanced":
            risk_rb = ReplayBufferBalanced(buffer_size=args.total_timesteps)
        else:
            risk_rb = ReplayBuffer(buffer_size=args.total_timesteps)
                          #, observation_space=envs.single_observation_space, action_space=envs.single_action_space)
        if args.risk_type == "quantile":
            weight_tensor = torch.Tensor([1]*args.quantile_num).to(device)
            weight_tensor[0] = args.weight
        elif args.risk_type == "binary":
            weight_tensor = torch.Tensor([1., args.weight]).to(device)
        if args.model_type == "bayesian":
            criterion = nn.NLLLoss(weight=weight_tensor)
        else:
            criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)


    if args.collect_data:
        storage_path = os.path.join(args.storage_path, args.env_id, run_name)
        make_dirs(storage_path, 0)

    action_size = np.prod(envs.action_space.shape)
    if args.use_risk:
        print("using risk")
        #if args.risk_type == "binary":
        actor = RiskActor(envs, risk_size).to(device)
        qf1 = QNetwork(envs).to(device)
        qf1_target = QNetwork(envs).to(device)
        target_actor = RiskActor(envs, risk_size).to(device)
        #Risk model
        risk_model = risk_model_class[args.model_type][args.risk_type](obs_size=np.array(envs.observation_space["observation"].shape).prod(), batch_norm=True, out_size=risk_size,\
                                                                     model_type="state_risk")
        if os.path.exists(args.risk_model_path):
            risk_model.load_state_dict(torch.load(args.risk_model_path, map_location=device))
            print("Pretrained risk model loaded successfully")

        risk_model.to(device)
        risk_model.eval()
        if args.fine_tune_risk != "None":
            # print("Fine Tuning risk")
            ## Freezing all except last layer of the risk model
            if args.freeze_risk_layers:
                for param in risk_model.parameters():
                    param.requires_grad = False 
                risk_model.out.weight.requires_grad = True
                risk_model.out.bias.requires_grad = True 
            opt_risk = optim.Adam(filter(lambda p: p.requires_grad, risk_model.parameters()), lr=args.risk_lr, eps=1e-10)
            risk_model.eval()
        else:
            print("No model in the path specified!!")
    else:
        actor = Actor(envs).to(device)
        qf1 = QNetwork(envs).to(device)
        qf1_target = QNetwork(envs).to(device)
        target_actor = Actor(envs).to(device)

    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    envs.observation_space.dtype = np.float32
    rb = HerReplayBuffer(
        args.buffer_size,
        envs.observation_space,
        envs.action_space,
        envs,
        device,
        handle_timeout_termination=False,
        goal_selection_strategy="future",
    )
    # print(envs.observation_space, envs.action_space)
    # rb = ReplayBuffer(
    #     args.buffer_size,
    #     envs.observation_space,
    #     envs.action_space,
    #     device,
    #     handle_timeout_termination=False,
    # )

    ## Finetuning data collection 
    f_obs = [None]*args.num_envs
    f_next_obs = [None]*args.num_envs
    f_risks = [None]*args.num_envs
    f_ep_len = [0]
    f_actions = [None]*args.num_envs
    f_rewards = [None]*args.num_envs
    f_dones = [None]*args.num_envs
    f_costs = [None]*args.num_envs
    f_risks_quant = [None]*args.num_envs
    start_time = time.time()
    num_successes, total_cost = 0, 0
    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    score, success_rate = [], []
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = envs.action_space.sample().reshape(1, 3)
        else:
            with torch.no_grad():
                risk = risk_model(torch.Tensor(obs["observation"]).to(device)) if args.use_risk else None
                actions = actor(convert_dict_to_tensor(obs, device), risk)
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                actions = actions.cpu().numpy().clip(envs.action_space.low, envs.action_space.high)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        if (args.fine_tune_risk != "None" and args.use_risk) or args.collect_data:
            for i in range(args.num_envs):
                f_obs[i] = torch.Tensor(obs["observation"][i]).unsqueeze(0).to(device) if f_obs[i] is None else torch.concat([f_obs[i], torch.Tensor(obs["observation"][i]).unsqueeze(0).to(device)], axis=0)
                f_next_obs[i] = torch.Tensor(next_obs["observation"][i]).unsqueeze(0).to(device) if f_next_obs[i] is None else torch.concat([f_next_obs[i], torch.Tensor(next_obs["observation"][i]).unsqueeze(0).to(device)], axis=0)
                f_actions[i] = torch.Tensor(actions[i]).unsqueeze(0).to(device) if f_actions[i] is None else torch.concat([f_actions[i], torch.Tensor(actions[i]).unsqueeze(0).to(device)], axis=0)
                f_obs[i] = torch.Tensor(next_obs["observation"][i]).unsqueeze(0).to(device) if f_obs[i] is None else torch.concat([f_obs[i], torch.Tensor(next_obs["observation"][i]).unsqueeze(0).to(device)], axis=0)
                f_actions[i] = torch.concat([f_actions[i], torch.zeros(action_size).unsqueeze(0).to(device)], axis=0)

                # f_rewards[i] = reward[i].unsqueeze(0).to(device) if f_rewards[i] is None else torch.concat([f_rewards[i], reward[i].unsqueeze(0).to(device)], axis=0)
                # # f_risks = risk_ if f_risks is None else torch.concat([f_risks, risk_], axis=0)
                # f_costs[i] = cost[i].unsqueeze(0).to(device) if f_costs[i] is None else torch.concat([f_costs[i], cost[i].unsqueeze(0).to(device)], axis=0)
                # f_dones[i] = next_done[i].unsqueeze(0).to(device) if f_dones[i] is None else torch.concat([f_dones[i], next_done[i].unsqueeze(0).to(device)], axis=0)

        # if args.fine_tune_risk == "off" and args.use_risk:
        #     if args.use_risk and (global_step > args.learning_starts and args.fine_tune_risk != "None") and total_cost % args.risk_update_period == 0:
        #         for epoch in range(args.num_risk_epochs):
        #             if args.finetune_risk_online:
        #                 print("I am online")
        #                 data = risk_rb.slice_data(-args.risk_batch_size*args.num_update_risk, 0)
        #             else:
        #                 print(args.risk_batch_size*args.num_update_risk)
        #                 data = risk_rb.sample(args.risk_batch_size*args.num_update_risk)
        #             risk_loss = train_risk(risk_model, data, criterion, opt_risk, device)
        #         writer.add_scalar("risk/risk_loss", risk_loss, global_step)         

        if args.fine_tune_risk == "off" and args.use_risk and global_step >= args.start_risk_update:
            if global_step % args.risk_update_period == 0:
                risk_data = risk_rb.sample(args.num_risk_samples)
                risk_dataset = RiskyDataset(risk_data["next_obs"].to('cpu'), None, risk_data["dist_to_fail"].to('cpu'), False, risk_type=args.risk_type,
                                        fear_clip=None, fear_radius=args.fear_radius, one_hot=True, quantile_size=args.quantile_size, quantile_num=args.quantile_num)
                risk_dataloader = DataLoader(risk_dataset, batch_size=args.risk_batch_size, shuffle=True)

                risk_loss = train_risk(risk_model, risk_dataloader, criterion, opt_risk, args.num_risk_epochs, device)
                writer.add_scalar("risk/risk_loss", risk_loss, global_step)
                risk_model.eval()
                risk_data, risk_dataset, risk_dataloader = None, None, None
                
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for i, info in enumerate(infos):
            if 'episode' in info.keys():
                ep_len = info["episode"]["l"]
                total_cost += infos[0]["cost"]
                # num_successes += int(infos[0]["cum_goal_met"])
                success_rate.append(int(infos[0]["is_success"]))
                score.append(info["episode"]['r'])
                if args.fine_tune_risk != "None":
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}, Replay buffer size = {len(risk_rb)}, Total cost={info['cost']}, Success Rate={np.mean(success_rate[-100:])}")
                else:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}, Ep Cost = {info['cost']}, Success Rate={np.mean(success_rate[-100:])}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                writer.add_scalar("charts/Total Success", num_successes, global_step)
                writer.add_scalar("charts/Success rate", np.mean(success_rate[-100:]), global_step)
                writer.add_scalar("charts/Avg. Return", np.mean(score[-100:]), global_step)
                writer.add_scalar("cost/ep_cost", info["cum_cost"], global_step)
                writer.add_scalar("cost/total_cost", total_cost, global_step)

                e_risks = torch.Tensor(np.array(list(reversed(range(int(ep_len))))) if info["cost"] > 0 else np.array([int(ep_len)]*int(ep_len))).repeat_interleave(2).numpy()
                # print(risks.size())
                e_risks_quant = torch.Tensor(np.apply_along_axis(lambda x: np.histogram(x, bins=risk_bins)[0], 1, np.expand_dims(e_risks, 1)))
                e_risks = torch.Tensor(e_risks)

                if args.use_risk and args.fine_tune_risk != "None":
                    if args.risk_type == "binary":
                        risk_rb.add(f_obs[i], f_next_obs[i], f_actions[i], None, None, None, (e_risks <= args.fear_radius).float(), e_risks.unsqueeze(1))
                    else:
                        risk_rb.add(f_obs[i], f_next_obs[i], f_actions[i], None, None, None, e_risks_quant, e_risks.unsqueeze(1))
                f_obs[i], f_next_obs[i], f_actions[i] = None, None, None

                f_risks[i] = e_risks if f_risks[i] is None else torch.concat([f_risks[i], e_risks], axis=0)
                ## Save all the data
                if args.collect_data:
                    torch.save(f_obs[i], os.path.join(storage_path, "obs.pt"))
                    torch.save(f_actions[i], os.path.join(storage_path, "actions.pt"))
                    torch.save(f_costs[i], os.path.join(storage_path, "costs.pt"))
                    torch.save(f_risks[i], os.path.join(storage_path, "risks.pt"))
                    # torch.save(torch.Tensor(f_ep_len), os.path.join(storage_path, "ep_len.pt"))
                    #make_dirs(storage_path, episode)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs = infos[idx]["terminal_observation"]
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_risk = risk_model(data.next_observations["observation"]) if args.use_risk else None
                next_state_actions = target_actor(data.next_observations, next_risk)
                
                # next_risk_q = risk_model(data.next_observations["observation"]) if args.use_risk else None
                qf1_next_target = qf1_target(data.next_observations, next_state_actions, next_risk)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (qf1_next_target).view(-1)

            risk_q = risk_model(data.observations["observation"]) if args.use_risk else None
            qf1_a_values = qf1(data.observations, data.actions, risk_q).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

            # optimize the model
            q_optimizer.zero_grad()
            qf1_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                with torch.no_grad():
                    risks = risk_model(data.observations["observation"]) if args.use_risk else None
                actions = actor(data.observations, risks)
                
                actor_loss = -qf1(data.observations, actions, risks).mean()
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

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save((actor.state_dict(), qf1.state_dict()), model_path)
        wandb.save(model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.ddpg_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=(Actor, QNetwork),
            device=device,
            exploration_noise=args.exploration_noise,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "DDPG", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
