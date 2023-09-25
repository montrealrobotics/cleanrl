# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
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
import torch.nn.functional as F
import torch.optim as optim
import stable_baselines3.common.buffers as buffers # import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

import gymnasium.spaces as spaces
from src.utils import * 
from src.models.risk_models import *
from src.datasets.risk_datasets import * 


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
        help="the user or org name of the model repository from the Hugging Face Hub")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="SafetyCarGoal1Gymnasium-v0",
        help="the id of the environment")
    parser.add_argument("--early-termination", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to terminate early i.e. when the catastrophe has happened")
    parser.add_argument("--term-cost", type=int, default=1,
        help="how many violations before you terminate")
    parser.add_argument("--failure-penalty", type=float, default=0.0,
        help="Reward Penalty when you fail")
    parser.add_argument("--action-scale", type=float, default=0.2,
        help="Reward Penalty when you fail")    
    parser.add_argument("--collect-data", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="store data while trianing")
    parser.add_argument("--storage-path", type=str, default="./data/ppo/term_1",
        help="the storage path for the data collected")
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--buffer-size", type=int, default=10000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.,
        help="the target network update rate")
    parser.add_argument("--target-network-frequency", type=int, default=500,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=128,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=10000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=10,
        help="the frequency of training")
    
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
    parser.add_argument("--model-type", type=str, default="mlp",
        help="specify the NN to use for the risk model")
    parser.add_argument("--risk-bnorm", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--risk-type", type=str, default="binary",
        help="whether the risk is binary or continuous")
    parser.add_argument("--fear-radius", type=int, default=5,
        help="fear radius for training the risk model")
    parser.add_argument("--num-risk-datapoints", type=int, default=1000,
        help="fear radius for training the risk model")
    parser.add_argument("--risk-update-period", type=int, default=1000,
        help="how frequently to update the risk model")
    parser.add_argument("--num-update-risk", type=int, default=10,
        help="number of sgd steps to update the risk model")
    parser.add_argument("--risk-lr", type=float, default=1e-7,
        help="the learning rate of the optimizer")
    parser.add_argument("--risk-batch-size", type=int, default=1000,
        help="number of epochs to update the risk model")
    parser.add_argument("--fine-tune-risk", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--finetune-risk-online", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--start-risk-update", type=int, default=10000,
        help="number of epochs to update the risk model") 
    parser.add_argument("--rb-type", type=str, default="simple",
        help="which type of replay buffer to use for ")
    parser.add_argument("--freeze-risk-layers", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--weight", type=float, default=1.0, 
        help="weight for the 1 class in BCE loss")
    parser.add_argument("--quantile-size", type=int, default=4, help="size of the risk quantile ")
    parser.add_argument("--quantile-num", type=int, default=5, help="number of quantiles to make")

    args = parser.parse_args()
    # fmt: on
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"

    return args


def make_env(cfg, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(cfg.env_id, render_mode="rgb_array", early_termination=cfg.early_termination, term_cost=cfg.term_cost, failure_penalty=cfg.failure_penalty)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(cfg.env_id, early_termination=cfg.early_termination, term_cost=cfg.term_cost, failure_penalty=cfg.failure_penalty)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env, action_size=2, risk_size=0):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod() + risk_size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, action_size),
        )

    def forward(self, x, risk=None):
        if risk is not None:
            x = torch.cat([x, risk], axis=-1)
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)



def train_risk(cfg, model, data, criterion, opt, device):
    model.train()
    dataset = RiskyDataset(data["next_obs"].to('cpu'), data["actions"].to('cpu'), data["risks"].to('cpu'), False, risk_type=cfg.risk_type,
                            fear_clip=None, fear_radius=cfg.fear_radius, one_hot=True, quantile_size=cfg.quantile_size, quantile_num=cfg.quantile_num)
    dataloader = DataLoader(dataset, batch_size=cfg.risk_batch_size, shuffle=True, num_workers=10, generator=torch.Generator(device='cpu'))
    net_loss = 0
    for batch in dataloader:
        pred = model(get_risk_obs(cfg, batch[0]).to(device))
        if cfg.model_type == "mlp":
            loss = criterion(pred, batch[1].squeeze().to(device))
        else:
            loss = criterion(pred, torch.argmax(batch[1].squeeze(), axis=1).to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()

        net_loss += loss.item()

    model.eval()
    return net_loss

def get_risk_obs(cfg, next_obs):
    if "goal" in cfg.risk_model_path.lower():
        if "push" in cfg.env_id.lower():
            #print("push")
            next_obs_risk = next_obs[:, :-16]
        elif "button" in cfg.env_id.lower():
            #print("button")
            next_obs_risk = next_obs[:, list(range(24)) + list(range(40, 88))]
        else:
            next_obs_risk = next_obs
    elif "button" in cfg.risk_model_path.lower():
        if "push" in cfg.env_id.lower():
            #print("push")
            next_obs_risk = next_obs[:, list(range(24)) + list(range(72, 88)) + list(range(24, 72))]
        elif "goal" in cfg.env_id.lower():
            #print("button")
            next_obs_risk = next_obs[:, list(range(24)) + list(range(24, 40)) + list(range(24, 72))]
        else:
            next_obs_risk = next_obs
    elif "push" in cfg.risk_model_path.lower():
        if "button" in cfg.env_id.lower():
            #print("push")
            next_obs_risk = next_obs[:, list(range(24)) + list(range(72, 88)) + list(range(24, 72))]
        elif "goal" in cfg.env_id.lower():
            #print("button")
            next_obs_risk = next_obs[:, :-16]
        else:
            next_obs_risk = next_obs
    else:
        next_obs_risk = next_obs
    return next_obs_risk        



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
    action_map = {0: np.array([args.action_scale, args.action_scale]), 1: np.array([-args.action_scale, args.action_scale]), 2: np.array([args.action_scale, -args.action_scale]), 3: np.array([-args.action_scale, -args.action_scale])}
    def action_map_fn(action):
        return action_map[action]

    def get_random_action():
        return random.choice(range(len(action_map)))
        

    if args.track:
        import wandb

        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True
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
        [make_env(args, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    # assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"




    args.use_risk = False if args.risk_model_path == "None" else True 

    risk_model_class = {"bayesian": {"continuous": BayesRiskEstCont, "binary": BayesRiskEst, "quantile": BayesRiskEst}, 
                        "mlp": {"continuous": RiskEst, "binary": RiskEst}} 

    risk_size_dict = {"continuous": 1, "binary": 2, "quantile": args.quantile_num}
    risk_size = risk_size_dict[args.risk_type] if args.use_risk else 0
    if args.fine_tune_risk:
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

    if "goal" in args.risk_model_path.lower():
        risk_obs_size = 72 
    elif args.risk_model_path == "scratch":
        risk_obs_size = np.array(envs.single_observation_space.shape).prod()
    else:
        risk_obs_size = 88



    if args.use_risk:
        risk_model = risk_model_class[args.model_type][args.risk_type](obs_size=risk_obs_size, batch_norm=True, out_size=risk_size)
        if os.path.exists(args.risk_model_path):
            risk_model.load_state_dict(torch.load(args.risk_model_path, map_location=device))
            print("Pretrained risk model loaded successfully")

        risk_model.to(device)
        risk_model.eval()
        if args.fine_tune_risk:
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



    q_network = QNetwork(envs, action_size=len(action_map), risk_size=risk_size).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs, action_size=len(action_map), risk_size=risk_size).to(device)
    target_network.load_state_dict(q_network.state_dict())


    rb = buffers.ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        spaces.MultiDiscrete(np.array([len(action_map)])),
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()


    ## Finetuning data collection 
    f_obs = None
    f_next_obs = None
    f_risks = None
    f_ep_len = [0]
    f_actions = None
    f_rewards = None
    f_dones = None
    f_costs = None

    risk, next_risk = None, None

    step_log = 0
    scores = []
    if args.collect_data:
        storage_path = os.path.join(args.storage_path, args.env_id, run.name)
        make_dirs(storage_path, 0) #episode)
    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    cum_cost, ep_cost, ep_risk_cost_int, cum_risk_cost_int, ep_risk, cum_risk = 0, 0, 0, 0, 0, 0
    last_step = 0
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([get_random_action() for _ in range(envs.num_envs)])
        else:
            risk = risk_model(get_risk_obs(args, torch.Tensor(obs).to(device))) if args.use_risk else None
            q_values = q_network(torch.Tensor(obs).to(device), risk)
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminated, truncated, infos = envs.step(map(action_map_fn, actions))
        done = np.logical_or(terminated, truncated)

        step_log += 1
        if not done:
            cost = torch.Tensor(infos["cost"]).to(device).view(-1)
            ep_cost += infos["cost"]; cum_cost += infos["cost"]
        else:
            cost = torch.Tensor(np.array([infos["final_info"][0]["cost"]])).to(device).view(-1)
            ep_cost += np.array([infos["final_info"][0]["cost"]]); cum_cost += np.array([infos["final_info"][0]["cost"]])



        if args.fine_tune_risk or args.collect_data:
            f_obs = torch.Tensor(obs) if f_obs is None else torch.concat([f_obs, torch.Tensor(obs)], axis=0)
            f_next_obs = torch.Tensor(next_obs) if f_next_obs is None else torch.concat([f_next_obs, torch.Tensor(next_obs)], axis=0)
            f_actions = torch.Tensor(actions) if f_actions is None else torch.concat([f_actions, torch.Tensor(actions)], axis=0)
            f_rewards = torch.Tensor(rewards) if f_rewards is None else torch.concat([f_rewards, torch.Tensor(rewards)], axis=0)
            f_costs = torch.Tensor(cost) if f_costs is None else torch.concat([f_costs, torch.Tensor(cost)], axis=0)


        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if "episode" not in info:
                    continue


                last_step = global_step
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_cost", ep_cost, global_step)
                writer.add_scalar("charts/cummulative_cost", cum_cost, global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                writer.add_scalar("charts/epsilon", epsilon, global_step)

                scores.append(info['episode']['r'])
                ep_cost = 0

                avg_mean_score = np.mean(scores[-100:])
                writer.add_scalar("Results/Avg_Return", avg_mean_score, global_step)
                ## Save all the data
                e_risks = torch.Tensor(np.array(list(reversed(range(int(info["episode"]["l"])))))).to(device) if cost > 0 else torch.Tensor(np.array([int(info["episode"]["l"])]*int(info["episode"]["l"]))).to(device)
                if args.fine_tune_risk and args.use_risk:
                    f_risks = e_risks
                    f_dist_to_fail = f_risks
                    f_dones = torch.Tensor(np.array([0]*(f_risks.size()[0])))
                    if args.rb_type == "balanced":
                        idx_risky = (f_dist_to_fail<=args.fear_radius)
                        idx_safe = (f_dist_to_fail>args.fear_radius)

                        risk_rb.add_risky(f_obs[idx_risky], f_next_obs[idx_risky], f_actions[idx_risky], f_rewards[idx_risky], f_dones[idx_risky], f_costs[idx_risky], f_risks[idx_risky], f_dist_to_fail.unsqueeze(1)[idx_risky])
                        risk_rb.add_safe(f_obs[idx_safe], f_next_obs[idx_safe], f_actions[idx_safe], f_rewards[idx_safe], f_dones[idx_safe], f_costs[idx_safe], f_risks[idx_safe], f_dist_to_fail.unsqueeze(1)[idx_safe])
                    else: 
                        risk_rb.add(f_obs, f_next_obs, f_actions, f_rewards, f_dones, f_costs, f_risks, f_risks.unsqueeze(1))

                        f_obs = None
                        f_next_obs = None
                        f_risks = None
                        f_ep_len = [0]
                        f_actions = None
                        f_rewards = None
                        f_dones = None
                        f_costs = None

                if args.collect_data:
                    f_risks = e_risks if f_risks is None else torch.cat([f_risks, e_risks], axis=0)
                    torch.save(f_obs, os.path.join(storage_path, "obs.pt"))
                    torch.save(f_actions, os.path.join(storage_path, "actions.pt"))
                    torch.save(f_costs, os.path.join(storage_path, "costs.pt"))
                    torch.save(f_risks, os.path.join(storage_path, "risks.pt"))

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(truncated):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminated, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:

            # if global_step % args.update_risk_model == 0 and args.fine_tune_risk:
            if args.use_risk and (global_step > args.learning_starts and args.fine_tune_risk) and global_step % args.risk_update_period == 0:
                if args.finetune_risk_online:
                    print("I am online")
                    data = risk_rb.slice_data(-args.risk_batch_size*args.num_update_risk, 0)
                else:
                    data = risk_rb.sample(args.risk_batch_size*args.num_update_risk)
                risk_loss = train_risk(args, risk_model, data, criterion, opt_risk, device)
                writer.add_scalar("risk/risk_loss", risk_loss, global_step)

            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    next_risk = risk_model(get_risk_obs(args, data.next_observations.float())) if args.use_risk else None
                    target_max, _ = target_network(data.next_observations.float(), next_risk).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                risk = risk_model(get_risk_obs(args, data.observations.float())) if args.use_risk else None
                old_val = q_network(data.observations.float(), risk).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.dqn_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=0.05,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "DQN", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()

