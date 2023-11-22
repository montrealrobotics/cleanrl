# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer as sb3buffer
from torch.utils.tensorboard import SummaryWriter

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
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--buffer-size", type=int, default=500000,
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
    parser.add_argument("--mode", type=str, default="train",
        help="whether to train or evaluate the policy")
    parser.add_argument("--pretrained-policy-path", type=str, default="None",
        help="the id of the environment")

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
    parser.add_argument("--risk-update-period", type=int, default=1000,
        help="how frequently to update the risk model")
    parser.add_argument("--num-risk-epochs", type=int, default=1,
        help="number of sgd steps to update the risk model")
    parser.add_argument("--num-update-risk", type=int, default=10,
        help="number of sgd steps to update the risk model")
    parser.add_argument("--risk-lr", type=float, default=1e-7,
        help="the learning rate of the optimizer")
    parser.add_argument("--risk-batch-size", type=int, default=1000,
        help="number of epochs to update the risk model")
    parser.add_argument("--fine-tune-risk", type=str, default="None",
        help="fine tune risk by which method")
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
    parser.add_argument("--quantile-size", type=int, default=2, help="size of the risk quantile ")
    parser.add_argument("--quantile-num", type=int, default=10, help="number of quantiles to make")
    parser.add_argument("--risk-penalty", type=float, default=0., help="penalty to impose for entering risky states")
    parser.add_argument("--risk-penalty-start", type=float, default=20., help="penalty to impose for entering risky states")
    args = parser.parse_args()
    # fmt: on
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"

    return args


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
class QNetwork(nn.Module):
    def __init__(self, env, risk_size=0):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space["image"].shape).prod()+risk_size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )

    def forward(self, x, risk=None):
        if risk is None:
            return self.network(x)
        else:
            return self.network(torch.cat([x, risk], axis=1))


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


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
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    args.use_risk = False if args.risk_model_path == "None" else True 
    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"


    risk_model_class = {"bayesian": {"continuous": BayesRiskEstCont, "binary": BayesRiskEst, "quantile": BayesRiskEst}, 
                        "mlp": {"continuous": RiskEst, "binary": RiskEst}} 

    risk_size_dict = {"continuous": 1, "binary": 2, "quantile": args.quantile_num}
    risk_size = risk_size_dict[args.risk_type] if args.use_risk else 0
    risk_bins = np.array([i*args.quantile_size for i in range(args.quantile_num)])

    if args.use_risk:
        risk_model = risk_model_class[args.model_type][args.risk_type](obs_size=np.array(envs.single_observation_space["image"].shape).prod(), batch_norm=True, out_size=risk_size)
        if os.path.exists(args.risk_model_path):
            risk_model.load_state_dict(torch.load(args.risk_model_path, map_location=device))
            print("Pretrained risk model loaded successfully")

        risk_model.to(device)
        risk_model.eval()
    

    if args.fine_tune_risk != "None" and args.use_risk:
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
        opt_risk = optim.Adam(filter(lambda p: p.requires_grad, risk_model.parameters()), lr=args.risk_lr, eps=1e-10)


    q_network = QNetwork(envs, risk_size=risk_size).to(device)

    if os.path.exists(args.pretrained_policy_path):
        q_network.load_state_dict(torch.load(args.pretrained_policy_path, map_location=device))
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs, risk_size=risk_size).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = sb3buffer(
        args.buffer_size,
        envs.single_observation_space["image"],
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()


    f_obs, f_next_obs, f_actions = [None]*args.num_envs, [None]*args.num_envs, [None]*args.num_envs
    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    obs = obs
    total_cost = 0
    scores = []
    total_goals = 0
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon and args.mode == "train":
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            obs_in = torch.Tensor(obs["image"]).reshape(args.num_envs, -1).to(device)
            with torch.no_grad():
                risk = risk_model(obs_in) if args.use_risk else None
            q_values = q_network(obs_in, risk)
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)
        cost = int(terminated) and (rewards == 0)
        if (args.fine_tune_risk != "None" and args.use_risk):
            for i in range(args.num_envs):
                f_obs[i] = torch.Tensor(obs["image"][i]).reshape(1, -1).to(device) if f_obs[i] is None else torch.concat([f_obs[i], torch.Tensor(obs["image"][i]).reshape(1, -1).to(device)], axis=0)
                f_next_obs[i] = torch.Tensor(next_obs["image"][i]).reshape(1, -1).to(device) if f_next_obs[i] is None else torch.concat([f_next_obs[i], torch.Tensor(next_obs["image"][i]).reshape(1, -1).to(device)], axis=0)
                f_actions[i] = torch.Tensor([actions[i]]).unsqueeze(0).to(device) if f_actions[i] is None else torch.concat([f_actions[i], torch.Tensor([actions[i]]).unsqueeze(0).to(device)], axis=0)
                # f_rewards[i] = reward[i].unsqueeze(0).to(device) if f_rewards[i] is None else torch.concat([f_rewards[i], rewards[i].unsqueeze(0).to(device)], axis=0)
                # f_risks = risk_ if f_risks is None else torch.concat([f_risks, risk_], axis=0)
                # f_costs[i] = cost[i].unsqueeze(0).to(device) if f_costs[i] is None else torch.concat([f_costs[i], cost[i].unsqueeze(0).to(device)], axis=0)
                # f_dones[i] = next_done[i].unsqueeze(0).to(device) if f_dones[i] is None else torch.concat([f_dones[i], next_done[i].unsqueeze(0).to(device)], axis=0)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for i, info in enumerate(infos["final_info"]):
                # Skip the envs that are not done
                if "episode" not in info:
                    continue
                total_cost += cost
                ep_len = info["episode"]["l"]

                if args.use_risk and args.fine_tune_risk != "None":
                    e_risks = np.array(list(reversed(range(int(ep_len))))) if cost > 0 else np.array([int(ep_len)]*int(ep_len))
                    e_risks_quant = torch.Tensor(np.apply_along_axis(lambda x: np.histogram(x, bins=risk_bins)[0], 1, np.expand_dims(e_risks, 1)))
                    e_risks = torch.Tensor(e_risks)
                    if args.risk_type == "binary":
                        risk_rb.add(f_obs[i], f_next_obs[i], f_actions[i], None, None, None, (e_risks <= args.fear_radius).float(), e_risks.unsqueeze(1))
                    else:
                        risk_rb.add(f_obs[i], f_next_obs[i], f_actions[i], None, None, None, e_risks_quant, e_risks.unsqueeze(1))

                f_obs[i], f_next_obs[i], f_actions[i] = None, None, None
                scores.append(info['episode']['r'])
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}, total cost={total_cost}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/Avg Return", np.mean(scores[-100:]), global_step)
                writer.add_scalar("charts/total_cost", total_cost, global_step)
                writer.add_scalar("charts/episodic_cost",   cost, global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                writer.add_scalar("charts/epsilon", epsilon, global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(truncated):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs["image"], real_next_obs["image"], actions, rewards, terminated, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0 and args.mode=="train":
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    next_risk = risk_model(data.next_observations.reshape(args.batch_size, -1).float()) if args.use_risk else None
                    target_max, _ = target_network(data.next_observations.reshape(args.batch_size, -1).float(), next_risk).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                    risk = risk_model(data.observations.reshape(args.batch_size, -1).float()) if args.use_risk else None
                old_val = q_network(data.observations.reshape(args.batch_size, -1).float(), risk).gather(1, data.actions).squeeze()
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


            torch.save(q_network.state_dict(), os.path.join(wandb.run.dir, "qnet.pt"))
            wandb.save("qnet.pt")
            ## Update Risk Network 
            if args.use_risk and args.fine_tune_risk != "None" and global_step % args.risk_update_period == 0 and args.mode=="train":
                risk_model.train()
                risk_data = risk_rb.sample(args.risk_batch_size)
                pred = risk_model(risk_data["next_obs"].to(device))
                risk_loss = criterion(pred, torch.argmax(risk_data["risks"].squeeze(), axis=1).to(device))
                opt_risk.zero_grad()
                risk_loss.backward()
                opt_risk.step()
                risk_model.eval()
                writer.add_scalar("charts/risk_loss", risk_loss.item(), global_step)
                torch.save(risk_model.state_dict(), os.path.join(wandb.run.dir, "risk_model.pt"))
                wandb.save("risk_model.pt")

            # update target network
            if global_step % args.target_network_frequency == 0 and args.mode=="train":
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
