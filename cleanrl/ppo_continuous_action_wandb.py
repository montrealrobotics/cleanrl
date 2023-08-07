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
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from comet_ml import Experiment

from src.models.risk_models import *
from src.utils import * 

import hydra
import os



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
    parser.add_argument("--env-id", type=str, default="SafetyCarGoal1Gymnasium-v0",
        help="the id of the environment")
    parser.add_argument("--early-termination", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to terminate early i.e. when the catastrophe has happened")
    parser.add_argument("--term-cost", type=int, default=1,
        help="how many violations before you terminate")
    parser.add_argument("--collect-data", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="store data while trianing")
    parser.add_argument("--storage-path", type=str, default="./data/ppo/term_1",
        help="the storage path for the data collected")
    
    parser.add_argument("--total-timesteps", type=int, default=10000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=2048,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--reward-penalty", type=float, default=0,
        help="coefficient of the value function")
    
    ## Arguments related to risk model 
    parser.add_argument("--use-risk", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Use risk model or not ")
    parser.add_argument("--risk-actor", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use risk model in the actor or not ")
    parser.add_argument("--risk-critic", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Use risk model in the critic or not ")
    parser.add_argument("--risk-model-path", type=str, default="./pretrained/agent.pt",
        help="the id of the environment")
    parser.add_argument("--binary-risk", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Use risk model in the critic or not ")
    parser.add_argument("--model-type", type=str, default="mlp",
        help="specify the NN to use for the risk model")
    parser.add_argument("--fear-radius", type=int, default=5,
        help="fear radius for training the risk model")
    parser.add_argument("--num-risk-datapoints", type=int, default=10000,
        help="fear radius for training the risk model")
    parser.add_argument("--update-risk-model", type=int, default=10000,
        help="number of epochs to update the risk model")
    parser.add_argument("--risk-epochs", type=int, default=10,
        help="number of epochs to update the risk model")
    parser.add_argument("--risk-lr", type=float, default=1e-5,
        help="the learning rate of the optimizer")
    parser.add_argument("--risk-batch-size", type=int, default=10,
        help="number of epochs to update the risk model")
    parser.add_argument("--fine-tune-risk", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args





def make_env(cfg, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video:
            env = gym.make(cfg.env_id, render_mode="rgb_array", early_termination=cfg.early_termination, term_cost=cfg.term_cost)
        else:
            env = gym.make(cfg.env_id, early_termination=cfg.early_termination, term_cost=cfg.term_cost)
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



class RiskAgent(nn.Module):
    def __init__(self, envs, risk_actor=True, risk_critic=False):
        super().__init__()
        ## Actor
        self.actor_fc1 = layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64))
        self.actor_fc2 = layer_init(nn.Linear(76, 76))
        self.actor_fc3 = layer_init(nn.Linear(76, np.prod(envs.single_action_space.shape)), std=0.01)
        ## Critic
        self.critic_fc1 = layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64))
        self.critic_fc2 = layer_init(nn.Linear(76, 76))
        self.critic_fc3 = layer_init(nn.Linear(76, 1), std=0.01)

        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))
        self.tanh = nn.Tanh()

        self.risk_encoder = nn.Sequential(
            layer_init(nn.Linear(2, 12)),
            nn.Tanh())

    def forward_actor(self, x, risk):
        risk = self.risk_encoder(risk)
        x = self.tanh(self.actor_fc1(x))
        x = self.tanh(self.actor_fc2(torch.cat([x, risk], axis=1)))
        x = self.tanh(self.actor_fc3(x))

        return x


    def get_value(self, x, risk):
        risk = self.risk_encoder(risk)
        x = self.tanh(self.critic_fc1(x))
        x = self.tanh(self.critic_fc2(torch.cat([x, risk], axis=1)))
        value = self.tanh(self.critic_fc3(x))

        return value

    def get_action_and_value(self, x, risk, action=None):
        action_mean = self.forward_actor(x, risk)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.get_value(x, risk)


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

class RiskDataset(nn.Module):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return self.inputs.size()[0]

    def __getitem__(self, idx):
        y = torch.zeros(2)
        y[int(self.targets[idx][0])] = 1.0
        return self.inputs[idx], y



def fine_tune_risk(cfg, model, inputs, targets, opt, device):
        model.train()
        dataset = RiskDataset(inputs, targets)
        weight = torch.sum(targets==0) / torch.sum(targets==1)
        if cfg.model_type == "bayesian":
            criterion = nn.NLLLoss(weight=torch.Tensor([1, weight]).to(device))
        else:
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1, weight]).to(device))

        dataloader = DataLoader(dataset, batch_size=cfg.risk_batch_size, shuffle=True, num_workers=10, generator=torch.Generator(device=device))
        for epoch in range(cfg.risk_epochs):
            net_loss = 0
            for batch in dataloader:
                pred = model(batch[0].to(device))
                if cfg.model_type == "mlp":
                    loss = criterion(pred, batch[1].to(device))
                else:
                    loss = criterion(pred, torch.argmax(batch[1].squeeze(), axis=1).to(device))
                opt.zero_grad()
                loss.backward()
                opt.step()
                # scheduler.step()
                net_loss += loss.item()
            print("Average Risk training loss: %.4f"%(net_loss / len(dataloader)))

        model.eval()
        return model
            
            




def train(cfg):
    # fmt: on

    run_name = f"{int(time.time())}"

    #experiment = Experiment(
    #    api_key="FlhfmY238jUlHpcRzzuIw3j2t",
    #    project_name="risk-aware-exploration",
    #    workspace="hbutsuak95",
    #)      
    #import wandb 
    #wandb.init(config=vars(cfg), entity="kaustubh95",
    #             project="risk_aware_exploration",
    #             name=run_name, monitor_gym=True,
    #             sync_tensorboard=True, save_code=True)

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(cfg).items()])),
    )

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(cfg, i, cfg.capture_video, run_name, cfg.gamma) for i in range(cfg.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    if cfg.model_type == "bayesian":
        risk_model_class = BayesRiskEst 
    else:
        risk_model_class = RiskEst
    # print(envs.single_observation_space.shape)

    if cfg.use_risk:
        agent = RiskAgent(envs=envs).to(device)
        if os.path.exists(cfg.risk_model_path):
            risk_model = risk_model_class(obs_size=np.array(envs.single_observation_space.shape).prod(), batch_norm=True)
            risk_model.load_state_dict(torch.load(cfg.risk_model_path, map_location=device))
            risk_model.to(device)
            if cfg.fine_tune_risk:
                opt_risk = optim.Adam(risk_model.parameters(), lr=cfg.risk_lr, eps=1e-5)
            risk_model.eval()
        else:
            raise("No model in the path specified!!")
    else:
        agent = Agent(envs=envs).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    # print(envs.single_observation_space.shape)
    # ALGO Logic: Storage setup
    obs = torch.zeros((cfg.num_steps, cfg.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((cfg.num_steps, cfg.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    rewards = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    dones = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    values = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    costs = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    risks = torch.zeros((cfg.num_steps, cfg.num_envs) + (2,)).to(device)

    all_costs = torch.zeros((cfg.total_timesteps, cfg.num_envs)).to(device)
    all_risks = torch.zeros((cfg.total_timesteps, cfg.num_envs)).to(device)


    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=cfg.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(cfg.num_envs).to(device)
    num_updates = cfg.total_timesteps // cfg.batch_size

    cum_cost, ep_cost, ep_risk_cost_int, cum_risk_cost_int, ep_risk, cum_risk = 0, 0, 0, 0, 0, 0
    cost = 0
    last_step = 0
    episode = 0
    step_log = 0

    ## Finetuning data collection 
    f_obs = next_obs
    f_risks = torch.Tensor([[0.]]).to(device)
    f_ep_len = [0]
    f_actions = None

    # print(f_obs.size(), f_risks.size())


    if cfg.collect_data:
        storage_path = os.path.join(cfg.storage_path, cfg.env_id, run_name)
        make_dirs(storage_path, episode)

    
    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if cfg.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * cfg.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, cfg.num_steps):
            risk = torch.Tensor([[0.]]).to(device)
            global_step += 1 * cfg.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            costs[step] = cost
            all_costs[global_step] = cost

            if cfg.use_risk:
                next_risk = risk_model(next_obs / 10.0).detach()
                if cfg.binary_risk:
                    id_risk = torch.argmax(next_risk, axis=1)
                    next_risk = torch.zeros_like(next_risk)
                    next_risk[:, id_risk] = 1
                # print(next_risk)
                risks[step] = next_risk
                all_risks[global_step] = torch.argmax(next_risk, axis=-1)


            # ALGO LOGIC: action logic
            with torch.no_grad():
                if cfg.use_risk:
                    action, logprob, _, value = agent.get_action_and_value(next_obs, next_risk)
                else:
                    action, logprob, _, value = agent.get_action_and_value(next_obs)

                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
                    
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)

            info_dict = {'reward': reward, 'done': done, 'cost': cost, 'prev_action': action} 
            if cfg.collect_data:
                store_data(next_obs, info_dict, storage_path, episode, step_log)
            
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            f_obs = torch.concat([f_obs, next_obs], axis=0)
            f_risks = torch.concat([f_risks, risk], axis=0)
            f_actions = action if f_actions is None else torch.concat([f_actions, action], axis=0)
            # print(f_risks.size(), f_obs.size())

            if cost > 0:
                fear_radius = min(cfg.fear_radius, step_log)
                f_risks[global_step-cfg.fear_radius:, 0] = 1.
            
            step_log += 1
            if not done:
                cost = torch.Tensor(infos["cost"]).to(device).view(-1)
                ep_cost += infos["cost"]; cum_cost += infos["cost"]
            else:
                cost = torch.Tensor(np.array([infos["final_info"][0]["cost"]])).to(device).view(-1)
                ep_cost += np.array([infos["final_info"][0]["cost"]]); cum_cost += np.array([infos["final_info"][0]["cost"]])

            if global_step % cfg.update_risk_model == 0 and cfg.fine_tune_risk:
                fine_tune_risk(cfg, risk_model, f_obs[-cfg.num_risk_datapoints:], f_risks[-cfg.num_risk_datapoints:], opt_risk, device)


            # Only print when at least 1 env is done
            if "final_info" not in infos:
                continue


            for info in infos["final_info"]:
                # Skip the envs that are not done
                if info is None:
                    continue

                ep_cost = torch.sum(all_costs[last_step:global_step]).item()
                cum_cost += ep_cost

                print(f"global_step={global_step}, episodic_return={info['episode']['r']}, episode_cost={ep_cost}")

                if cfg.use_risk:
                    ep_risk = torch.sum(all_risks[last_step:global_step]).item()
                    cum_risk += ep_risk

                    risk_cost_int = torch.logical_and(f_risks[last_step:global_step], all_risks[last_step:global_step])
                    ep_risk_cost_int = torch.sum(risk_cost_int).item()
                    cum_risk_cost_int += ep_risk_cost_int

                    writer.add_scalar("charts/episodic_risk", ep_risk, global_step)
                    writer.add_scalar("charts/cummulative_risk", cum_risk, global_step)
                    writer.add_scalar("charts/episodic_risk_&&_cost", ep_risk_cost_int, global_step)
                    writer.add_scalar("charts/cummulative_risk_&&_cost", cum_risk_cost_int, global_step)


                    #experiment.log_metric("charts/episodic_risk", ep_risk, global_step)
                    #experiment.log_metric("charts/cummulative_risk", cum_risk, global_step)
                    #experiment.log_metric("charts/episodic_risk_&&_cost", ep_risk_cost_int, global_step)
                    #experiment.log_metric("charts/cummulative_risk_&&_cost", cum_risk_cost_int, global_step)

                    print(f"global_step={global_step}, ep_Risk_cost_int={ep_risk_cost_int}, cum_Risk_cost_int={cum_risk_cost_int}")
                    print(f"global_step={global_step}, episodic_risk={ep_risk}, cum_risks={cum_risk}, cum_costs={cum_cost}")



                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                writer.add_scalar("charts/episodic_cost", ep_cost, global_step)
                writer.add_scalar("charts/cummulative_cost", cum_cost, global_step)
                #print(info["episode"]["l"])
                f_ep_len.append(f_ep_len[-1]+info["episode"]["l"][0])

#                wandb.log({"charts/episodic_return":info["episode"]["r"]}, step=global_step)

#experiment.log_metric("charts/episodic_return", info["episode"]["r"], global_step)
                #experiment.log_metric("charts/episodic_length", info["episode"]["l"], global_step)
                #experiment.log_metric("charts/episodic_cost", ep_cost, global_step)
                #experiment.log_metric("charts/cummulative_cost", cum_cost, global_step)
                last_step = global_step
                episode += 1
                step_log = 0
                if cfg.collect_data:
                    make_dirs(storage_path, episode)

        # bootstrap value if not done
        with torch.no_grad():
            if cfg.use_risk:
                next_value = agent.get_value(next_obs, next_risk).reshape(1, -1)
            else:
                next_value = agent.get_value(next_obs).reshape(1, -1)   
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(cfg.num_steps)):
                if t == cfg.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + cfg.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_risks = risks.reshape((-1, ) + (2, ))


        # Optimizing the policy and value network
        b_inds = np.arange(cfg.batch_size)
        clipfracs = []
        for epoch in range(cfg.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_inds = b_inds[start:end]

                if cfg.use_risk:
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_risks[mb_inds], b_actions[mb_inds])
                else:
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if cfg.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if cfg.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -cfg.clip_coef,
                        cfg.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + v_loss * cfg.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

            if cfg.target_kl is not  None:
                if approx_kl > cfg.target_kl:
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





        # TRY NOT TO MODIFY: record rewards for plotting purposes
        #experiment.log_metric("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        #experiment.log_metric("losses/value_loss", v_loss.item(), global_step)
        #experiment.log_metric("losses/policy_loss", pg_loss.item(), global_step)
        #experiment.log_metric("losses/entropy", entropy_loss.item(), global_step)
        #experiment.log_metric("losses/old_approx_kl", old_approx_kl.item(), global_step)
        #experiment.log_metric("losses/approx_kl", approx_kl.item(), global_step)
        #experiment.log_metric("losses/clipfrac", np.mean(clipfracs), global_step)
        #experiment.log_metric("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        #experiment.log_metric("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
    ## Save all the data
    if cfg.collect_data:
        torch.save(f_obs, os.path.join(storage_path, "obs.pt"))
        torch.save(f_actions, os.path.join(storage_path, "actions.pt"))
        torch.save(f_risks, os.path.join(storage_path, "risks.pt"))
        torch.save(torch.Tensor(f_ep_len), os.path.join(storage_path, "ep_len.pt"))
    print(f_ep_len)
    envs.close()
    writer.close()
    return 1 



if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
