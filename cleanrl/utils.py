import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



class BayesRiskEst(nn.Module):
    def __init__(self, obs_size=64, fc_units=64, risk_size=10):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, fc_units)
        self.fc2 = nn.Linear(fc_units, fc_units)
        self.out = nn.Linear(fc_units, risk_size)

        ## Batch Norm layers
        self.bnorm1 = nn.LayerNorm(fc_units)
        self.bnorm2 = nn.LayerNorm(fc_units)

        # Activation functions
        self.activation = nn.ReLU()

        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
    	x = self.bnorm1(self.activation(self.fc1(x)))
    	x = self.dropout(self.bnorm2(self.activation(self.fc2(x))))
    	out = self.logsoftmax(self.out(x))
    	return out



class ReplayBuffer:
	def __init__(self, buffer_size=1000000, filter_risks=False):
		self.obs = None 
		# self.actions = None 
		self.risks = None 
		self.dist_to_fail = None 

		self.buffer_size = buffer_size

	def add(self, obs, risks, dist_to_fail):
		self.obs = torch.cat([self.obs, obs], axis=0) if self.obs is not None else obs 
		# self.actions = torch.cat([self.actions, actions], axis=0) if self.actions is not None else actions 
		self.risks = torch.cat([self.risks, risks], axis=0) if self.risks is not None else risks 
		self.dist_to_fail = torch.cat([self.dist_to_fail, dist_to_fail], axis=0) if self.dist_to_fail is not None else dist_to_fail 

	def __len__(self):
		return self.obs.size()[0] if self.obs is not None else 0

	def sample(self, sample_size):
		## fixing replay buffer size 		
	    if self.obs.size()[0] > self.buffer_size:
	        self.obs = self.next_obs[-self.buffer_size:]
	        self.risks = self.risks[-self.buffer_size:]

	    idx = range(self.obs.size()[0])
	    sample_idx = np.random.choice(idx, sample_size)
	    print(self.dist_to_fail.size(), max(idx), min(idx))
	    return {"obs": self.obs[sample_idx],
	            # "actions": self.actions[sample_idx],
	            "risks": self.risks[sample_idx], 
	            "dist_to_fail": self.dist_to_fail[sample_idx]}


import tqdm

def train_risk(cfg, model, data, criterion, opt, num_epochs, device):
	model.train()
	dataset = RiskyDataset(data["obs"].to('cpu'), None, data["risks"].to('cpu'), False, risk_type=cfg.risk_type,
	                fear_clip=None, fear_radius=cfg.fear_radius, one_hot=True, quantile_size=cfg.quantile_size, quantile_num=cfg.quantile_num)
	dataloader = DataLoader(dataset, batch_size=cfg.risk_batch_size, shuffle=True, num_workers=10, generator=torch.Generator(device='cpu'))
	net_loss = 0
	for _ in range(num_epochs):
		for batch in dataloader:
			X, y = batch[0], batch[1]
			pred = model(X.to(device))
			loss = criterion(pred, torch.argmax(y.squeeze(), axis=1).to(device))
			opt.zero_grad()
			loss.backward()
			opt.step()
			net_loss += loss.item()
	model.eval()
	return net_loss / (num_epochs * len(dataloader))


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
    parser.add_argument("--reward-goal", type=float, default=1.0, 
        help="reward to give when the goal is achieved")
    parser.add_argument("--reward-distance", type=float, default=1.0, 
        help="reward to give when the goal is achieved")
    parser.add_argument("--early-termination", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to terminate early i.e. when the catastrophe has happened")
    parser.add_argument("--unifying-lidar", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="what kind of sensor is used (same for every environment?)")
    parser.add_argument("--term-cost", type=int, default=1,
        help="how many violations before you terminate")
    parser.add_argument("--failure-penalty", type=float, default=0.0,
        help="Reward Penalty when you fail")
    parser.add_argument("--collect-data", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="store data while trianing")
    parser.add_argument("--storage-path", type=str, default="./data/ppo/term_1",
        help="the storage path for the data collected")

    parser.add_argument("--total-timesteps", type=int, default=1000000,
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

    ## Arguments related to risk model 
    parser.add_argument("--use-risk", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Use risk model or not ")
    parser.add_argument("--risk-input", type=str, default="state_action",
        help="specify the NN to use for the risk model")  
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
    parser.add_argument("--num-risk-samples", type=int, default=20000,
        help="fear radius for training the risk model")
    parser.add_argument("--risk-update-period", type=int, default=10000,
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
    parser.add_argument("--start-risk-update", type=int, default=20000,
        help="number of epochs to update the risk model") 
    parser.add_argument("--rb-type", type=str, default="simple",
        help="which type of replay buffer to use for ")
    parser.add_argument("--freeze-risk-layers", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--weight", type=float, default=1.0, 
        help="weight for the 1 class in BCE loss")
    parser.add_argument("--risk-penalty", type=float, default=0.0, 
        help="risk penalty to discourage risky state visitation")
    parser.add_argument("--risk-penalty-start", type=float, default=100, 
        help="risk penalty to discourage risky state visitation")
    parser.add_argument("--quantile-size", type=int, default=4, help="size of the risk quantile ")
    parser.add_argument("--quantile-num", type=int, default=5, help="number of quantiles to make")


    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


class RiskyDataset(nn.Module):
    def __init__(self, obs, actions, risks, action=False, risk_type="discrete", fear_clip=None, fear_radius=None, one_hot=True, quantile_size=4, quantile_num=5):
        self.obs = obs
        self.risks = risks
        self.actions = actions
        self.one_hot = one_hot
        self.action = action
        self.fear_clip = fear_clip 
        self.fear_radius = fear_radius
        self.risk_type = risk_type

        self.quantile_size = quantile_size
        self.quantile_num = quantile_num

    def __len__(self):
        return self.obs.size()[0]
    
    def get_quantile_risk(self, idx):
        risk = self.risks[idx]
        y = torch.zeros(self.quantile_num)
        quant = self.quantile_size
        label = None
        for i in range(self.quantile_num-1):
            if risk < quant:
                label = i
                break
            else:
                quant += self.quantile_size
        if label is None:
            label = self.quantile_num-1

        y[label] = 1.0 
        return y

    def get_binary_risk(self, idx):
        if self.one_hot:
            y = torch.zeros(2)
            y[int(self.risks[idx] <= self.fear_radius)] = 1.0
        else:
            y = int(self.risks[idx] <= self.fear_radius)
        return y
    
    def get_continuous_risk(self, idx):
        if self.fear_clip is not None:
            return 1. / torch.clip(self.risks[idx]+1.0, 1, self.fear_clip)
        else:
            return 1. / self.risks[idx]

    def __getitem__(self, idx):
        if self.risk_type == "continuous":
            y = self.get_continuous_risk(idx)
        elif self.risk_type == "binary":
            y = self.get_binary_risk(idx)
        elif self.risk_type == "quantile":
            y = self.get_quantile_risk(idx)

        if self.action:
            return self.obs[idx], self.actions[idx], y
        else:
            return self.obs[idx], y