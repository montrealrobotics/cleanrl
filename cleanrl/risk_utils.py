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
            self.dist_to_fail = self.dist_to_fail[-self.buffer_size:]
        idx = range(self.obs.size()[0])
        print(self.obs.size()[0])
        sample_idx = np.random.choice(idx, sample_size)
        return {"obs": self.obs[sample_idx],
        # "actions": self.actions[sample_idx],
        "risks": self.risks[sample_idx], 
        "dist_to_fail": self.dist_to_fail[sample_idx]}


import tqdm

def train_risk(cfg, model, data, criterion, opt, num_epochs, device):
	model.train()
	dataset = RiskyDataset(data["obs"].to('cpu'), None, data["dist_to_fail"].to('cpu'), False, risk_type="quantile",
	                fear_clip=None, fear_radius=20, one_hot=True, quantile_size=cfg.quantile_size, quantile_num=cfg.quantile_num)
	dataloader = DataLoader(dataset, batch_size=cfg.risk_batch_size, shuffle=True, num_workers=4, generator=torch.Generator(device='cpu'))
	net_loss = 0
	for _ in tqdm.tqdm(range(num_epochs)):
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