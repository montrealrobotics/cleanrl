# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy

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
from torch.nn.functional import softplus
from torch.distributions.normal import Normal
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
    parser.add_argument("--wandb-project-name", type=str, default="SafeRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Safexp-PointGoal1-v0",
        help="the id of the environment")
    parser.add_argument("--collect-data", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--total-timesteps", type=int, default=10000000, #
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--penalty-lr", type=float, default=5e-2,
        help="the learning rate of the penalty optimizer")
    parser.add_argument("--xlambda", type=float, default=1.0,
        help="the initial lambda")
    parser.add_argument("--vf-lr", type=float, default=1e-3,
        help="the learning rate of the value function optimizer")
    parser.add_argument("--cost-limit", type=float, default=1,
        help="the limit of cost per episode")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=30000,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=80,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=1,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=0.012,
        help="the target KL divergence threshold")


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
    parser.add_argument("--rb-type", type=str, default="balanced",
        help="which type of replay buffer to use for ")
    parser.add_argument("--freeze-risk-layers", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--weight", type=float, default=1.0, 
        help="weight for the 1 class in BCE loss")
    parser.add_argument("--quantile-size", type=int, default=4, help="size of the risk quantile ")
    parser.add_argument("--quantile-num", type=int, default=5, help="number of quantiles to make")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id,  early_termination=False, failure_penalty=0)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        #env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.neure = 256
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape[0]).prod(), self.neure)),
            nn.Tanh(),
            layer_init(nn.Linear(self.neure, self.neure)),
            nn.Tanh(),
            layer_init(nn.Linear(self.neure, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape[0]).prod(), self.neure)),
            nn.Tanh(),
            layer_init(nn.Linear(self.neure, self.neure)),
            nn.Tanh(),
            layer_init(nn.Linear(self.neure, envs.single_action_space.shape[0]), std=0.01),
        )
        self.coster = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape[0]).prod(), self.neure)),
            nn.Tanh(),
            layer_init(nn.Linear(self.neure, self.neure)),
            nn.Tanh(),
            layer_init(nn.Linear(self.neure, 1), std=0.01),
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(2)))

    def get_value(self, x):
        return self.critic(x)

    def get_cvalue(self,x):
        return self.coster(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x), self.coster(x)


class RiskAgent(nn.Module):
    def __init__(self, envs, risk_size=2):
        super().__init__()
        self.neure = 256+12
        self.actor_fc1 = layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), self.neure-12))
        self.actor_fc2 = layer_init(nn.Linear(self.neure, self.neure))
        self.actor_fc3 = layer_init(nn.Linear(self.neure, np.prod(envs.single_action_space.shape)), std=0.01)
        ## Critic
        self.critic_fc1 = layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), self.neure-12))
        self.critic_fc2 = layer_init(nn.Linear(self.neure, self.neure))
        self.critic_fc3 = layer_init(nn.Linear(self.neure, 1), std=0.01)

        self.coster_fc1 = layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), self.neure-12))
        self.coster_fc2 = layer_init(nn.Linear(self.neure, self.neure))
        self.coster_fc3 = layer_init(nn.Linear(self.neure, 1), std=0.01)

        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(2)))
        self.tanh = nn.Tanh()

        self.risk_encoder_actor = nn.Sequential(
            layer_init(nn.Linear(risk_size, 12)),
            nn.Tanh())

        self.risk_encoder_critic = nn.Sequential(
            layer_init(nn.Linear(risk_size, 12)),
            nn.Tanh())

        self.risk_encoder_coster = nn.Sequential(
            layer_init(nn.Linear(risk_size, 12)),
            nn.Tanh())
        
    def forward_actor(self, x, risk):
        risk = self.risk_encoder_actor(risk)
        x = self.tanh(self.actor_fc1(x))
        x = self.tanh(self.actor_fc2(torch.cat([x, risk], axis=1)))
        x = self.tanh(self.actor_fc3(x))

        return x


    def get_value(self, x, risk):
        risk = self.risk_encoder_critic(risk)
        x = self.tanh(self.critic_fc1(x))
        x = self.tanh(self.critic_fc2(torch.cat([x, risk], axis=1)))
        value = self.tanh(self.critic_fc3(x))

        return value

    def get_cvalue(self, x, risk):
        risk = self.risk_encoder_coster(risk)
        x = self.tanh(self.coster_fc1(x))
        x = self.tanh(self.coster_fc2(torch.cat([x, risk], axis=1)))
        value = self.tanh(self.coster_fc3(x))

        return value

    def get_action_and_value(self, x, risk, action=None):
        action_mean = self.forward_actor(x, risk)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.get_value(x, risk), self.get_cvalue(x, risk)


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
    #print(next_obs_risk.size())
    return next_obs_risk        


def train_risk(args, model, data, criterion, opt, device):
    model.train()
    dataset = RiskyDataset(data["next_obs"].to('cpu'), data["actions"].to('cpu'), data["risks"].to('cpu'), False, risk_type=args.risk_type,
                            fear_clip=None, fear_radius=args.fear_radius, one_hot=True, quantile_size=args.quantile_size, quantile_num=args.quantile_num)
    dataloader = DataLoader(dataset, batch_size=args.risk_batch_size, shuffle=True, num_workers=10, generator=torch.Generator(device='cpu'))
    net_loss = 0
    for batch in dataloader:
        pred = model(get_risk_obs(args, batch[0]).to(device))
        if args.model_type == "mlp":
            loss = criterion(pred, batch[1].squeeze().to(device))
        else:
            loss = criterion(pred, torch.argmax(batch[1].squeeze(), axis=1).to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()

        net_loss += loss.item()
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, "risk_model.pt"))
    wandb.save("risk_model.pt")
    model.eval()
    return net_loss

if __name__ == "__main__":
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
    args.use_risk = False if args.risk_model_path == "None" else True 

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    risk_model_class = {"bayesian": {"continuous": BayesRiskEstCont, "binary": BayesRiskEst, "quantile": BayesRiskEst}, 
                        "mlp": {"continuous": RiskEst, "binary": RiskEst}} 

    risk_size_dict = {"continuous": 1, "binary": 2, "quantile": args.quantile_num}
    risk_size = risk_size_dict[args.risk_type]
    if args.fine_tune_risk:
        if args.rb_type == "balanced":
            rb = ReplayBufferBalanced(buffer_size=args.total_timesteps)
        else:
            rb = ReplayBuffer(buffer_size=args.total_timesteps)
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


    penalty_param = torch.tensor(args.xlambda,requires_grad=True).float()

    if args.use_risk:
        print("using risk")
        #if args.risk_type == "binary":
        agent = RiskAgent(envs=envs, risk_size=risk_size).to(device)
        #else:
        #    agent = ContRiskAgent(envs=envs).to(device)
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
        actor_optimizer = optim.Adam(list(agent.actor_fc1.parameters()) + list(agent.actor_fc2.parameters()) + list(agent.actor_fc3.parameters()) + list(agent.risk_encoder_actor.parameters()), lr=args.learning_rate, eps=1e-5)
        critic_optimizer = optim.Adam(list(agent.critic_fc1.parameters()) + list(agent.critic_fc2.parameters()) + list(agent.critic_fc3.parameters()) + list(agent.risk_encoder_critic.parameters()), lr=args.vf_lr, eps=1e-5)
        cost_optimizer = optim.Adam(list(agent.coster_fc1.parameters()) + list(agent.coster_fc2.parameters()) + list(agent.coster_fc3.parameters()) + list(agent.risk_encoder_coster.parameters()), lr=args.vf_lr, eps=1e-5)
        penalty_optimizer =optim.Adam([penalty_param], lr=args.penalty_lr)#惩罚系数优化器

    else:
        agent = Agent(envs=envs).to(device)

        actor_optimizer = optim.Adam(agent.actor.parameters(), lr=args.learning_rate, eps=1e-5)
        critic_optimizer = optim.Adam(agent.critic.parameters(), lr=args.vf_lr, eps=1e-5)
        cost_optimizer = optim.Adam(agent.coster.parameters(), lr=args.vf_lr, eps=1e-5)
        penalty_optimizer =optim.Adam([penalty_param], lr=args.penalty_lr)#惩罚系数优化器

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    costs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    cvalues = torch.zeros((args.num_steps, args.num_envs)).to(device) 
    risks = torch.zeros((args.num_steps, args.num_envs) + (risk_size,)).to(device)


    cum_cost, ep_cost, ep_risk_cost_int, cum_risk_cost_int, ep_risk, cum_risk = 0, 0, 0, 0, 0, 0
    cost = 0
    last_step = 0
    episode = 0
    step_log = 0

    ## Finetuning data collection 
    f_obs = None
    f_next_obs = None
    f_risks = None
    f_ep_len = [0]
    f_actions = None
    f_rewards = None
    f_dones = None
    f_costs = None

    scores = []
    # risk_ = torch.Tensor([[1., 0.]]).to(device)
    # print(f_obs.size(), f_risks.size())
    all_data = None

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()[0]).to(device)
    obs_ = next_obs
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    ep_cost = np.zeros(args.num_envs)
    total_cost = 0
    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            actor_optimizer.param_groups[0]["lr"] = lrnow

        count = 0
        reward_pool = []
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            if args.use_risk:
                with torch.no_grad():
                    next_obs_risk = get_risk_obs(args, next_obs)
                    next_risk = torch.Tensor(risk_model(next_obs_risk.to(device))).to(device)
                    if args.risk_type == "continuous":
                        next_risk = next_risk.unsqueeze(0)
                #print(next_risk.size())
                if args.binary_risk and args.risk_type == "binary":
                    id_risk = torch.argmax(next_risk, axis=1)
                    next_risk = torch.zeros_like(next_risk)
                    next_risk[:, id_risk] = 1
                elif args.binary_risk and args.risk_type == "continuous":
                    id_risk = int(next_risk[:,0] >= 1 / (args.fear_radius + 1))
                    next_risk = torch.zeros_like(next_risk)
                    next_risk[:, id_risk] = 1
                # print(next_risk)
                risks[step] = next_risk
                # all_risks[global_step] = next_risk#, axis=-1)


            # ALGO LOGIC: action logic
            with torch.no_grad():
                if args.use_risk:
                    action, logprob, _, value, cvalue = agent.get_action_and_value(next_obs, next_risk)
                else:
                    action, logprob, _, value, cvalue = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
                cvalues[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            cost_array = np.zeros(args.num_envs,dtype=np.float32)
            done = np.logical_or(terminated, truncated)
            for i in range(args.num_envs):
                if not done[i]:
                    cost_array[i] = infos['cost'][i]
                else:
                    cost_array[i] = infos['final_info'][i]['cost']
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            costs[step] = torch.tensor(cost_array).to(device).view(-1)
            next_obs, next_done, reward, cost = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device), torch.Tensor(reward).to(device), torch.tensor(cost_array).to(device).view(-1)
            if args.fine_tune_risk or args.collect_data:
                f_obs = obs_.to("cpu") if f_obs is None else torch.concat([f_obs, obs_.to("cpu")], axis=0)
                f_next_obs = next_obs.to("cpu") if f_next_obs is None else torch.concat([f_next_obs, next_obs.to("cpu")], axis=0)
                f_actions = action.to("cpu") if f_actions is None else torch.concat([f_actions, action.to("cpu")], axis=0)
                f_rewards = reward.to("cpu") if f_rewards is None else torch.concat([f_rewards, reward.to("cpu")], axis=0)
                # f_risks = risk_ if f_risks is None else torch.concat([f_risks, risk_], axis=0)
                f_costs = cost.to("cpu") if f_costs is None else torch.concat([f_costs, cost.to("cpu")], axis=0)
                f_dones = next_done.to("cpu") if f_dones is None else torch.concat([f_dones, next_done.to("cpu")], axis=0)

            obs_ = next_obs


            if args.use_risk and (global_step > args.start_risk_update and args.fine_tune_risk) and global_step % args.risk_update_period == 0:
                #print(global_step)
                # update_risk = 0
                # while update_risk < args.num_update_risk:
                if args.finetune_risk_online:
                    print("I am online")
                    data = rb.slice_data(-args.risk_batch_size*args.num_update_risk, 0)
                else:
                    data = rb.sample(args.risk_batch_size*args.num_update_risk)
                risk_loss = train_risk(args, risk_model, data, criterion, opt_risk, device)
                writer.add_scalar("risk/risk_loss", risk_loss, global_step)
            #print(info)
#            for item in info:
            ep_cost += cost_array
            # Only print when at least 1 env is done
            if "final_info" not in infos:
                continue

            for info in infos["final_info"]:
                if info is None:
                    continue
                #if "episode" in item.keys():
                count += 1
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}, episodic_cost={ep_cost}")
                total_cost += ep_cost
                reward_pool.append(info['episode']['r'])
                writer.add_scalar("costs/episodic_cost", ep_cost, global_step)
                ep_cost = 0
                if count == 30:
                    writer.add_scalar("charts/episodic_return", np.mean(reward_pool), global_step)
                    writer.add_scalar("costs/Cummulative Cost", total_cost, global_step)
                    count = 0
                    # writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                f_ep_len.append(f_ep_len[-1] + int(info["episode"]["l"]))
                # f_dist_to_fail = torch.Tensor(np.array(list(reversed(range(f_obs.size()[0]))))).to(device) if cost > 0 else torch.Tensor(np.array([f_obs.size()[0]]*f_obs.shape[0])).to(device)
                e_risks = torch.Tensor(np.array(list(reversed(range(int(info["episode"]["l"])))))) if cost > 0 else torch.Tensor(np.array([int(info["episode"]["l"])]*int(info["episode"]["l"])))
                # print(risks.size())
                
                if args.fine_tune_risk or args.collect_data:
                    f_risks = e_risks.unsqueeze(1) if f_risks is None else torch.cat([f_risks, e_risks.unsqueeze(1)], axis=0)
                    f_risks_discrete = torch.zeros_like(f_risks)
                    f_risks_discrete[-args.fear_radius:] = 1 
                if args.fine_tune_risk:
                    f_dist_to_fail = e_risks
                    if args.rb_type == "balanced":
                        idx_risky = (f_dist_to_fail<=args.fear_radius)
                        idx_safe = (f_dist_to_fail>args.fear_radius)

                        rb.add_risky(f_obs[idx_risky], f_next_obs[idx_risky], f_actions[idx_risky], f_rewards[idx_risky], f_dones[idx_risky], f_costs[idx_risky], f_risks[idx_risky], f_dist_to_fail.unsqueeze(1)[idx_risky])
                        rb.add_safe(f_obs[idx_safe], f_next_obs[idx_safe], f_actions[idx_safe], f_rewards[idx_safe], f_dones[idx_safe], f_costs[idx_safe], f_risks[idx_safe], f_dist_to_fail.unsqueeze(1)[idx_safe])
                    else:
                        rb.add(f_obs, f_next_obs, f_actions, f_rewards, f_dones, f_costs, f_risks, e_risks.unsqueeze(1))

                    f_obs = None    
                    f_next_obs = None
                    f_risks = None
                    #f_ep_len = None
                    f_actions = None
                    f_rewards = None
                    f_dones = None
                    f_costs = None

                break

        # bootstrap value if not done
        with torch.no_grad():
            if args.use_risk:
                next_value = agent.get_value(next_obs, next_risk).reshape(1, -1)
            else:
                next_value = agent.get_value(next_obs).reshape(1, -1)   
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

            if args.use_risk:
                next_cvalue = agent.get_cvalue(next_obs, next_risk).reshape(1, -1)
            else:
                next_cvalue = agent.get_cvalue(next_obs).reshape(1, -1)
            
            c_advantages = torch.zeros_like(costs).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextcvalues = next_cvalue
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextcvalues = cvalues[t + 1]
                delta = costs[t] + args.gamma * nextcvalues * nextnonterminal - cvalues[t]
                c_advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            creturns = c_advantages + cvalues

            #calculate average ep cost
            episode_cost = []
            accumulate_cost = torch.zeros([args.num_envs],dtype=torch.float32).to(device)
            for t in range(args.num_steps):
                if torch.eq(dones[t],torch.zeros([args.num_envs],dtype=torch.float32).to(device)).all().item() is True :
                    accumulate_cost += costs[t]
                else: 
                    indx = torch.where(dones[t]==1)
                    for add in indx[0]:
                        episode_cost.append(accumulate_cost[add.item()].item())
                        accumulate_cost[add.item()] = 0
            if episode_cost == []:
                average_ep_cost = 0
            else:
                average_ep_cost = np.mean(episode_cost)

        writer.add_scalar("charts/episodic_cost", average_ep_cost, global_step)
        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_costs = costs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_cadvantages = c_advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_creturns = creturns.reshape(-1)
        b_values = values.reshape(-1)
        b_cvalues = cvalues.reshape(-1)
        b_risks = risks.reshape((-1, ) + (risk_size, ))

        # Optimizing the lambda
        cost_devitation = average_ep_cost - args.cost_limit
        #update lambda
        loss_penalty = - penalty_param * cost_devitation
        penalty_optimizer.zero_grad()
        loss_penalty.backward()
        penalty_optimizer.step()

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                if args.use_risk:
                    _, newlogprob, entropy, newvalue, newcvalue = agent.get_action_and_value(b_obs[mb_inds], b_risks[mb_inds], b_actions[mb_inds])
                else:
                    _, newlogprob, entropy, newvalue, newcvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
            
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                mb_cadvantages = b_cadvantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                    mb_cadvantages = (mb_cadvantages - mb_cadvantages.mean()) / (mb_cadvantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = mb_advantages * ratio
                pg_loss2 = mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.min(pg_loss1, pg_loss2).mean()
                
                cpg_loss = ratio * mb_cadvantages
                cpg_loss = cpg_loss.mean()

                p = softplus(penalty_param)
                penalty_item = p.item()

                entropy_loss = entropy.mean()
                # Create policy objective function, including entropy regularization
                objective = pg_loss + args.ent_coef * entropy_loss

                # Possibly include cpg_loss in objective
                objective -= penalty_item * cpg_loss
                objective = -objective/(1+penalty_item)

                actor_optimizer.zero_grad()
                objective.backward()
                actor_optimizer.step()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                newcvalue = newcvalue.view(-1)
                if args.clip_vloss:
                    cv_loss_unclipped = (newcvalue - b_creturns[mb_inds]) ** 2
                    cv_clipped = b_cvalues[mb_inds] + torch.clamp(
                        newcvalue - b_cvalues[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    cv_loss_clipped = (cv_clipped - b_creturns[mb_inds]) ** 2
                    cv_loss_max = torch.max(cv_loss_unclipped, cv_loss_clipped)
                    cv_loss = 0.5 * cv_loss_max.mean()
                else:
                    cv_loss = 0.5 * ((newcvalue - b_creturns[mb_inds]) ** 2).mean()                    

                critic_optimizer.zero_grad()
                v_loss.backward()
                critic_optimizer.step()

                cost_optimizer.zero_grad()
                cv_loss.backward()
                cost_optimizer.step()


            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", actor_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/cost_value_loss", cv_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
