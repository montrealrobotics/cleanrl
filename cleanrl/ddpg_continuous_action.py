# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import safety_gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from src.models.risk_models import *
from src.datasets.risk_datasets import *
import src.utils as utils 

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
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "risk-aware-exploration"
    """the wandb's project name"""
    wandb_entity: str = "kaustubh_umontreal"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "SafetyCarGoal1Gymnasium-v0"
    """the environment id of the Atari game"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 25e3
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""

    # Risk arguments 
    # risk learning rate 
    risk_lr: float = 1e-6
    # risk batch size 
    risk_batch_size: int = 5000
    # num of epochs for risk update 
    num_risk_epochs: int = 10
    # fine tune risk or not 
    fine_tune_risk: str = "None" 
    # whether to use risk or not 
    use_risk: bool = False
    # model path 
    risk_model_path: str = "None"
    # risk update period 
    risk_update_period: int = 1000

    # Env parameters 
    early_termination: bool = True 
    term_cost: int = 1
    failure_penalty: int = 0
    reward_goal: int = 10
    reward_distance: int = 0





def make_env(cfg, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video:
            env = gym.make(cfg.env_id, render_mode="rgb_array", early_termination=cfg.early_termination, term_cost=cfg.term_cost, failure_penalty=cfg.failure_penalty, reward_goal=cfg.reward_goal, reward_distance=cfg.reward_distance)
        else:
            env = gym.make(cfg.env_id, early_termination=cfg.early_termination, term_cost=cfg.term_cost, failure_penalty=cfg.failure_penalty, reward_goal=cfg.reward_goal, reward_distance=cfg.reward_distance)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(torch.max((q-1) * errors, q * errors).unsqueeze(1))
        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss

class QRNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, quantiles):
        super(QRNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size * len(quantiles))
        self.quantiles = quantiles

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = out.view(out.size(0), len(self.quantiles))  # Reshape output to separate quantiles
        return out



# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env, risk_size=0):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape) + risk_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a, risk=None):
        x = x if risk is None else torch.cat([x, risk], axis=-1)
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env, risk_size=0):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod()+risk_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x, risk=None):
        x = x if risk is None else torch.cat([x, risk], axis=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias

class RiskDataset(nn.Module):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return self.inputs.size()[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def train_risk(cfg, model, data, criterion, opt, device):
    model.train()
    dataset = RiskDataset(data["next_obs"].to('cpu'), data["dist_to_fail"].to('cpu'))
    dataloader = DataLoader(dataset, batch_size=cfg.risk_batch_size, shuffle=True, num_workers=10, generator=torch.Generator(device='cuda'))
    net_loss = 0
    for batch in dataloader:
        pred = model(batch[0].to(device))
        loss = criterion(pred, batch[1].to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()

        net_loss += loss.item()
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, "risk_model.pt"))
    wandb.save("risk_model.pt")
    model.eval()
    print("risk_loss:", net_loss)
    return net_loss

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
    if torch.cuda.is_available() and args.cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor) 
    else:
        torch.set_default_tensor_type(torch.FloatTensor)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    quantiles = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    args.use_risk = False if args.risk_model_path == "None" else True 
    risk_size = len(quantiles) if args.use_risk else 0

    if args.fine_tune_risk != "None":
        risk_rb = utils.ReplayBuffer(buffer_size=args.total_timesteps)
        criterion = QuantileLoss(quantiles)

    if args.use_risk:
        print("using risk")
        risk_model = QRNN(np.array(envs.single_observation_space.shape).prod(), 64, 64, 1, quantiles).to(device)
        
        if os.path.exists(args.risk_model_path):
            risk_model.load_state_dict(torch.load(args.risk_model_path, map_location=device))
            print("Pretrained risk model loaded successfully")

        risk_model.to(device)
        risk_model.eval()
        if args.fine_tune_risk != "None":
            opt_risk = optim.Adam(risk_model.parameters(), lr=args.risk_lr, eps=1e-10)
            risk_model.eval()

    actor = Actor(envs, risk_size).to(device)
    qf1 = QNetwork(envs, risk_size).to(device)
    qf1_target = QNetwork(envs, risk_size).to(device)
    target_actor = Actor(envs, risk_size).to(device)
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
    goal_met = 0;  ep_goal_met = 0

    ## Finetuning data collection 
    f_next_obs = [None]
    scores = []; goal_scores = []
    cum_cost, ep_cost, ep_risk_cost_int, cum_risk_cost_int, ep_risk, cum_risk = 0, 0, 0, 0, 0, 0
    cost = 0
    last_step = 0
    episode = 0
    step_log = 0

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                risk = risk_model(torch.Tensor(obs).to(device)) if args.use_risk else None
                actions = actor(torch.Tensor(obs).to(device), risk)
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)


        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if (args.fine_tune_risk != "None" and args.use_risk):
            for i in range(1):
                f_next_obs[i] = torch.Tensor(next_obs[i]).unsqueeze(0) if f_next_obs[i] is None else torch.concat([f_next_obs[i], torch.Tensor(next_obs[i]).unsqueeze(0)], axis=0)


        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for i, info in enumerate(infos["final_info"]):
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                ep_cost = info["cost_sum"]
                cum_cost += ep_cost
                ep_len = info["episode"]["l"][0]
                goal_met += info["cum_goal_met"]
                #print(f"global_step={global_step}, episodic_return={info['episode']['r']}, episode_cost={ep_cost}")
                scores.append(info['episode']['r'])
                goal_scores.append(info["cum_goal_met"])
                writer.add_scalar("goals/Ep Goal Achieved ", info["cum_goal_met"], global_step)
                writer.add_scalar("goals/Avg Ep Goal", np.mean(goal_scores[-100:]))
                writer.add_scalar("goals/Total Goal Achieved", goal_met, global_step)

                if args.use_risk and args.fine_tune_risk != "None":
                    e_risks = np.array(list(reversed(range(int(ep_len))))) if cum_cost > 0 else np.array([int(ep_len)]*int(ep_len))
                    # print(risks.size())
                    e_risks = torch.Tensor(e_risks)
                    if args.fine_tune_risk != "None" and args.use_risk:
                        f_risks = e_risks.unsqueeze(1)
                    elif args.collect_data:
                        f_risks = e_risks.unsqueeze(1) if f_risks is None else torch.concat([f_risks, e_risks.unsqueeze(1)], axis=0)

                    if args.fine_tune_risk in ["off", "sync"] and args.use_risk:
                        f_dist_to_fail = e_risks
                        risk_rb.add(None, f_next_obs[i], None, None, None, None, f_risks, f_risks)
                    f_next_obs[i] = None

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
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_risk = risk_model(data.next_observations) if args.use_risk else None
                state_risk = risk_model(data.observations) if args.use_risk else None 
                next_state_actions = target_actor(data.next_observations, next_state_risk)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions, next_state_risk)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (qf1_next_target).view(-1)


            qf1_a_values = qf1(data.observations.float(), data.actions.float(), state_risk).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

            # optimize the model
            q_optimizer.zero_grad()
            qf1_loss.backward()
            q_optimizer.step()

            if args.fine_tune_risk == "off" and args.use_risk:
                if args.use_risk and (global_step > args.learning_starts and args.fine_tune_risk) and global_step % args.risk_update_period == 0:
                    for epoch in tqdm.tqdm(range(args.num_risk_epochs)):
                        risk_data = risk_rb.sample(args.risk_batch_size*args.num_risk_epochs)
                        risk_loss = train_risk(args, risk_model, risk_data, criterion, opt_risk, device)
                    writer.add_scalar("risk/risk_loss", risk_loss, global_step)   

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(data.observations.float(), actor(data.observations.float(), state_risk), state_risk).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save((actor.state_dict(), qf1.state_dict()), model_path)
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
