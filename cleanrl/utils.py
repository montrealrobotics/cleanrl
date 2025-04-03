import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt


def calculate_mc_returns(rewards, gamma):
    """Calculate Monte Carlo returns for a list of rewards."""
    mc_returns = []
    mc_return = 0
    for r in reversed(rewards):
        mc_return = r + gamma * mc_return
        mc_returns.insert(0, mc_return)
    return mc_returns


def perform_rollout(obs, env, actor, qf1, qf2, device, args, max_steps=1000, safety=False, use_mean_action=False):
    """Perform a single rollout in the environment and store results as tensors."""
    done = False
    step = 0
    episode_rewards = []
    q1_values = []
    q2_values = []
    cdq_values = []
    entropy_list = []

    while not done and step < max_steps:
        with torch.no_grad():
            state = torch.FloatTensor(obs).to(device)
            action, entropy, action_mean = actor.get_action(state.unsqueeze(0))
            if use_mean_action:
                action = action_mean
            q1 = qf1(state.unsqueeze(0), action)
            q2 = qf2(state.unsqueeze(0), action)

        # Store q1 and q2 values
        q1_values.append(q1.item())
        q2_values.append(q2.item())
        cdq_values.append(min(q1.item(), q2.item()))
        entropy_list.append(entropy.item())

        action = action.squeeze().detach().cpu().numpy()
        obs, reward, terminated, truncated, _ = env.step(action)
        if safety:
            reward = int(terminated)
        done = terminated or truncated
        episode_rewards.append(reward)
        step += 1

    # Calculate Monte Carlo returns
    mc_returns = calculate_mc_returns(episode_rewards, args.gamma)
    mc_returns_tensor = np.array(mc_returns, dtype=np.float32)
    entropy_tensor = np.array(entropy_list)

    return q1_values, q2_values,cdq_values, mc_returns_tensor, entropy_tensor

def evaluate_value_estimates(args, env, actor, qf1, qf2, device, safety=False, use_mean_action=False, num_episodes=10, num_trials=10, max_steps=1000):
    """
    Evaluates Q-value overestimation by comparing Q-values to Monte Carlo returns.
    
    Args:
        env: The environment to evaluate in
        actor: The policy network
        qf1, qf2: The Q-networks
        device: The device to run computations on
        num_episodes: Number of episodes to average over
        num_trials: Number of trials per episode
        max_steps: Maximum steps per episode
        
    Returns:
        mean_q_error: Average error between Q-values and MC returns
        mean_q_values: Average predicted Q-values
        mean_mc_values: Average Monte Carlo returns
    """
    q1_tensor = np.zeros((num_episodes, num_trials, max_steps), dtype=np.float32)
    q2_tensor = np.zeros((num_episodes, num_trials, max_steps), dtype=np.float32)
    cdq_tensor = np.zeros((num_episodes, num_trials, max_steps), dtype=np.float32)
    entropy_tensor = np.zeros((num_episodes, num_trials, max_steps), dtype=np.float32)
    mc_returns_tensor = np.zeros((num_episodes, num_trials, max_steps), dtype=np.float32)

    for episode in range(num_episodes):
        start_state, _ = env.reset() # Reset without saved state 
        env.save_state() # save the start state so that you can do multiple trials 
        for trial in range(num_trials):
            if trial: # if its not the first trial (reset to the saved state)
                start_state, _ = env.reset(use_saved_state=True)
            
            q1_values, q2_values, cdq_values, mc_returns, entropy = perform_rollout(start_state, env, actor, qf1, qf2, device, args, max_steps, safety, use_mean_action)
            q1_tensor[episode, trial, :len(q1_values)] = np.array(q1_values)
            q2_tensor[episode, trial, :len(q2_values)] = np.array(q2_values)
            cdq_tensor[episode, trial, :len(cdq_values)] = np.array(cdq_values)
            entropy_tensor[episode, trial, :len(entropy)] = np.array(entropy)
            mc_returns_tensor[episode, trial, :len(mc_returns)] = mc_returns

    mean_q_error = np.mean(q1_tensor - mc_returns_tensor) + np.mean(q2_tensor - mc_returns_tensor)
    mean_q_values = (np.mean(q1_tensor) + np.mean(q2_tensor)) / 2
    mean_mc_values = np.mean(mc_returns_tensor)

    results = {
        "mean_q_error": mean_q_error,
        "mean_q_values": mean_q_values,
        "mean_mc_values": mean_mc_values,
        "entropy": entropy_tensor,
        "q1_tensor": q1_tensor,
        "q2_tensor": q2_tensor,
        "cdq_tensor": cdq_tensor,
        "mc_returns_tensor": mc_returns_tensor
    }
    return results


def analyze_estimation(writer, global_step, results):
    q1, q2, cdq, mc = results["q1_tensor"], results["q2_tensor"], results["cdq_tensor"], results["mc_returns_tensor"]
    q1_s0, q2_s0, cdq_s0, mc_s0 = q1[:, :, 0], q2[:, :, 0], cdq[:, :, 0], mc[:, :, 0]

    q_std, mc_std, mc_mean = np.abs((q1 - q2) / 2), np.std(mc, axis=1), np.mean(mc, axis=1)

    estimation_error = (cdq - mc_mean)

    q_std_s0, mc_std_s0, estimation_error_s0 = q_std[:, :, 0], mc_std[:, 0], estimation_error[:, :, 0]
    std_err_s0 = np.mean(np.abs(np.mean(q_std_s0, axis=1) - mc_std_s0))

    q1_estimation_err = np.mean(q1_s0 - mc_mean[:, 0])
    q2_estimation_err = np.mean(q2_s0 - mc_mean[:, 0])
    cdq_estimation_err = np.mean(cdq_s0 - mc_mean[:, 0])
    mean_q_estimation_err = np.mean((q1_s0+q2_s0) / 2 - mc_mean[:, 0])

    q1_rel_error = np.mean((q1_s0 - mc_mean[:, 0])) / np.mean(mc_mean[:, 0])
    q2_rel_error = np.mean((q2_s0 - mc_mean[:, 0])) / np.mean(mc_mean[:, 0])
    cdq_rel_error = np.mean((cdq_s0 - mc_mean[:, 0])) / np.mean(mc_mean[:, 0])
    mean_q_rel_error = np.mean(((q1_s0 + q2_s0)/2 - mc_mean[:, 0])) / np.mean(mc_mean[:, 0])

    ## Let's see if there is a correlation between variance and overestimation
    fig, ax = plt.subplots()
    print(q_std_s0.shape, estimation_error_s0.shape)
    ax.scatter(q_std_s0.flatten(), estimation_error_s0.flatten())
    ax.set_xlabel('Q-Value Standard Deviation')
    ax.set_ylabel('Estimation Error')
    ax.set_title('Q-Value Standard Deviation vs Estimation Error')
    wandb.log({f"Q-Value_Std_vs_Estimation_Error_{global_step}": wandb.Image(fig)})

    corrcoef_s0 = np.corrcoef(q_std_s0.flatten(), estimation_error_s0.flatten())[0, 1]

    writer.add_scalar("Estimation/correlation (s0)", corrcoef_s0, global_step)
    writer.add_scalar("Estimation/StdDev Err (s0)", std_err_s0, global_step)
    writer.add_scalar("Estimation/Mean MC StdDev (s0)", np.mean(mc_std_s0), global_step)
    writer.add_scalar("Estimation/Mean Q StdDev (s0)", np.mean(q_std_s0), global_step)
    writer.add_scalar("Estimation/Mean Q1 (s0)", np.mean(q1_s0), global_step)
    writer.add_scalar("Estimation/Mean Q2 (s0)", np.mean(q2_s0), global_step)
    writer.add_scalar("Estimation/Mean MC (s0)", np.mean(mc_mean), global_step)
    writer.add_scalar("Estimation/Mean CDQ (s0)", np.mean(cdq_s0), global_step)



    writer.add_scalar("EstimationErrors/Q1_Error (s0)", q1_estimation_err, global_step)
    writer.add_scalar("EstimationErrors/Q2_Error (s0)", q2_estimation_err, global_step)
    writer.add_scalar("EstimationErrors/CDQ_Error (s0)", cdq_estimation_err, global_step)
    writer.add_scalar("EstimationErrors/Mean_Q_Error (s0)", mean_q_estimation_err, global_step)
    writer.add_scalar("EstimationErrors/Q1_Relative_Error (s0)", q1_rel_error, global_step)
    writer.add_scalar("EstimationErrors/Q2_Relative_Error (s0)", q2_rel_error, global_step)
    writer.add_scalar("EstimationErrors/CDQ_Relative_Error (s0)", cdq_rel_error, global_step)
    writer.add_scalar("EstimationErrors/Mean_Q_Relative_Error (s0)", mean_q_rel_error, global_step)

    writer.add_scalar("Entropy/Mean Entropy (s0)", np.mean(results["entropy"][:, :, 0]), global_step)









