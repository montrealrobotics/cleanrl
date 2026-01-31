import os
import torch
import wandb
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


def correlation_plot(x, y, xlabel="", ylabel="", title="", wandb_label="", distances=None):
    fig, ax = plt.subplots()
    if distances is not None:
        scatter = ax.scatter(x, y, c=distances, alpha=0.3, cmap='viridis')
        plt.colorbar(scatter, label='Distance from Start State')
    else:
        ax.scatter(x, y, alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    wandb.log({f"{wandb_label} (plots)/{title}": wandb.Image(fig)})
    # plt.close(fig)

def analyze_reward_safety_correlation(q_values, safety_q_values, q_mean, safety_q_mean, 
                                     q_std, safety_q_std, mc_returns, safety_mc_returns, 
                                     q1_values, safety_q1_values, q2_values, safety_q2_values, 
                                     cdq_values, safety_cdq_values, state, global_step,
                                     distances_to_failure=None):
    """
    Analyze correlations between reward and safety value estimates
    
    Args:
        q_values, safety_q_values: Q-values for reward and safety
        q_mean, safety_q_mean: Mean Q-values for reward and safety
        q_std, safety_q_std: Q-value standard deviations for reward and safety
        mc_returns, safety_mc_returns: Monte Carlo returns for reward and safety
        q1_values, safety_q1_values: Q1 values for reward and safety
        q2_values, safety_q2_values: Q2 values for reward and safety
        cdq_values, safety_cdq_values: CDQ values for reward and safety
        state: String indicating state (e.g., "(s0)")
        global_step: Current training step
        distances_to_failure: Array of distances to failure for each state (optional)
    """
    # Compute correlations between reward and safety values
    if len(q_values) != len(safety_q_values):
        print(f"Warning: Length mismatch between reward ({len(q_values)}) and safety ({len(safety_q_values)}) arrays")
        return
    
    # Q-value correlations
    metrics = {}
    metrics[f"Reward_Safety_Correlation {state}/Q_Values"] = np.corrcoef(q_values, safety_q_values)[0, 1]
    metrics[f"Reward_Safety_Correlation {state}/Q1_Values"] = np.corrcoef(q1_values, safety_q1_values)[0, 1]
    metrics[f"Reward_Safety_Correlation {state}/Q2_Values"] = np.corrcoef(q2_values, safety_q2_values)[0, 1]
    metrics[f"Reward_Safety_Correlation {state}/CDQ_Values"] = np.corrcoef(cdq_values, safety_cdq_values)[0, 1]
    metrics[f"Reward_Safety_Correlation {state}/Q_Mean_Values"] = np.corrcoef(q_mean, safety_q_mean)[0, 1]
    metrics[f"Reward_Safety_Correlation {state}/Q_Std_Values"] = np.corrcoef(q_std, safety_q_std)[0, 1]
    
    # MC returns correlations
    metrics[f"Reward_Safety_Correlation {state}/MC_Returns"] = np.corrcoef(mc_returns, safety_mc_returns)[0, 1]
    metrics[f"Reward_Safety_Correlation {state}/MC_Returns vs Q-values"] = np.corrcoef(safety_mc_returns, q_values)[0, 1]
    metrics[f"Reward_Safety_Correlation {state}/safetyQ_vs_rewardQ"] = np.corrcoef(safety_q_values, q_values)[0, 1]
    
    # Estimation errors
    reward_error = np.array(q_values) - np.array(mc_returns)
    safety_error = np.array(safety_q_values) - np.array(safety_mc_returns)
    metrics[f"Reward_Safety_Correlation {state}/Estimation_Errors"] = np.corrcoef(reward_error, safety_error)[0, 1]
    
    # Relative errors
    reward_rel_error = (np.array(q_values) - np.array(mc_returns)) / (np.array(mc_returns) + 1)
    safety_rel_error = (np.array(safety_q_values) - np.array(safety_mc_returns)) / (np.array(safety_mc_returns) + 1)
    metrics[f"Reward_Safety_Correlation {state}/Relative_Estimation_Errors"] = np.corrcoef(reward_rel_error, safety_rel_error)[0, 1]
    
    # Std correlation
    metrics[f"Reward_Safety_Correlation {state}/Std_Correlation"] = np.corrcoef(q_std, safety_q_std)[0, 1]
    
    # Trade-off metrics
    reward_positive = np.array(q_values) > 0
    safety_risk = np.array(safety_q_values) > 0.5
    metrics[f"Reward_Safety_Tradeoff {state}/Disagreement_Rate"] = np.mean(np.logical_xor(reward_positive, safety_risk))
    
    # Log all metrics to wandb
    wandb.log(metrics, step=global_step)
    
    # Generate correlation plots
    correlation_plot(q_values, safety_q_values, 'Reward Q-Values', 'Safety Q-Values', 
                    'Reward vs Safety Q-Values', f"Reward_Safety_Correlation {state}", distances_to_failure)
    correlation_plot(safety_mc_returns, q_values, 'Reward MC Returns', 'Reward Q-Values', 
                    'Safety MC Returns vs Reward Q-Values', f"Reward_Safety_Correlation {state}", distances_to_failure)
    correlation_plot(q_std, safety_q_std, 'Reward Q-Value Std', 'Safety Q-Value Std', 
                    'Reward vs Safety Q-Value Standard Deviation', f"Reward_Safety_Correlation {state}", distances_to_failure)
    correlation_plot(mc_returns, safety_mc_returns, 'Reward MC Returns', 'Safety MC Returns', 
                    'Reward vs Safety MC Returns', f"Reward_Safety_Correlation {state}", distances_to_failure)
    correlation_plot(reward_error, safety_error, 'Reward Estimation Error', 'Safety Estimation Error', 
                    'Reward vs Safety Estimation Error', f"Reward_Safety_Correlation {state}", distances_to_failure)

def analyze_values(q_values, q_mean, q_std, mc_returns, q1_values, q2_values, cdq_values, state, global_step, prefix="", distances=None, policy_entropies=None):
    """
    Analyze the relationship between predicted values and Monte Carlo returns
    
    Args:
        prefix: String prefix for the wandb metrics ("Reward" or "Safety")
        distances: Array of distances from start state for each data point
        policy_entropies: Array of policy entropies for each state
    """
    if prefix:
        prefix = prefix + "_"
    
    metrics = {}
    
    # Mean values
    metrics[f"{prefix}Estimation {state}/Mean_Q_Values"] = np.mean(q_values)
    metrics[f"{prefix}Estimation {state}/Mean_Q1_Values"] = np.mean(q1_values)
    metrics[f"{prefix}Estimation {state}/Mean_Q2_Values"] = np.mean(q2_values)
    metrics[f"{prefix}Estimation {state}/Mean_CDQ_Values"] = np.mean(cdq_values)
    metrics[f"{prefix}Estimation {state}/Mean_MC_Values"] = np.mean(mc_returns)

    # Estimation errors
    mean_q_error = np.array(q_values) - np.array(mc_returns)
    mean_q1_error = np.array(q1_values) - np.array(mc_returns)
    mean_q2_error = np.array(q2_values) - np.array(mc_returns)
    mean_cdq_error = np.array(cdq_values) - np.array(mc_returns)
    mean_q_mean_error = np.array(q_mean) - np.array(mc_returns)

    metrics[f"{prefix}EstimationErrors {state}/Mean_Q_Error"] = np.mean(mean_q_error)
    metrics[f"{prefix}EstimationErrors {state}/Mean_Q1_Error"] = np.mean(mean_q1_error)
    metrics[f"{prefix}EstimationErrors {state}/Mean_Q2_Error"] = np.mean(mean_q2_error)
    metrics[f"{prefix}EstimationErrors {state}/Mean_CDQ_Error"] = np.mean(mean_cdq_error)
    metrics[f"{prefix}EstimationErrors {state}/Mean_Q_Mean_Error"] = np.mean(mean_q_mean_error)

    # Relative errors
    mean_q_rel_error = (np.array(q_values) - np.array(mc_returns)) / (np.array(mc_returns) + 1)
    mean_q1_rel_error = (np.array(q1_values) - np.array(mc_returns)) / (np.array(mc_returns) + 1)
    mean_q2_rel_error = (np.array(q2_values) - np.array(mc_returns)) / (np.array(mc_returns) + 1)
    mean_cdq_rel_error = (np.array(cdq_values) - np.array(mc_returns)) / (np.array(mc_returns) + 1)
    mean_q_mean_rel_error = (np.array(q_mean) - np.array(mc_returns)) / (np.array(mc_returns) + 1)

    metrics[f"{prefix}RelEstimationErrors {state}/Mean_Q_Relative_Error"] = np.mean(mean_q_rel_error)
    metrics[f"{prefix}RelEstimationErrors {state}/Mean_Q1_Relative_Error"] = np.mean(mean_q1_rel_error)
    metrics[f"{prefix}RelEstimationErrors {state}/Mean_Q2_Relative_Error"] = np.mean(mean_q2_rel_error)
    metrics[f"{prefix}RelEstimationErrors {state}/Mean_CDQ_Relative_Error"] = np.mean(mean_cdq_rel_error)
    metrics[f"{prefix}RelEstimationErrors {state}/Mean_Q_Mean_Relative_Error"] = np.mean(mean_q_mean_rel_error)

    # Policy entropy metrics if provided
    if policy_entropies is not None:
        metrics[f"{prefix}EntropyStats {state}/Mean_Entropy"] = np.mean(policy_entropies)
        metrics[f"{prefix}EntropyStats {state}/Std_Entropy"] = np.std(policy_entropies)
        metrics[f"{prefix}EntropyStats {state}/Max_Entropy"] = np.max(policy_entropies)
        metrics[f"{prefix}EntropyStats {state}/Min_Entropy"] = np.min(policy_entropies)

    # Q-value standard deviation statistics
    metrics[f"{prefix}Q_Std_Stats {state}/Mean_Q_Std"] = np.mean(q_std)
    metrics[f"{prefix}Q_Std_Stats {state}/Max_Q_Std"] = np.max(q_std)
    metrics[f"{prefix}Q_Std_Stats {state}/Q_Std_25th_Percentile"] = np.percentile(q_std, 25)
    metrics[f"{prefix}Q_Std_Stats {state}/Q_Std_50th_Percentile"] = np.percentile(q_std, 50)
    metrics[f"{prefix}Q_Std_Stats {state}/Q_Std_75th_Percentile"] = np.percentile(q_std, 75)
    metrics[f"{prefix}Q_Std_Stats {state}/Q_Std_90th_Percentile"] = np.percentile(q_std, 90)
    metrics[f"{prefix}Q_Std_Stats {state}/Q_Std_95th_Percentile"] = np.percentile(q_std, 95)
    metrics[f"{prefix}Q_Std_Stats {state}/Q_Std_99th_Percentile"] = np.percentile(q_std, 99)

    # Relative Q-value standard deviation statistics
    relative_q_std = q_std / (np.abs(q_values) + 1)
    metrics[f"{prefix}Q_Std_Stats {state}/Mean_Q_Std_Relative"] = np.mean(relative_q_std)
    metrics[f"{prefix}Q_Std_Stats {state}/Max_Q_Std_Relative"] = np.max(relative_q_std)
    metrics[f"{prefix}Q_Std_Stats {state}/Q_Std_25th_Percentile_Relative"] = np.percentile(relative_q_std, 25)
    metrics[f"{prefix}Q_Std_Stats {state}/Q_Std_50th_Percentile_Relative"] = np.percentile(relative_q_std, 50)
    metrics[f"{prefix}Q_Std_Stats {state}/Q_Std_75th_Percentile_Relative"] = np.percentile(relative_q_std, 75)
    metrics[f"{prefix}Q_Std_Stats {state}/Q_Std_90th_Percentile_Relative"] = np.percentile(relative_q_std, 90)
    metrics[f"{prefix}Q_Std_Stats {state}/Q_Std_95th_Percentile_Relative"] = np.percentile(relative_q_std, 95)
    metrics[f"{prefix}Q_Std_Stats {state}/Q_Std_99th_Percentile_Relative"] = np.percentile(relative_q_std, 99)

    # Log all metrics to wandb
    wandb.log(metrics, step=global_step)

    # Generate correlation plots
    correlation_plot(q_std, mean_q_error, 'Q-Value Standard Deviation', 'Estimation Error', 
                    'Q-Value Standard Deviation vs Estimation Error', f"{prefix}Correlation Plots {state}", distances)
    correlation_plot(q_std, mean_q1_error, 'Q-Value Standard Deviation', 'Q1 Estimation Error', 
                    'Q-Value Standard Deviation vs Q1 Estimation Error', f"{prefix}Correlation Plots {state}", distances)
    correlation_plot(q_std, mean_q2_error, 'Q-Value Standard Deviation', 'Q2 Estimation Error', 
                    'Q-Value Standard Deviation vs Q2 Estimation Error', f"{prefix}Correlation Plots {state}", distances)
    correlation_plot(q_std, mean_cdq_error, 'Q-Value Standard Deviation', 'CDQ Estimation Error', 
                    'Q-Value Standard Deviation vs CDQ Estimation Error', f"{prefix}Correlation Plots {state}", distances)
    correlation_plot(q_std, mean_q_mean_error, 'Q-Value Standard Deviation', 'Mean Q Estimation Error', 
                    'Q-Value Standard Deviation vs Mean Q Estimation Error', f"{prefix}Correlation Plots {state}", distances)
    
    # MC vs Q plots
    correlation_plot(mc_returns, q_values, 'MC Returns', 'Q-Values', 
                    'MC Returns vs Q-Values', f"{prefix}MC vs Q Plots {state}", distances)
    correlation_plot(mc_returns, q1_values, 'MC Returns', 'Q1-Values', 
                    'MC Returns vs Q1-Values', f"{prefix}MC vs Q Plots {state}", distances)
    correlation_plot(mc_returns, q2_values, 'MC Returns', 'Q2-Values', 
                    'MC Returns vs Q2-Values', f"{prefix}MC vs Q Plots {state}", distances)
    correlation_plot(mc_returns, cdq_values, 'MC Returns', 'CDQ-Values', 
                    'MC Returns vs CDQ-Values', f"{prefix}MC vs Q Plots {state}", distances)
    correlation_plot(mc_returns, q_mean, 'MC Returns', 'Mean Q-Values', 
                    'MC Returns vs Mean Q-Values', f"{prefix}MC vs Q Plots {state}", distances)

    # Policy entropy plots if provided
    if policy_entropies is not None:
        correlation_plot(policy_entropies, q_values, 'Policy Entropy', 'Q-Values', 
                        'Policy Entropy vs Q-Values', f"{prefix}Entropy Correlation {state}", distances)
        correlation_plot(policy_entropies, q1_values, 'Policy Entropy', 'Q1-Values', 
                        'Policy Entropy vs Q1-Values', f"{prefix}Entropy Correlation {state}", distances)
        correlation_plot(policy_entropies, q2_values, 'Policy Entropy', 'Q2-Values', 
                        'Policy Entropy vs Q2-Values', f"{prefix}Entropy Correlation {state}", distances)
        correlation_plot(policy_entropies, cdq_values, 'Policy Entropy', 'CDQ-Values', 
                        'Policy Entropy vs CDQ-Values', f"{prefix}Entropy Correlation {state}", distances)
        correlation_plot(policy_entropies, mc_returns, 'Policy Entropy', 'MC Returns', 
                        'Policy Entropy vs MC Returns', f"{prefix}Entropy Correlation {state}", distances)

def evaluate_value_estimates(
    global_step,
    args,
    env,
    actor,
    qf1,
    qf2,
    device,
    safety_qf1=None,
    safety_qf2=None,
    safety_mode="both",
    num_episodes=100,
    max_steps=1000,
    max_episode_steps=2000,
):
    """
    Evaluates Q-value overestimation by comparing Q-values to Monte Carlo returns for both reward and safety critics.
    
    Args:
        global_step: Current training step
        args: Training arguments
        env: The environment to evaluate in
        actor: The policy network
        qf1, qf2: The reward Q-networks
        safety_qf1, safety_qf2: The safety Q-networks (optional)
        device: The device to run computations on
        safety_mode: One of "reward", "safety", or "both"
        num_episodes: Number of episodes to average over
        max_steps: Number of visited states to evaluate per episode
        max_episode_steps: Rollout horizon for MC estimation
    """
    def compute_truncated_returns(rewards, gamma, horizon):
        rewards_np = np.asarray(rewards, dtype=np.float32)
        seq_len = len(rewards_np)
        if seq_len == 0:
            return []
        full_returns = np.zeros(seq_len + 1, dtype=np.float32)
        for t in range(seq_len - 1, -1, -1):
            full_returns[t] = rewards_np[t] + gamma * full_returns[t + 1]
        if horizon <= 0:
            return np.zeros(seq_len, dtype=np.float32).tolist()
        truncated_returns = np.zeros(seq_len, dtype=np.float32)
        gamma_h = gamma ** horizon
        for t in range(seq_len):
            if t + horizon < seq_len:
                truncated_returns[t] = full_returns[t] - gamma_h * full_returns[t + horizon]
            else:
                truncated_returns[t] = full_returns[t]
        return truncated_returns.tolist()

    # Initialize arrays for reward critic
    q1_values, q2_values, cdq_values, q_values, q_std, q_mean = [], [], [], [], [], []
    mc_returns = []
    policy_entropies = []  # New array for policy entropies

    # Initialize arrays for safety critic if provided
    if safety_qf1 is not None and safety_qf2 is not None and (safety_mode == "safety" or safety_mode == "both"):
        safety_q1_values, safety_q2_values = [], []
        safety_cdq_values, safety_q_values = [], []
        safety_q_std, safety_q_mean = [], []
        safety_mc_returns = []

    # Arrays for storing data across all steps
    q_array = np.zeros((num_episodes, max_steps), dtype=np.float32)
    q1_array = np.zeros((num_episodes, max_steps), dtype=np.float32)
    q2_array = np.zeros((num_episodes, max_steps), dtype=np.float32)
    q_mean_array = np.zeros((num_episodes, max_steps), dtype=np.float32)
    cdq_array = np.zeros((num_episodes, max_steps), dtype=np.float32)
    q_std_array = np.zeros((num_episodes, max_steps), dtype=np.float32)
    mc_returns_array = np.zeros((num_episodes, max_steps), dtype=np.float32)
    policy_entropy_array = np.zeros((num_episodes, max_steps), dtype=np.float32)  # New array for policy entropies
    
    distances_to_term_array = np.zeros((num_episodes, max_steps), dtype=np.float32)
    distances_from_start_array = np.zeros((num_episodes, max_steps), dtype=np.float32)
    
    # Safety arrays if needed
    if safety_qf1 is not None and safety_qf2 is not None and (safety_mode == "safety" or safety_mode == "both"):
        safety_q_array = np.zeros((num_episodes, max_steps), dtype=np.float32)
        safety_q1_array = np.zeros((num_episodes, max_steps), dtype=np.float32)
        safety_q2_array = np.zeros((num_episodes, max_steps), dtype=np.float32)
        safety_q_mean_array = np.zeros((num_episodes, max_steps), dtype=np.float32)
        safety_cdq_array = np.zeros((num_episodes, max_steps), dtype=np.float32)
        safety_q_std_array = np.zeros((num_episodes, max_steps), dtype=np.float32)
        safety_mc_returns_array = np.zeros((num_episodes, max_steps), dtype=np.float32)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        episode_rewards = []
        episode_safety = []  # For tracking safety (termination) events
        
        # Run episode and collect rewards
        while not done and step < max_episode_steps:
            with torch.no_grad():
                state = torch.FloatTensor(obs).to(device)
                action, log_prob, entropy = actor.get_action(state.unsqueeze(0))  # Get entropy from policy
                
                # Get reward Q-values
                q1 = qf1(state.unsqueeze(0), action)
                q2 = qf2(state.unsqueeze(0), action)
                
                # Get safety Q-values if needed
                if safety_qf1 is not None and safety_qf2 is not None and (safety_mode == "safety" or safety_mode == "both"):
                    safety_q1 = safety_qf1(state.unsqueeze(0), action)
                    safety_q2 = safety_qf2(state.unsqueeze(0), action)

            # Record first step values for reward critic
            if step == 0:
                q1_values.append(q1.item())
                q2_values.append(q2.item())
                cdq_values.append(torch.min(q1, q2).item())
                q_std.append(torch.abs(q1-q2).item())
                q_mean.append(((q1+q2)/2).item())
                policy_entropies.append(log_prob.mean().item())  # Take mean of entropy if it's a multi-element tensor

                if args.use_cdq == "True":
                    q_value = torch.min(q1, q2).item()
                elif args.use_cdq == "beta":
                    q_value = ((q1+q2)/2 - args.beta * (q1-q2)/2).item()
                else:
                    q_value = q1.item()
            
                q_values.append(q_value)
                
                # Record first step values for safety critic if needed
                if safety_qf1 is not None and safety_qf2 is not None and (safety_mode == "safety" or safety_mode == "both"):
                    safety_q1_values.append(safety_q1.item())
                    safety_q2_values.append(safety_q2.item())
                    safety_cdq_values.append(torch.max(safety_q1, safety_q2).item())  # For safety, we take the max
                    safety_q_std.append(torch.abs(safety_q1-safety_q2).item())
                    safety_q_mean.append(((safety_q1+safety_q2)/2).item())
                    
                    if args.use_cdq == "True":
                        safety_q_value = torch.max(safety_q1, safety_q2).item()  # For safety, we take the max
                    elif args.use_cdq == "beta":
                        safety_q_value = ((safety_q1+safety_q2)/2 + args.beta * (safety_q1-safety_q2)/2).item()
                    else:
                        safety_q_value = safety_q1.item()
                    
                    safety_q_values.append(safety_q_value)
            
            if step < max_steps:
                # Record all step values for reward critic
                q1_array[episode, step] = q1.item()
                q2_array[episode, step] = q2.item()
                cdq_array[episode, step] = torch.min(q1, q2).item()
                q_array[episode, step] = q_value if step == 0 else 0  # Only for first step
                q_std_array[episode, step] = torch.abs(q1 - q2).item() / 2
                q_mean_array[episode, step] = ((q1 + q2)/2).item()
                policy_entropy_array[episode, step] = log_prob.mean().item()  # Take mean of entropy if it's a multi-element tensor
                
                # Record all step values for safety critic if needed
                if safety_qf1 is not None and safety_qf2 is not None and (safety_mode == "safety" or safety_mode == "both"):
                    safety_q1_array[episode, step] = safety_q1.item()
                    safety_q2_array[episode, step] = safety_q2.item()
                    safety_cdq_array[episode, step] = torch.max(safety_q1, safety_q2).item()  # For safety, take max
                    safety_q_array[episode, step] = safety_q_value if step == 0 else 0  # Only for first step
                    safety_q_std_array[episode, step] = torch.abs(safety_q1 - safety_q2).item() / 2
                    safety_q_mean_array[episode, step] = ((safety_q1 + safety_q2)/2).item()

            # Execute action in environment
            action = action.squeeze().detach().cpu().numpy()
            obs, reward, terminated, truncated, _ = env.step(action)
            
            # Record reward and safety outcomes
            episode_rewards.append(reward)
            episode_safety.append(int(terminated))  # 1 if terminated, 0 otherwise
            
            done = terminated or truncated
            step += 1

        # Calculate Monte Carlo returns for reward
        # Calculate Monte Carlo returns and distances
        mc_return = compute_truncated_returns(episode_rewards, args.gamma, max_steps)
        distance_to_term = [
            len(episode_rewards) - 1 - i for i in range(len(episode_rewards))
        ]
        distance_from_start = list(range(len(episode_rewards)))

        if mc_return:
            mc_returns.append(mc_return[0])  # First step return
        mc_len = min(len(mc_return), max_steps)
        if mc_len > 0:
            mc_returns_array[episode, :mc_len] = np.array(mc_return[:mc_len])
            distances_to_term_array[episode, :mc_len] = np.array(distance_to_term[:mc_len]) 
            distances_from_start_array[episode, :mc_len] = np.array(distance_from_start[:mc_len])
        
        # Calculate Monte Carlo returns for safety if needed
        if safety_qf1 is not None and safety_qf2 is not None and (safety_mode == "safety" or safety_mode == "both"):
            safety_mc_return = compute_truncated_returns(episode_safety, args.gamma, max_steps)
            if safety_mc_return:
                safety_mc_returns.append(safety_mc_return[0])  # First step return
            safety_len = min(len(safety_mc_return), max_steps)
            if safety_len > 0:
                safety_mc_returns_array[episode, :safety_len] = np.array(safety_mc_return[:safety_len])

    # Save results
    import pickle

    Q_dict = {
        'q1_values_s0': q1_values,
        'q2_values_s0': q2_values,
        'cdq_values_s0': cdq_values,
        'q_values_s0': q_values,
        'q_std_s0': q_std,
        'q_mean_s0': q_mean,
        'policy_entropies_s0': policy_entropies,  # Add policy entropies to saved data

        "q1_array": q1_array,
        "q2_array": q2_array,
        "q_array": q_array,
        "mc_array": mc_returns_array,
        "cdq_array": cdq_array,
        "q_std_array": q_std_array,
        "q_mean_array": q_mean_array,
        "policy_entropy_array": policy_entropy_array,  # Add policy entropy array to saved data
    }
    
    # Add safety values to dictionary if needed
    if safety_qf1 is not None and safety_qf2 is not None and (safety_mode == "safety" or safety_mode == "both"):
        safety_dict = {
            'safety_q1_values_s0': safety_q1_values,
            'safety_q2_values_s0': safety_q2_values,
            'safety_cdq_values_s0': safety_cdq_values,
            'safety_q_values_s0': safety_q_values,
            'safety_q_std_s0': safety_q_std,
            'safety_q_mean_s0': safety_q_mean,

            "safety_q1_array": safety_q1_array,
            "safety_q2_array": safety_q2_array,
            "safety_q_array": safety_q_array,
            "safety_mc_array": safety_mc_returns_array,
            "safety_cdq_array": safety_cdq_array,
            "safety_q_std_array": safety_q_std_array,
            "safety_q_mean_array": safety_q_mean_array,
        }
        Q_dict.update(safety_dict)

    with open(os.path.join(wandb.run.dir, f'results_{global_step}.pkl'), 'wb') as f:
        pickle.dump(Q_dict, f)

    wandb.save(os.path.join(wandb.run.dir, f'results_{global_step}.pkl'))
    
    # Analyze reward critic values
    if safety_mode == "reward" or safety_mode == "both":
        analyze_values(q_values, q_mean, q_std, mc_returns, q1_values, q2_values, cdq_values, "(s0)", global_step, "Reward", distances_to_term_array[:, 0], policy_entropies)
        analyze_values(q_array.flatten(), q_mean_array.flatten(), q_std_array.flatten(), mc_returns_array.flatten(),
                      q1_array.flatten(), q2_array.flatten(), cdq_array.flatten(), "", global_step, "Reward", distances_to_term_array.flatten(), policy_entropy_array.flatten())
    
    # Analyze safety critic values if needed
    if safety_qf1 is not None and safety_qf2 is not None and (safety_mode == "safety" or safety_mode == "both"):
        analyze_values(safety_q_values, safety_q_mean, safety_q_std, safety_mc_returns, safety_q1_values, safety_q2_values, 
                      safety_cdq_values, "(s0)", global_step, "Safety", distances_to_term_array[:, 0], policy_entropies)
        analyze_values(safety_q_array.flatten(), safety_q_mean_array.flatten(), safety_q_std_array.flatten(), 
                      safety_mc_returns_array.flatten(), safety_q1_array.flatten(), safety_q2_array.flatten(), 
                      safety_cdq_array.flatten(), "", global_step, "Safety", distances_to_term_array.flatten(), policy_entropy_array.flatten())
    
    # Analyze reward-safety correlations if both are being evaluated
    if safety_qf1 is not None and safety_qf2 is not None and safety_mode == "both":
        analyze_reward_safety_correlation(q_values, safety_q_values, q_mean, safety_q_mean, 
                                         q_std, safety_q_std, mc_returns, safety_mc_returns, 
                                         q1_values, safety_q1_values, q2_values, safety_q2_values, 
                                         cdq_values, safety_cdq_values, "(s0)", global_step, distances_to_term_array[:, 0])
        
        # Also analyze correlations across all steps
        analyze_reward_safety_correlation(
            q_array.flatten(), safety_q_array.flatten(),
            q_mean_array.flatten(), safety_q_mean_array.flatten(),
            q_std_array.flatten(), safety_q_std_array.flatten(),
            mc_returns_array.flatten(), safety_mc_returns_array.flatten(),
            q1_array.flatten(), safety_q1_array.flatten(),
            q2_array.flatten(), safety_q2_array.flatten(),
            cdq_array.flatten(), safety_cdq_array.flatten(),
            "", global_step, distances_to_term_array.flatten())