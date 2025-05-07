import os
import torch
import wandb
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


def correlation_plot(x, y, xlabel="", ylabel="", title="", wandb_label="", c=None, cmap="viridis"):
    fig, ax = plt.subplots()
    if c is not None:
        scatter = ax.scatter(x, y, c=c, cmap=cmap, alpha=0.2)  # Added alpha=0.5 for translucency
        plt.colorbar(scatter, label='Distance to Termination')
    else:
        ax.scatter(x, y, alpha=0.2)  # Added alpha=0.5 for translucency
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    wandb.log({f"{wandb_label}/{title}": wandb.Image(fig)})
    plt.close(fig)



def evaluate_value_estimates(global_step, writer, args, env, actor, qf1, qf2, device, safety=False, num_episodes=100, max_steps=1000):
    """
    Evaluates Q-value overestimation by comparing Q-values to Monte Carlo returns.
    
    Args:
        env: The environment to evaluate in
        actor: The policy network
        qf1, qf2: The Q-networks
        device: The device to run computations on
        num_episodes: Number of episodes to average over
        max_steps: Maximum steps per episode
        
    Returns:
        mean_q_error: Average error between Q-values and MC returns
        mean_q_values: Average predicted Q-values
        mean_mc_values: Average Monte Carlo returns
    """
    q1_values, q2_values, cdq_values, q_values, q_std, q_mean  = [], [], [], [], [], []
    mc_returns = []

    q_array = np.zeros((num_episodes, max_steps), dtype=np.float32)
    q1_array = np.zeros((num_episodes, max_steps), dtype=np.float32)
    q2_array = np.zeros((num_episodes, max_steps), dtype=np.float32)
    q_mean_array = np.zeros((num_episodes, max_steps), dtype=np.float32)
    cdq_array = np.zeros((num_episodes, max_steps), dtype=np.float32)
    q_std_array = np.zeros((num_episodes, max_steps), dtype=np.float32)
    mc_returns_array = np.zeros((num_episodes, max_steps), dtype=np.float32)
    distances_to_term_array = np.zeros((num_episodes, max_steps), dtype=np.float32)
    distances_from_start_array = np.zeros((num_episodes, max_steps), dtype=np.float32)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        episode_rewards = []
        
        # Run episode and collect rewards
        while not done and step < max_steps:

            with torch.no_grad():
                state = torch.FloatTensor(obs).to(device)
                action, _, _ = actor.get_action(state.unsqueeze(0))
                q1 = qf1(state.unsqueeze(0), action)
                q2 = qf2(state.unsqueeze(0), action)

            if step == 0:
                q1_values.append(q1.item())
                q2_values.append(q2.item())
                cdq_values.append(torch.min(q1, q2).item())
                q_std.append(torch.abs(q1-q2).item())
                q_mean.append(((q1+q2)/2).item())

                if args.use_cdq == "True":
                    q_value = torch.min(q1, q2).item()
                elif args.use_cdq == "beta":
                    q_value = ((q1+q2)/2 - args.beta * (q1-q2)/2).item()
                else:
                    q_value = q1.item()
            
                q_values.append(q_value)
            
            q1_array[episode, step] = q1.item()
            q2_array[episode, step] = q2.item()
            cdq_array[episode, step] = torch.min(q1, q2).item()
            q_array[episode, step] = q_value
            q_std_array[episode, step] = torch.abs(q1 - q2).item() / 2
            q_mean_array[episode, step] = ((q1 + q2)/2).item()

            action = action.squeeze().detach().cpu().numpy()
            obs, reward, terminated, truncated, _ = env.step(action)
            if safety:
                reward = int(terminated)
            done = terminated or truncated
            episode_rewards.append(reward)
            step += 1

        distance_to_term = []  # Distance to termination state
        distance_from_start = []  # Distance from start state
        
        # Calculate Monte Carlo return
        mc_return = []
        return_ = 0
        for i, r in enumerate(reversed(episode_rewards)):
            return_ = r + args.gamma * return_
            mc_return.insert(0, return_)
            distance_to_term.insert(0, i)  # Steps until end
        
        for i in range(len(episode_rewards)):
            distance_from_start.append(i)  # Steps since start
            
        mc_returns.append(return_)
        mc_returns_array[episode, :len(mc_return)] = np.array(mc_return)
        distances_to_term_array[episode, :len(distance_to_term)] = np.array(distance_to_term) 
        distances_from_start_array[episode, :len(distance_from_start)] = np.array(distance_from_start)
       
    import pickle

    Q_dict = {
        'q1_values_s0': q1_values,
        'q2_values_s0': q2_values,
        'cdq_values_s0': cdq_values,
        'q_values_s0': q_values,
        'q_std_s0': q_std,
        'q_mean_s0': q_mean,

        "q1_array": q1_array,
        "q2_array": q2_array,
        "q_array": q_array,
        "mc_array": mc_returns_array,
        "cdq_array": cdq_array,
        "q_std_array": q_std_array,
        "q_mean_array": q_mean_array,

    }

    with open(os.path.join(wandb.run.dir, f'results_{global_step}.pkl'), 'wb') as f:
        pickle.dump(Q_dict, f)

    wandb.save(os.path.join(wandb.run.dir, f'results_{global_step}.pkl'))
    analyze_values(q_values, q_mean, q_std, mc_returns, q1_values, q2_values, cdq_values, distances_to_term_array[:, 0].flatten(), "(s0)", global_step, writer)
    analyze_values(q_array.flatten(), q_mean_array.flatten(), q_std_array.flatten(), mc_returns_array.flatten(),\
                    q1_array.flatten(), q2_array.flatten(), cdq_array.flatten(), distances_from_start_array.flatten(), "", global_step, writer)

def analyze_values(q_values, q_mean, q_std, mc_returns, q1_values, q2_values, cdq_values, distances_to_term, state, global_step, writer):
    mean_q_values = np.mean(q_values)
    mean_mc_values = np.mean(mc_returns)
    mean_q1_values = np.mean(q1_values)
    mean_q2_values = np.mean(q2_values)
    mean_cdq_values = np.mean(cdq_values)

    writer.add_scalar(f"Estimation {state}/Mean_Q_Values", mean_q_values, global_step)
    writer.add_scalar(f"Estimation {state}/Mean_Q1_Values", mean_q1_values, global_step)
    writer.add_scalar(f"Estimation {state}/Mean_Q2_Values", mean_q2_values, global_step)
    writer.add_scalar(f"Estimation {state}/Mean_CDQ_Values", mean_cdq_values, global_step)
    writer.add_scalar(f"Estimation {state}/Mean_MC_Values", mean_mc_values, global_step)

    # Computing estimation errors wrt MC returns
    mean_q_error = np.array(q_values) - np.array(mc_returns)
    mean_q1_error = np.array(q1_values) - np.array(mc_returns)
    mean_q2_error = np.array(q2_values) - np.array(mc_returns)
    mean_cdq_error = np.array(cdq_values) - np.array(mc_returns)
    mean_q_mean_error = np.array(q_mean) - np.array(mc_returns)

    writer.add_scalar(f"EstimationErrors {state}/Mean_Q_Error", np.mean(mean_q_error), global_step)
    writer.add_scalar(f"EstimationErrors {state}/Mean_Q1_Error", np.mean(mean_q1_error), global_step)
    writer.add_scalar(f"EstimationErrors {state}/Mean_Q2_Error", np.mean(mean_q2_error), global_step)
    writer.add_scalar(f"EstimationErrors {state}/Mean_CDQ_Error", np.mean(mean_cdq_error), global_step)
    writer.add_scalar(f"EstimationErrors {state}/Mean_Q_Mean_Error", np.mean(mean_q_mean_error), global_step)

    mean_q_rel_error = (np.array(q_values) - np.array(mc_returns)) / (np.array(mc_returns) + 1)
    mean_q1_rel_error = (np.array(q1_values) - np.array(mc_returns)) / (np.array(mc_returns) + 1)
    mean_q2_rel_error = (np.array(q2_values) - np.array(mc_returns)) / (np.array(mc_returns) + 1)
    mean_cdq_rel_error = (np.array(cdq_values) - np.array(mc_returns)) / (np.array(mc_returns) + 1)
    mean_q_mean_rel_error = (np.array(q_mean) - np.array(mc_returns)) / (np.array(mc_returns) + 1)

    writer.add_scalar(f"RelEstimationErrors {state}/Mean_Q_Relative_Error", np.mean(mean_q_rel_error), global_step)
    writer.add_scalar(f"RelEstimationErrors {state}/Mean_Q1_Relative_Error", np.mean(mean_q1_rel_error), global_step)
    writer.add_scalar(f"RelEstimationErrors {state}/Mean_Q2_Relative_Error", np.mean(mean_q2_rel_error), global_step)
    writer.add_scalar(f"RelEstimationErrors {state}/Mean_CDQ_Relative_Error", np.mean(mean_cdq_rel_error), global_step)
    writer.add_scalar(f"RelEstimationErrors {state}/Mean_Q_Mean_Relative_Error", np.mean(mean_q_mean_rel_error), global_step)

    # Add distance to termination coloring to correlation plots
    correlation_plot(q_std, mean_q_error, 'Q-Value Standard Deviation', 'Estimation Error', 'Q-Value Standard Deviation vs Estimation Error', f"Correlation Plots {state}", c=distances_to_term)
    correlation_plot(q_std, mean_q1_error, 'Q-Value Standard Deviation', 'Q1 Estimation Error', 'Q-Value Standard Deviation vs Q1 Estimation Error', f"Correlation Plots {state}", c=distances_to_term)
    correlation_plot(q_std, mean_q2_error, 'Q-Value Standard Deviation', 'Q2 Estimation Error', 'Q-Value Standard Deviation vs Q2 Estimation Error', f"Correlation Plots {state}", c=distances_to_term)
    correlation_plot(q_std, mean_cdq_error, 'Q-Value Standard Deviation', 'CDQ Estimation Error', 'Q-Value Standard Deviation vs CDQ Estimation Error', f"Correlation Plots {state}", c=distances_to_term)
    correlation_plot(q_std, mean_q_mean_error, 'Q-Value Standard Deviation', 'Mean Q Estimation Error', 'Q-Value Standard Deviation vs Mean Q Estimation Error', f"Correlation Plots {state}", c=distances_to_term)
    correlation_plot(q_std, (np.array(q_values) - np.array(mc_returns)) / (np.array(mc_returns)+1), 'Q-Value Standard Deviation', 'Relative Estimation Error', 'Q-Value Standard Deviation vs Relative Estimation Error', f"Relative Correlation Plot {state}", c=distances_to_term)
    correlation_plot(q_std, (np.array(q1_values) - np.array(mc_returns)) / (np.array(mc_returns)+1), 'Q-Value Standard Deviation', 'Q1 Relative Estimation Error', 'Q-Value Standard Deviation vs Q1 Relative Estimation Error', f"Relative Correlation Plot {state}", c=distances_to_term)
    correlation_plot(q_std, (np.array(q2_values) - np.array(mc_returns)) / (np.array(mc_returns)+1), 'Q-Value Standard Deviation', 'Q2 Relative Estimation Error', 'Q-Value Standard Deviation vs Q2 Relative Estimation Error', f"Relative Correlation Plot {state}", c=distances_to_term)
    correlation_plot(q_std, (np.array(cdq_values) - np.array(mc_returns)) / (np.array(mc_returns)+1), 'Q-Value Standard Deviation', 'CDQ Relative Estimation Error', 'Q-Value Standard Deviation vs CDQ Relative Estimation Error', f"Relative Correlation Plot {state}", c=distances_to_term)
    correlation_plot(q_std, (np.array(q_mean) - np.array(mc_returns)) / (np.array(mc_returns)+1), 'Q-Value Standard Deviation', 'Mean Q Relative Estimation Error', 'Q-Value Standard Deviation vs Mean Q Relative Estimation Error', f"Relative Correlation Plot {state}", c=distances_to_term)

    corrcoef_q = np.corrcoef(q_std, mean_q_error)[0, 1]
    corrcoef_q1 = np.corrcoef(q_std, mean_q1_error)[0, 1]
    corrcoef_q2 = np.corrcoef(q_std, mean_q2_error)[0, 1]
    corrcoef_cdq = np.corrcoef(q_std, mean_cdq_error)[0, 1]
    corrcoef_q_mean = np.corrcoef(q_std, mean_q_mean_error)[0, 1]

    writer.add_scalar(f"Correlation {state}/Q-Std_vs_Q-Estimation_Error", corrcoef_q, global_step)
    writer.add_scalar(f"Correlation {state}/Q-Std_vs_Q1-Estimation_Error", corrcoef_q1, global_step)
    writer.add_scalar(f"Correlation {state}/Q_Std_vs_Q2-Estimation_Error", corrcoef_q2, global_step)
    writer.add_scalar(f"Correlation {state}/Q_Std_vs_CDQ-Estimation_Error", corrcoef_cdq, global_step)
    writer.add_scalar(f"Correlation {state}/Q_Std_vs_Q_mean_Estimation_Error", corrcoef_q_mean, global_step)

    # Log Q-value standard deviation statistics
    writer.add_scalar(f"Q_Std_Stats {state}/Mean", np.mean(q_std), global_step)
    writer.add_scalar(f"Q_Std_Stats {state}/Median", np.median(q_std), global_step)
    writer.add_scalar(f"Q_Std_Stats {state}/25th_Percentile", np.percentile(q_std, 25), global_step)
    writer.add_scalar(f"Q_Std_Stats {state}/75th_Percentile", np.percentile(q_std, 75), global_step)
    writer.add_scalar(f"Q_Std_Stats {state}/Min", np.min(q_std), global_step)
    writer.add_scalar(f"Q_Std_Stats {state}/Max", np.max(q_std), global_step)

    # Log Q-value statistics
    writer.add_scalar(f"Q_Value_Stats {state}/Mean", np.mean(q_values), global_step)
    writer.add_scalar(f"Q_Value_Stats {state}/Median", np.median(q_values), global_step)
    writer.add_scalar(f"Q_Value_Stats {state}/25th_Percentile", np.percentile(q_values, 25), global_step)
    writer.add_scalar(f"Q_Value_Stats {state}/75th_Percentile", np.percentile(q_values, 75), global_step)
    writer.add_scalar(f"Q_Value_Stats {state}/Min", np.min(q_values), global_step)
    writer.add_scalar(f"Q_Value_Stats {state}/Max", np.max(q_values), global_step)

    # Log MC return statistics
    writer.add_scalar(f"MC_Return_Stats {state}/Mean", np.mean(mc_returns), global_step)
    writer.add_scalar(f"MC_Return_Stats {state}/Median", np.median(mc_returns), global_step)
    writer.add_scalar(f"MC_Return_Stats {state}/25th_Percentile", np.percentile(mc_returns, 25), global_step)
    writer.add_scalar(f"MC_Return_Stats {state}/75th_Percentile", np.percentile(mc_returns, 75), global_step)
    writer.add_scalar(f"MC_Return_Stats {state}/Min", np.min(mc_returns), global_step)
    writer.add_scalar(f"MC_Return_Stats {state}/Max", np.max(mc_returns), global_step)




# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Old Utils
# def calculate_return(rewards, gamma):
#     """Calculate Monte Carlo returns for a list of rewards."""
#     return_ = 0
#     for r in reversed(rewards):
#         return_ = r + gamma * return_
#     return return_


# def perform_rollout(obs, env, actor, qf1, qf2, device, args, max_steps=1000, safety=False, use_mean_action=False):
#     """Perform a single rollout in the environment and store results as tensors."""
#     done = False
#     step = 0
#     episode_rewards = []
#     q1_values = []
#     q2_values = []
#     cdq_values = []
#     entropy_list = []

#     while not done and step < max_steps:
#         with torch.no_grad():
#             state = torch.FloatTensor(obs).to(device)
#             action, entropy, action_mean = actor.get_action(state.unsqueeze(0)) # Fix action 
#             if use_mean_action:
#                 action = action_mean
#             q1 = qf1(state.unsqueeze(0), action)
#             q2 = qf2(state.unsqueeze(0), action)

#         # Store q1 and q2 values
#         q1_values.append(q1.item())
#         q2_values.append(q2.item())
#         cdq_values.append(min(q1.item(), q2.item()))
#         entropy_list.append(entropy.item())

#         action = action.squeeze().detach().cpu().numpy()
#         obs, reward, terminated, truncated, _ = env.step(action)
#         if safety:
#             reward = int(terminated)
#         done = terminated or truncated
#         episode_rewards.append(reward)
#         step += 1

#     # Calculate Monte Carlo returns
#     return_ = calculate_return(episode_rewards, args.gamma)
#     entropy_tensor = np.array(entropy_list)

#     return q1_values, q2_values,cdq_values, return_, entropy_tensor

# def evaluate_value_estimates(args, env, actor, qf1, qf2, device, safety=False, use_mean_action=False, num_episodes=10, num_trials=10, max_steps=1000):
#     """
#     Evaluates Q-value overestimation by comparing Q-values to Monte Carlo returns.
    
#     Args:
#         env: The environment to evaluate in
#         actor: The policy network
#         qf1, qf2: The Q-networks
#         device: The device to run computations on
#         num_episodes: Number of episodes to average over
#         num_trials: Number of trials per episode
#         max_steps: Maximum steps per episode
        
#     Returns:
#         mean_q_error: Average error between Q-values and MC returns
#         mean_q_values: Average predicted Q-values
#         mean_mc_values: Average Monte Carlo returns
#     """
#     q1_tensor = np.zeros((num_episodes, num_trials, max_steps), dtype=np.float32)
#     q2_tensor = np.zeros((num_episodes, num_trials, max_steps), dtype=np.float32)
#     cdq_tensor = np.zeros((num_episodes, num_trials, max_steps), dtype=np.float32)
#     entropy_tensor = np.zeros((num_episodes, num_trials, max_steps), dtype=np.float32)
#     mc_returns_tensor = np.zeros((num_episodes, num_trials), dtype=np.float32)

#     for episode in range(num_episodes):
#         start_state, _ = env.reset() # Reset without saved state 
#         env.save_state() # save the start state so that you can do multiple trials 
#         for trial in range(num_trials):
#             if trial: # if its not the first trial (reset to the saved state)
#                 start_state, _ = env.reset(use_saved_state=True)
            
#             q1_values, q2_values, cdq_values, return_, entropy = perform_rollout(start_state, env, actor, qf1, qf2, device, args, max_steps, safety, use_mean_action)
#             q1_tensor[episode, trial, :len(q1_values)] = np.array(q1_values)
#             q2_tensor[episode, trial, :len(q2_values)] = np.array(q2_values)
#             cdq_tensor[episode, trial, :len(cdq_values)] = np.array(cdq_values)
#             entropy_tensor[episode, trial, :len(entropy)] = np.array(entropy)
#             mc_returns_tensor[episode, trial] = return_

#     # mean_q_error = np.mean(q1_tensor - mc_returns_tensor) + np.mean(q2_tensor - mc_returns_tensor)
#     # mean_q_values = (np.mean(q1_tensor) + np.mean(q2_tensor)) / 2
#     # mean_mc_values = np.mean(mc_returns_tensor)

#     results = {
#         # "mean_q_error": mean_q_error,
#         # "mean_q_values": mean_q_values,
#         # "mean_mc_values": mean_mc_values,
#         "entropy": entropy_tensor,
#         "q1_tensor": q1_tensor,
#         "q2_tensor": q2_tensor,
#         "cdq_tensor": cdq_tensor,
#         "mc_returns_tensor": mc_returns_tensor
#     }
#     return results


# def analyze_estimation(writer, global_step, results):
#     q1, q2, cdq, mc_s0 = results["q1_tensor"], results["q2_tensor"], results["cdq_tensor"], results["mc_returns_tensor"]
#     q1_s0, q2_s0, cdq_s0 = q1[:, :, 0], q2[:, :, 0], cdq[:, :, 0]

#     q_std, mc_mean = np.abs((q1 - q2) / 2), np.mean(mc_s0, axis=1)

#     estimation_error_s0 = (cdq_s0 - mc_mean)
#     q_std_s0 = q_std[:, :, 0]
#     # std_err_s0 = np.mean(np.abs(np.mean(q_std_s0, axis=1) - mc_std_s0))

#     q1_estimation_err = np.mean(q1_s0 - mc_mean)
#     q2_estimation_err = np.mean(q2_s0 - mc_mean)
#     cdq_estimation_err = np.mean(cdq_s0 - mc_mean)
#     mean_q_estimation_err = np.mean((q1_s0+q2_s0) / 2 - mc_mean)

#     q1_rel_error = np.mean((q1_s0 - mc_mean)) / np.mean(mc_mean)
#     q2_rel_error = np.mean((q2_s0 - mc_mean)) / np.mean(mc_mean)
#     cdq_rel_error = np.mean((cdq_s0 - mc_mean)) / np.mean(mc_mean)
#     mean_q_rel_error = np.mean(((q1_s0 + q2_s0)/2 - mc_mean)) / np.mean(mc_mean)

#     ## Let's see if there is a correlation between variance and overestimation
#     fig, ax = plt.subplots()
#     ax.scatter(q_std_s0.flatten(), estimation_error_s0.flatten())
#     ax.set_xlabel('Q-Value Standard Deviation')
#     ax.set_ylabel('Estimation Error')
#     ax.set_title('Q-Value Standard Deviation vs Estimation Error')
#     wandb.log({f"Q-Value_Std_vs_Estimation_Error_{global_step}": wandb.Image(fig)})

#     corrcoef_s0 = np.corrcoef(q_std_s0.flatten(), estimation_error_s0.flatten())[0, 1]

#     writer.add_scalar("Estimation/correlation (s0)", corrcoef_s0, global_step)
#     # writer.add_scalar("Estimation/StdDev Err (s0)", std_err_s0, global_step)
#     # writer.add_scalar("Estimation/Mean MC StdDev (s0)", np.mean(mc_std_s0), global_step)
#     writer.add_scalar("Estimation/Mean Q StdDev (s0)", np.mean(q_std_s0), global_step)
#     writer.add_scalar("Estimation/Mean Q1 (s0)", np.mean(q1_s0), global_step)
#     writer.add_scalar("Estimation/Mean Q2 (s0)", np.mean(q2_s0), global_step)
#     writer.add_scalar("Estimation/Mean MC (s0)", np.mean(mc_mean), global_step)
#     writer.add_scalar("Estimation/Mean CDQ (s0)", np.mean(cdq_s0), global_step)


#     writer.add_scalar("EstimationErrors/Q1_Error (s0)", q1_estimation_err, global_step)
#     writer.add_scalar("EstimationErrors/Q2_Error (s0)", q2_estimation_err, global_step)
#     writer.add_scalar("EstimationErrors/CDQ_Error (s0)", cdq_estimation_err, global_step)
#     writer.add_scalar("EstimationErrors/Mean_Q_Error (s0)", mean_q_estimation_err, global_step)
#     writer.add_scalar("EstimationErrors/Q1_Relative_Error (s0)", q1_rel_error, global_step)
#     writer.add_scalar("EstimationErrors/Q2_Relative_Error (s0)", q2_rel_error, global_step)
#     writer.add_scalar("EstimationErrors/CDQ_Relative_Error (s0)", cdq_rel_error, global_step)
#     writer.add_scalar("EstimationErrors/Mean_Q_Relative_Error (s0)", mean_q_rel_error, global_step)

#     writer.add_scalar("Entropy/Mean Entropy (s0)", np.mean(results["entropy"][:, :, 0]), global_step)



# class MuJoCoStateWrapper(gym.Env):
#      def __init__(self, env):
#          super().__init__()
#          self.env = env
#          self.saved_state = None
 
#      def save_state(self):
#          """Save the current state of the environment."""
#          self.saved_state = {
#              "qpos": self.env.data.qpos.copy(),
#              "qvel": self.env.data.qvel.copy(),
#              "cinert": self.env.data.cinert.copy(),
#              "cvel": self.env.data.cvel.copy(),
#              "qfrc_actuator": self.env.data.qfrc_actuator.copy(),
#              "cfrc_ext": self.env.data.cfrc_ext.copy(),
#          }
 
#      def reset(self, use_saved_state=False, **kwargs):
#          if use_saved_state and self.saved_state is not None:
#              # Reset to the saved state
#              self.env.set_state(self.saved_state["qpos"], self.saved_state["qvel"])
#              self.env.data.cinert[:] = self.saved_state["cinert"]
#              self.env.data.cvel[:] = self.saved_state["cvel"]
#              self.env.data.qfrc_actuator[:] = self.saved_state["qfrc_actuator"]
#              self.env.data.cfrc_ext[:] = self.saved_state["cfrc_ext"]
#              observation = self.env.unwrapped._get_obs()
#              info = {}
#          else:
#              observation, info = self.env.reset(**kwargs)
#          return observation, info
 
#      def step(self, action):
#          return self.env.step(action)
 
#      def render(self, mode='human'):
#          return self.env.render(mode)
 
#      def close(self):
#          self.env.close()
 
#      def __getattr__(self, name):
#          return getattr(self.env, name)


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Old Utils