import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import argparse
import time
from typing import Tuple, Optional, Dict
import torch

from env import MazeEnv
from agent import MyAgents
from replay import ReplayBuffer

def simulation_config(config_path: str, new_agent: bool = True):
    """
    Configure the environment and optionally an agent using a JSON configuration file.

    Args:
        config_path (str): Path to the configuration JSON file.
        new_agent (bool): Whether to initialize the agent. Defaults to True.

    Returns:
        Tuple[MazeEnv, Optional[MyAgent], Dict]: Configured environment, agent (if new), and the configuration dictionary.
    """

    # Read config
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    # Env configuration
    env = MazeEnv(
        size=config.get('grid_size'),                               # Grid size
        walls_proportion=config.get('walls_proportion'),            # Walls proportion in the grid
        num_dynamic_obstacles=config.get('num_dynamic_obstacles'),  # Number of dynamic obstacles
        num_agents=config.get('num_agents'),                        # Number of agents
        communication_range=config.get('communication_range'),      # Maximum distance for agent communications
        max_lidar_dist_main=config.get('max_lidar_dist_main'),      # Maximum distance for main LIDAR scan
        max_lidar_dist_second=config.get('max_lidar_dist_second'),  # Maximum distance for secondary LIDAR scan
        max_episode_steps=config.get('max_episode_steps'),          # Number of steps before episode termination
        render_mode=None,
        seed=config.get('seed', None)                               # Seed for reproducibility
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_agents = env.num_agents
    # Agent configuration
    agents = MyAgents(state_size=98,action_size=env.action_space.n,n_agents=num_agents, lr=3e-4, device = device) if new_agent else None

    return env, agents, config


def plot_cumulated_rewards(rewards: list, interval: int = 10):
    """
    Plot and save the rewards over episodes with an optional smoothed curve.

    Args:
        rewards (list): List of total rewards per episode.
        interval (int): Interval for the moving average (default is 100).
    """
    plt.figure(figsize=(10, 6))

    # Plot the basic reward curve
    # plt.plot(range(1, len(rewards) + 1), rewards, color='blue', linestyle='-', label='Cumulated Rewards')

    try:
        # Calculate the moving average
        if len(rewards) >= interval:
            moving_avg = np.convolve(rewards, np.ones(interval) / interval, mode='valid')
            # Plot the moving average curve
            plt.plot(range(interval, interval + len(moving_avg)), moving_avg, color='blue', linestyle='-', label='Moving Average')
        else:
            print("Interval is larger than the number of episodes. Skipping moving average.")
    except Exception as e:
        print(f"Error calculating moving average: {e}")

    plt.title('Total Cumulated Rewards per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulated Rewards')

    # Adjust x-ticks to display every 'interval' episodes
    interval_ticks = max(1, len(rewards) // interval)
    xticks = range(1, len(rewards) + 1, interval_ticks)
    plt.xticks(xticks)

    plt.grid(True)
    plt.legend()
    plt.savefig('reward_curve_per_episode.png', dpi=300)
    plt.show()

def plot_evacuated(evacuated: list, interval: int = 10):
    """
    Plot and save the number of drones evacuated over episodes with an optional smoothed curve.

    Args:
        evacuated (list): List of the number of drones evacuated per episode.
        interval (int): Interval for the moving average (default is 100).
    """
    plt.figure(figsize=(10, 6))

    # Plot the basic evacuated curve
    # plt.plot(range(1, len(evacuated) + 1), evacuated, color='green', linestyle='-', label='Evacuated Drones')

    try:
        # Calculate the moving average
        if len(evacuated) >= interval:
            moving_avg = np.convolve(evacuated, np.ones(interval) / interval, mode='valid')
            # Plot the moving average curve
            plt.plot(range(interval, interval + len(moving_avg)), moving_avg, color='orange', linestyle='--', label='Moving Average')
        else:
            print("Interval is larger than the number of episodes. Skipping moving average.")
    except Exception as e:
        print(f"Error calculating moving average: {e}")

    plt.title('Number of Drones Evacuated Over Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Number of Drones')

    # Adjust x-ticks to display every 'interval' episodes
    interval_ticks = max(1, len(evacuated) // interval)
    xticks = range(1, len(evacuated) + 1, interval_ticks)
    plt.xticks(xticks)

    plt.grid(True)
    plt.legend()
    plt.savefig('evacuated_per_episode.png', dpi=300)
    plt.show()

def plot_deactivated(deactivated: list, interval: int = 10):
    """
    Plot and save the number of drones deactivated over episodes with an optional smoothed curve.

    Args:
        deactivated (list): List of the number of drones deactivated per episode.
        interval (int): Interval for the moving average (default is 100).
    """
    plt.figure(figsize=(10, 6))

    # Plot the basic deactivated curve
    # plt.plot(range(1, len(deactivated) + 1), deactivated, color='red', linestyle='-', label='Deactivated Drones')

    try:
        # Calculate the moving average
        if len(deactivated) >= interval:
            moving_avg = np.convolve(deactivated, np.ones(interval) / interval, mode='valid')
            # Plot the moving average curve
            plt.plot(range(interval, interval + len(moving_avg)), moving_avg, color='purple', linestyle='--', label='Moving Average')
        else:
            print("Interval is larger than the number of episodes. Skipping moving average.")
    except Exception as e:
        print(f"Error calculating moving average: {e}")

    plt.title('Number of Drones Deactivated Over Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Number of Drones')

    # Adjust x-ticks to display every 'interval' episodes
    interval_ticks = max(1, len(deactivated) // interval)
    xticks = range(1, len(deactivated) + 1, interval_ticks)
    plt.xticks(xticks)

    plt.grid(True)
    plt.legend()
    plt.savefig('deactivated_per_episode.png', dpi=300)
    plt.show()


def train(config_path: str) -> MyAgents:
    """
    Train an agent on the configured environment.

    Args:
        config_path (str): Path to the configuration JSON file.

    Returns:
        MyAgent: The trained agent.
    """

    # Environment and agent configuration
    env, agent, config = simulation_config(config_path)
    max_episodes = config.get('max_episodes')
    buffer = ReplayBuffer()
    
    # Metrics to follow the performance
    all_rewards = []
    total_reward = 0
    episode_count = 0
    
    # Initial reset of the environment
    states, info = env.reset()
    #time.sleep(1)

    try:
        while episode_count < max_episodes:
            
            # Determine agents actions
            actions, log_probs = agent.select_actions(states) 
            actions = actions.tolist()
            log_probs = log_probs.tolist()
            # Execution of a simulation step
            next_states, rewards, dones, truncated ,info= env.step(actions)
            
            # Stocker les expériences pour chaque agent
            for i in range(env.num_agents):
                buffer.store((states[i], actions[i], log_probs[i], rewards[i], next_states[i], dones))

            states = next_states
            total_reward += np.sum(rewards)

            # Update agent policy
            agent.update_policy(buffer)

            # Display of the step information
            print(f"\rEpisode {episode_count + 1}, Step {info['current_step']}, "
                  f"Reward: {total_reward:.2f}, "
                  f"Evacuated: {len(info['evacuated_agents'])}, "
                  f"Deactivated: {len(info['deactivated_agents'])}", end='')
            
            # Pause
            time.sleep(0.1)
            
            # If the episode is terminated
            if dones or truncated:
                print("\r")
                episode_count += 1
                all_rewards.append(total_reward)
                total_reward = 0
                
                if episode_count < max_episodes:
                    states, info = env.reset()

    except KeyboardInterrupt:
        print("\nSimulation interrupted by the user")
    
    finally:
        env.close()

    return agent, all_rewards


def evaluate(configs_paths: list, trained_agent: MyAgents, num_episodes: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate a trained agent on multiple configurations, calculate metrics, and visualize results.

    Args:
        config_path (list): List of paths to the configuration JSON files.
        trained_agent (MyAgent): A pre-trained agent to evaluate.
        num_episodes (int): Number of episodes to run for evaluation per configuration. Defaults to 10.

    Returns:
        pd.DataFrame: A DataFrame containing evaluation metrics for each episode and configuration.
    """

    # Evaluation results
    all_results = pd.DataFrame()

    for config_path in configs_paths:
        print(f"\n--- Evaluating Configuration: {config_path} ---")

        # Environment configuration
        env, _, config = simulation_config(config_path, new_agent=False)

        # Metrics to follow the performance
        metrics = []
        total_reward = 0
        episode_count = 0
        
        # Initial reset of the environment
        state, info = env.reset()
        time.sleep(1) 
   
        # Run evaluation for the specified number of episodes
        try:
            while episode_count < num_episodes:
                # Determine agents actions
                actions = trained_agent.get_action(state, evaluation=True)

                # Execution of a simulation step
                state, rewards, terminated, truncated, info = env.step(actions)
                total_reward += np.sum(rewards)

                # Display of the step information
                print(f"\rEpisode {episode_count + 1}/{num_episodes}, Step {info['current_step']}, "
                    f"Reward: {total_reward:.2f}, "
                    f"Evacuated: {len(info['evacuated_agents'])}, "
                    f"Deactivated: {len(info['deactivated_agents'])}", end='')
            
                # Pause
                time.sleep(1)

                # If the episode is terminated
                if terminated or truncated:
                    print("\r")
                    # Save metrics
                    metrics.append({
                        "config_path": config_path,
                        "episode": episode_count + 1,
                        "steps": info['current_step'],
                        "reward": total_reward,
                        "evacuated": len(info['evacuated_agents']),
                        "deactivated": len(info['deactivated_agents'])
                    })

                    episode_count += 1
                    total_reward = 0

                    if episode_count < num_episodes:
                        state, info = env.reset()
        
        except KeyboardInterrupt:
            print("\nSimulation interrupted by the user")
        
        finally:
            env.close()

        # Convert the current configuration's metrics to a DataFrame
        config_results = pd.DataFrame(metrics)
        all_results = pd.concat([all_results, config_results], ignore_index=True)
    
    env.close()

    all_results.to_csv('all_results.csv', index=False)

    return all_results