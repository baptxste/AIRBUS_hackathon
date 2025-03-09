import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import argparse
import time
from typing import Tuple, Optional, Dict
import torch
from env import MazeEnv
from agent import MAPPOAgent, Critic, Actor
from process_state import StateNormalizer



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
        render_mode=config.get('render_mode', None),
        seed=config.get('seed', None)                               # Seed for reproducibility
    )

    # Agent configuration
    if new_agent == True:
        agents = MAPPOAgent(state_size=98,action_size=env.action_space.n,n_agents=env.num_agents, grid_size=env.grid_size)
        # agents = MAPPOAgent(state_size=env.single_agent_state_size,action_size=env.action_space.n,n_agents=env.num_agents)
          

    elif new_agent == False : 
        # actor = torch.load("./models/actor.pth", weights_only=False)
        # critic = torch.load("./models/critic.pth", weights_only=False)
        # agents = MAPPOAgent(state_size=env.single_agent_state_size,action_size=env.action_space.n,n_agents=env.num_agents, actor=actor, critic=critic)

        agents = MAPPOAgent(state_size=98,action_size=env.action_space.n,n_agents=env.num_agents, grid_size=env.grid_size)
        agents.actor.load_state_dict(torch.load('./models/actor_best.pth'))
        agents.critic.load_state_dict(torch.load('./models/critic_best.pth'))

    else : 
        agents = None

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
            plt.plot(range(interval, interval + len(moving_avg)), moving_avg, color='green', linestyle='-', label='Moving Average')
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
            plt.plot(range(interval, interval + len(moving_avg)), moving_avg, color='red', linestyle='-', label='Moving Average')
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

def train(config_path = 'config.json'):
    env, agent, config = simulation_config(config_path,new_agent=True)
    n_agents = env.num_agents
    buffer = ReplayBuffer()
    rewards_over_episodes = []
    evacuated = []
    deactivated = []
    try : 
        for episode in range(config.get('max_episodes')):
            states, info = env.reset()  # (n_agents, obs_dim)
            episode_rewards = 0
            # states = process_state(states, env.grid_size)
            buffer.clear()

            for step in range(config.get('max_episode_steps')):
                with torch.no_grad():
                    actions, log_probs = agent.select_actions(states)
                # actions, log_probs = agent.select_actions(states)  # Récupérer toutes les actions
                actions = actions.tolist()
                log_probs = log_probs.tolist()
                next_states, rewards, dones, _ ,info= env.step(actions)  # Exécuter toutes les actions

                rewards = np.where(rewards == 100, rewards +  (config.get('max_episode_steps')-step)/config.get('max_episode_steps'), rewards)  # Si la valeur est a, on multiplie par 10
                rewards = np.where(rewards == -50, rewards -  (config.get('max_episode_steps')-step)/config.get('max_episode_steps'), rewards) 

                # Stocker les expériences pour chaque agent
                for i in range(n_agents):
                    buffer.store((states[i], actions[i], log_probs[i], rewards[i], next_states[i], dones))

                states = next_states
                episode_rewards += sum(rewards)

                if dones: break  # Fin de l'épisode si tous les agents sont terminés
            
            # Train the agent after collecting data
            data = buffer.get_data()
            states, actions, log_probs_old, rewards, next_states, dones = zip(*data)
            dones = list(dones)
            actor_loss, critic_loss = agent.compute_loss(states, actions, log_probs_old, rewards, dones)
            
            # Update Actor
            agent.optim_actor.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.optim_actor.step()

            # Update Critic
            agent.optim_critic.zero_grad()
            critic_loss.backward()
            agent.optim_critic.step()

            rewards_over_episodes.append(episode_rewards)
            evacuated.append(len(info['evacuated_agents']))
            deactivated.append(len(info['deactivated_agents']))
            if all(element == env.num_agents for element in evacuated[-15:]):
                print("le modèle a convergé")
                raise KeyboardInterrupt
            print(f"Episode {episode}, Reward: {episode_rewards:.2f},Evacuated: {len(info['evacuated_agents'])},Deactivated: {len(info['deactivated_agents'])}, Actor Loss: {actor_loss:.2f}, Critic Loss: {critic_loss:.2f}")
    except KeyboardInterrupt:
        print("\nSimulation interrupted by the user") 
    plot_cumulated_rewards(rewards_over_episodes)
    plot_evacuated(evacuated)
    plot_deactivated(deactivated)
    agent.save()
    return agent

def finetune(config_path = 'config.json'):
    env, agent, config = simulation_config(config_path,new_agent=False)
    n_agents = env.num_agents
    buffer = ReplayBuffer()
    rewards_over_episodes = []
    evacuated = []
    deactivated = []
    try : 
        for episode in range(config.get('max_episodes')):
            states, info = env.reset()  # (n_agents, obs_dim)
            episode_rewards = 0
            buffer.clear()

            for step in range(config.get('max_episode_steps')):
                with torch.no_grad():
                    actions, log_probs = agent.select_actions(states)
                # actions, log_probs = agent.select_actions(states)  # Récupérer toutes les actions
                actions = actions.tolist()
                log_probs = log_probs.tolist()
                next_states, rewards, dones, _ ,info= env.step(actions)  # Exécuter toutes les actions

                # Stocker les expériences pour chaque agent
                for i in range(n_agents):
                    buffer.store((states[i], actions[i], log_probs[i], rewards[i], next_states[i], dones))

                states = next_states
                episode_rewards += sum(rewards)

                if dones: break  # Fin de l'épisode si tous les agents sont terminés
            
            # Train the agent after collecting data
            data = buffer.get_data()
            states, actions, log_probs_old, rewards, next_states, dones = zip(*data)
            dones = list(dones)
            actor_loss, critic_loss = agent.compute_loss(states, actions, log_probs_old, rewards, dones)
            
            # Update Actor
            agent.optim_actor.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.optim_actor.step()

            # Update Critic
            agent.optim_critic.zero_grad()
            critic_loss.backward()
            agent.optim_critic.step()

            rewards_over_episodes.append(episode_rewards)
            evacuated.append(len(info['evacuated_agents']))
            deactivated.append(len(info['deactivated_agents']))
            if all(element == env.num_agents for element in evacuated[-15:]):
                print("le modèle a convergé")
                raise KeyboardInterrupt
            print(f"Episode {episode}, Reward: {episode_rewards},Evacuated: {len(info['evacuated_agents'])},Deactivated: {len(info['deactivated_agents'])}, Actor Loss: {actor_loss:.2f}, Critic Loss: {critic_loss:.2f}")
    except KeyboardInterrupt:
        print("\nSimulation interrupted by the user") 
    plot_cumulated_rewards(rewards_over_episodes)
    plot_evacuated(evacuated)
    plot_deactivated(deactivated)
    agent.save()
    return agent

def evaluate(configs_paths: list, trained_agent: MAPPOAgent, num_episodes: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        evacuated_total = 0
        # Initial reset of the environment
        states, info = env.reset()
        
   
        # Run evaluation for the specified number of episodes
        try:
            while episode_count < num_episodes:
                # Determine agents actions
                actions, log_probs = trained_agent.select_actions(states)  # Récupérer toutes les actions
                actions = actions.tolist()
                
                next_states, rewards, dones, truncated ,info= env.step(actions)  # Exécuter toutes les actions
                
                total_reward += sum(rewards)
                # Display of the step information
                print(f"\rEpisode {episode_count + 1}/{num_episodes}, Step {info['current_step']}, "
                    f"Reward: {total_reward:.2f}, "
                    f"Evacuated: {len(info['evacuated_agents'])}, "
                    f"Deactivated: {len(info['deactivated_agents'])}", end='')
                states = next_states
                # Pause
                #time.sleep(1)

                # If the episode is terminated
                if dones or truncated:
                    evacuated_total += len(info['evacuated_agents'])
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
                        states, info = env.reset()
        
        except KeyboardInterrupt:
            print("\nSimulation interrupted by the user")
        
        finally:
            env.close()

        # Convert the current configuration's metrics to a DataFrame
        config_results = pd.DataFrame(metrics)
        all_results = pd.concat([all_results, config_results], ignore_index=True)
    all_results.to_csv('all_results.csv', index=False)
    env.close()

    return all_results, evacuated_total

class ReplayBuffer:
    def __init__(self, max_size=10000):  # ajustéer la taille en fonction des ressources GPU
        self.memory = []
        self.max_size = max_size

    def store(self, trajectory):
        if len(self.memory) >= self.max_size:
            self.memory.pop(0)  # Supprime la plus ancienne transition
        self.memory.append(trajectory)

    def get_data(self):
        return self.memory

    def clear(self):
        self.memory = []
        torch.cuda.empty_cache()

