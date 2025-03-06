import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
import json
import gymnasium as gym
from env import MazeEnv  # Assurez-vous que ce chemin est correct
from gymnasium import spaces
import numpy as np
from agent import MyAgent

from ray.rllib.algorithms.ppo import PPOConfig
from gymnasium import spaces



# Read config
with open("config.json", 'r') as config_file:
    config = json.load(config_file)

# Env configuration
env_conf = {
    "size": config.get('grid_size'),                            # Taille de la grille
    "walls_proportion": config.get('walls_proportion'),          # Proportion de murs dans la grille
    "num_dynamic_obstacles": config.get('num_dynamic_obstacles'),# Nombre d'obstacles dynamiques
    "num_agents": config.get('num_agents'),                      # Nombre d'agents
    "communication_range": config.get('communication_range'),    # Distance maximale pour les communications des agents
    "max_lidar_dist_main": config.get('max_lidar_dist_main'),    # Distance maximale pour le scan LIDAR principal
    "max_lidar_dist_second": config.get('max_lidar_dist_second'),# Distance maximale pour le scan LIDAR secondaire
    "max_episode_steps": config.get('max_episode_steps'),        # Nombre d'étapes avant la terminaison de l'épisode
    "render_mode": config.get('render_mode', None),              # Mode de rendu
    "seed": config.get('seed', None)                             # Graine pour la reproductibilité
}

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

# from gymnasium.envs.registration import register

# register(
#     id='MazeEnv-v0',
#     entry_point='env:MazeEnv',
# )
from ray.tune.registry import register_env
register_env("MazeEnv-v0", lambda cfg: MazeEnv(cfg))

# Définir les espaces d'observation et d'action
observation_space = spaces.Box(low=-30, high=30, shape=(12,), dtype=np.float32)
action_space = spaces.Discrete(7)



# Configuration de PPO avec l'environnement multi-agent
config = (
    PPOConfig()
    .environment(env='MazeEnv-v0', env_config = env_conf)
    .multi_agent(
        policies={f"policy_{i}": (None, observation_space, action_space, {})
                  for i in range(env_conf["num_agents"])},
        policy_mapping_fn=lambda agent_id: f"policy_{agent_id.split('_')[1]}"
    )
    .framework("torch")
)

# Construire l'algorithme PPO et lancer l'entraînement
ppo = config.build_algo()
ppo.train()