import numpy as np
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import json

with open("config.json", 'r') as config_file:
    config = json.load(config_file)

class MultiAgentMazeEnv(MultiAgentEnv):
    def __init__(self, config):
        super().__init__()
        self.num_agents = config.get('num_agents', 2)
        self.observation_space = spaces.Box(low=-30, high=30, shape=(12,), dtype=np.float32)
        self.action_space = spaces.Discrete(7)

    def reset(self, *, seed=None, options=None):
        obs = {f"agent_{i}": self.observation_space.sample() for i in range(self.num_agents)}
        return obs, {}

    def step(self, action_dict):
        obs = {f"agent_{i}": self.observation_space.sample() for i in range(self.num_agents)}
        rewards = {f"agent_{i}": np.random.rand() for i in range(self.num_agents)}
        dones = {f"agent_{i}": False for i in range(self.num_agents)}
        dones["__all__"] = np.random.rand() < 0.1
        infos = {f"agent_{i}": {} for i in range(self.num_agents)}
        return obs, rewards, dones, infos

from ray.rllib.algorithms.ppo import PPOConfig

# Configuration de l'environnement
env_config = {
    "num_agents": 2,
    # Autres paramètres de configuration
}

# Configuration de PPO avec l'environnement multi-agent
config = (
    PPOConfig()
    .environment(MultiAgentMazeEnv, env_config=env_config)
    .multi_agent(
        policies={f"policy_{i}": (None, spaces.Box(low=-30, high=30, shape=(12,), dtype=np.float32), spaces.Discrete(7), {})
                  for i in range(env_config["num_agents"])},
        policy_mapping_fn=lambda agent_id: f"policy_{agent_id.split('_')[1]}"
    )
    .framework("torch")
)

# Construire l'algorithme PPO et lancer l'entraînement
ppo = config.build()
ppo.train()
