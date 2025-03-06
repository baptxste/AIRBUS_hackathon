from pettingzoo import ParallelEnv
from env import MazeEnv
import json
from copy import copy
import functools
from gymnasium.spaces import Discrete, MultiDiscrete

class CustomEnv(ParallelEnv):
    def __init__(self, config_path: str):
        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)

        self.possible_agents = [f"agent_{i}" for i  in range(self.config.get('num_agents'))] 
        self.env = MazeEnv(
                size=self.config.get('grid_size'),                               # Grid size
                walls_proportion=self.config.get('walls_proportion'),            # Walls proportion in the grid
                num_dynamic_obstacles=self.config.get('num_dynamic_obstacles'),  # Number of dynamic obstacles
                num_agents=self.config.get('num_agents'),                        # Number of agents
                communication_range=self.config.get('communication_range'),      # Maximum distance for agent communications
                max_lidar_dist_main=self.config.get('max_lidar_dist_main'),      # Maximum distance for main LIDAR scan
                max_lidar_dist_second=self.config.get('max_lidar_dist_second'),  # Maximum distance for secondary LIDAR scan
                max_episode_steps=self.config.get('max_episode_steps'),          # Number of steps before episode termination
                render_mode=self.config.get('render_mode', None),
                seed=self.config.get('seed', None)                               # Seed for reproducibility
                )

    def reset(self, seed = None, options = None):
        self.agents = copy(self.possible_agents)
        """ Réinitialise l'environnement et retourne un dict d'observations par agent. """
        observations = self.env.reset()  # Doit renvoyer une liste d'états
        observations  = observations[0].tolist()
        infos = {a: {} for a in self.possible_agents}
        return {agent: obs for agent, obs in zip(self.possible_agents, observations)}, infos

    def step(self, actions):
        """
        Exécute une étape de simulation.
        actions : dict {agent: action}
        Retourne : observations, récompenses, terminaisons, infos (dicts par agent)
        """
        # Convertir le dict d'actions en liste ordonnée
        action_list = [actions[agent] for agent in self.agents]

        # Exécuter l'étape dans l'environnement sous-jacent
        observations, rewards, dones, truncated, infos = self.env.step(action_list)
        print("Step outputs:")
        print("Observations:", observations, type(observations))
        print("Rewards:", rewards, type(rewards))
        print("Dones:", dones, type(dones))
        print("Infos:", infos, type(infos))

        observations = {
            a: observations[index].tolist()
            for index, a in enumerate(self.agents)
        }
        rewards = {
            a:rewards[index]
            for index, a in enumerate(self.agents)
        }
        done = {a: False for a in self.agents}
        for e in infos["deactivated_agents"] : 
            done[f"agent_{e}"] = True
        for e in infos["evacuated_agents"] : 
            done[f"agent_{e}"] = True
        truncations = {a: False for a in self.agents}
        if self.env.current_step > self.env.max_episode_steps:
            truncations = {a: True for a in self.agents}
        self.env.current_step += 1

        infos = {a: {} for a in self.possible_agents}

        self.agents = [a for a in self.agents if not done[a]]

        return observations, rewards, done, truncations, infos

        # # Convertir les résultats en dictionnaires
        # return (
        #     {agent: obs for agent, obs in zip(self.possible_agents, observations)},  
        #     {agent: reward for agent, reward in zip(self.possible_agents, rewards)},  
        #     {agent: done for agent, done in zip(self.possible_agents, dones)},  
        #     {agent: info for agent, info in zip(self.possible_agents, infos)},  
        # )

    def close(self):
        """ Ferme l'environnement sous-jacent. """
        self.env.close()

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return MultiDiscrete([7 * 7] * 3)

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(4)