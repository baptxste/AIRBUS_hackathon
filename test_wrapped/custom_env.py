# from pettingzoo import ParallelEnv
# from env import MazeEnv
# import json
# from copy import copy
# import functools
# from gymnasium.spaces import Discrete, MultiDiscrete
# import numpy as np
# from gymnasium.spaces import Box

# class CustomEnv(ParallelEnv):
#     metadata = {
#         "name": "custom_environment_v0",
#         "is_parallelizable": False  # Indique si l'environnement peut être parallélisé
#     }
#     def __init__(self, config_path: str):
#         with open(config_path, 'r') as config_file:
#             self.config = json.load(config_file)

#         self.possible_agents = [f"agent_{i}" for i  in range(self.config.get('num_agents'))] 
#         self.env = MazeEnv(
#                 size=self.config.get('grid_size'),                               # Grid size
#                 walls_proportion=self.config.get('walls_proportion'),            # Walls proportion in the grid
#                 num_dynamic_obstacles=self.config.get('num_dynamic_obstacles'),  # Number of dynamic obstacles
#                 num_agents=self.config.get('num_agents'),                        # Number of agents
#                 communication_range=self.config.get('communication_range'),      # Maximum distance for agent communications
#                 max_lidar_dist_main=self.config.get('max_lidar_dist_main'),      # Maximum distance for main LIDAR scan
#                 max_lidar_dist_second=self.config.get('max_lidar_dist_second'),  # Maximum distance for secondary LIDAR scan
#                 max_episode_steps=self.config.get('max_episode_steps'),          # Number of steps before episode termination
#                 render_mode=self.config.get('render_mode', None),
#                 seed=self.config.get('seed', None)                               # Seed for reproducibility
#                 )
#         self.render_mode = self.env.render_mode
#         self.agents = copy(self.possible_agents)

#     def reset(self, seed = None, options = None):
#         self.agents = copy(self.possible_agents)
#         """ Réinitialise l'environnement et retourne un dict d'observations par agent. """
#         observations = self.env.reset()  # Doit renvoyer une liste d'états
#         observations  = observations[0].astype(np.int64).tolist()
#         infos = {a: {} for a in self.possible_agents}
#         return {agent: obs for agent, obs in zip(self.possible_agents, observations)}, infos

#     def step(self, actions):
#         """
#         Exécute une étape de simulation.
#         actions : dict {agent: action}
#         Retourne : observations, récompenses, terminaisons, infos (dicts par agent)
#         """

#         action_list = [actions[agent] for agent in self.agents]

#         observations, rewards, dones, truncated, infos = self.env.step(action_list)

#         observations = {
#             a: observations[index].astype(np.int64).tolist()
#             for index, a in enumerate(self.agents)
#         }
#         rewards = {
#             a:rewards[index]
#             for index, a in enumerate(self.agents)
#         }



#         done = {a: False for a in self.agents}  # Initialisez avec False pour chaque agent actif
#         truncations = {a: False for a in self.agents}  # Idem pour truncations

#         for e in infos["deactivated_agents"]:
#             done[f"agent_{e}"] = True
#         for e in infos["evacuated_agents"]:
#             done[f"agent_{e}"] = True

#         # Assurez-vous que truncations est mis à jour correctement
#         if self.env.current_step > self.env.max_episode_steps:
#             truncations = {a: True for a in self.agents}

#             # Veillez à ce que la longueur de done et truncations soit cohérente
#         done = {a: done.get(a, False) for a in self.agents}  # Garantit que tous les agents sont pris en compte
#         truncations = {a: truncations.get(a, False) for a in self.agents}  # Idem pour truncations


#         infos = {a: {} for a in self.possible_agents}

#         self.agents = [a for a in self.agents if not done[a]]

#         return observations, rewards, done, truncations, infos


#     def close(self):
#         """ Ferme l'environnement sous-jacent. """
#         self.env.close()

#     # Observation space should be defined here.
#     # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
#     # If your spaces change over time, remove this line (disable caching).
#     @functools.lru_cache(maxsize=None)
#     def observation_space(self, agent):
#         # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
#         # return MultiDiscrete([30] * 22)
#         return Box(low=0, high=30, shape=(12 + 10* (self.num_agents -1),), dtype=np.int32)
#     # Action space should be defined here.
#     # If your spaces change over time, remove this line (disable caching).
#     @functools.lru_cache(maxsize=None)
#     def action_space(self, agent):
#         return Discrete(7)



from pettingzoo import ParallelEnv
from env import MazeEnv
import json
from copy import copy
import functools
from gymnasium.spaces import Discrete, Box
import numpy as np

class CustomEnv(ParallelEnv):
    metadata = {
        "name": "custom_environment_v0",
        "is_parallelizable": False
    }
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)

        self.possible_agents = [f"agent_{i}" for i in range(self.config.get('num_agents'))] 
        self.env = MazeEnv(
            size=self.config.get('grid_size'),
            walls_proportion=self.config.get('walls_proportion'),
            num_dynamic_obstacles=self.config.get('num_dynamic_obstacles'),
            num_agents=self.config.get('num_agents'),
            communication_range=self.config.get('communication_range'),
            max_lidar_dist_main=self.config.get('max_lidar_dist_main'),
            max_lidar_dist_second=self.config.get('max_lidar_dist_second'),
            max_episode_steps=self.config.get('max_episode_steps'),
            render_mode=self.config.get('render_mode', None),
            seed=self.config.get('seed', None)
        )
        self.render_mode = self.env.render_mode
        self.agents = copy(self.possible_agents)
    
    def reset(self, seed=None, options=None):
        # print("ATTENTION RESET")
        self.agents = copy(self.possible_agents)
        observations = self.env.reset()[0].astype(np.int64).tolist()
        infos = {a: {} for a in self.possible_agents}
        return {agent: obs for agent, obs in zip(self.agents, observations)}, infos
    
    def step(self, actions):
        action_list = [actions[agent] for agent in self.agents]
        self.last_action = action_list
        observations, rewards, dones, truncated, env_infos = self.env.step(action_list)
        
        observations = {a: observations[i].astype(np.int64).tolist() for i, a in enumerate(self.agents)}
        rewards = {a: env_infos['individual_rewards'][i] for i, a in enumerate(self.agents)}
        
        done = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        
        for idx in env_infos["deactivated_agents"]:
            done[f"agent_{idx}"] = True
        for idx in env_infos["evacuated_agents"]:
            done[f"agent_{idx}"] = True
        
        if env_infos["current_step"] >= self.env.max_episode_steps:
            truncations = {a: True for a in self.agents}
            # print("OK")
            self.reset()
        
        done = {a: done.get(a, False) for a in self.agents}
        truncations = {a: truncations.get(a, False) for a in self.agents}
        
        infos = {"agent_0":env_infos } #{a: env_infos for a in self.possible_agents}
        
        
        self.agents = [a for a in self.agents if not done[a]]
        
        return observations, rewards, done, truncations, infos
    
    def close(self):
        self.env.close()
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(low=0, high=30, shape=(12 + 10 * (self.config.get('num_agents') - 1),), dtype=np.int32)
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(7)
