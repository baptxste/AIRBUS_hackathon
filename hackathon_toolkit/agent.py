import numpy as np

def manhattan_distance(pos1, pos2):
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


class MyAgents():
    def __init__(self, env, num_agents: int):        
        self.rng = np.random.default_rng()
        self.num_agents = num_agents
        self.grid_size = env.grid_size  # Récupération dynamique
        self.communication_range = env.communication_range
        self.max_lidar_dist_main = env.max_lidar_dist_main
        self.max_episode_steps = env.max_episode_steps

    def get_action(self, state: list, evaluation: bool = False):
        # Choose random action
        actions = self.rng.integers(low=0, high=6, size=self.num_agents)
        return actions.tolist()

    def update_policy(self, actions: list, state: list, reward: float):
        # Do nothing
        pass

    def process_states(self, list_state: list):
        final = []
        grid_size = self.grid_size

        for state in list_state:
            agent_pos = (state[0], state[1])
            goal_pos = (state[4], state[5])
            agent_status = 0  # Actif par défaut

            if state[3] == 1:  # Évacué
                agent_status = 1
            elif state[3] == 2:  # Désactivé
                agent_status = 2

            # Distance normalisée au goal (Manhattan)
            max_possible_distance = grid_size * 2
            distance_to_goal = manhattan_distance(agent_pos, goal_pos) / max_possible_distance

            # Lidar distances pour obstacles proches (3 distances)
            lidar_distances = np.array([state[6], state[8], state[10]]) / self.max_lidar_dist_main

            # Nombre de voisins dans la communication range
            num_neighbors = 0
            other_state = state[12:]
            num_drones = len(other_state) // 10  # Chaque drone a 10 valeurs stockées

            for i in range(num_drones):
                other_agent_pos = (other_state[10 * i], other_state[10 * i + 1])
                if other_state[10 * i + 3] == 0:  # Si actif
                    if manhattan_distance(agent_pos, other_agent_pos) <= self.communication_range:
                        num_neighbors += 1
            if self.num_agents != 1:
                num_neighbors_normalized = num_neighbors / (self.num_agents - 1)
            else:
                num_neighbors_normalized = 0.0

            # État final avec distances relatives aux obstacles
            processed_state = [
                distance_to_goal,
                *lidar_distances,  # Ajout des 3 distances normalisées du lidar
                num_neighbors_normalized,
                agent_status / 2,  # 0, 0.5 ou 1
            ]

            final.append(processed_state)

        return final
