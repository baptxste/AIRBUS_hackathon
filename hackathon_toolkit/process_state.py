import numpy as np

class StateNormalizer:
    def __init__(self, grid_size, max_lidar_range, max_agents=3):
        self.grid_size = grid_size
        self.max_lidar_range = max_lidar_range
        self.max_agents = max_agents  # Maximum agents within communication range

    def normalize_position(self, pos):
        """Normalize position (x or y) to [0,1], handle deactivated agents (-1)"""
        return 0.0 if pos == -1 else pos / self.grid_size

    def one_hot_orientation(self, orientation):
        """One-hot encode orientation (0,1,2,3), handle deactivated agents (-1)"""
        orientation = int(orientation)
        if orientation == -1:
            return np.zeros(4)  # Mask out
        one_hot = np.zeros(4)
        one_hot[orientation] = 1
        return one_hot

    def one_hot_status(self, status):
        """One-hot encode status (0: running, 1: evacuated, 2: deactivated), handle deactivated (-1)"""
        status = int(status)
        if status == -1:
            return np.zeros(3)  # Mask out
        one_hot = np.zeros(3)
        one_hot[status] = 1
        return one_hot

    def normalize_lidar(self, lidar_distance, lidar_type):
        """Normalize LIDAR distance and one-hot encode obstacle type, handle deactivated (-1)"""
        lidar_type = int(lidar_type)
        if lidar_distance == -1 or lidar_type == -1:
            return np.zeros(5)  # Mask out
        dist = lidar_distance / self.max_lidar_range
        type_one_hot = np.zeros(4)
        type_one_hot[lidar_type] = 1
        return np.concatenate([[dist], type_one_hot])

    def normalize_agent_state(self, state_list):
        """Normalize an agent's state from list format"""
        idx = 0  # Track index in list
        
        # Normalize own position and goal position
        x, y = self.normalize_position(state_list[idx]), self.normalize_position(state_list[idx + 1])
        idx += 2

        # One-hot encode orientation and status
        orientation = self.one_hot_orientation(state_list[idx])
        status = self.one_hot_status(state_list[idx + 1])
        idx += 2

        # Normalize goal position
        goal_x, goal_y = self.normalize_position(state_list[idx]), self.normalize_position(state_list[idx + 1])
        idx += 2

        # Normalize LIDAR readings
        lidar_main = self.normalize_lidar(state_list[idx], state_list[idx + 1])
        lidar_right = self.normalize_lidar(state_list[idx + 2], state_list[idx + 3])
        lidar_left = self.normalize_lidar(state_list[idx + 4], state_list[idx + 5])
        idx += 6

        # Concatenate the normalized values
        normalized_state = np.concatenate(
            [[x, y], orientation, status, [goal_x, goal_y], lidar_main, lidar_right, lidar_left]
        )

        # Process nearby agents within communication range
        nearby_states = []
        for _ in range(self.max_agents):  # Assuming a fixed number of agents
            if idx >= len(state_list):  # Stop if no more agents in the list
                break

            agent_x, agent_y = self.normalize_position(state_list[idx]), self.normalize_position(state_list[idx + 1])
            agent_orientation = self.one_hot_orientation(state_list[idx + 2])
            agent_status = self.one_hot_status(state_list[idx + 3])
            idx += 4

            agent_lidar_main = self.normalize_lidar(state_list[idx], state_list[idx + 1])
            agent_lidar_right = self.normalize_lidar(state_list[idx + 2], state_list[idx + 3])
            agent_lidar_left = self.normalize_lidar(state_list[idx + 4], state_list[idx + 5])
            idx += 6

            agent_state = np.concatenate(
                [[agent_x, agent_y], agent_orientation, agent_status, agent_lidar_main, agent_lidar_right, agent_lidar_left]
            )
            nearby_states.append(agent_state)

        # Flatten nearby agents' states
        if nearby_states:
            normalized_nearby_agents = np.concatenate(nearby_states)
        else:
            normalized_nearby_agents = np.zeros(normalized_state.shape)  # Empty communication data

        return np.concatenate([normalized_state, normalized_nearby_agents])
    
    def normalize_state(self, state):
        """Normalize the state for all agents"""
        return [self.normalize_agent_state(agent_state) for agent_state in state]