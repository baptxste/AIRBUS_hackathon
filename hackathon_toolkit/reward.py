import numpy as np

def manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance between two positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def compute_reward(num_agents, old_positions, agent_positions, evacuated_agents, deactivated_agents, goal_area):
    rewards = np.zeros(num_agents)

    for i, (old_pos, new_pos) in enumerate(zip(old_positions, agent_positions)):
        if i in evacuated_agents:
            continue
        elif i in deactivated_agents:   # Penalties for each deactivated agent
            # only receive this penalty once
            if not np.array_equal(old_pos, new_pos):
                rewards[i] = 50 
            else:
                rewards[i] = 0.0
        elif tuple(new_pos) in goal_area:   # One-time reward for each agent reaching the goal
            rewards[i] = 100 
            evacuated_agents.add(i)
        else:
            # if the agent came closer of the goal their is a small  postive reward 
            # if if move away small negative reward
            d_old = manhattan_distance(old_positions[i], goal_area[i])
            d_new= manhattan_distance(agent_positions[i], goal_area[i])
            if d_old > d_new : 
                rewards[i] = 10
                
            else : rewards[i] = -20

    return rewards, evacuated_agents