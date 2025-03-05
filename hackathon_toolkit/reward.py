# import numpy as np

# def compute_reward(num_agents, old_positions, agent_positions, evacuated_agents, deactivated_agents, goal_area):
#     rewards = np.zeros(num_agents)

#     # Compute reward for each agent
#     for i, (old_pos, new_pos) in enumerate(zip(old_positions, agent_positions)):
#         if i in evacuated_agents:
#             continue
#         elif i in deactivated_agents:   # Penalties for each deactivated agent
#             rewards[i] = -10.0
#         elif tuple(new_pos) in goal_area:   # One-time reward for each agent reaching the goal
#             rewards[i] = 100.0
#             evacuated_agents.add(i)
#         else:
#             # Penalties for not finding the goal
#             rewards[i] = -0.1

#     return rewards, evacuated_agents

import numpy as np

def manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance between two positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

# ajouter un cas pour traier laction =-1 ( pour le cas ou  l'agent n'est plsu en jeu)
# def compute_reward(num_agents, old_positions, agent_positions, evacuated_agents, deactivated_agents, goal_area):
#     rewards = np.zeros(num_agents)
    # print( "OLD : ", old_positions)
    # print("NEW : ",agent_positions)
    # print("GOALS : ", goal_area)



#     # Compute reward for each agent
#     for i, (old_pos, new_pos) in enumerate(zip(old_positions, agent_positions)):
#         if i in evacuated_agents:
#             continue
#         elif i in deactivated_agents:   # Penalties for each deactivated agent
#             rewards[i] = -10.0
#         elif tuple(new_pos) in goal_area:   # One-time reward for each agent reaching the goal
#             rewards[i] = 1000.0
#             evacuated_agents.add(i)
#         else:
#             # if the agent came closer of the goal their is a small  postive reward 
#             # if if move away small negative reward
            

#             d_old = manhattan_distance(old_positions[i], goal_area[i])
#             d_new= manhattan_distance(agent_positions[i], goal_area[i])
#             if d_old < d_new : 
#                 rewards[i] = 100
#             else : rewards[i] = -10
            

    # return rewards, evacuated_agents


def compute_reward(num_agents, old_positions, agent_positions, evacuated_agents, deactivated_agents, goal_area, current_step, max_episode_steps):
    rewards = np.zeros(num_agents)

    for i, (old_pos, new_pos) in enumerate(zip(old_positions, agent_positions)):
        if i in evacuated_agents:
            continue
        elif i in deactivated_agents:
            death_penalty = -50 * (1 - current_step / max_episode_steps)
            rewards[i] = death_penalty
        elif tuple(new_pos) in goal_area:
            rewards[i] = 1000.0
            evacuated_agents.add(i)
        else:
            d_old = manhattan_distance(old_positions[i], goal_area[i])
            d_new = manhattan_distance(agent_positions[i], goal_area[i])
            delta = d_old - d_new

            if delta > 0:
                rewards[i] = delta * 20
            elif delta < 0:
                rewards[i] = delta * 20
            else:
                rewards[i] = -10

            #bonus de survie
            rewards[i] += 5

    #bonus collectif si tout le monde a réussi
    if len(evacuated_agents) == num_agents:
        total_bonus = 500
        rewards += total_bonus / num_agents

    return rewards, evacuated_agents
