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



# def compute_reward(num_agents, old_positions, agent_positions, evacuated_agents, deactivated_agents, goal_area, current_step, max_episode_steps):
#     rewards = np.zeros(num_agents)

#     for i, (old_pos, new_pos) in enumerate(zip(old_positions, agent_positions)):
#         if i in evacuated_agents:
#             continue
#         elif i in deactivated_agents:
#             death_penalty = -50 * (1 - current_step / max_episode_steps)
#             rewards[i] = death_penalty
#         elif tuple(new_pos) in goal_area:
#             rewards[i] = 1000.0
#             evacuated_agents.add(i)
#         else:
#             d_old = manhattan_distance(old_positions[i], goal_area[i])
#             d_new = manhattan_distance(agent_positions[i], goal_area[i])
#             delta = d_old - d_new

#             if delta > 0:
#                 rewards[i] = delta * 20
#             elif delta < 0:
#                 rewards[i] = delta * 20
#             else:
#                 rewards[i] = -10

#             #bonus de survie
#             rewards[i] += 50

#     #bonus collectif si tout le monde a réussi
#     if len(evacuated_agents) == num_agents:
#         total_bonus = 500
#         rewards += total_bonus / num_agents

#     return rewards, evacuated_agents


def compute_reward(num_agents, old_positions, agent_positions, evacuated_agents, deactivated_agents, goal_area, current_step, max_episode_steps):
    rewards = np.zeros(num_agents)
    reward_scale = 1000.0  #facteur de normalisation

    for i, (old_pos, new_pos) in enumerate(zip(old_positions, agent_positions)):
        if i in evacuated_agents:
            continue

        elif i in deactivated_agents:
            #pénalité de mort, pondérée par le temps (plus tôt = plus sévère)
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
                rewards[i] = delta * 20  #rapproche
            elif delta < 0:
                rewards[i] = delta * 20  #éloigne
                rewards[i] = -10  #surplace

            #bonus de survie à chaque step
            rewards[i] += 50

    #bonus collectif si tous les agents évacués
    if len(evacuated_agents) == num_agents:
        total_bonus = 500
        rewards += total_bonus / num_agents

    #normalisation des rewards
    rewards /= reward_scale

    return rewards, evacuated_agents
