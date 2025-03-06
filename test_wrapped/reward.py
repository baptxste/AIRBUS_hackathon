import numpy as np

def manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance between two positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    

# #recompense normalisée pour le dqnn
# def compute_reward(num_agents, old_positions, agent_positions, evacuated_agents, deactivated_agents, goal_area):
#     rewards = np.zeros(num_agents)
#     # print( "OLD : ", old_positions)
#     # print("NEW : ",agent_positions)
#     # print("GOALS : ", goal_area)

#     # Compute reward for each agent
#     for i, (old_pos, new_pos) in enumerate(zip(old_positions, agent_positions)):
#         if i in evacuated_agents:
#             continue
#         elif i in deactivated_agents:   # Penalties for each deactivated agent
#             # print(old_pos, new_pos)
#             # # print(i)
#             if  not np.array_equal(old_pos, new_pos): # le drone vient d'être eliminé
#                 rewards[i] = -1
#             else : rewards[i] = 0
#         elif tuple(new_pos) in goal_area:   # One-time reward for each agent reaching the goal
#             rewards[i] = 1
#             evacuated_agents.add(i)
#         else:
#             # if the agent came closer of the goal their is a small  postive reward 
#             # if if move away small negative reward
#             d_old = manhattan_distance(old_positions[i], goal_area[i])
#             d_new= manhattan_distance(agent_positions[i], goal_area[i])
#             if d_old < d_new : 
#                 rewards[i] = 1
#             else : rewards[i] = -0.1
#     return rewards, evacuated_agents


# def compute_reward(num_agents, old_positions, agent_positions, evacuated_agents, deactivated_agents, goal_area):
#     rewards = np.zeros(num_agents)

#     for i, (old_pos, new_pos) in enumerate(zip(old_positions, agent_positions)):
#         if i in evacuated_agents:
#             continue
        
#         elif i in deactivated_agents:  
#             # Si l'agent est désactivé, il prend une grosse pénalité une seule fois
#             if not np.array_equal(old_pos, new_pos): 
#                 rewards[i] = -2  # Plus sévère pour décourager d'être désactivé
#             else:
#                 rewards[i] = 0  # Pas de récompense en restant immobile
        
#         elif tuple(new_pos) in goal_area:  
#             rewards[i] = 1  # Récompense en atteignant l'objectif
#             evacuated_agents.add(i)
        
#         else:
#             # Encourager les agents à se rapprocher de leur objectif
#             d_old = manhattan_distance(old_pos, goal_area[i])
#             d_new = manhattan_distance(new_pos, goal_area[i])
            
#             if d_new < d_old:  # Si l'agent se rapproche
#                 rewards[i] = 0.2  # Récompense progressive
#             elif d_new > d_old:  # Si l'agent s'éloigne
#                 rewards[i] = -0.2  # Pénalité plus importante
#             else:  # Si l'agent ne bouge pas
#                 rewards[i] = -0.5  # Encourager à bouger

#     return rewards, evacuated_agents

import numpy as np

def compute_reward(num_agents, old_positions, agent_positions, evacuated_agents, deactivated_agents, goal_area):
    """
    Calcule la récompense pour chaque agent en fonction de ses déplacements.

    - num_agents : nombre total d'agents
    - old_positions : liste des anciennes positions [array([x, y])]
    - agent_positions : liste des nouvelles positions [array([x, y])]
    - evacuated_agents : set des agents ayant atteint leur but
    - deactivated_agents : set des agents désactivés
    - goal_area : liste des objectifs [(goal_x, goal_y)]
    
    Retourne : tableau des récompenses et liste mise à jour des agents évacués
    """

    rewards = np.zeros(num_agents)

    for i in range(num_agents):
        old_pos = tuple(old_positions[i])  # Ancienne position (x, y)
        new_pos = tuple(agent_positions[i])  # Nouvelle position (x, y)
        goal_pos = goal_area[i]  # Objectif (x, y)

        if i in evacuated_agents:  # Si déjà évacué, pas de récompense
            continue
        elif i in deactivated_agents:  # Si désactivé, grosse pénalité
            if old_pos != new_pos:  # Détecte le moment où il est désactivé
                rewards[i] = -5
            else:
                rewards[i] = 0
        elif new_pos == goal_pos:  # Atteindre l'objectif
            rewards[i] = 500
            evacuated_agents.add(i)
        else:
            # Distance Manhattan avant/après
            d_old = manhattan_distance(old_pos, goal_pos)
            d_new = manhattan_distance(new_pos, goal_pos)

            # Récompense en fonction de la direction du déplacement
            if d_new < d_old:
                rewards[i] = 1  # Bonus pour se rapprocher
            else:
                rewards[i] = -0.2  # Pénalité si éloignement

    return rewards, evacuated_agents
