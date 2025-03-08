import numpy as np

def manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance between two positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

# ajouter un cas pour traier laction =-1 ( pour le cas ou  l'agent n'est plsu en jeu)
# def compute_reward(num_agents, old_positions, agent_positions, evacuated_agents, deactivated_agents, goal_area):#, current_step, max_steps):
#     rewards = np.zeros(num_agents)
#     # print( "OLD : ", old_positions)
#     # print("NEW : ",agent_positions)
#     # print("GOALS : ", goal_area)
#     # Compute reward for each agent
#     for i, (old_pos, new_pos) in enumerate(zip(old_positions, agent_positions)):
#         if i in evacuated_agents:
#             continue
#         elif i in deactivated_agents:   # Penalties for each deactivated agent
#             # only receive this penalty once
#             if not np.array_equal(old_pos, new_pos):
#                 rewards[i] = -100.0 
#             else:
#                 rewards[i] = 0.0
#         elif tuple(new_pos) in goal_area:   # One-time reward for each agent reaching the goal
#             rewards[i] = 100
#             evacuated_agents.add(i)
#         else:
#             # if the agent came closer of the goal their is a small  postive reward 
#             # if if move away small negative reward
#             d_old = manhattan_distance(old_positions[i], goal_area[i])
#             d_new= manhattan_distance(agent_positions[i], goal_area[i])
#             if d_old > d_new : 
#                 rewards[i] = 20.0

#             elif d_old == d_new : 
#                 rewards[i] = -10
                
#             else : rewards[i] = -15

#     return rewards, evacuated_agents
def compute_reward(num_agents, old_positions, agent_positions, evacuated_agents, deactivated_agents, goal_area):
    rewards = np.zeros(num_agents)

    # Conversion des positions en tableau numpy pour accélérer les calculs
    old_positions = np.array(old_positions)
    agent_positions = np.array(agent_positions)

    # Calcul des distances de Manhattan globales pour l'ensemble de l'essaim
    d_old_all = np.array([manhattan_distance(old_pos, goal_area[i]) for i, old_pos in enumerate(old_positions)])
    d_new_all = np.array([manhattan_distance(new_pos, goal_area[i]) for i, new_pos in enumerate(agent_positions)])

    # Vérifier si l'essaim se rapproche globalement du but (en comparant la somme des distances anciennes et nouvelles)
    total_distance_old = np.sum(d_old_all)
    total_distance_new = np.sum(d_new_all)
    swarm_progress = total_distance_old - total_distance_new

    # Ajouter une pénalité globale si l'essaim ne se rapproche pas du but
    global_penalty = 0
    if swarm_progress <= 0:  # Si l'essaim n'a pas fait de progrès vers le but
        global_penalty = -1.0  # Pénalité modérée pour l'ensemble de l'essaim

    for i, (old_pos, new_pos) in enumerate(zip(old_positions, agent_positions)):
        if i in evacuated_agents:
            continue
        elif i in deactivated_agents:   # Pénalité pour les drones désactivés
            if not np.array_equal(old_pos, new_pos):
                rewards[i] = -100.0 
            else:
                rewards[i] = 0.0
        elif tuple(new_pos) in goal_area:   # Récompense unique quand un drone atteint son objectif
            rewards[i] = 100
            evacuated_agents.add(i)
        else:
            # Récompense/pénalité en fonction de la distance au but
            d_old = manhattan_distance(old_positions[i], goal_area[i])
            d_new = manhattan_distance(agent_positions[i], goal_area[i])
            
            if d_old > d_new: 
                rewards[i] = 20.0  # Récompense si le drone se rapproche du but
            elif d_old == d_new: 
                rewards[i] = -10   # Légère pénalité s'il ne bouge pas efficacement
            else: 
                rewards[i] = -15   # Pénalité s'il s'éloigne

    # **Calcul de la position moyenne de l'essaim (barycentre)**
    swarm_center = np.mean(agent_positions, axis=0)  # Moyenne des positions (x, y)
    
    min_distance = 2  # Distance minimale souhaitée du centre de l'essaim
    max_distance = 6  # Distance maximale avant pénalité

    # Calcul de la distance par rapport au centre pour chaque agent
    distances = np.array([manhattan_distance(agent_positions[i], swarm_center) for i in range(num_agents)])

    # Pénalité si l'agent est trop près ou trop loin du centre de l'essaim
    for i, distance in enumerate(distances):
        if i in evacuated_agents:
            continue  # Ignorer les drones évacués

        if distance < min_distance:
            penalty = -10 * (min_distance - distance)  # Pénalité si trop proche du centre
            rewards[i] += penalty
        elif distance > max_distance:
            penalty = -5 * (distance - max_distance)  # Pénalité si trop loin du centre
            rewards[i] += penalty

    # Appliquer la pénalité globale à tous les agents (plus petite que les pénalités individuelles)
    rewards += global_penalty

    # **Normalisation des récompenses : Moyenne et écart-type en ligne**
    reward_mean_new = np.mean(rewards) 
    reward_std_new = np.std(rewards) 
    # Normalisation des récompenses actuelles
    normalized_rewards = (rewards ) / (reward_std_new*reward_mean_new+ 1)  # Ajout d'un epsilon pour éviter la division par zéro

    return normalized_rewards, evacuated_agents  # Pas besoin de renvoyer la moyenne et l'écart-type si pas nécessaire