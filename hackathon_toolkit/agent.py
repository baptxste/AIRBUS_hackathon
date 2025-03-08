import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
import os

# Vérifie si un GPU est disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def get_possible_actions(states):
    listes_action_possibles = []
    for state in states:
        x, y, orientation = state[0], state[1], int(state[2])
        lidar_main_dist, lidar_main_type = state[6], state[7]
        lidar_right_dist, lidar_right_type = state[8], state[9]
        lidar_left_dist, lidar_left_type = state[10], state[11]

        possible_actions = {0, 5, 6}  # steady et rotations toujours possibles

        # Déterminer quel lidar correspond à quelle direction dans la grille
        directions = [
            (lidar_main_dist, lidar_main_type),
            (lidar_left_dist, lidar_left_type),
            (lidar_right_dist, lidar_right_type)
        ]

        if orientation == 0:  # Agent orienté vers le haut
            up, left, right, down = directions[0], directions[1], directions[2], None
        elif orientation == 1:  # Agent orienté vers la droite
            up, down, right, left = directions[1], directions[2], directions[0], None
        elif orientation == 2:  # Agent orienté vers le bas
            down, left, right, up = directions[0], directions[2], directions[1], None
        else:  # Agent orienté vers la gauche
            up, down, left, right = directions[2], directions[1], directions[0], None

        # Vérifier les déplacements en fonction des obstacles détectés
        def check_direction(direction):
            return direction and (direction[1] in {0, 2} or direction[0] > 1)

        if check_direction(up):
            possible_actions.add(1)
        if check_direction(down):
            possible_actions.add(2)
        if check_direction(left):
            possible_actions.add(3)
        if check_direction(right):
            possible_actions.add(4)

        listes_action_possibles.append(list(possible_actions))
    return listes_action_possibles

# class Actor(nn.Module):
#     def __init__(self, state_size, action_size, hidden_size=64):
#         super(Actor, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(state_size, hidden_size),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(hidden_size, action_size)
#         ).to(device)

#     def forward(self, x):
#         return self.model(x)

# class Critic(nn.Module):
#     def __init__(self, state_size, hidden_size=64):
#         super(Critic, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(state_size, hidden_size),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(hidden_size, 1)
#         ).to(device)

#     def forward(self, x):
#         return self.model(x)



class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256, num_heads=4, num_layers=4):
        super().__init__()

        self.embedding = nn.Linear(state_size, hidden_size)  # Embedding linéaire des états
        self.norm = nn.LayerNorm(hidden_size)  # Normalisation pour stabiliser l'apprentissage

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 2, 
            dropout=0.1, activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.fc_out = nn.Linear(hidden_size, action_size)  # Projection vers l'espace des actions
        self.to(device)  # Déplacement du modèle sur le GPU

    def forward(self, x):
        x = self.embedding(x)  # Transformation de l'entrée
        x = self.norm(x)  # Normalisation
        x = x.unsqueeze(0)  # Ajout d'une dimension batch (nécessaire pour Transformer)
        x = self.transformer_encoder(x)  # Passage dans l'encodeur Transformer
        x = x.squeeze(0)  # Suppression de la dimension batch
        return self.fc_out(x)  # Projection finale vers les actions


class Critic(nn.Module):
    def __init__(self, state_size, hidden_size=256, num_heads=4, num_layers=4):
        super().__init__()

        self.embedding = nn.Linear(state_size, hidden_size)  # Embedding linéaire des états
        self.norm = nn.LayerNorm(hidden_size)  # Normalisation pour stabiliser l'apprentissage

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 2, 
            dropout=0.1, activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.fc_out = nn.Linear(hidden_size, 1)  # Projection vers la valeur de l'état
        self.to(device)  # Déplacement du modèle sur le GPU

    def forward(self, x):
        x = self.embedding(x)  # Transformation de l'entrée
        x = self.norm(x)  # Normalisation
        x = x.unsqueeze(0)  # Ajout d'une dimension batch
        x = self.transformer_encoder(x)  # Passage dans l'encodeur Transformer
        x = x.squeeze(0)  # Suppression de la dimension batch
        return self.fc_out(x)  # Projection finale vers une valeur unique




class MAPPOAgent:
    def __init__(self, state_size, action_size, n_agents, lr=3e-5):
        self.n_agents = n_agents
        self.actor = Actor(state_size, action_size).to(device)
        self.critic = Critic(state_size).to(device)
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=lr)

    # def select_actions(self, states):
    #     states = torch.tensor(states, dtype=torch.float32).to(device)
    #     probs = self.actor(states)

    #     possible_actions = get_possible_actions(states)
    #     mask = torch.full(probs.shape, float('-inf')).to(device)

    #     for i, valid_actions in enumerate(possible_actions):
    #         mask[i, valid_actions] = 0

    #     masked_probs = probs + mask
    #     distribution = dist.Categorical(logits=masked_probs)
    #     actions = distribution.sample()
    #     log_probs = distribution.log_prob(actions)

    #     return actions.detach().cpu().numpy(), log_probs.detach().cpu().numpy()

    def select_actions(self, states):
        states = torch.tensor(states, dtype=torch.float32).to(device)
        probs = self.actor(states)

        possible_actions = get_possible_actions(states.cpu().numpy())  # Convertir en numpy pour éviter les conversions multiples
        mask = torch.full(probs.shape, float('-inf')).to(device)

        for i, valid_actions in enumerate(possible_actions):
            mask[i, valid_actions] = 0

        masked_probs = probs + mask
        distribution = dist.Categorical(logits=masked_probs)
        actions = distribution.sample()
        log_probs = distribution.log_prob(actions)

        return actions.detach().cpu().numpy(), log_probs.detach().cpu().numpy()

    def compute_loss(self, states, actions, log_probs_old, rewards, dones, gamma=0.99, clip_eps=0.1):
        with torch.no_grad():
            states = torch.tensor(states, dtype=torch.float32).to(device)
            actions = torch.tensor(actions, dtype=torch.long).to(device)
            log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            dones = torch.tensor(dones, dtype=torch.float32).to(device)

        values = self.critic(states).squeeze()
        returns = rewards + gamma * values * (1 - dones)
        advantage = returns - values.detach()

        probs = self.actor(states)
        dist_new = dist.Categorical(logits=probs)
        log_probs_new = dist_new.log_prob(actions)

        ratio = torch.exp(log_probs_new - log_probs_old)
        clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
        actor_loss = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()

        critic_loss = F.mse_loss(values, returns)

        return actor_loss, critic_loss

    def save(self):
        if not os.path.isdir('./models'):
            os.makedirs("./models")
        torch.save(self.actor, "./models/actor.pth")
        torch.save(self.critic, "./models/critic.pth")






# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.distributions as dist
# import numpy as np
# # Vérifie si un GPU est disponible
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# import os 



# def get_possible_actions(states):
#     listes_action_possibles = []
#     for state in states:
#         x, y, orientation = state[0], state[1], int(state[2])
#         lidar_main_dist, lidar_main_type = state[6], state[7]
#         lidar_right_dist, lidar_right_type = state[8], state[9]
#         lidar_left_dist, lidar_left_type = state[10], state[11]

#         possible_actions = {0, 5, 6}  # steady et rotations toujours possibles

#         # Déterminer quel lidar correspond à quelle direction dans la grille
#         if orientation == 0:  # Agent orienté vers le haut
#             up, left, right = (lidar_main_dist, lidar_main_type), (lidar_left_dist, lidar_left_type), (lidar_right_dist, lidar_right_type)
#             down = None
#         elif orientation == 1:  # Agent orienté vers la droite
#             up, down, right = (lidar_left_dist, lidar_left_type), (lidar_right_dist, lidar_right_type), (lidar_main_dist, lidar_main_type)
#             left = None
#         elif orientation == 2:  # Agent orienté vers le bas
#             down, left, right = (lidar_main_dist, lidar_main_type), (lidar_right_dist, lidar_right_type), (lidar_left_dist, lidar_left_type)
#             up = None
#         else:  # Agent orienté vers la gauche
#             up, down, left = (lidar_right_dist, lidar_right_type), (lidar_left_dist, lidar_left_type), (lidar_main_dist, lidar_main_type)
#             right = None

#         # Vérifier les déplacements en fonction des obstacles détectés
#         def check_direction(direction):
#             if direction and (direction[1] in {0, 2} or direction[0] > 1):
#                 return True
#             return False

#         if check_direction(up):
#             possible_actions.add(1)
#         if check_direction(down):
#             possible_actions.add(2)
#         if check_direction(left):
#             possible_actions.add(3)
#         if check_direction(right):
#             possible_actions.add(4)

#         listes_action_possibles.append(list(possible_actions))
#     return listes_action_possibles


# class Actor(nn.Module):
#     def __init__(self, state_size, action_size, hidden_size=64):
#         super(Actor, self).__init__()
        
#         # Définition des couches sous forme séquentielle
#         self.model = nn.Sequential(
#             nn.Linear(state_size, hidden_size),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(hidden_size, action_size)  # Pas d'activation ici pour les logits d'action
#         ).to(device)

#     def forward(self, x):
#         return self.model(x)


# class Critic(nn.Module):
#     def __init__(self, state_size, hidden_size=64):
#         super(Critic, self).__init__()

#         # Définition des couches sous forme séquentielle
#         self.model = nn.Sequential(
#             nn.Linear(state_size, hidden_size),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(hidden_size, 1)  # Sortie unique pour la valeur de l'état
#         ).to(device)

#     def forward(self, x):
#         return self.model(x)
    

# class MAPPOAgent:
#     def __init__(self, state_size, action_size, n_agents, lr=3e-5):
#         self.n_agents = n_agents
#         self.actor = Actor(state_size, action_size).to(device)  # Déplace le modèle sur le GPU
#         self.critic = Critic(state_size).to(device)  # Déplace le modèle sur le GPU
#         self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
#         self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=lr)

#     # def select_actions(self, states):
#     #     """ Sélectionne les actions pour tous les agents en parallèle """
#     #     states = torch.tensor(states, dtype=torch.float32).to(device)  # Déplace les données sur le GPU
#     #     probs = self.actor(states)  # (n_agents, act_dim)
#     #     distribution = dist.Categorical(logits=probs)
#     #     actions = distribution.sample()  # (n_agents,)
#     #     # actions = probs.argmax()  # (n_agents,)
#     #     log_probs = distribution.log_prob(actions)  # (n_agents,)
#     #     # return actions.detach().cpu().numpy(), log_probs.detach().cpu().numpy()  # Déplace les résultats vers le CPU
#     #     actions = actions.detach().cpu().numpy()
#     #     log_probs = log_probs.detach().cpu().numpy()

#     #     # Correction des actions non valides
#     #     for i, valid_actions in enumerate(get_possible_actions(states)):
#     #         if actions[i] not in valid_actions:
#     #             actions[i] = np.random.choice(valid_actions)  # Remplace par une action valide au hasard

#     #     return actions, log_probs
#     def select_actions(self, states):
#         """ Sélectionne les actions pour tous les agents en parallèle en ne permettant que les actions valides """
#         states = torch.tensor(states, dtype=torch.float32).to(device)  # Déplace les données sur le GPU
#         probs = self.actor(states)  # (n_agents, act_dim)

#         possible_actions = get_possible_actions(states)  # Récupère les actions valides pour chaque agent
#         mask = torch.full(probs.shape, float('-inf')).to(device)  # Initialisation du masque avec -inf
        
#         # Applique le masque : met 0 pour les actions valides
#         for i, valid_actions in enumerate(possible_actions):
#             mask[i, valid_actions] = 0  

#         masked_probs = probs + mask  # Applique le masque aux logits
#         distribution = dist.Categorical(logits=masked_probs)  # Crée la distribution avec les actions valides
#         actions = distribution.sample()  # Sélectionne une action parmi les valides
#         log_probs = distribution.log_prob(actions)  # Récupère le log-prob correspondant

#         return actions.detach().cpu().numpy(), log_probs.detach().cpu().numpy()  # Déplace vers le CPU pour éviter les erreurs



#     def compute_loss(self, states, actions, log_probs_old, rewards, dones, gamma=0.99, clip_eps=0.1):
#         states  = np.array(states)# speed up the tensor conversion
#         states = torch.tensor(states, dtype=torch.float32).to(device)  # Déplace les données sur le GPU
#         actions = torch.tensor(actions, dtype=torch.long).to(device)  # Déplace les données sur le GPU
#         log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32).to(device)  # Déplace les données sur le GPU
#         rewards = torch.tensor(rewards, dtype=torch.float32).to(device)  # Déplace les données sur le GPU
#         dones = torch.tensor(dones, dtype=torch.float32).to(device)  # Déplace les données sur le GPU
        
#         # Compute advantage
#         values = self.critic(states).squeeze()
#         returns = rewards + gamma * values * (1 - dones)
#         advantage = returns - values.detach()

#         # Compute new log_probs
#         probs = self.actor(states)
#         dist_new = dist.Categorical(logits=probs)
#         log_probs_new = dist_new.log_prob(actions)

#         # PPO Clip loss
#         ratio = torch.exp(log_probs_new - log_probs_old)
#         clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
#         actor_loss = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()

#         # Critic loss (MSE loss)
#         critic_loss = F.mse_loss(values, returns)

#         return actor_loss, critic_loss

#     def save(self): 
#         if not os.path.isdir('./models'): 
#             os.makedirs("./models")
#         torch.save(self.actor, "./models/actor.pth")
#         torch.save(self.critic, "./models/critic.pth")



