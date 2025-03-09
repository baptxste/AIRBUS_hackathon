import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
import os
from process_state import StateNormalizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=32, low_policy_weights_init=True):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.fc(x)


class Critic(nn.Module):
    def __init__(self, state_size, hidden_size=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x)


class MAPPOAgent:
    def __init__(self, state_size, action_size, n_agents, lr=5e-5, actor = None, critic= None, grid_size=30, max_lidar_range = 8):

        self.n_agents = n_agents
        if actor != None: 
            self.actor = actor.to(device)
        else : 
            self.actor = Actor(state_size, action_size).to(device)
        if critic != None : 
            self.critic = critic.to(device)
        else : 
            self.critic = Critic(state_size).to(device)
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.normalizer = StateNormalizer(grid_size, max_lidar_range)



    def select_actions(self, states):
        """ Sélectionne les actions pour tous les agents en parallèle """
        states = np.array([self.normalizer.normalize_agent_state(s) for s in states])
        states = torch.tensor(states, dtype=torch.float32, device=device)  # (n_agents, obs_dim)
        probs = self.actor(states)  # (n_agents, act_dim)
        distribution = dist.Categorical(logits=probs)
        actions = distribution.sample()  # (n_agents,)
        log_probs = distribution.log_prob(actions)  # (n_agents,)
        return actions.cpu().detach().numpy(), log_probs.cpu().detach().numpy()  # Retourne toutes les actions et log_probs

        # possible_actions = get_possible_actions(torch.tensor(states, dtype=torch.float32).to(device))  # Convertir en numpy pour éviter les conversions multiples
        # states = np.array([self.normalizer.normalize_agent_state(s) for s in states])
        # states = torch.tensor(states, dtype=torch.float32, device=device)
        # probs = self.actor(states)

        # mask = torch.full(probs.shape, float('-inf')).to(device)

        # for i, valid_actions in enumerate(possible_actions):
        #     mask[i, valid_actions] = 0

        # masked_probs = probs + mask
        # distribution = dist.Categorical(logits=masked_probs)
        # actions = distribution.sample()
        # log_probs = distribution.log_prob(actions)

        # return actions.detach().cpu().numpy(), log_probs.detach().cpu().numpy()


    def compute_loss(self, states, actions, log_probs_old, rewards, dones, gamma=0.99, clip_eps=0.2):
        states = np.array([self.normalizer.normalize_agent_state(s) for s in states])
        states = torch.tensor(states, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

        # Compute advantage
        values = self.critic(states).squeeze()
        dones = torch.tensor(dones, dtype=torch.float32,device=device)
        returns = rewards + gamma * values * (1 - dones)
        advantage = returns - values.detach()

        probs = self.actor(states)
        dist_new = dist.Categorical(logits=probs)
        log_probs_new = dist_new.log_prob(actions)

        # PPO Clip loss
        ratio = torch.exp(log_probs_new - log_probs_old)
        clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)

        # Calcul de l'entropie pour encourager l'exploration
        entropy = dist_new.entropy().mean()  # Mesure d'incertitude de la politique
        entropy_bonus = 0.01 * entropy  # Poids ajustable de l'entropie

        actor_loss = -torch.min(ratio * advantage, clipped_ratio * advantage).mean() - entropy_bonus

        # Critic loss (MSE loss)
        noise = torch.randn_like(returns) * 1 # Petit bruit aléatoire
        returns_noisy = returns + noise
        critic_loss = F.mse_loss(values, returns_noisy)

        return actor_loss, critic_loss


    def save(self):
        if not os.path.isdir('./models'):
            os.makedirs("./models")
        # torch.save(self.actor, "./models/actor.pth")
        # torch.save(self.critic, "./models/critic.pth")
        torch.save(self.actor.state_dict(), './models/actor.pth')
        torch.save(self.critic.state_dict(), './models/critic.pth')



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