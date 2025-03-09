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
    def __init__(self, state_size, action_size, n_agents, lr=1e-5, actor = None, critic= None, grid_size=30, max_lidar_range = 8):

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
        torch.save(self.actor, "./models/actor.pth")
        torch.save(self.critic, "./models/critic.pth")

