import torch.distributions as dist
from process_state import StateNormalizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from network import Actor, Critic

from itertools import islice

class MyAgents:
    def __init__(self, state_size, action_size, n_agents, lr=1e-4, grid_size = 30, max_lidar_range = 5, device='cpu'):
        self.device = device
        self.n_agents = n_agents
        self.actor = Actor(state_size, action_size).to(self.device)
        self.critic = Critic(state_size).to(self.device)
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.normalizer = StateNormalizer(grid_size, max_lidar_range)

        self.batch_size = 64
        
    def get_forbidden_actions(self,state):
        """ Renvoie les actions interdites en fonction des data du LIDAR """
        if int(state[3]) == -1: # agent desactivé
          return []
        OBSTACLE_THRESHOLD = 1.0
        ACTION_FORWARD = 1
        ACTION_RIGHT = 4
        ACTION_LEFT = 3
        LIDAR_MAIN_INDEX = 6
        LIDAR_RIGHT_INDEX = 8
        LIDAR_LEFT_INDEX = 10
        
        lidar_directions = {
            0: [ACTION_FORWARD, ACTION_RIGHT, ACTION_LEFT],   # Face haut
            1: [ACTION_LEFT, ACTION_FORWARD, ACTION_RIGHT],   # Face gauche
            2: [ACTION_FORWARD, ACTION_RIGHT, ACTION_LEFT],   # Face bas
            3: [ACTION_RIGHT, ACTION_FORWARD, ACTION_LEFT]    # Face droite
        }
        forbidden = []

        lidar_main = state[LIDAR_MAIN_INDEX]   # Distance main
        lidar_right = state[LIDAR_RIGHT_INDEX]  # Distance à right
        lidar_left = state[LIDAR_LEFT_INDEX]  # Distance à left

        if lidar_main < OBSTACLE_THRESHOLD:
            forbidden.append(lidar_directions[int(state[3])][0])

        if lidar_right < OBSTACLE_THRESHOLD:
            forbidden.append(lidar_directions[int(state[3])][1])

        if lidar_left < OBSTACLE_THRESHOLD:
            forbidden.append(lidar_directions[int(state[3])][2])

        return forbidden

    def select_actions(self, states):
        """ Sélectionne les actions en enlevant les actions interdites """
        # Mask les actions interdites
        mask = torch.ones(size=(4,7), dtype=torch.bool, device=self.device)

        STATUS_INDEX = 4
        for i, state in enumerate(states):
            # Récupérer les actions interdites
            forbidden_actions = self.get_forbidden_actions(state)
            mask[i, forbidden_actions] = False  # Désactiver

        # Normalisation des états
        states = np.array([self.normalizer.normalize_agent_state(s) for s in states])
        states = torch.tensor(states, dtype=torch.float32, device=self.device)  # (n_agents, obs_dim)

        probs = self.actor(states)  # (n_agents, act_dim)

        # Appliquer le mask
        masked_probs = probs.masked_fill(~mask, float('-inf'))
        masked_probs = nn.functional.softmax(masked_probs, dim=-1)
        distribution = dist.Categorical(probs=masked_probs)
        actions = distribution.sample()  # (n_agents,)
        log_probs = distribution.log_prob(actions)  # (n_agents,)
        return actions.cpu().detach().numpy(), log_probs.cpu().detach().numpy()

    #def select_actions(self, states):
    #    """ Sélectionne les actions pour tous les agents en parallèle """
    #    states = np.array([self.normalizer.normalize_agent_state(s) for s in states])
    #    states = torch.tensor(states, dtype=torch.float32, device=device)  # (n_agents, obs_dim)
    #    probs = self.actor(states)  # (n_agents, act_dim)
    #    distribution = dist.Categorical(logits=probs)
    #    actions = distribution.sample()  # (n_agents,)
    #    log_probs = distribution.log_prob(actions)  # (n_agents,)
    #    return actions.cpu().detach().numpy(), log_probs.cpu().detach().numpy()  # Retourne toutes les actions et log_probs

    def compute_loss(self, states, actions, log_probs_old, rewards, dones, gamma=0.99, clip_eps=0.2):
        """ Renvoie les loss de l'acteur et du critique """
        states = np.array([self.normalizer.normalize_agent_state(s) for s in states])
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        # Compute advantage
        values = self.critic(states).squeeze()
        dones = torch.tensor(dones, dtype=torch.float32,device=self.device)
        returns = rewards + gamma * values * (1 - dones)
        advantage = returns - values.detach()
        # Normalisation avantage
        advantage = (advantage - advantage.mean()) / (advantage.std()+1e-5)
        # Calcul log_probs
        probs = self.actor(states)
        dist_new = dist.Categorical(logits=probs)
        log_probs_new = dist_new.log_prob(actions)

        # PPO Clip loss
        ratio = torch.exp(log_probs_new - log_probs_old)
        clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)

        # Loss entropie
        entropy = dist_new.entropy().mean()
        entropy_bonus = 0.01 * entropy

        # Loss acteur
        actor_loss = -torch.min(ratio * advantage, clipped_ratio * advantage).mean() - entropy_bonus

        # Loss critic
        noise = torch.randn_like(returns) * 1 # Pas sur que ça aide
        returns_noisy = returns + noise
        critic_loss = F.mse_loss(values, returns_noisy)

        return actor_loss, critic_loss
    
    
    def update_policy(self, buffer):
        data = buffer.get_data()
        
        if len(data) < self.batch_size:
            return
        # Selection d'un batch
        data = islice(data, len(data)-self.batch_size, len(data))
        
        states, actions, log_probs_old, rewards, next_states, dones = zip(*data)
        dones = list(dones)
        actor_loss, critic_loss = self.compute_loss(states, actions, log_probs_old, rewards, dones)

        # Update Actor
        self.optim_actor.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.optim_actor.step()

        # Update Critic
        self.optim_critic.zero_grad()
        critic_loss.backward()
        self.optim_critic.step()
