import numpy as np
import qlearn
import DQN
import torch
import random
import env
from collections import deque

class MyAgents():
    def __init__(self, num_agents: int, state_dim: int, action_dim: int):        
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # define the AI
        self.q_networks = [DQN.DQN(input_dim=state_dim, output_dim=action_dim).to(self.device) for _ in range(self.num_agents)]
        self.target_networks = [DQN.DQN(input_dim=state_dim, output_dim=action_dim).to(self.device) for _ in range(self.num_agents)]
        self.mix_network = DQN.MixingNetwork(num_agents).to(self.device)
        
        self.optimizers = [torch.optim.Adam(q.parameters(), lr=0.005) for q in self.q_networks]
        self.mix_optimizer = torch.optim.Adam(self.mix_network.parameters(), lr=0.005)
        
        self.loss_fn = torch.nn.MSELoss()
        self.batch_size = 32
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.replay_buffer = deque(maxlen=10000)
        self.gamma = 0.99
        
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return [random.choice(np.arange(self.action_dim)) for _ in range(self.num_agents)]
        
        with torch.no_grad():
            actions = []
            for i,s in enumerate(state):
                state_tensor = s.unsqueeze(0)  # Add batch dimension
                q_values = torch.softmax(self.q_networks[i](state_tensor),dim=-1)
                actions.append(q_values.argmax(dim=-1).item())
            return actions

    def update_policy(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = zip(*batch)

        q_values = []
        next_q_values = []
        rewards = torch.FloatTensor(batch_rewards).to(self.device)
        dones = torch.FloatTensor(batch_dones).to(self.device)
        for i in range(self.num_agents):
            states = torch.FloatTensor([s[i] for s in batch_states], device=self.device)
            actions = torch.LongTensor([a[i] for a in batch_actions], device=self.device)
            next_states = torch.FloatTensor([s[i] for s in batch_next_states], device=self.device)

            q_values.append(self.q_networks[i](states).gather(1, actions.unsqueeze(1)).squeeze(1))
            with torch.no_grad():
                next_q_values.append(self.target_networks[i](next_states).max(1)[0])

        q_values = torch.stack(q_values, dim=1)
        next_q_values = torch.stack(next_q_values, dim=1)
        
        # QMIX - Construction de la Q-value globale
        q_total = self.mix_network(q_values)
        next_q_total = self.mix_network(next_q_values)

        # Cible pour l'apprentissage
        targets = rewards.mean(dim=-1) + self.gamma * next_q_total * (1 - dones.unsqueeze(1))

        # Loss et optimisation
        loss = self.loss_fn(q_total, targets.detach())
        self.mix_optimizer.zero_grad()
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        loss.backward()
        self.mix_optimizer.step()

        for optimizer in self.optimizers:
            optimizer.step()
    
    
    def update_target_networks(self):
        for i in range(self.num_agents):
            self.target_networks[i].load_state_dict(self.q_networks[i].state_dict())