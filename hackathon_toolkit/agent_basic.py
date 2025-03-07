import numpy as np
import DQN
import torch
import random
from collections import deque

class MyAgents():
    def __init__(self, state_dim: int, action_dim: int, device : str, lr: float):
        self.device = torch.device(device)        
        # state dim = 11 pour preprocess_states
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # define the Networks
        self.q_network = DQN.DQN(input_dim=state_dim, output_dim=action_dim).to(self.device)
        self.target_network = DQN.DQN(input_dim=state_dim, output_dim=action_dim).to(self.device)
        
        # define the optimizer
        self.optimizers = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = torch.nn.SmoothL1Loss()
        
        # define the hyperparameters
        self.batch_size = 128
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.99
        self.TAU = 0.005
        
        # define the replay buffer
        self.replay_buffer = deque(maxlen=10000)
    
    def process_states(self, state: list) :
        final = []
        # print(len(list_state))
        if state[3]==0: # état du drone
            main_state = np.concatenate((state[:3], state[6:12]))
            goal = (state[4], state[5])
            main_state[0] = main_state[0] - goal[0]
            main_state[1] = main_state[1] - goal[1]
            other_state = state[12:]
            # print( len(state))
            # print( len(main_state))
            # print(len(other_state))
            num_drone = int(len(other_state)/10)
            avg_position = [(state[0], state[1])]

            for i in range(num_drone): # on récupère la postion de chaque drone pour calculer la position moyenne de l'essaim 
                if other_state[10*i+3] == 0 :
                    avg_position.append(( other_state[10*i+0],  other_state[10*i+1]))
            mean_position = np.round(np.mean(avg_position, axis=0)).astype(int) - goal
            
            # l'état final contient la position du drone, du goal et les infos lidar et la position moyenne de l'essaim
            final.append(np.concatenate((main_state, mean_position)).astype(int))
        if state[3]!=0: # état du drone
            final.append(np.full((11), -1))
        return final
        
    def get_action(self, state):
        state = self.process_states(state)
        if np.random.rand() < self.epsilon:
            return random.choice(np.arange(self.action_dim))
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # Add batch dimension
            q_values = torch.softmax(self.q_network(state_tensor),dim=-1)
            pi = torch.distributions.Categorical(q_values)
            action = pi.sample().item()
            # action = torch.argmax(q_values).item()
            # action = max(0, min(action, self.action_dim - 1))
            return action

    def update_target_network(self):
        target_net_state_dict = self.target_network.state_dict()
        policy_net_state_dict = self.q_network.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
        self.target_network.load_state_dict(target_net_state_dict)
        
    def update_policy(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.choices(self.replay_buffer, k=self.batch_size)
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = zip(*batch)
        
        #clean states
        batch_dim = len(batch_states)
        clean_batch_states = []
        clean_batch_next_states = []
        for i in range(batch_dim):
            clean_batch_states.append(self.process_states(batch_states[i]))
            clean_batch_next_states.append(self.process_states(batch_next_states[i]))
        batch_states = np.array(clean_batch_states)
        batch_next_states = np.array(clean_batch_next_states)
        
        #convert to tensor
        states = torch.FloatTensor(batch_states).squeeze(1).to(self.device)
        actions = torch.LongTensor(batch_actions).to(self.device)
        rewards = torch.FloatTensor(batch_rewards).to(self.device)
        next_states = torch.FloatTensor(batch_next_states).squeeze(1).to(self.device)
        dones = torch.FloatTensor(batch_dones).to(self.device)
        
        # compute actual q values
        q_values = self.q_network(states)
        # Retrieve the q values for the actions that were taken
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # compute target q values
        with torch.no_grad():
            target_q_values = self.target_network(next_states)
            target_q_values = rewards + self.gamma * target_q_values.max(dim=1).values * (1 - dones)
        
        # compute loss
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizers.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_value_(self.q_network.parameters(), 100)
        self.optimizers.step()
        
        # update epsilon
        #self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()