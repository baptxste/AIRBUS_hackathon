import numpy as np
import random
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim

class MyAgent():
    def __init__(self, num_agents: int):        
        # Random number generator for random actions pick
        self.rng = np.random.default_rng()
        self.num_agents = num_agents
        # self.actions = [0,1,3,4,5,6] # on enlève la marche arrière car pas de lidar
        self.actions = [1,2,3,4,5]
        self.ai = QLearn(actions=self.actions, alpha=0.5, gamma=0.2, epsilon=0.5)
        # self.ai = DQN(state_dim=11, action_dim=7, alpha=0.01, gamma=0.1, epsilon=0.1)
        # self.ai =  DQN(10, 7, 0.001, 0.999, 0.9, 64)
        self.lastState = None
        self.lastAction = None

    def process_states(self, list_state: list) :
        final = []
        # print(len(list_state))
        for state in list_state : 
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
        # return list_state

    def get_action(self, list_state: list, evaluation: bool = False):
        self.lastAction = []
        self.lastState = []
        for state in self.process_states(list_state) : 
            if not np.all(np.equal(state, np.full((len(state)), -1))): # pas "dead"
                if isinstance(self.ai, QLearn) :
                    # state = ''.join(map(str, state))
                    state = state.tobytes()
                    pass

                if isinstance(self.ai, DQN) :
                    pass
                self.lastState.append(state)
                action = self.ai.choose_action(state)
                self.lastAction.append(action)
            else : 
                print("action 0 ", state)
                self.lastAction.append(0)# 0 pour immobile
        return self.lastAction
 
    # def update_policy(self, actions: list, state: list, reward: float):
    #     for i, (s, a) in enumerate(zip(self.lastState, self.lastAction)):
    #         s = tuple(s) 
    #         next_s = tuple(state[i])  
    #         self.ai.learn(s, a, next_s, reward[i])
    def update_policy(self, actions: list, state: list, reward: float):

        clean_states = self.process_states(state)

        # for i, (s1, a, s2) in enumerate(zip(self.lastState, self.lastAction, clean_states)):
        for i, (s1, a, s2) in enumerate(zip(self.lastState, actions, clean_states)):
            # print(i)
            # print("s1 : ", s1)
            # print("a : ",a)
            # print("s2 :",s2)
            # print(reward[i])
            # s = tuple(s1)
            if not isinstance(s1, bytes):
                s1 = s1.tobytes()
            # # next_s = tuple(s2)
            if not isinstance(s2, bytes):
                s2 = s2.tobytes()
            # next_s = s2.tobytes()
 
            self.ai.learn(s1, a, s2, reward[i])

    def save(self) :
        if isinstance(self.ai, DQN):
            if not os.path.exists("./dqn_models"):
                os.makedirs("./dqn_models")
            torch.save(self.ai.q_network, "./dqn_models/dqn.pth")
        elif isinstance(self.ai, QLearn): 
            if not os.path.exists("./qlearning_models"):
                os.makedirs("./qlearning_models")
            q_dict_serializable = {str(key): value for key, value in self.ai.q.items()} 
            with open("./qlearning_models/model.json", 'w') as json_file:
                json.dump(q_dict_serializable, json_file, indent=4)
        else : 
            print( "WARNING SAVE NOT IMPLEMENTED")





class QLearn:
    """
    Q-learning:
        Q(s, a) += alpha * (reward(s,a) + gamma * max(Q(s', a') - Q(s,a))

        * alpha is the learning rate.
        * gamma is the value of the future reward.
    It use the best next choice of utility in later state to update the former state.
    """
    def __init__(self, actions, alpha, gamma, epsilon):
        self.q = {}
        self.alpha = alpha
        self.gamma = gamma
        self.actions = actions  # collection of choices
        self.epsilon = epsilon  # exploration constant

    # Get the utility of an action in certain state, default is 0.0.
    def get_utility(self, state, action):
        return self.q.get((state, action), 0.0)

    # When in certain state, find the best action while explore new grid by chance.
    def choose_action(self, state):
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            q = [self.get_utility(state, act) for act in self.actions]
            max_utility = max(q)
            # In case there're several state-action max values
            # we select a random one among them
            if q.count(max_utility) > 1:
                best_actions = [self.actions[i] for i in range(len(self.actions)) if q[i] == max_utility]
                action = random.choice(best_actions)
            else:
                action = self.actions[q.index(max_utility)]
        return action

    # learn
    def learn(self, state1, action, state2, reward):
        old_utility = self.q.get((state1, action), None)
        if old_utility is None:
            self.q[(state1, action)] = reward

        # update utility
        else:
            next_max_utility = max([self.get_utility(state2, a) for a in self.actions])
            self.q[(state1, action)] = old_utility + self.alpha * (reward + self.gamma * next_max_utility - old_utility)

class DQN:
    """
    Deep Q-learning:
        Q(s, a) = reward(s, a) + gamma * max(Q(s', a'))

        * alpha is the learning rate.
        * gamma is the value of the future reward.
    """
    def __init__(self, state_dim, action_dim, alpha, gamma, epsilon, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = self.build_network(state_dim, action_dim).to(self.device)
        self.target_network = self.build_network(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=alpha)
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_dim = action_dim

    def build_network(self, state_dim, action_dim):
        """
        Build a simple neural network for Q-function approximation.
        """

        return nn.Sequential(
            nn.Linear(state_dim, 128),  
            nn.ReLU(),  
            nn.Dropout(0.2), 

            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, action_dim) 
        )


    def get_utility(self, state, action):
        """
        Get the Q-value for a state-action pair using the Q-network.
        """
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)  # Convert to tensor
        q_values = self.q_network(state)
        return q_values[0, action].item()

    def choose_action(self, state):
        """
        Choose an action based on epsilon-greedy strategy.
        """
        if random.random() < self.epsilon:
            return random.choice(range(self.action_dim))
        else:
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_network(state)
            return torch.argmax(q_values, dim=1).item()

    def learn(self, state1, action, state2, reward):
        """
        Perform one step of learning.
        """
        state1 = torch.tensor(state1, dtype=torch.float32, device=self.device).unsqueeze(0)
        state2 = torch.tensor(state2, dtype=torch.float32, device=self.device).unsqueeze(0)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)

        # Compute Q(s', a') for the next state using target network
        next_q_values = self.target_network(state2).detach()  # No gradient needed
        next_max_q = next_q_values.max(1)[0]

        # Compute the target for the Q-value
        target = reward + self.gamma * next_max_q

        # Compute the current Q-value for the state-action pair
        q_values = self.q_network(state1)
        current_q = q_values[0, action]

        # Compute the loss
        loss = nn.MSELoss()(current_q, target)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Optionally, update the target network every few episodes
        self.update_target_network()

    def update_target_network(self):
        """
        Copy the Q-network weights to the target network.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    