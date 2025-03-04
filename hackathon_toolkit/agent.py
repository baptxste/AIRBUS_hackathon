import numpy as np
import random
# class MyAgent():
#     def __init__(self, num_agents: int):        
#         # Random number generator for random actions pick
#         self.rng = np.random.default_rng()
#         self.num_agents = num_agents
#     def get_action(self, state: list, evaluation: bool = False):
#         # Choose random action
#         actions = self.rng.integers(low=0, high=6, size=self.num_agents)
#         return actions.tolist()

#     def update_policy(self, actions: list, state: list, reward: float):
#         # Do nothing
#         pass



class MyAgent():
    def __init__(self, num_agents: int):        
        # Random number generator for random actions pick
        self.rng = np.random.default_rng()
        self.num_agents = num_agents
        self.ai = QLearn(actions=range(7), alpha=0.1, gamma=0.9, epsilon=0.3)
        # self.ai =  DQN(10, 7, 0.001, 0.999, 0.9, 64)
        self.lastState = None
        self.lastAction = None
    def get_action(self, list_state: list, evaluation: bool = False):
        self.lastAction = []
        self.lastState = []
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
                final_state = np.concatenate((main_state, mean_position))

                state = np.array2string(final_state)

                self.lastState.append(state)
                self.lastAction.append(self.ai.choose_action(state))
            # else : 
            #     self.lastAction.append(-1)
        return self.lastAction
 
    def update_policy(self, actions: list, state: list, reward: float):
        for i, (s, a) in enumerate(zip(self.lastState, self.lastAction)):
            s = tuple(s) 
            next_s = tuple(state[i])  
            self.ai.learn(s, a, next_s, reward)




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


# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from collections import deque


# class MyAgent():
#     def __init__(self, num_agents: int):
#         # Initialisation du générateur de nombres aléatoires
#         self.rng = np.random.default_rng()
#         self.num_agents = num_agents
        
#         # Initialisation du modèle DQN
#         self.dqn = DQN(state_dim = 11, action_dim = 7, alpha=0.001, gamma=0.999, epsilon=0.9, batch_size=64)
#         self.lastState = None
#         self.lastAction = None

#     def get_action(self, list_state: list, evaluation: bool = False):
#         """ Sélectionne une action en utilisant le modèle DQN """
#         self.lastAction = []
#         self.lastState = []
        
#         for state in list_state:
#             if state[3] == 0:  # Vérification de l'état du drone
#                 state = np.concatenate((state[:3], state[4:]))  # Suppression de l'élément 3
#                 state = np.array(state, dtype=np.float32)  # Conversion en np.array
                
#                 self.lastState.append(state)
                
#                 if evaluation:
#                     action = self.dqn.choose_action(state, greedy=True)
#                 else:
#                     action = self.dqn.choose_action(state)
                
#                 self.lastAction.append(action)

#         return self.lastAction

#     def update_policy(self, actions: list, next_states: list, reward: float):
#         """ Mise à jour du modèle DQN en utilisant les transitions """
#         for i, (s, a) in enumerate(zip(self.lastState, self.lastAction)):
#             next_s = np.concatenate((next_states[i][:3], next_states[i][4:]))  # Suppression de l'élément 3
#             next_s = np.array(next_s, dtype=np.float32)

#             done = False  # À modifier si on a une condition de fin d'épisode
            
#             # Stocker l'expérience dans la mémoire
#             self.dqn.store_experience(s, a, reward, next_s, done)

#         # Apprentissage à partir de la mémoire d'expérience
#         self.dqn.learn()

#         # Mise à jour du réseau cible périodiquement
#         self.dqn.update_target_network()


# class DQN:
#     def __init__(self, state_dim, action_dim, alpha, gamma, epsilon, batch_size, memory_size=10000):
#         # Parameters
#         self.state_dim = state_dim  # dimension de l'état
#         self.action_dim = action_dim  # nombre d'actions possibles
#         self.alpha = alpha  # taux d'apprentissage
#         self.gamma = gamma  # facteur de réduction des récompenses futures
#         self.epsilon = epsilon  # taux d'exploration
#         self.batch_size = batch_size  # taille du batch pour l'expérience de réexamen
#         self.memory = deque(maxlen=memory_size)  # mémoire de réexamen
        
#         # Réseau de neurones Q (approximateur de la fonction Q)
#         self.q_network = self.build_model()
#         self.target_network = self.build_model()  # Réseau cible
#         self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.alpha)
        
#         # Initialisation du réseau cible avec les poids du réseau Q
#         self.target_network.load_state_dict(self.q_network.state_dict())
    
#     def build_model(self):
#         # """Crée un réseau de neurones simple pour approximer la fonction Q"""
#         # model = nn.Sequential(
#         #     nn.Linear(self.state_dim, 64),
#         #     nn.ReLU(),
#         #     nn.Linear(64, 64),
#         #     nn.ReLU(),
#         #     nn.Linear(64, self.action_dim)
#         # )
#         # return model
#         model = QNetwork(state_dim = self.state_dim, action_dim=self.action_dim)
#         return model


    
#     def get_utility(self, state):
#         """Prévoit les valeurs Q pour un état donné à partir du réseau de neurones"""
#         state = torch.FloatTensor(state).unsqueeze(0)  # Convertir l'état en tensor
#         q_values = self.q_network(state)
#         return q_values
    
#     def choose_action(self, state):
#         """Choisir une action en fonction de la stratégie epsilon-greedy"""
#         if random.random() < self.epsilon:
#             return random.choice(range(self.action_dim))  # Exploration
#         else:
#             q_values = self.get_utility(state)
#             return torch.argmax(q_values).item()  # Exploitation (choisir l'action avec la meilleure Q)
    
#     def store_experience(self, state, action, reward, next_state, done):
#         """Enregistre l'expérience dans la mémoire de réexamen"""
#         self.memory.append((state, action, reward, next_state, done))
    
#     def sample_experience(self):
#         """Échantillonne un batch d'expériences de la mémoire"""
#         return random.sample(self.memory, self.batch_size)
    
#     def learn(self):
#         """Met à jour les poids du réseau en utilisant un batch d'expériences"""
#         if len(self.memory) < self.batch_size:
#             return
        
#         # Échantillonner un batch d'expériences
#         batch = self.sample_experience()
#         states, actions, rewards, next_states, dones = zip(*batch)
        
#         states = torch.FloatTensor(states)
#         next_states = torch.FloatTensor(next_states)
#         actions = torch.LongTensor(actions)
#         rewards = torch.FloatTensor(rewards)
#         dones = torch.BoolTensor(dones)
        
#         # Calcul des Q-values pour les états actuels
#         # q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
#         q_values = self.q_network(states).gather(1, actions.view(-1, 1))

        
#         # Calcul des cibles (Q-targets) à partir du réseau cible
#         with torch.no_grad():
#             next_q_values = self.target_network(next_states).max(1)[0]
#             target_q_values = rewards + (1 - dones.float()) * self.gamma * next_q_values
        
#         # Calcul de la perte (MSE) entre Q-values et cibles
#         loss = nn.MSELoss()(q_values.squeeze(1), target_q_values)
        
#         # Optimiser le réseau Q
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
    
#     def update_target_network(self):
#         """Met à jour le réseau cible avec les poids du réseau Q"""
#         self.target_network.load_state_dict(self.q_network.state_dict())



# import torch
# import torch.nn as nn

# class QNetwork(nn.Module):
#     def __init__(self, state_dim, action_dim, num_heads=4, num_layers=2, d_model=64):
#         super(QNetwork, self).__init__()
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.d_model = d_model

#         # Embedding Layer pour convertir l'entrée en une représentation de dimension d_model
#         self.embedding = nn.Linear(state_dim, d_model)
        
#         # Positional Encoding pour injecter de l'information temporelle
#         # self.positional_encoding = nn.Parameter(torch.zeros(1, state_dim, d_model))
#         self.positional_encoding = nn.Parameter(torch.zeros(1, d_model))  # Supprime state_dim
        
#         # Définition des couches Transformer
#         encoder_layers = nn.TransformerEncoderLayer(
#             d_model=d_model, 
#             nhead=num_heads, 
#             dim_feedforward=128, 
#             activation='relu'
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

#         # Couches de sortie pour prédire les valeurs Q
#         self.fc_out = nn.Linear(d_model, action_dim)

#     def forward(self, x):
#         # Ajout de l'embedding et du positional encoding
#         x = self.embedding(x) + self.positional_encoding

#         # Passage dans le Transformer
#         x = self.transformer(x)

#         # Agrégation des features avant la sortie
#         x = x.mean(dim=1)  # Moyenne sur la séquence pour obtenir un vecteur unique

#         # Prédiction des valeurs Q
#         return self.fc_out(x)