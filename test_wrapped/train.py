import numpy as np
import os
from custom_env import CustomEnv
from stable_baselines3 import PPO, DQN, SAC, A2C
import supersuit as ss
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
from supersuit.multiagent_wrappers.black_death import black_death_par
from stable_baselines3.common.vec_env import DummyVecEnv
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

config_path ="config.json"

def plot_results(data):

    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    num_agent = config.get('num_agents')
    cumulative_rewards = np.zeros(num_agent) 
    steps = []
    cumulative = [ [] for _ in range(num_agent)]

    for entry in data:
        steps.append(len(steps))
        cumulative_rewards += entry['individual_rewards']

        for i in range(num_agent):
            cumulative[i].append(cumulative_rewards[i])

    plt.figure(figsize=(10, 6))
    for i in range(num_agent):
        plt.plot(steps,cumulative[i], label=f'Agent {i}')

    plt.title("Récompenses cumulées des agents au fil des étapes")
    plt.xlabel("Étapes")
    plt.ylabel("Récompenses cumulées")
    plt.legend()
    plt.grid(True)
    plt.savefig('results.png')
    plt.show()
    # plt.close()

def process_file_and_plot_histogram(data):

    counter = Counter(data)
    
    numbers = list(counter.keys())
    frequencies = list(counter.values())


    plt.figure(figsize=(10, 5))
    plt.bar(numbers, frequencies, color='blue', alpha=0.7)
    plt.xlabel('Nombre')
    plt.ylabel('Fréquence')
    plt.title('Fréquence des nombres dans les listes')
    plt.xticks(numbers)  
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('movements_freq.png')

    plt.show()


class InfoCollectorCallback(BaseCallback):
    """
    Custom callback to collect and store the 'info' dictionary returned by the environment.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.infos = []
        self.actions  = []

    def _on_step(self) -> bool:
        """
        This method is called at each step of the training.
        Here we collect the 'info' dictionary returned by the environment.
        """
        # if 'infos' in self.locals:
        #     for info in self.locals['infos']:
        #         self.infos.append(info)
       
        if 'infos' in self.locals:
            for info in self.locals['infos']:
                try : 
                    info['individual_rewards'] = info['individual_rewards'].tolist()
                    self.infos.append(info)
                    info["terminal_observation"] = info["terminal_observation"].tolist()
                     
                except : 
                    # print(info.keys())
                    pass

        if 'actions' in self.locals : 
            try : 
                self.actions.append(self.locals['actions'])
            except : 
                pass



        return True

    def _on_training_end(self) -> None:
        """
        This method is called when training ends.
        Here we save the collected 'info' dictionaries to a JSON file.
        """
        with open('collected_infos.json', 'w') as f:
            f.write('[\n')
            for i, d in enumerate(self.infos):
                json.dump(d, f)
                if i < len(self.infos) - 1:  # Vérifie si ce n'est pas le dernier élément
                    f.write(',\n')
            f.write('\n]')

        with open('all_actions.json', 'w') as f:
            for e in self.actions: 
                f.write(str(e.tolist())+'\n')

    def get_collected_infos(self):
        """
        Returns the collected 'info' dictionaries.
        """
        return self.infos


if __name__ == "__main__":
    env = CustomEnv("config.json")
    env = black_death_par(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=20, base_class="stable_baselines3")

    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)

    info_callback = InfoCollectorCallback()

    # model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    # model = DQN("MlpPolicy", env, learning_rate=3e-4, gamma=0.99, batch_size=64, verbose=1, tensorboard_log=log_dir)
    # model = DQN("MlpPolicy", env,
    #         learning_rate=1e-3,
    #         gamma=0.98, 
    #         batch_size=128,
    #         buffer_size=100000,  # Taille du buffer mémoire replay
    #         exploration_fraction=0.1,  # Durée de l'exploration epsilon-greedy
    #         exploration_final_eps=0.02,  # Valeur finale d'epsilon
    #         target_update_interval=1000,  # Fréquence de mise à jour du réseau cible
    #         verbose=1, 
    #         tensorboard_log=log_dir)

    # model = DQN("MlpPolicy", env,

    #         learning_rate=1e-3,
    #         gamma=0.98, 
    #         batch_size=128,
    #         buffer_size=100000,  # Taille du buffer mémoire replay
    #         exploration_fraction=0.1,  # Durée de l'exploration epsilon-greedy
    #         exploration_final_eps=0.02,  # Valeur finale d'epsilon
    #         target_update_interval=1000,  # Fréquence de mise à jour du réseau cible
    #         verbose=1, 
    #         tensorboard_log=log_dir)

    model = A2C("MlpPolicy", env, 
            learning_rate=7e-4,
            gamma=0.90,
            n_steps=1,  # Nombre de steps avant une mise à jour
            ent_coef=0.01,
            vf_coef=0.5,  # Pondération de la loss sur la valeur
            max_grad_norm=0.5,  # Clip du gradient pour éviter explosion des gradients
            verbose=1, 
            tensorboard_log=log_dir)

    # model = SAC("MlpPolicy", env,
    #         learning_rate=3e-4,
    #         gamma=0.99,
    #         batch_size=256, 
    #         buffer_size=500000,  
    #         tau=0.005,  # Taux de mise à jour du réseau cible
    #         ent_coef='auto',  # Coefficient d'entropie, auto-ajusté
    #         verbose=1, 
    #         tensorboard_log=log_dir)
    
    

    model.learn(total_timesteps=10000, callback=info_callback)

    model.save("ppo_pettingzoo_model")
    print("Entraînement terminé. Les informations collectées ont été sauvegardées dans 'collected_infos.json'.")




with open("./collected_infos.json",'r') as file : 
    data = json.load(file)

plot_results(data)

data = []
with open("./all_actions.json", 'r') as f:
    for line in f:
        numbers = list(line.strip().replace('[','').replace(']','').replace(' ','').split(','))  # Convertir en entiers
        data.extend(numbers)  # Ajouter tous les nombres à une seule liste

process_file_and_plot_histogram(data)