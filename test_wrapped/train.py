# from custom_env import CustomEnv
# from stable_baselines3 import PPO
# import supersuit as ss
# from pettingzoo.test import parallel_api_test

# from supersuit.multiagent_wrappers.black_death import black_death_par


# if __name__ == "__main__":
#     env = CustomEnv("config.json")
#     # parallel_api_test(env, num_cycles=1_000)


#     env = black_death_par(env)
#     # env = ss.pad_observations_v0(env)

#     # Application des prétraitements nécessaires
#     env = ss.pettingzoo_env_to_vec_env_v1(env)
#     env = ss.concat_vec_envs_v1(env, 1, base_class="stable_baselines3")
#     model = PPO("MlpPolicy", env, verbose=1)

#     # Entraînement de l'agent
#     model.learn(total_timesteps=1000000)

#     # Sauvegarde du modèle entraîné
#     model.save("ppo_pettingzoo_model")








# import numpy as np
# import os
# from custom_env import CustomEnv
# from stable_baselines3 import PPO
# import supersuit as ss
# from stable_baselines3.common.callbacks import BaseCallback
# import matplotlib.pyplot as plt
# from supersuit.multiagent_wrappers.black_death import black_death_par
# from stable_baselines3.common.vec_env import DummyVecEnv


# # # Callback personnalisé pour enregistrer les métriques dans TensorBoard
# # class CustomTensorboardCallback(BaseCallback):
# #     """
# #     Custom callback for plotting additional values (like cumulative rewards) in TensorBoard.
# #     """

# #     def __init__(self, verbose=0):
# #         super().__init__(verbose)
# #         self.cumulative_rewards = []
# #         self.episode_rewards = []  # Liste des récompenses cumulées par épisode

# #     def _on_step(self) -> bool:
# #         """
# #         This method is called every time the model steps.
# #         Here we log cumulative rewards per episode.
# #         """
# #         # Récupérer la récompense cumulée pour chaque agent à partir de l'environnement
# #         if 'infos' in self.locals:
# #             for info in self.locals['infos']:
# #                 if 'episode' in info:
# #                     # Accumuler la récompense par épisode
# #                     self.episode_rewards.append(info['episode']['r'])
# #                     # Log de la récompense cumulée dans TensorBoard
# #                     self.logger.record("episode_reward", info['episode']['r'])

# #         return True

# #     def _on_training_end(self):
# #         """
# #         This method is called when training ends, allowing you to plot the cumulative rewards.
# #         """
# #         # Calculer et enregistrer la récompense cumulée totale
# #         total_reward = np.sum(self.episode_rewards)
# #         self.logger.record("total_reward", total_reward)
# #         # Affichage de la courbe des récompenses cumulées à la fin de l'entraînement
# #         plt.plot(self.episode_rewards)
# #         plt.xlabel('Episodes')
# #         plt.ylabel('Cumulative Reward')
# #         plt.title('Learning Curve')
# #         plt.savefig('learning_curve.png')
# #         plt.close()


# class CustomTensorboardCallback(BaseCallback):
#     """
#     Custom callback for plotting additional values (like cumulative rewards) in TensorBoard.
#     """

#     def __init__(self, verbose=0):
#         super().__init__(verbose)
#         self.episode_rewards = []

#     def _on_step(self) -> bool:
#         """
#         This method is called every time the model steps.
#         Here we log cumulative rewards per episode.
#         """
#         if 'infos' in self.locals:
#             for info in self.locals['infos']:
#                 if 'episode' in info:
#                     self.episode_rewards.append(info['episode']['r'])
#                     self.logger.record("episode_reward", info['episode']['r'])
#                     self.logger.record("mean_episode_reward", np.mean(self.episode_rewards))
#                     self.logger.record("reward_variance", np.var(self.episode_rewards))

#         return True

#     def _on_training_end(self):
#         """
#         This method is called when training ends, allowing you to plot the cumulative rewards.
#         """
#         total_reward = np.sum(self.episode_rewards)
#         self.logger.record("total_reward", total_reward)

#         plt.plot(self.episode_rewards)
#         plt.xlabel('Episodes')
#         plt.ylabel('Cumulative Reward')
#         plt.title('Learning Curve')
#         plt.savefig('learning_curve.png')
#         plt.close()


# if __name__ == "__main__":
#     # Charger l'environnement personnalisé
#     env = CustomEnv("config.json")

#     # Appliquer les prétraitements
#     env = black_death_par(env)
#     env = ss.pettingzoo_env_to_vec_env_v1(env)
#     env = ss.concat_vec_envs_v1(env,num_vec_envs = 1, num_cpus=20, base_class="stable_baselines3")


#     # Créer un dossier pour les logs
#     log_dir = "./logs"
#     os.makedirs(log_dir, exist_ok=True)

#     # Créer un callback personnalisé pour enregistrer les métriques
#     custom_callback = CustomTensorboardCallback()

#     # Créer et entraîner le modèle PPO
#     model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

#     model.learn(total_timesteps=1000000, callback=custom_callback)

#     model.save("ppo_pettingzoo_model")
#     print("Entraînement terminé. Les métriques sont disponibles dans TensorBoard.")








import numpy as np
import os
from custom_env import CustomEnv
from stable_baselines3 import PPO, DQN
import supersuit as ss
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
from supersuit.multiagent_wrappers.black_death import black_death_par
from stable_baselines3.common.vec_env import DummyVecEnv
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_results(data):
    cumulative_rewards = np.zeros(4)  # 4 agents
    steps = []

    reward1_cumulative = []
    reward2_cumulative = []
    reward3_cumulative = []
    reward4_cumulative = []

    # Parcourir les données pour accumuler les récompenses
    step = 0
    for entry in data:

        steps.append(step)
        step +=1
        
        # Ajouter les récompenses individuelles à la récompense cumulée
        cumulative_rewards += entry['individual_rewards']
        
        # Ajouter les récompenses cumulées à leurs listes respectives
        reward1_cumulative.append(cumulative_rewards[0])
        reward2_cumulative.append(cumulative_rewards[1])
        reward3_cumulative.append(cumulative_rewards[2])
        reward4_cumulative.append(cumulative_rewards[3])

    # Tracer les courbes des récompenses cumulées
    plt.figure(figsize=(10, 6))
    plt.plot(steps, reward1_cumulative, label='Agent 1', marker='o')
    plt.plot(steps, reward2_cumulative, label='Agent 2', marker='o')
    plt.plot(steps, reward3_cumulative, label='Agent 3', marker='o')
    plt.plot(steps, reward4_cumulative, label='Agent 4', marker='o')

    # Ajouter des labels et un titre
    plt.title("Récompenses cumulées des agents au fil des étapes")
    plt.xlabel("Étapes")
    plt.ylabel("Récompenses cumulées")
    plt.legend()

    # Afficher le graphique
    plt.grid(True)
    plt.savefig('results.png')
    plt.show()
    # plt.close()



class InfoCollectorCallback(BaseCallback):
    """
    Custom callback to collect and store the 'info' dictionary returned by the environment.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.infos = []

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
                if info!={} :
                    info['individual_rewards'] = info['individual_rewards'].tolist()

                    self.infos.append(info)

        return True

    def _on_training_end(self) -> None:
        """
        This method is called when training ends.
        Here we save the collected 'info' dictionaries to a JSON file.
        """
        with open('collected_infos.json', 'w') as f:
            json.dump(self.infos, f, indent=4)
            # f.write(str(self.infos))

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
    model = DQN("MlpPolicy", env, learning_rate=3e-4, gamma=0.99, batch_size=64, verbose=1, tensorboard_log=log_dir)

    model.learn(total_timesteps=10000, callback=info_callback)

    model.save("ppo_pettingzoo_model")
    print("Entraînement terminé. Les informations collectées ont été sauvegardées dans 'collected_infos.json'.")




with open("./collected_infos.json",'r') as file : 
    data = json.load(file)

plot_results(data)