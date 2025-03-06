from custom_env import CustomEnv
from stable_baselines3 import PPO
import supersuit as ss
from pettingzoo.test import parallel_api_test

from supersuit.multiagent_wrappers.black_death import black_death_par


if __name__ == "__main__":
    env = CustomEnv("config.json")
    # parallel_api_test(env, num_cycles=1_000)


    env = black_death_par(env)
    # env = ss.pad_observations_v0(env)

    # Application des prétraitements nécessaires
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, base_class="stable_baselines3")
    model = PPO("MlpPolicy", env, verbose=1)

    # Entraînement de l'agent
    model.learn(total_timesteps=10000)

    # Sauvegarde du modèle entraîné
    model.save("ppo_pettingzoo_model")