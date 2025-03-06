from pettingzoo.utils.conversions import parallel_wrapper_fn
from custom_env import CustomEnv 

def env_creator(args):
    return parallel_wrapper_fn(CustomEnv("config.json"))  # Converti en mode parallèle

# Enregistre l'environnement
from ray.tune.registry import register_env
register_env("CustomEnv-v0", lambda config: env_creator(config))