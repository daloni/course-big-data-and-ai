import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium import ObservationWrapper, spaces


class MultiInputCarRacingWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Dict({
            "obs": env.observation_space,
            "info": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        })

    def observation(self, observation):
        speed = getattr(self.env.unwrapped.car, "speed", 0.0)
        angle = getattr(self.env.unwrapped.car, "angle", 0.0)
        info = np.array([speed / 100.0, angle / 180.0], dtype=np.float32)
        return {
            "obs": observation,
            "info": info
        }


# Crear entorno con render humano para visualizar
def make_env():
    env = gym.make("CarRacing-v3", render_mode="human")
    env = MultiInputCarRacingWrapper(env)
    return env

env = DummyVecEnv([make_env])

# Cargar el modelo entrenado
model = PPO.load("ppo_carracing")
# model = PPO.load("./logs/best_model/best_model", env=env)

# Evaluar por unos episodios
NUM_EPISODES = 5

for ep in range(NUM_EPISODES):
    obs = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += reward

    print(f"Episodio {ep+1}: recompensa total = {episode_reward[0]:.2f}")
