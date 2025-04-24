import gymnasium as gym
import numpy as np
import datetime
import cv2
import os
from gymnasium import ObservationWrapper, RewardWrapper
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback

render_mode = "human"

class OffTrackExponentialPenaltyWrapper(RewardWrapper):
    def __init__(self, env):
        """
        Termina el episodio si el coche está fuera de los tiles (fuera del circuito).
        
        Args:
            env: El entorno de Gymnasium.
        """
        super().__init__(env)

    def step(self, action):
        # Llama al método original `step` del entorno
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Accede al entorno base
        base_env = self.env
        while hasattr(base_env, "env"):
            base_env = base_env.env

        # Verifica si el coche está fuera de los tiles
        car = getattr(base_env, "car", None)
        if car is not None and hasattr(car, "tiles") and len(car.tiles) == 0:
            # Termina el episodio si el coche está fuera del circuito
            terminated = True
            info["off_track"] = True  # Agrega información adicional al diccionario `info`

        return obs, reward, terminated, truncated, info

class GrayScaleObservation(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=(shape[0], shape[1], 1), dtype=np.uint8)

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return np.expand_dims(gray, -1)

# Crear entorno con render humano para visualizar
def make_env():
    env = gym.make("CarRacing-v3", render_mode=render_mode)
    env = OffTrackExponentialPenaltyWrapper(env)
    env = GrayScaleObservation(env)
    return env

env = DummyVecEnv([make_env])
env = VecFrameStack(env, n_stack=8)

# model = PPO.load("ppo_carracing")
model = PPO.load("./logs/best_model/best_model", env=env)

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
