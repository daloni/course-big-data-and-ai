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

TEST_MODE = False
CONTINUE_FROM_CHECKPOINT = True
render_mode = "human" if TEST_MODE else "rgb_array"

# class OffTrackExponentialPenaltyWrapper(RewardWrapper):
#     def __init__(self, env, penalty_scale=5.0, base=1.1):
#         super().__init__(env)
#         self.penalty_scale = penalty_scale
#         self.base = base
#         self.off_track_steps = 0

#     def reward(self, reward):
#         base_env = self.env
#         while hasattr(base_env, "env"):
#             base_env = base_env.env

#         car = getattr(base_env, "car", None)
#         if car is not None and hasattr(car, "tiles") and len(car.tiles) == 0:
#             self.off_track_steps += 1
#             penalty = self.penalty_scale * (self.base ** self.off_track_steps)
#             reward -= penalty
#         else:
#             self.off_track_steps = 0

#         return reward
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

def make_env(render_mode=None):
    def _init():
        env = gym.make("CarRacing-v3", render_mode=render_mode)
        env = OffTrackExponentialPenaltyWrapper(env)
        env = GrayScaleObservation(env)
        return env
    return _init

if __name__ == "__main__":
    env = SubprocVecEnv([make_env(render_mode) for _ in range(4)])
    env = VecFrameStack(env, n_stack=8)

    eval_callback = EvalCallback(
        env,
        best_model_save_path="./logs/best_model/",
        log_path="./logs/results/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=10,
        ent_coef=0.01,
        learning_rate=2.5e-4,
        clip_range=0.2,
        tensorboard_log="./ppo_carracing_tensorboard/",
        policy_kwargs={"normalize_images": True}
    )

    if CONTINUE_FROM_CHECKPOINT and os.path.exists("./logs/best_model/best_model.zip"):
        print("Loading model from checkpoint...")
        checkpoint_path = "./logs/best_model/best_model.zip"
        model = PPO.load(checkpoint_path, env=env)
        print(f"Loaded model from {checkpoint_path}")

    model.learn(total_timesteps=1_000_000, callback=eval_callback)

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    model_name = f"ppo_carracing_{now}"
    model.save(model_name)
