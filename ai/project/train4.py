import gymnasium as gym
import numpy as np
import datetime
import cv2
import os
from gymnasium import ObservationWrapper, spaces
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback

TEST_MODE = False
CONTINUE_FROM_CHECKPOINT = False
OUT_TIME_LIMIT = 3.0
# CHECKPOINT_FROM_PATH = "./logs/best_model/best_model.zip"
CHECKPOINT_FROM_PATH = "./ppo_carracing_20250520_2210.zip"
VEC_NORMALIZED_MODEL_PATH = f"{CHECKPOINT_FROM_PATH[:-4]}_vecnormalize.pkl"
render_mode = "human" if TEST_MODE else "rgb_array"

class GrayScaleObservation(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=0.0, high=1.0, shape=(64, 64, 1), dtype=np.float32)

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (64, 64))
        gray = gray.astype(np.float32) / 255.0
        return np.expand_dims(gray, -1)

class MultiInputCarRacingWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Dict({
            "image": env.observation_space,
            "info": spaces.Box(
                low=np.array([0.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32
            )
        })

    def observation(self, observation):
        car = self.env.unwrapped.car
        if car is not None:
            speed = np.linalg.norm(car.hull.linearVelocity)
            wheel_angle = (car.wheels[0].joint.angle + car.wheels[1].joint.angle) / 2.0
        else:
            speed = 0.0
            wheel_angle = 0.0

        # Normaliza los datos
        normalized_speed = speed / 100.0
        normalized_wheel_angle = wheel_angle / 0.4

        info = np.array([normalized_speed, normalized_wheel_angle], dtype=np.float32)

        return {
            "image": observation,
            "info": info
        }

def make_env(render_mode=None):
    def _init():
        env = gym.make("CarRacing-v3", render_mode=render_mode)
        env = GrayScaleObservation(env)
        env = MultiInputCarRacingWrapper(env)
        return env
    return _init

if __name__ == "__main__":
    env = SubprocVecEnv([make_env(render_mode) for _ in range(16 if not TEST_MODE else 1)])
    env = VecFrameStack(env, n_stack=8)

    # if CONTINUE_FROM_CHECKPOINT and os.path.exists(VEC_NORMALIZED_MODEL_PATH):
    #     print("Loading VecNormalize model...")
    #     env = VecNormalize.load(VEC_NORMALIZED_MODEL_PATH, venv=env)
    # else:
    #     env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # env = VecTransposeImage(env)

    eval_callback = EvalCallback(
        env,
        best_model_save_path="./logs/best_model/",
        log_path="./logs/results/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    def linear_schedule(initial_value):
        def schedule(progress_remaining):
            return progress_remaining * initial_value
        return schedule

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        n_steps=1024,
        batch_size=128,
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=10,
        ent_coef=0.001,
        learning_rate=linear_schedule(3e-4),
        clip_range=0.1,
        tensorboard_log="./ppo_carracing_tensorboard/",
        policy_kwargs={"normalize_images": True}
    )

    if CONTINUE_FROM_CHECKPOINT and os.path.exists(CHECKPOINT_FROM_PATH):
        print("Loading model from checkpoint...")
        model = PPO.load(CHECKPOINT_FROM_PATH, env=env)
        print(f"Loaded model from {CHECKPOINT_FROM_PATH}")

    model.learn(total_timesteps=1_000_000, callback=eval_callback)

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    model_name = f"ppo_carracing_{now}"
    model.save(model_name)
