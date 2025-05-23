import gymnasium as gym
import numpy as np
import datetime
import cv2
import os
from gymnasium import ObservationWrapper, RewardWrapper, spaces
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback

FPS = 50 # Obtained from the environment
TEST_MODE = True
CONTINUE_FROM_CHECKPOINT = False
render_mode = "human" if TEST_MODE else "rgb_array"

class OffTrackExponentialPenaltyWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_tile_count = 0
        self.negative_reward_time = 0.0

    # def reward(self, reward):
    #     base_env = self.env
    #     while hasattr(base_env, "env"):
    #         base_env = base_env.env

    #     # car = getattr(base_env, "car", None)
    #     # if car is not None and hasattr(car, "tiles") and len(car.tiles) == 0:
    #     #     reward -= 10.0

    #     if hasattr(base_env, "tile_visited_count") and hasattr(base_env, "t"):
    #         tile_count = base_env.tile_visited_count
    #         time_elapsed = base_env.t

    #         if tile_count > self.last_tile_count:
    #             reward += (tile_count - self.last_tile_count) / (time_elapsed + 1e-5) * 10.0

    #         self.last_tile_count = tile_count

    #     if reward < 0:
    #         self.negative_reward_time += 1.0 / 50.0
    #     else:
    #         self.negative_reward_time = 0.0

    #     # If time is greater than 3 seconds in negative reward, apply an additional penalty
    #     if self.negative_reward_time > 2.0:
    #         reward -= 100.0
    #         self.negative_reward_time = 0.0

    #     return reward
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        base_env = self.env
        while hasattr(base_env, "env"):
            base_env = base_env.env

        # car = getattr(base_env, "car", None)
        # if car is not None and hasattr(car, "tiles") and len(car.tiles) == 0:
        #     reward -= 10.0

        # Increment the reward for tiles traversed quickly
        if hasattr(base_env, "tile_visited_count") and hasattr(base_env, "t"):
            tile_count = base_env.tile_visited_count
            time_elapsed = base_env.t

            if tile_count > self.last_tile_count:
                reward += (tile_count - self.last_tile_count) / (time_elapsed + 1e-5) * 10.0

            self.last_tile_count = tile_count

        # If the reward is negative, accumulate the time
        if reward < 0:
            self.negative_reward_time += 1.0 / FPS
        else:
            self.negative_reward_time = 0.0

        # If the accumulated time with negative reward exceeds 5 seconds, terminate the episode
        if self.negative_reward_time > 5.0:
            terminated = True
            reward -= 100.0
            info["terminated_due_to_negative_reward"] = True

        return obs, reward, terminated, truncated, info

class GrayScaleObservation(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=(shape[0], shape[1], 1), dtype=np.uint8)

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

        # Show the grayscale image
        # cv2.imshow("Observation", gray)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return np.expand_dims(gray, -1)

class MultiInputCarRacingWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Dict({
            "image": env.observation_space,  # Imagen RGB o en escala de grises
            "info": spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)  # Ajusta la dimensión a (8,)
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

        # Ajusta la dimensión del vector "info" para que sea de tamaño (8,)
        info = np.array([normalized_speed, normalized_wheel_angle] * 4, dtype=np.float32)  # Repite para alcanzar dimensión 8

        return {
            "image": observation,
            "info": info
        }

def make_env(render_mode=None):
    def _init():
        env = gym.make("CarRacing-v3", render_mode=render_mode)
        env = OffTrackExponentialPenaltyWrapper(env)
        env = GrayScaleObservation(env)
        env = MultiInputCarRacingWrapper(env)
        return env
    return _init

if __name__ == "__main__":
    env = SubprocVecEnv([make_env(render_mode) for _ in range(8 if not TEST_MODE else 1)])
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
        # "CnnPolicy",
        "MultiInputPolicy",
        env,
        verbose=1,
        n_steps=1024,
        batch_size=128,
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=20,
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
