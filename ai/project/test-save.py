import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import RecordVideo
import os

def make_env():
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    env = Monitor(env)
    env = RecordVideo(env, video_folder="videos", episode_trigger=lambda episode_id: True)
    return env

if __name__ == "__main__":
    # Make sure the output folder exists
    os.makedirs("videos", exist_ok=True)

    # Load environment
    env = DummyVecEnv([make_env])
    env = VecTransposeImage(env)

    # Load the trained model
    model = PPO.load("ppo_car_racing_final", env=env)

    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward

    print(f"Total reward: {total_reward}")
    env.close()
