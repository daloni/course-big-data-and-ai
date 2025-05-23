import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor

def make_env():
    env = gym.make("CarRacing-v3", render_mode="human")  # 'human' to render in a window
    env = Monitor(env)
    return env

if __name__ == "__main__":
    # Load the environment
    env = DummyVecEnv([make_env])
    env = VecTransposeImage(env)

    # Load the trained model
    model = PPO.load("ppo_car_racing_final4", env=env)
    # model = PPO.load("./ppo_car_racing_best/best_model", env=env)

    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward

    print(f"Total reward: {total_reward}")
    env.close()
