from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecTransposeImage
from train2 import make_env

render_mode = "human"

env = DummyVecEnv([make_env(render_mode)])
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
