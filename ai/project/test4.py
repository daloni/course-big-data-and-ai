from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from train4 import make_env

render_mode = "human"

MODEL_PATH = "./ppo_carracing_20250521_0030.zip"
# MODEL_PATH = "./logs/best_model/best_model"
# VEC_NORMALIZED_MODEL_PATH = f"{MODEL_PATH[:-4]}_vecnormalize.pkl"
env = DummyVecEnv([make_env(render_mode)])
env = VecFrameStack(env, n_stack=8)
# env = VecNormalize.load(VEC_NORMALIZED_MODEL_PATH, venv=env)
# env = VecTransposeImage(env)

model = PPO.load(MODEL_PATH, env=env)
# model = PPO.load("./logs/best_model/best_model", env=env)

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
