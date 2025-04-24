import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium import ObservationWrapper, spaces

TEST_MODE = False
render_mode = "human" if TEST_MODE else "rgb_array"

# class MultiInputCarRacingWrapper(ObservationWrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         self.observation_space = spaces.Dict({
#             "obs": env.observation_space,  # imagen (96x96x3)
#             "info": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)  # [speed, angle]
#         })

#     def observation(self, observation):
#         speed = getattr(self.env.unwrapped.car, "speed", 0.0)
#         angle = getattr(self.env.unwrapped.car, "angle", 0.0)
#         info = np.array([speed / 100.0, angle / 180.0], dtype=np.float32)
#         return {
#             "obs": observation,
#             "info": info
#         }

# def make_env():
#     env = gym.make("CarRacing-v3", render_mode=render_mode)
#     env = MultiInputCarRacingWrapper(env)
#     return env

# env = DummyVecEnv([make_env])

def make_env(render_mode=None):
    def _init():
        env = gym.make("CarRacing-v3", render_mode=render_mode)
        return env
    return _init


env = SubprocVecEnv([make_env(render_mode) for _ in range(4)])
env = VecFrameStack(env, n_stack=4)
env = VecTransposeImage(env)

eval_callback = EvalCallback(
    env,
    best_model_save_path="./logs/best_model/",
    log_path="./logs/results/",
    eval_freq=10000,
    deterministic=True,
    render=False
)

# model = PPO(
#     "MultiInputPolicy",
#     env,
#     verbose=1,
#     n_steps=2048,
#     batch_size=64,
#     gae_lambda=0.95,
#     gamma=0.99,
#     n_epochs=10,
#     ent_coef=0.01,
#     learning_rate=2.5e-4,
#     clip_range=0.2,
#     tensorboard_log="./ppo_multi_tensorboard/",
# )
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
    policy_kwargs={"normalize_images": False},
)
# model = SAC(
#     "CnnPolicy",
#     env,
#     verbose=1,
#     batch_size=256,
#     learning_rate=3e-4,
#     buffer_size=100_000,
#     learning_starts=10000,
#     tau=0.005,
#     gamma=0.99,
#     train_freq=1,
#     gradient_steps=1,
#     tensorboard_log="./sac_carracing_tensorboard/",
#     policy_kwargs={"normalize_images": False}
# )

model.learn(total_timesteps=100_000, callback=eval_callback)
model.save("ppo_carracing")
