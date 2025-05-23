import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecTransposeImage, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor

N_ENVS = 4
LOAD_PREVIOUS = False

if __name__ == "__main__":
    def make_env():
        def _init():
            env = gym.make("CarRacing-v3", render_mode="rgb_array")
            env = Monitor(env)
            return env
        return _init

    # Wrap the environment so that PPO can handle images (convnet)
    env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])
    env = VecTransposeImage(env)

    # StopTraining on Reward Threshold
    # stop_callback = StopTrainingOnRewardThreshold(reward_threshold=900, verbose=1)
    eval_callback = EvalCallback(
        env,
        # callback_on_new_best=stop_callback,
        eval_freq=5000,
        best_model_save_path="./ppo_car_racing_best",
        verbose=1,
    )

    # Create the PPO model
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=2.5e-4,
        n_steps=516,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log="./ppo_car_racing_tensorboard/"
    )

    if LOAD_PREVIOUS:
        model = PPO.load("./ppo_car_racing_model", env=env)
        print("Model loaded")

    # Training
    model.learn(total_timesteps=1_000_000, callback=eval_callback)

    # Save the model
    model.save("ppo_car_racing_model")
