import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import sys
sys.path.append('/root/ws')
import competevo
import gym_compete

from train.CleanRL.PPO_hydra import FirstItemWrapper
# # Parallel environments
# vec_env = make_vec_env("CartPole-v1", n_envs=4)

env = gym.make("robo-sumo-ants-v0", cfg={'use_parse_reward': True},render_mode=None)
env = FirstItemWrapper(env)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200000)
model.save("ppo_cartpole")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render("human")