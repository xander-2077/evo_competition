import sys
sys.path.append('/root/ws')

import competevo
import gym_compete

import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.wrappers import TransformObservation, TransformReward
import numpy as np
from config.env.config import Config
import argparse

class SelectFirstObservationWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # 假设原始env的observation_space是一个Tuple
        # 我们只取Tuple中的第一个Box作为新的observation_space
        self.observation_space = env.observation_space[0]

    def step(self, observation):
        # 从原始观测值Tuple中只选择第一个元素作为新的观测值
        return observation[0]

class FirstItemWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        # obs, reward, done, info = self.env.step(action)
        state, reward, terminated, truncated, info = self.env.step(action)
        terminated = any(terminated)
        # 同样，只取第一项的数据
        return state, reward, terminated, truncated, info[0]



def str2bool(input_str):
    """Converts a string to a boolean value.

    Args:
        input_str (str): The string to be converted.

    Returns:
        bool: The boolean representation of the input string.
    """
    true_values = ["true", "yes", "1", "on", "y", "t"]
    false_values = ["false", "no", "0", "off", "n", "f"]

    lowercase_str = input_str.lower()
    if lowercase_str in true_values:
        return True
    elif lowercase_str in false_values:
        return False
    else:
        raise ValueError("Invalid input string. Could not convert to boolean.")

parser = argparse.ArgumentParser(description="User's arguments from terminal.")
parser.add_argument("--cfg", 
                    dest="cfg_file", 
                    help="Config file", 
                    required=True, 
                    type=str)
parser.add_argument('--use_cuda', type=str2bool, default=True)
parser.add_argument('--gpu_index', type=int, default=0)
parser.add_argument('--num_threads', type=int, default=1)
parser.add_argument('--epoch', type=str, default='0')
args = parser.parse_args()
# Load config file
cfg = Config(args.cfg_file)

env = gym.make(cfg.env_name, cfg=cfg, render_mode="human")
# env = SelectFirstObservationWrapper(env)
# env = TransformObservation(env, lambda obs: obs[0])
# env = TransformReward(env, lambda reward: reward[0])
# env = FirstItemWrapper(env)
import pdb; pdb.set_trace()

obs, _ = env.reset()

for _ in range(10000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)
   import pdb; pdb.set_trace()

   if any(terminated) or truncated:
      observation, info = env.reset()
env.close()