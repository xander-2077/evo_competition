import competevo
import gym_compete

import gymnasium as gym
from config.env_cfg.config import Config
import argparse

from stable_baselines3 import PPO
from torch.utils.tensorboard import SummaryWriter

# Tensorboard
writer = SummaryWriter(log_dir='./result/log')

class EnvSingleWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.tuple.Tuple), "env.observation_space must be gym.spaces.tuple.Tuple"
        self.observation_space = env.observation_space.spaces[0]

    def step(self, action):
        # observation, reward, terminated, truncated, info = env.step(action)
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation[0], reward, terminated, truncated, info

    def reset(self):
        # observation, _ = env.reset()
        observation, _ = self.env.reset()
        return observation[0], _


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

env = gym.make(cfg.env_name, cfg=cfg, render_mode=None)
observation, _ = env.reset()

# import pdb; pdb.set_trace()

env_ = EnvSingleWrapper(env)

model = PPO("MlpPolicy", env_, verbose=1)
model.learn(total_timesteps=10000)

for _ in range(10000):
    import pdb; pdb.set_trace()
#   obs: Tuple(Box(-1.0, 1.0, (8,), float32), Box(-1.0, 1.0, (8,), float32))
    action, _ = model.predict(observation[0])
    # import pdb; pdb.set_trace()
    action_all = env.action_space.sample()  # this is where you would insert your policy
    action_1 = action_all[1]

    action = (action, action_1)

    observation, reward, terminated, truncated, info = env.step(action)

    if any(terminated) or truncated:
       observation, info = env.reset()
env.close()