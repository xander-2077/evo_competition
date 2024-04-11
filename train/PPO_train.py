import sys
sys.path.append('/root/ws')

import competevo
import gym_compete

import gymnasium as gym
from config.config import Config
import argparse

import os
import glob
import time
from datetime import datetime, timedelta

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from train.PPO_policy import PPO

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


################################### Training ###################################
def train(cfg):
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    env_name = cfg.env_name

    has_continuous_action_space = True  # continuous action space; else discrete

    max_ep_len = 1000                   # max timesteps in one episode
    max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e5)          # save model frequency (in num timesteps)

    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4      # update policy every n timesteps
    K_epochs = 80               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    random_seed = 0         # set random seed if required (0 = no random seed)
    #####################################################

    print("training environment name : " + env_name)

    # env = gym.make(env_name, cfg=cfg, render_mode="human")
    env = gym.make(env_name, cfg=cfg, render_mode=None)
    # import pdb; pdb.set_trace()

    # state space dimension
    state_dim = env.observation_space[0].shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space[0].shape[0]
    else:
        action_dim = env.action_space.n

    ###################### logging ######################
    
    
    




    #### log files for multiple runs are NOT overwritten
    root_dir = "./result"
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    root_dir = root_dir + '/' + env_name + '/'
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    current_time = (datetime.now()+ timedelta(hours=10)).strftime('%m-%d_%H-%M')

    root_dir = root_dir + '/' + current_time + '_PPO'
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    writer = SummaryWriter(log_dir=root_dir)

    # #### get number of log files in log directory
    # run_num = 0
    # current_num_files = next(os.walk(log_dir))[2]
    # run_num = len(current_num_files)

    # #### create new log file for each run
    # log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    # print("current logging run number for " + env_name + " : ", run_num)
    # print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    # run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

    # directory = "./result/model"
    # if not os.path.exists(directory):
    #       os.makedirs(directory)

    # directory = directory + '/' + env_name + '/'
    # if not os.path.exists(directory):
    #       os.makedirs(directory)


    # checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)

    checkpoint_path = root_dir + '/ppoagent.pth'

    # print("save checkpoint path : " + checkpoint_path)
    #####################################################


    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    # log_f = open(log_f_name,"w+")
    # log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= max_training_timesteps:

        state, _ = env.reset()
        state = state[0]
        current_ep_reward = 0

        for t in range(1, max_ep_len+1):

            # select action with policy
            action_0 = ppo_agent.select_action(state)
            action_1 = env.action_space.sample()[1]
            action = (action_0, action_1)
            # import pdb; pdb.set_trace()
            # state, reward, done, _ = env.step(action)
            state, reward, terminated, truncated, info = env.step(action)

            state = state[0]
            reward = reward[0]


            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)

            done = any(terminated) or truncated
            ppo_agent.buffer.is_terminals.append(done)

            time_step +=1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                writer.add_scalar('log_avg_reward', log_avg_reward, time_step)
                # log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                # log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    # log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
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

    train(cfg)