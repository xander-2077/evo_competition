import sys
sys.path.append('/root/ws')
import competevo
import gym_compete

import gymnasium as gym
import hydra
from omegaconf import OmegaConf

import os
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from train.PPO_policy import PPO


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def train(cfg):
    env_name = cfg.env.env_name
    has_continuous_action_space = cfg.algo.has_continuous_action_space
    max_ep_len = cfg.algo.max_ep_len     # TODO: 这取决于环境的最大步数      
    max_training_timesteps = cfg.algo.max_training_timesteps   
    print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    save_model_freq = cfg.algo.save_model_freq
    action_std = cfg.algo.action_std            
    action_std_decay_rate = cfg.algo.action_std_decay_rate       
    min_action_std = cfg.algo.min_action_std        
    action_std_decay_freq = cfg.algo.action_std_decay_freq 
    update_timestep = max_ep_len * 4      # update policy every n timesteps
    K_epochs = cfg.algo.K_epochs             
    eps_clip = cfg.algo.eps_clip        
    gamma = cfg.algo.gamma       
    lr_actor = cfg.algo.lr_actor      
    lr_critic = cfg.algo.lr_critic     
    random_seed = cfg.algo.random_seed       

    ################ Env setting #########################

    env = gym.make(env_name, cfg=cfg.env, render_mode=cfg.render_mode)

    # state space dimension
    state_dim = env.observation_space[0].shape[0]   # choose the first agent's observation space

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space[0].shape[0]   # choose the first agent's action space
    else:
        action_dim = env.action_space[0].n

    ###################### logging ######################
    root_dir = "./result"
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    root_dir = root_dir + '/' + env_name + '/'
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    current_time = datetime.now().strftime('%m-%d_%H-%M')

    root_dir = root_dir + '/' + current_time + '_' + cfg.algo.algo_name + '/'
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # tensorboard writer
    writer = SummaryWriter(log_dir=root_dir)

    # checkpoint path
    checkpoint_path = root_dir + '/ppoagent.pth'

    # save config
    with open(os.path.join(root_dir, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))

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

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    print("============================================================================================")

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    # log_running_reward = 0
    # log_running_episodes = 0

    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= max_training_timesteps:

        state, _ = env.reset()
        current_ep_reward = 0

        for t in range(1, max_ep_len+1):

            # select action with policy, agent1的动作由ppo_agent选择，agent2的动作随机选择
            action_0 = ppo_agent.select_action(state[0])
            action_1 = env.action_space.sample()[1]
            action = (action_0, action_1)
            state, reward, terminated, truncated, info = env.step(action)

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

            # # log in logging file
            # if time_step % log_freq == 0:

            #     # log average reward till last episode
            #     log_avg_reward = log_running_reward / log_running_episodes
            #     log_avg_reward = round(log_avg_reward, 4)

            #     writer.add_scalar('log_avg_reward', log_avg_reward, time_step)

            #     log_running_reward = 0
            #     log_running_episodes = 0

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
                ppo_agent.save(checkpoint_path)

            # break; if the episode is over
            if done:
                break
        
        writer.add_scalar('episode/episode_reward', current_ep_reward, i_episode)
        writer.add_scalar('episode/episode_length', t, i_episode)
        writer.add_scalar('timestep/reward', current_ep_reward, time_step)
        
        print_running_reward += current_ep_reward
        print_running_episodes += 1

        # log_running_reward += current_ep_reward
        # log_running_episodes += 1

        i_episode += 1

    env.close()


if __name__ == '__main__':
    train()