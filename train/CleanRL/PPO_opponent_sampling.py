import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append('/root/ws')
import competevo
import gym_compete
import hydra
from pprint import pprint
from config.config import Config
import argparse
from utils.tools import str2bool


class FirstItemWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env.observation_space = env.observation_space[0]
        self.env.action_space = gym.spaces.Box(-1.0, 1.0, (16,), np.float32)
    
    def step(self, actions):
        num_actions = actions.shape[-1] // 2
        action_self = actions[:num_actions]
        action_opponent = actions[num_actions:]
        action = (action_self, action_opponent)
        observation, reward, terminated, truncated, info = self.env.step(action)
        info[0]['opponent_observation'] = observation[1]
        return observation[0], reward[0], any(terminated), truncated, info[0]
    
    def reset(self):
        observation, info = self.env.reset()
        info['opponent_observation'] = observation[1]
        return observation[0], info


def make_env(env_id, cfg, gamma, render_mode=None):
    def thunk():
        env = gym.make(env_id, cfg=cfg, render_mode=render_mode)
        env = FirstItemWrapper(env)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(gym.spaces.Box(-1.0, 1.0, (8,), np.float32).shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(gym.spaces.Box(-1.0, 1.0, (8,), np.float32).shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


@hydra.main(config_path="../../cfg", config_name="PPO_opponent_sampling_cleanrl", version_base="1.3")
def main(args):
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    args.iteration_alpha_anneal = int(args.num_iterations * 0.15)
    if args.target_kl == 'None': args.target_kl = None

    print("batch_size: ", args.batch_size)
    print("minibatch_size: ", args.minibatch_size)
    print("num_iterations: ", args.num_iterations)
    print("iteration_alpha_anneal: ", args.iteration_alpha_anneal)
    
    input()  # Press enter to continue!

    writer = SummaryWriter(log_dir='.')
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    parser = argparse.ArgumentParser(description="User's arguments from terminal.")
    parser.add_argument("--cfg", 
                        dest="cfg_file", 
                        help="Config file", 
                        default='/root/ws/config/robo-sumo-ants-v0.yaml',
                        type=str)
    parser.add_argument('--use_cuda', type=str2bool, default=True)
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--num_threads', type=int, default=72)
    parser.add_argument('--epoch', type=str, default='0')
    argument = parser.parse_args()
    cfg = Config(argument.cfg_file)

    envs0 = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, cfg, args.gamma, render_mode=args.render_mode) for _ in range(args.num_envs)]
    )
    envs1 = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, cfg, args.gamma, render_mode=args.render_mode) for _ in range(args.num_envs)]
    )
    envses = [envs0, envs1]

    # Agent
    agent0 = Agent(envses[0]).to(device)
    optimizer0 = optim.Adam(agent0.parameters(), lr=args.learning_rate, eps=1e-5)
    agent1 = Agent(envses[1]).to(device)
    optimizer1 = optim.Adam(agent1.parameters(), lr=args.learning_rate, eps=1e-5)

    agent_opponent = Agent(envses[1]).to(device)

    agents = [agent0, agent1]
    optimizers = [optimizer0, optimizer1]

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envses[0].single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + gym.spaces.Box(-1.0, 1.0, (8,), np.float32).shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    obs_buffer = [obs, obs.clone()]
    actions_buffer = [actions, actions.clone()]
    logprobs_buffer = [logprobs, logprobs.clone()]
    rewards_buffer = [rewards, rewards.clone()]
    dones_buffer = [dones, dones.clone()]
    values_buffer = [values, values.clone()]

    model_list = []

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    alpha = 1.0
    start_time = time.time()

    for iteration in range(1, args.num_iterations + 1): 
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            for idx in range(2):
                optimizers[idx].param_groups[0]["lr"] = lrnow
        
        alpha = 1.0 - iteration / args.iteration_alpha_anneal if iteration < args.iteration_alpha_anneal else 0.0
        writer.add_scalar("charts/alpha", alpha, global_step)
        
        for idx in range(2):
            # TODO: reset condition
            if iteration == 1:
                next_obs, _ = envses[idx].reset()
                next_obs = torch.Tensor(next_obs).to(device)
                next_done = torch.zeros(args.num_envs).to(device)

            victories = 0
            defeats = 0
            episode_in_iteration = 0

            for step in range(0, args.num_steps):
                global_step += args.num_envs
                obs_buffer[idx][step] = next_obs
                dones_buffer[idx][step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agents[idx].get_action_and_value(next_obs)
                    values_buffer[idx][step] = value.flatten()
                actions_buffer[idx][step] = action
                logprobs_buffer[idx][step] = logprob

                if len(model_list) == 0:
                    action_opponent = np.zeros_like(action.cpu().numpy())
                else:
                    start_index = int(len(model_list) * args.delta)
                    sampled_model_idx = random.choice(model_list[start_index:])
                    agent_opponent.load_state_dict(torch.load(f"agent{1-idx}/iter_{sampled_model_idx}.pth"))
                    agent_opponent.eval()
                    with torch.no_grad():
                        action_opponent, _, _, _ = agent_opponent.get_action_and_value(opponent_obs)
                    action_opponent = action_opponent.cpu().numpy()


                # TRY NOT TO MODIFY: execute the game and log data.
                action = np.concatenate((action.cpu().numpy(), action_opponent), axis=-1)
                next_obs, reward, terminations, truncations, infos = envses[idx].step(action)
                opponent_obs = torch.tensor(np.stack(infos['opponent_observation']), dtype=torch.float32).to(device)
                reward = alpha * infos["reward_dense"] + (1-alpha) * infos["reward_parse"]  # Curriculum learning
                next_done = np.logical_or(terminations, truncations)
                rewards_buffer[idx][step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            print(f"global_step={global_step}, agent{idx} episodic_return={info['episode']['r']}")
                            writer.add_scalar(f"agent{idx}/episodic_return", info["episode"]["r"], global_step)
                            writer.add_scalar(f"agent{idx}/episodic_length", info["episode"]["l"], global_step)

                            episode_in_iteration += 1
                            if info['win_reward'] > 0: 
                                victories += 1
                            if info['lose_penalty'] < 0: 
                                defeats += 1

            if episode_in_iteration > 0:
                writer.add_scalar(f"agent{idx}/success_rate", victories/episode_in_iteration, iteration)
                writer.add_scalar(f"agent{idx}/loss_rate", defeats/episode_in_iteration, iteration)
                writer.add_scalar(f"agent{idx}/num_episode_per_iteration", episode_in_iteration, iteration)

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agents[idx].get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards_buffer[idx]).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones_buffer[idx][t + 1]
                        nextvalues = values_buffer[idx][t + 1]
                    delta = rewards_buffer[idx][t] + args.gamma * nextvalues * nextnonterminal - values_buffer[idx][t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values_buffer[idx]

            # flatten the batch
            b_obs = obs_buffer[idx].reshape((-1,) + envses[idx].single_observation_space.shape)
            b_logprobs = logprobs_buffer[idx].reshape(-1)
            b_actions = actions_buffer[idx].reshape((-1,) + envses[idx].single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values_buffer[idx].reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agents[idx].get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizers[idx].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agents[idx].parameters(), args.max_grad_norm)
                    optimizers[idx].step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            if episode_in_iteration > 0:
                writer.add_scalar(f"agent{idx}/success_rate", victories/episode_in_iteration, iteration)
                writer.add_scalar(f"agent{idx}/loss_rate", defeats/episode_in_iteration, iteration)
                writer.add_scalar(f"agent{idx}/num_episode_per_iteration", episode_in_iteration, iteration)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", optimizers[idx].param_groups[0]["lr"], global_step)
            writer.add_scalar(f"agent{idx}/value_loss", v_loss.item(), global_step)
            writer.add_scalar(f"agent{idx}/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar(f"agent{idx}/entropy", entropy_loss.item(), global_step)
            writer.add_scalar(f"agent{idx}/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar(f"agent{idx}/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar(f"agent{idx}/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar(f"agent{idx}/explained_variance", explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            if args.save_model and iteration % args.save_model_interval == 0:
                model_path = f"agent{idx}/iter_{iteration}.pth"
                if idx == 1 : model_list.append(iteration)
                torch.save(agents[idx].state_dict(), model_path)
                print(f"model saved to {model_path}")

    writer.close()


if __name__ == "__main__":
    main()