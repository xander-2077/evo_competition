import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append('/root/ws')
import competevo
import gym_compete
import hydra
from datetime import datetime
from omegaconf import OmegaConf
from pprint import pprint


class FirstItemWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env.observation_space = env.observation_space[0]
        self.env.action_space = env.action_space[0]
    
    def step(self, action, opponent_action):
        action = (action, opponent_action)
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation[0], reward[0], any(terminated), truncated, info[0]
    
    def reset(self):
        observation, _ = self.env.reset()
        return observation[0], _


def make_env(env_id, idx, capture_video, run_name, gamma, render_mode=None):
    def thunk():
        env = gym.make(env_id, cfg={}, render_mode=render_mode)
            
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
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

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


@hydra.main(config_path="../../cfg", config_name="config", version_base="1.3")
def main(args):
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    print("batch_size: ", args.batch_size)
    print("minibatch_size: ", args.minibatch_size)
    print("num_iterations: ", args.num_iterations)
    
    import pdb; pdb.set_trace()

    run_name = f"{datetime.now().strftime('%m-%d_%H-%M')}_{args.exp_name}_{args.seed}"

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
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma, render_mode=args.render_mode) for i in range(args.num_envs)]
    )

    agent0 = Agent(envs).to(device)
    optimizer0 = optim.Adam(agent0.parameters(), lr=args.learning_rate, eps=1e-5)
    agent1 = Agent(envs).to(device)
    optimizer1 = optim.Adam(agent1.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage for opponent parameters
    opponent_parameters = []

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    opponent_obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    opponent_actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    opponent_logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    opponent_rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    opponent_dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    opponent_values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    alpha = 1.0
    start_time = time.time()
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer0.param_groups[0]["lr"] = lrnow
            optimizer1.param_groups[0]["lr"] = lrnow
        
        victories = 0
        defeats = 0
        episode_in_iteration = 0
        alpha = 1.0-iteration/args.iteration_alpha_anneal if iteration < args.iteration_alpha_anneal else 0.0
        writer.add_scalar("charts/alpha", alpha, global_step)
        
        # Sample opponent parameters
        if opponent_parameters:
            delta = random.uniform(0, 1)
            sampled_opponent_idx = int((1 - delta) * len(opponent_parameters))
            agent1.load_state_dict(opponent_parameters[sampled_opponent_idx])

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent0.get_action_and_value(next_obs)
                opponent_action, opponent_logprob, _, opponent_value = agent1.get_action_and_value(next_obs)
                values[step] = value.flatten()
                opponent_values[step] = opponent_value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            opponent_actions[step] = opponent_action
            opponent_logprobs[step] = opponent_logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy(), opponent_action.cpu().numpy())
            # Curriculum learning
            reward = alpha * infos["reward_dense"] + (1-alpha) * infos["reward_parse"]
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            opponent_rewards[step] = torch.tensor([info["reward_dense"] for info in infos[1]]).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

                        episode_in_iteration += 1
                        if info['win_reward'] > 0: victories += 1
                        if info['lose_penalty'] < 0: defeats += 1

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent0.get_value(next_obs).reshape(1, -1)
            next_opponent_value = agent1.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            opponent_advantages = torch.zeros_like(opponent_rewards).to(device)
            lastgaelam = 0
            last_opponent_gaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                    next_opponent_values = next_opponent_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                    next_opponent_values = opponent_values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                opponent_delta = opponent_rewards[t] + args.gamma * next_opponent_values * nextnonterminal - opponent_values[t]
                opponent_advantages[t] = last_opponent_gaelam = opponent_delta + args.gamma * args.gae_lambda * nextnonterminal * last_opponent_gaelam
            returns = advantages + values
            opponent_returns = opponent_advantages + opponent_values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_opponent_obs = opponent_obs.reshape((-1,) + envs.single_observation_space.shape)
        b_opponent_logprobs = opponent_logprobs.reshape(-1)
        b_opponent_actions = opponent_actions.reshape((-1,) + envs.single_action_space.shape)
        b_opponent_advantages = opponent_advantages.reshape(-1)
        b_opponent_returns = opponent_returns.reshape(-1)
        b_opponent_values = opponent_values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent0.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

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

                optimizer0.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent0.parameters(), args.max_grad_norm)
                optimizer0.step()

                _, opponent_newlogprob, opponent_entropy, opponent_newvalue = agent1.get_action_and_value(b_opponent_obs[mb_inds], b_opponent_actions[mb_inds])
                opponent_logratio = opponent_newlogprob - b_opponent_logprobs[mb_inds]
                opponent_ratio = opponent_logratio.exp()

                with torch.no_grad():
                    opponent_old_approx_kl = (-opponent_logratio).mean()
                    opponent_approx_kl = ((opponent_ratio - 1) - opponent_logratio).mean()
                    clipfracs += [((opponent_ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_opponent_advantages = b_opponent_advantages[mb_inds]
                if args.norm_adv:
                    mb_opponent_advantages = (mb_opponent_advantages - mb_opponent_advantages.mean()) / (mb_opponent_advantages.std() + 1e-8)

                opponent_pg_loss1 = -mb_opponent_advantages * opponent_ratio
                opponent_pg_loss2 = -mb_opponent_advantages * torch.clamp(opponent_ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                opponent_pg_loss = torch.max(opponent_pg_loss1, opponent_pg_loss2).mean()

                opponent_newvalue = opponent_newvalue.view(-1)
                if args.clip_vloss:
                    opponent_v_loss_unclipped = (opponent_newvalue - b_opponent_returns[mb_inds]) ** 2
                    opponent_v_clipped = b_opponent_values[mb_inds] + torch.clamp(
                        opponent_newvalue - b_opponent_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    opponent_v_loss_clipped = (opponent_v_clipped - b_opponent_returns[mb_inds]) ** 2
                    opponent_v_loss_max = torch.max(opponent_v_loss_unclipped, opponent_v_loss_clipped)
                    opponent_v_loss = 0.5 * opponent_v_loss_max.mean()
                else:
                    opponent_v_loss = 0.5 * ((opponent_newvalue - b_opponent_returns[mb_inds]) ** 2).mean()

                opponent_entropy_loss = opponent_entropy.mean()
                opponent_loss = opponent_pg_loss - args.ent_coef * opponent_entropy_loss + opponent_v_loss * args.vf_coef

                optimizer1.zero_grad()
                opponent_loss.backward()
                nn.utils.clip_grad_norm_(agent1.parameters(), args.max_grad_norm)
                optimizer1.step()

        # Save the current parameters of the opponent
        opponent_parameters.append(agent1.state_dict())

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        if len(opponent_parameters) > args.max_opponent_versions:
            opponent_parameters.pop(0)

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
