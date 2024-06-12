from typing import Callable

import gymnasium as gym
import torch


def evaluate(
    model_path0: str,
    model_path1: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = True,
    gamma: float = 0.99,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, capture_video, run_name, gamma)])
    agent = Model(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
        next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
        obs = next_obs

    return episodic_returns


if __name__ == "__main__":
    import sys
    sys.path.append('/root/ws/')
    from train.CleanRL.PPO_opponent_sampling import Agent, make_env

    model_path0 = '/root/ws/agent_dl/agent0_iter_1860.pth'
    model_path1 = '/root/ws/agent_dl/agent1_iter_1860.pth'

    # evaluate(
    #     model_path0,
    #     model_path1,
    #     make_env,
    #     "robo-sumo-ants-v0",
    #     eval_episodes=10,
    #     run_name=f"eval",
    #     Model=Agent,
    #     device="cpu",
    #     capture_video=False,
    # )

    

