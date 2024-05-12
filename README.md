# Evo Competition

## File Architecture

`train/` folder is some scipts for training algorithms. 
Now `PPO-Pytorch/` is available. `StableBaselines3/` is useless.
What we are promoting is `CleanRL/`.
`test/` folder is some scipts for test.

For `PPO-Pytorch/`, you can just run 

```python
python train/PPO_train.py
# or enable the render mode
python train/PPO_train.py render_mode=human 
```

to train PPO algorithm. You'll find `/runs/env_name/$TIME$_$algoname$` folder which contains `config.yaml`, `events...` and `ppoagent.pth`. There are for config record, tensorboard and model saving, respectively.

We use `hydra` to manage configs. Ref to https://hydra.cc/docs/intro/. `cfg/config.yaml` is the main config file.


## Environment INFO

- Action space

    Tuple(Box(-1.0, 1.0, (8,), float32), Box(-1.0, 1.0, (8,), float32))

- Observation space

    Tuple(Box(-inf, inf, (118,), float32), Box(-inf, inf, (118,), float32))

- Reward

    Tuple(float32, float32)

- Termination

    Tuple(bool, bool)

- Truncated

    bool

- Info

    Tuple({'ctrl_reward': float32, 'alive_reward': float32, 'lose_penalty': float32, 'win_reward': float32, 'reward_parse': float32, 'move_to_opp_reward': float32, 'push_opp_reward': float32, 'reward_dense': float32}, {'ctrl_reward': float32, 'alive_reward': float32, 'lose_penalty': float32, 'win_reward': float32, 'reward_parse': float32, 'move_to_opp_reward': float32, 'push_opp_reward': float32, 'reward_dense': float32})

    reward_dense = alive_reward + ctrl_reward + push_opp_reward + move_to_opp_reward

    reward_parse = win_reward + lose_penalty

    reward = reward_dense + reward_parse

## Trouble Shooting

If you encounter `TypeError: reset() got an unexpected keyword argument 'seed'`, try to fix codes in `gymnasium/core.py", line 462`:

```python
# obs, info = self.env.reset(seed=seed, options=options)
obs, info = self.env.reset()
```



