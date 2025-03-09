import os
import numpy as np
import gymnasium as gym
import torch

from dataclasses import dataclass


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def make_env(env_id, cfg, trajectory_dataset=None):
    def thunk():
        if trajectory_dataset is not None:
            env = gym.make(env_id, cfg=cfg, trajectory_dataset=trajectory_dataset)
        else:
            env = gym.make(env_id, cfg=cfg)
        env = gym.wrappers.FlattenObservation(
            env
        )  # think if to flatten or not later. mostly the answer is no.
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.NormalizeReward(env)
        return env

    return thunk
