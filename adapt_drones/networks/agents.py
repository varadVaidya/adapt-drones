import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from adapt_drones.networks.base_policy import (
    EnvironmentalEncoder,
    Actor,
    Critic,
    TrajectoryEncoder,
)
from adapt_drones.utils.learning import layer_init


class SimpleActorCritic(nn.Module):
    def __init__(self, state_shape, action_shape):
        super(SimpleActorCritic, self).__init__()
        base_policy_layers = [64, 64, 64]
        base_policy_input_size = np.prod(state_shape)

        self.critic = Critic(base_policy_input_size, *base_policy_layers, output_size=1)
        self.actor_mean = Actor(
            base_policy_input_size,
            *base_policy_layers,
            output_size=np.prod(action_shape)
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(action_shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


class RMA(nn.Module):
    def __init__(self, priv_info_shape, state_obs_shape, action_shape):
        super(RMA, self).__init__()
        # self.env_encoder_input_size = envs.get_attr("priv_info_shape")[0]
        self.env_encoder_input_size = priv_info_shape
        # self.state_obs_shape = envs.get_attr("state_obs_shape")[0]
        self.state_obs_shape = state_obs_shape
        env_encoder_output_size = 6
        env_encoder_layers = [64, 64]

        # both actor and critic share the same base policy architecture
        base_policy_input_size = env_encoder_output_size + self.state_obs_shape
        base_policy_layers = [64, 64, 64]

        actor_input = env_encoder_output_size + self.state_obs_shape
        actor_output = np.prod(action_shape)
        critic_input = env_encoder_output_size + self.state_obs_shape

        self.env_encoder = EnvironmentalEncoder(
            self.env_encoder_input_size,
            *env_encoder_layers,
            output_size=env_encoder_output_size
        )

        self.critic = Critic(critic_input, *base_policy_layers, output_size=1)
        self.actor_mean = Actor(
            actor_input, *base_policy_layers, output_size=actor_output
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(actor_output)))

    def get_value(self, x):
        # x = observation.
        state_obs = x[:, : self.state_obs_shape]
        env_obs = x[:, self.state_obs_shape :]

        env_encoder = self.env_encoder(env_obs)
        x = torch.cat((state_obs, env_encoder), dim=-1)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        # x = observation.
        state_obs = x[:, : self.state_obs_shape]
        env_obs = x[:, self.state_obs_shape :]

        env_encoder = self.env_encoder(env_obs)
        x = torch.cat((state_obs, env_encoder), dim=-1)

        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


class RMA_DATT(nn.Module):
    """
    RMA class that is supposed to be used with the trajv_2 environment
    """

    def __init__(
        self,
        priv_info_shape,
        state_shape,
        traj_shape,
        action_shape,
    ):
        super(RMA_DATT, self).__init__()
        # self.env_encoder_input_size = envs.get_attr("priv_info_shape")[0]
        self.priv_info_shape = priv_info_shape
        # self.state_obs_shape = envs.get_attr("state_obs_shape")[0]
        self.state_obs_shape = state_shape
        self.traj_encoder_input_size = traj_shape

        env_encoder_output_size = 8
        traj_encoder_output_size = 32
        env_encoder_layers = [64, 64]

        # both actor and critic share the same base policy architecture
        base_policy_input_size = (
            env_encoder_output_size + self.state_obs_shape + traj_encoder_output_size
        )
        base_policy_layers = [64, 64, 64]

        actor_input = base_policy_input_size
        actor_output = np.prod(action_shape)
        critic_input = base_policy_input_size

        self.env_encoder = EnvironmentalEncoder(
            self.priv_info_shape,
            *env_encoder_layers,
            output_size=env_encoder_output_size
        )

        self.traj_encoder = TrajectoryEncoder(
            input_size=self.traj_encoder_input_size,
            output_size=traj_encoder_output_size,
        )

        self.critic = Critic(critic_input, *base_policy_layers, output_size=1)
        self.actor_mean = Actor(
            actor_input, *base_policy_layers, output_size=actor_output
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(actor_output)))

    def get_value(self, x):
        # x = observation.

        env_obs = x[:, : self.priv_info_shape]
        state_obs = x[
            :, self.priv_info_shape : self.priv_info_shape + self.state_obs_shape
        ]

        traj_obs = x[:, self.priv_info_shape + self.state_obs_shape :]

        env_encoder = self.env_encoder(env_obs)
        traj_encoder = self.traj_encoder(traj_obs)

        x = torch.cat((state_obs, env_encoder, traj_encoder), dim=-1)
        return self.critic(x)

    def get_action_and_value(
        self,
        x,
        action=None,
        predicited_enc=None,
    ):
        # x = observation.
        env_obs = x[:, : self.priv_info_shape]
        state_obs = x[
            :, self.priv_info_shape : self.priv_info_shape + self.state_obs_shape
        ]

        traj_obs = x[:, self.priv_info_shape + self.state_obs_shape :]
        env_encoder = (
            self.env_encoder(env_obs) if predicited_enc is None else predicited_enc
        )
        traj_encoder = self.traj_encoder(traj_obs)

        x = torch.cat((state_obs, env_encoder, traj_encoder), dim=-1)

        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )

    def forward(self, x, predicited_enc):
        """.
        Method to call the forward pass of the model, during final evaluation

        """
        # x = observation.
        env_obs = x[:, : self.priv_info_shape]
        state_obs = x[
            :, self.priv_info_shape : self.priv_info_shape + self.state_obs_shape
        ]

        traj_obs = x[:, self.priv_info_shape + self.state_obs_shape :]
        env_encoder = predicited_enc
        traj_encoder = self.traj_encoder(traj_obs)

        x = torch.cat((state_obs, env_encoder, traj_encoder), dim=-1)

        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)

        probs = Normal(action_mean, action_std)

        return probs.sample()
