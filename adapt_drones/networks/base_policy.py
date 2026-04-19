import numpy as np
import torch
import torch.nn as nn

from adapt_drones.utils.learning import layer_init


class EnvironmentalEncoder(nn.Module):
    """
    This class contains the environmental encoder network.
    """

    def __init__(self, input_size, *hidden_layers, output_size=6):
        super(EnvironmentalEncoder, self).__init__()
        layers = []
        current_size = input_size

        for hidden_layer in hidden_layers:
            layers.append(layer_init(nn.Linear(current_size, hidden_layer)))
            layers.append(nn.Tanh())
            current_size = hidden_layer
        layers.append(layer_init(nn.Linear(current_size, output_size)))

        # convert to nn.Sequential
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class TrajectoryEncoder(nn.Module):
    """
    This class contains the trajectory encoder network.
    The feedforward encoder architecture consists of 3 1-D convolution layers with ReLU activations
    that project the reference positions into a 32-dim representation for input to the main policy.
    Each 1-D convolution has 16 filters with a kernel size of 3
    """

    def __init__(self, input_size, output_size=16):
        super(TrajectoryEncoder, self).__init__()
        self.features_per_waypoint = 6
        self.n_filters = 8
        self.conv1 = nn.Conv1d(self.features_per_waypoint, self.n_filters, kernel_size=5, stride=2)
        self.n_filters2 = 4
        self.conv2 = nn.Conv1d(self.n_filters, self.n_filters2, kernel_size=5, stride=2)
        self.attn = nn.Linear(self.n_filters2, 1)
        self.out_proj = nn.Linear(self.n_filters2, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1, self.features_per_waypoint)  # (batch, waypoints, 6)
        x = x.transpose(1, 2)  # (batch, 6, waypoints)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.transpose(1, 2)  # (batch, seq, filters)
        attn_weights = torch.softmax(self.attn(x), dim=1)  # (batch, seq, 1)
        x = (x * attn_weights).sum(dim=1)  # (batch, filters)
        return self.out_proj(x)


class Actor(nn.Module):
    """
    This class contains the actor network
    """

    def __init__(self, input_size, *hidden_layers, output_size=6):
        super(Actor, self).__init__()
        layers = []

        current_size = input_size
        for hidden_layer in hidden_layers:
            layers.append(layer_init(nn.Linear(current_size, hidden_layer)))
            layers.append(nn.Tanh())
            current_size = hidden_layer
        layers.append(layer_init(nn.Linear(current_size, output_size), std=0.01))
        layers.append(nn.Tanh())

        # convert to nn.Sequential
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Critic(nn.Module):
    """
    This class contains the critic netwrk,
    """

    def __init__(self, input_size, *hidden_layers, output_size=1):
        super(Critic, self).__init__()
        layers = []

        current_size = input_size
        for hidden_layer in hidden_layers:
            layers.append(layer_init(nn.Linear(current_size, hidden_layer)))
            layers.append(nn.Tanh())
            current_size = hidden_layer
        layers.append(layer_init(nn.Linear(current_size, output_size), std=1.0))

        # convert to nn.Sequential
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    # Test the network
    input_size = 12
    hidden_layers = [64, 64]
    output_size = 6

    env_encoder = EnvironmentalEncoder(
        input_size, *hidden_layers, output_size=output_size
    )
    print("Environmental Encoder")
    print(env_encoder)
    actor = Actor(input_size, *hidden_layers, output_size=output_size)
    critic = Critic(input_size, *hidden_layers, output_size=1)
    print("Actor")
    print(actor)
    print("Critic")
    print(critic)
