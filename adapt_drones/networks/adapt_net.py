import os
import random
import datetime
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym


from adapt_drones.utils.learning import layer_init, make_env
from adapt_drones.networks.agents import *
from adapt_drones.cfgs.config import Config


class AdaptationNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 4, kernel_size=8, stride=4)
        self.conv2 = nn.Conv1d(4, 4, kernel_size=5, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(4)
        self.linear = nn.Linear(4 * 4, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)


def adapt_train_datt_rma(cfg: Config, envs, best_model: bool = True):
    print("RMA Adaptation Training")

    num_envs = cfg.learning.num_envs
    num_iterations = cfg.learning.total_timesteps // num_envs

    # seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.learning.torch_deterministic

    # loading the model from the run of the cfg file given
    run_folder = (
        "runs/"
        + cfg.experiment.wandb_project_name
        + "/"
        + cfg.grp_name
        + "/"
        + cfg.run_name
        + "/"
    )
    results_folder = run_folder + "results/"
    datadump_folder = results_folder + "datadump/"
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(datadump_folder, exist_ok=True)

    model_path = (
        run_folder + "best_model.pt" if best_model else run_folder + "final_model.pt"
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.learning.cuda else "cpu"
    )
    print("Training on Model: ", model_path)

    priv_info_shape = envs.get_attr("priv_info_shape")[0]
    state_shape = envs.get_attr("state_obs_shape")[0]
    traj_shape = envs.get_attr("reference_traj_shape")[0]
    action_shape = envs.single_action_space.shape[0]

    # Use only core drone state (12D) for adaptation, excluding obstacle obs
    core_state_shape = 12
    state_action_shape = core_state_shape + action_shape
    time_horizon = cfg.network.adapt_time_horizon

    adapt_input = time_horizon * state_action_shape
    adapt_output = cfg.network.env_encoder_output

    state_action_buffer = torch.zeros((num_envs, state_action_shape, time_horizon)).to(
        device
    )

    if cfg.agent not in ("RMA_DATT", "RMA_DATT_Safe"):
        raise ValueError("Agent not valid for this evaluation")

    agent_cls = RMA_DATT_Safe if cfg.agent == "RMA_DATT_Safe" else RMA_DATT
    agent = agent_cls(
        priv_info_shape=priv_info_shape,
        state_shape=state_shape,
        traj_shape=traj_shape,
        action_shape=action_shape,
    ).to(device)

    agent.load_state_dict(torch.load(model_path, weights_only=True))
    agent.eval()

    # init the adaptation network
    adapt_net = AdaptationNetwork(adapt_input, adapt_output).to(device)
    adapt_optim = torch.optim.Adam(
        adapt_net.parameters(), lr=cfg.learning.init_lr, eps=1e-5
    )

    global_step = 0
    next_ob, _ = envs.reset(seed=cfg.seed)
    next_ob = torch.tensor(next_ob).to(device)

    start_time = datetime.datetime.now()

    for itr in range(num_iterations):
        global_step += num_envs
        ob = next_ob

        env_ob = next_ob[:, :priv_info_shape]
        traj_obs = next_ob[:, priv_info_shape : priv_info_shape + traj_shape]
        state_ob = next_ob[:, priv_info_shape + traj_shape :]
        with torch.no_grad():
            action = agent.get_action_and_value(ob)[0]

        core_state_ob = state_ob[:, :core_state_shape]
        state_action = torch.cat((core_state_ob, action), dim=-1)
        state_action_buffer = torch.cat(
            (state_action.unsqueeze(-1), state_action_buffer[:, :, :-1].clone()), dim=-1
        )
        next_ob, reward, next_termination, next_truncation, info = envs.step(
            action.cpu().numpy()
        )
        # Correct next obervation (for vec gym)
        real_next_ob = next_ob.copy()
        for idx, trunc in enumerate(next_truncation):
            if trunc:
                real_next_ob[idx] = info["final_observation"][idx]
        next_ob = torch.Tensor(real_next_ob).to(device)

        predicted_encoder = adapt_net(state_action_buffer.view(num_envs, -1))
        encoder_output = agent.env_encoder(env_ob)

        adapt_optim.zero_grad()
        loss = nn.MSELoss()(predicted_encoder, encoder_output)
        loss.backward()
        adapt_optim.step()

        step_time = datetime.datetime.now()

        if itr % 10_000 == 0:
            print(
                f"Gloabl step: {global_step}, Loss: {loss.item()}, Time: {(step_time - start_time).seconds}"
            )

    torch.save(adapt_net.state_dict(), run_folder + "/adapt_network.pt")
