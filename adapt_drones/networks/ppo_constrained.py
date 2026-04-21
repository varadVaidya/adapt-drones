#### Constrained PPO-Lagrangian implementation
#### Based on ppo.py, with cost critic and Lagrange multiplier additions.

import os
import random
import time
from dataclasses import dataclass
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from adapt_drones.networks.agents import *
from adapt_drones.cfgs.config import Config


def ppo_constrained_train(args: Config, envs):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.learning.torch_deterministic

    print(f"Using seed {args.seed}")
    prjt_name = args.experiment.wandb_project_name
    grp_name = args.experiment.grp_name
    run_name = args.experiment.run_name

    layout = {
        "info": {
            "error": ["Multiline", ["error/pos", "error/vel", "error/margin"]],
            "rewards": [
                "Multiline",
                [
                    "rewards/distance",
                    "rewards/velocity",
                    "rewards/yaw",
                    "rewards/control",
                    "rewards/angular_velocity",
                    "rewards/close_distance",
                ],
            ],
            "safety": [
                "Multiline",
                [
                    "costs/episode_cost",
                    "costs/lagrange_multiplier",
                    "costs/cost_value_loss",
                    "costs/cost_policy_loss",
                    "safety/min_obstacle_distance",
                ],
            ],
        },
    }
    writer = SummaryWriter(f"runs/{prjt_name}/{grp_name}/{run_name}/tb")
    writer.add_custom_scalars(layout)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.learning.cuda else "cpu"
    )
    print(f"Using device: {device}")

    if args.agent == "RMA_DATT_Safe":
        agent = RMA_DATT_Safe(
            priv_info_shape=envs.get_attr("priv_info_shape")[0],
            state_shape=envs.get_attr("state_obs_shape")[0],
            traj_shape=envs.get_attr("reference_traj_shape")[0],
            action_shape=envs.single_action_space.shape,
        ).to(device)
    else:
        raise ValueError("Agent not recognized for constrained PPO: %s" % args.agent)

    print(f"Warm start: {args.warm_start}")
    if args.warm_start:
        if args.warm_model is None:
            raise ValueError("Warm start requested but no model provided")
        warm_model_path = (
            f"runs/{prjt_name}/{grp_name}/{args.warm_model}/final_model.pt"
        )
        agent.load_state_dict(torch.load(warm_model_path, weights_only=True))
        agent.traj_encoder.requires_grad_(False)
        agent.env_encoder.requires_grad_(False)
        print(f"Loaded model from {warm_model_path}")

    optimizer = optim.Adam(agent.parameters(), lr=args.learning.init_lr, eps=1e-5)

    num_params = sum(p.numel() for p in agent.parameters())
    print(f"Number of parameters in the model: {num_params:,}")

    # Storage
    obs = torch.zeros(
        (args.learning.num_steps, args.learning.num_envs)
        + envs.single_observation_space.shape
    ).to(device)

    next_obs = torch.zeros(
        (args.learning.num_steps, args.learning.num_envs)
        + envs.single_observation_space.shape
    ).to(device)

    actions = torch.zeros(
        (args.learning.num_steps, args.learning.num_envs)
        + envs.single_action_space.shape
    ).to(device)

    logprobs = torch.zeros((args.learning.num_steps, args.learning.num_envs)).to(device)
    rewards = torch.zeros((args.learning.num_steps, args.learning.num_envs)).to(device)

    next_dones = torch.zeros((args.learning.num_steps, args.learning.num_envs)).to(
        device
    )

    next_terminations = torch.zeros(
        (args.learning.num_steps, args.learning.num_envs)
    ).to(device)

    values = torch.zeros((args.learning.num_steps, args.learning.num_envs)).to(device)

    # Cost storage
    costs = torch.zeros((args.learning.num_steps, args.learning.num_envs)).to(device)
    cost_values = torch.zeros((args.learning.num_steps, args.learning.num_envs)).to(
        device
    )

    # Lagrange multiplier — raw dual gradient descent
    lagrange_lr = 0.01
    cost_limit = args.environment.cost_limit
    log_lagrange = torch.tensor(np.log(0.1), device=device)

    # Start
    global_step = 0
    best_avg_reward = -np.inf
    best_reward = -np.inf
    avg_rewards = deque(maxlen=50)
    for _ in range(len(avg_rewards)):
        avg_rewards.append(float("-inf"))
    start_time = time.time()
    next_ob, _ = envs.reset(seed=args.seed)
    next_ob = torch.Tensor(next_ob).to(device)
    next_done = torch.zeros(args.learning.num_envs).to(device)
    next_termination = torch.zeros(args.learning.num_envs).to(device)

    for iteration in range(1, args.learning.num_iterations + 1):
        if args.learning.anneal_lr:
            frac = 1.0 - (iteration / args.learning.num_iterations)
            lrnow = frac * args.learning.init_lr + (1 - frac) * args.learning.final_lr
            optimizer.param_groups[0]["lr"] = lrnow

        plot_once_iter = True
        for step in range(0, args.learning.num_steps):
            global_step += args.learning.num_envs
            ob = next_ob

            with torch.no_grad():
                action, logprob, _, value, cost_value = agent.get_action_value_cost(ob)

            next_ob, reward, next_termination, next_truncation, info = envs.step(
                action.cpu().numpy()
            )

            # Extract cost from info
            step_cost = torch.zeros(args.learning.num_envs, device=device)
            if "cost" in info:
                step_cost = torch.tensor(
                    info["cost"], dtype=torch.float32, device=device
                )

            real_next_ob = next_ob.copy()
            for idx, trunc in enumerate(next_truncation):
                if trunc:
                    real_next_ob[idx] = info["final_observation"][idx]
            next_ob = torch.Tensor(next_ob).to(device)

            obs[step] = torch.Tensor(ob).to(device)
            next_obs[step] = torch.Tensor(real_next_ob).to(device)
            actions[step] = torch.Tensor(action).to(device)
            logprobs[step] = torch.Tensor(logprob).to(device)
            values[step] = torch.Tensor(value.flatten()).to(device)
            next_terminations[step] = torch.Tensor(next_termination).to(device)
            next_dones[step] = torch.Tensor(
                np.logical_or(next_termination, next_truncation)
            ).to(device)

            rewards[step] = torch.tensor(reward).to(device).view(-1)
            costs[step] = step_cost
            cost_values[step] = cost_value.flatten()

            if "final_info" in info:
                for fi in info["final_info"]:
                    if fi and "episode" in fi:
                        if plot_once_iter:
                            writer.add_scalar(
                                "charts/episodic_return",
                                fi["episode"]["r"],
                                global_step,
                            )
                            avg_rewards.append(fi["episode"]["r"])
                            current_avg_reward = np.mean(
                                np.array(avg_rewards).flatten()
                            )
                            if current_avg_reward > best_avg_reward:
                                best_avg_reward = current_avg_reward
                                if args.learning.save_model:
                                    model_path = f"runs/{prjt_name}/{grp_name}/{run_name}/best_model.pt"
                                    torch.save(agent.state_dict(), model_path)
                            writer.add_scalar(
                                "charts/episodic_length",
                                fi["episode"]["l"],
                                global_step,
                            )
                            pos_error = fi["pos_error"] / fi["episode"]["l"]
                            vel_error = fi["vel_error"] / fi["episode"]["l"]
                            writer.add_scalar("info/pos_error", pos_error, global_step)
                            writer.add_scalar("info/vel_error", vel_error, global_step)
                            writer.add_scalar(
                                "rewards/distance",
                                fi["distance_reward"],
                                global_step,
                            )
                            writer.add_scalar(
                                "rewards/velocity",
                                fi["velocity_reward"],
                                global_step,
                            )
                            writer.add_scalar(
                                "rewards/yaw", fi["yaw_reward"], global_step
                            )
                            writer.add_scalar(
                                "rewards/control",
                                fi["action_reward"],
                                global_step,
                            )
                            writer.add_scalar(
                                "rewards/angular_velocity",
                                fi["angular_velocity_reward"],
                                global_step,
                            )
                            writer.add_scalar(
                                "rewards/close_distance",
                                fi["close_distance_reward"],
                                global_step,
                            )
                            writer.add_scalar(
                                "rewards/closest_distance",
                                fi["closest_distance_reward"],
                                global_step,
                            )
                            if "min_obstacle_distance" in fi:
                                writer.add_scalar(
                                    "safety/min_obstacle_distance",
                                    fi["min_obstacle_distance"],
                                    global_step,
                                )

                            if "obstacle_collision" in fi:
                                writer.add_scalar(
                                    "safety/obstacle_collision",
                                    fi["obstacle_collision"],
                                    global_step,
                                )

                            plot_once_iter = False

        # GAE for rewards
        with torch.no_grad():
            next_values = torch.zeros_like(values[0]).to(device)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.learning.num_steps)):
                if t == args.learning.num_steps - 1:
                    next_values = agent.get_value(next_obs[t]).flatten()
                else:
                    value_mask = next_dones[t].bool()
                    next_values[value_mask] = agent.get_value(
                        next_obs[t][value_mask]
                    ).flatten()
                    next_values[~value_mask] = values[t + 1][~value_mask]
                delta = (
                    rewards[t]
                    + args.learning.gamma * next_values * (1 - next_terminations[t])
                    - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta
                    + args.learning.gamma
                    * args.learning.gae_lambda
                    * (1 - next_dones[t])
                    * lastgaelam
                )
            returns = advantages + values

        # GAE for costs
        with torch.no_grad():
            next_cost_values = torch.zeros_like(cost_values[0]).to(device)
            cost_advantages = torch.zeros_like(costs).to(device)
            cost_lastgaelam = 0
            for t in reversed(range(args.learning.num_steps)):
                if t == args.learning.num_steps - 1:
                    next_cost_values = agent.get_cost_value(next_obs[t]).flatten()
                else:
                    cost_value_mask = next_dones[t].bool()
                    next_cost_values[cost_value_mask] = agent.get_cost_value(
                        next_obs[t][cost_value_mask]
                    ).flatten()
                    next_cost_values[~cost_value_mask] = cost_values[t + 1][
                        ~cost_value_mask
                    ]
                cost_delta = (
                    costs[t]
                    + args.learning.gamma
                    * next_cost_values
                    * (1 - next_terminations[t])
                    - cost_values[t]
                )
                cost_advantages[t] = cost_lastgaelam = (
                    cost_delta
                    + args.learning.gamma
                    * args.learning.gae_lambda
                    * (1 - next_dones[t])
                    * cost_lastgaelam
                )
            cost_returns = cost_advantages + cost_values

        # Update Lagrange multiplier
        episode_cost = costs.mean()
        constraint_error = (episode_cost - cost_limit).item()
        log_lagrange = log_lagrange + lagrange_lr * constraint_error
        lagrange = torch.clamp(log_lagrange.exp(), min=0.0, max=100.0)

        # Flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_cost_advantages = cost_advantages.reshape(-1)
        b_cost_returns = cost_returns.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.learning.batch_size)
        clipfracs = []
        for epoch in range(args.learning.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(
                0, args.learning.batch_size, args.learning.minibatch_size
            ):
                end = start + args.learning.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.learning.clip_coef)
                        .float()
                        .mean()
                        .item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.learning.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss (reward)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.learning.clip_coef, 1 + args.learning.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Cost policy loss
                mb_cost_advantages = b_cost_advantages[mb_inds]
                if args.learning.norm_adv:
                    mb_cost_advantages = (
                        mb_cost_advantages - mb_cost_advantages.mean()
                    ) / (mb_cost_advantages.std() + 1e-8)

                cost_pg_loss1 = mb_cost_advantages * ratio
                cost_pg_loss2 = mb_cost_advantages * torch.clamp(
                    ratio, 1 - args.learning.clip_coef, 1 + args.learning.clip_coef
                )
                cost_pg_loss = torch.max(cost_pg_loss1, cost_pg_loss2).mean()

                # Reward value loss
                newvalue = newvalue.view(-1)
                if args.learning.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.learning.clip_coef,
                        args.learning.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Cost value loss
                new_cost_value = agent.get_cost_value(b_obs[mb_inds]).view(-1)
                cost_v_loss = (
                    0.5 * ((new_cost_value - b_cost_returns[mb_inds]) ** 2).mean()
                )

                entropy_loss = entropy.mean()
                loss = (
                    pg_loss
                    + lagrange.detach() * cost_pg_loss
                    - args.learning.ent_coef * entropy_loss
                    + v_loss * args.learning.vf_coef
                    + cost_v_loss * args.learning.vf_coef
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    agent.parameters(), args.learning.max_grad_norm
                )
                optimizer.step()

            if (
                args.learning.target_kl is not None
                and approx_kl > args.learning.target_kl
            ):
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )
        writer.add_scalar("losses/total_loss", loss.item(), global_step)

        writer.add_scalar("costs/episode_cost", episode_cost.item(), global_step)
        writer.add_scalar("costs/lagrange_multiplier", lagrange.item(), global_step)
        writer.add_scalar("costs/cost_value_loss", cost_v_loss.item(), global_step)
        writer.add_scalar("costs/cost_policy_loss", cost_pg_loss.item(), global_step)
        writer.add_scalar("costs/constraint_error", constraint_error, global_step)

    if args.learning.save_model:
        model_path = f"runs/{prjt_name}/{grp_name}/{run_name}/final_model.pt"
        torch.save(agent.state_dict(), model_path)
        print(f"Model saved at {model_path}")

    envs.close()
    writer.close()
