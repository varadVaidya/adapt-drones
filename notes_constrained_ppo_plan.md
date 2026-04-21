# Constrained PPO-Lagrangian Implementation Plan

Detailed implementation plan for adding obstacle avoidance to the dynamics-invariant quadrotor controller. This adds a new environment with obstacles, a constrained PPO training loop with Lagrangian cost, and corresponding train/eval scripts.

All new files live alongside existing ones. The existing codebase is untouched — no modifications to existing files except `__init__.py` registrations and config imports.

---

## 1. New Files Overview

```
adapt_drones/
├── __init__.py                          # ADD: register "traj_obs_v3"
├── envs/
│   ├── __init__.py                      # ADD: import TrajObstacleAviaryv3
│   ├── TrajObstacleAviaryv3.py          # NEW: env with obstacles, inherits TrajAviaryv3
│   └── assets/
│       └── quad_obstacles.xml           # NEW: MuJoCo XML with obstacle geoms
├── cfgs/
│   ├── config.py                        # ADD: TrajObstacleAviaryv3Config to env_maps
│   └── environment_cfg.py               # ADD: TrajObstacleAviaryv3Config dataclass
├── networks/
│   ├── ppo_constrained.py               # NEW: constrained PPO with Lagrangian
│   └── agents.py                        # ADD: RMA_DATT_Safe agent with cost critic
├── train_obs.py                         # NEW: training script
├── eval_obs.py                          # NEW: evaluation script
└── utils/
    └── eval_obs.py                      # NEW: eval functions for obstacle env
```

---

## 2. Environment: TrajObstacleAviaryv3

### 2.1 Inheritance strategy

```
BaseAviary (unchanged)
  └── TrajAviaryv3 (unchanged)
        └── TrajObstacleAviaryv3 (NEW)
```

Inherit from `TrajAviaryv3` directly. No `BaseObstacleAviary` needed — obstacles are specific to the trajectory tracking task (you need a trajectory to fly through an obstacle field). If we later need hover+obstacles, we can refactor then.

### 2.2 MuJoCo XML: `quad_obstacles.xml`

Copy `quad.xml` and add obstacle bodies. Obstacles are **static spheres** added to `<worldbody>`:

```xml
<!-- Template: obstacles are repositioned programmatically via model.body_pos -->
<body name="obstacle_0" pos="1 0 1">
  <geom name="obs_geom_0" type="sphere" size="0.15" rgba="0.8 0.2 0.2 0.5"
        contype="1" conaffinity="1"/>
</body>
<body name="obstacle_1" pos="-1 0.5 1.5">
  <geom name="obs_geom_1" type="sphere" size="0.15" rgba="0.8 0.2 0.2 0.5"
        contype="1" conaffinity="1"/>
</body>
<!-- ... up to N_OBSTACLES (e.g., 5) -->
```

Key points:
- `contype`/`conaffinity` set so MuJoCo detects collisions between drone and obstacles
- Obstacle positions are **overwritten programmatically** each reset — the XML values are just placeholders
- Obstacle radii can also be varied per-reset
- Semi-transparent so they're visible but don't occlude the drone in recordings

### 2.3 TrajObstacleAviaryv3 class

```python
class TrajObstacleAviaryv3(TrajAviaryv3):
```

**`__init__` changes:**
- Load `quad_obstacles.xml` instead of `quad.xml`. Override the XML path before calling `super().__init__()` or set it after. Since `BaseAviary.__init__` loads the XML directly from a hardcoded path, the cleanest approach: override the model loading in `__init__` after calling super, or pass the XML path as a parameter.
- Practical approach: call `super().__init__()` then reload the model from the new XML. OR: monkey-patch `pkg_resources.resource_filename` temporarily. OR: add an `xml_file` parameter to `BaseAviary` (preferred — small, clean change to BaseAviary.__init__).
- **Recommended**: Add an optional `xml_file=None` kwarg to `BaseAviary.__init__`. If None, use the default `quad.xml`. `TrajObstacleAviaryv3` passes `quad_obstacles.xml`. This is a 2-line change to BaseAviary.
- Store obstacle body IDs: `self.obstacle_ids = [self.data.body(f"obstacle_{i}").id for i in range(N_OBS)]`
- Store obstacle geom IDs for collision detection
- Set `self.n_obstacles` and `self.obstacle_radius_range = [0.1, 0.3]`

**Observation space changes — `_observation_space()`:**

Add obstacle relative positions to the observation dict:

```python
def _observation_space(self):
    obs_space = super()._observation_space()

    # K nearest obstacles: relative position (3) per obstacle
    n_obs_in_obs = self.n_obstacles  # or a fixed K
    obstacle_box = spaces.Box(
        low=-np.inf * np.ones(n_obs_in_obs * 3),
        high=np.inf * np.ones(n_obs_in_obs * 3),
        dtype=np.float32,
    )
    self.obstacle_obs_shape = n_obs_in_obs * 3

    obs_space.spaces["obstacles"] = obstacle_box
    return obs_space
```

After `FlattenObservation`, the full flattened observation becomes:
`[priv_info(10), obstacles(15), trajectory(600), state(12)]` = 637 dims

**Important**: The obstacle observation must be placed consistently in the flattened vector. `FlattenObservation` sorts dict keys alphabetically. So the order will be: `obstacles`, `priv_info`, `state`, `trajectory`. Keep this in mind when slicing in the agent.

Actually, to avoid confusion with the existing slicing logic in `RMA_DATT` (which assumes `priv_info` comes first), the cleanest approach is to **append obstacle info to the state observation** rather than adding a new dict key:

```python
def _compute_obs(self):
    obs = super()._compute_obs()
    obstacle_obs = self._get_obstacle_obs()  # (n_obstacles * 3,)
    obs["state"] = np.concatenate([obs["state"], obstacle_obs])
    return obs
```

This way:
- `priv_info` stays at index 0:10 (unchanged)
- `state` becomes 12 + 15 = 27 dims (was 12)
- `trajectory` stays at 600 dims
- The existing agent slicing logic (`priv_info_shape`, `state_obs_shape`) still works because the env reports the updated shapes
- The new agent just has a larger state input — no architectural headache

**`_get_obstacle_obs()` method:**

```python
def _get_obstacle_obs(self):
    """Returns relative positions of obstacles w.r.t. drone, sorted by distance."""
    rel_positions = []
    for obs_id in self.obstacle_ids:
        obs_pos = self.model.body_pos[obs_id]  # obstacle world position
        rel_pos = obs_pos - self.position
        rel_positions.append(rel_pos)

    # Sort by distance (closest first)
    rel_positions = sorted(rel_positions, key=lambda p: np.linalg.norm(p))
    return np.concatenate(rel_positions).astype(np.float32)
```

**`reset()` — obstacle randomization:**

Override `reset()` to randomize obstacle positions along the trajectory corridor:

```python
def reset(self, seed=None, options=None):
    obs, info = super().reset(seed=seed, options=options)

    # Place obstacles near the trajectory but not on the start/end
    traj_positions = self.reference_trajectory[:, 1:4]  # (T, 3)
    for i, obs_id in enumerate(self.obstacle_ids):
        # Pick a random point along the trajectory
        t_idx = self.np_random.integers(20, len(traj_positions) - 20)
        base_pos = traj_positions[t_idx]

        # Offset perpendicular to trajectory direction
        offset = self.np_random.uniform(-0.5, 0.5, 3)
        self.model.body_pos[obs_id] = base_pos + offset

        # Randomize radius
        radius = self.np_random.uniform(*self.obstacle_radius_range)
        geom_id = self.obstacle_geom_ids[i]
        self.model.geom_size[geom_id][0] = radius

    mujoco.mj_forward(self.model, self.data)

    # Recompute obs with obstacle info
    obs = self._compute_obs()
    obs = self._flatten_obs(obs)  # or let the wrapper handle it
    return obs, info
```

**Cost signal — `_compute_info()`:**

Override `_compute_info()` to include a cost signal:

```python
def _compute_info(self):
    info = super()._compute_info()

    # Cost: distance-based (1.0 if within danger radius, 0.0 if safe)
    min_distance = self._min_obstacle_distance()
    cost = 1.0 if min_distance < self.danger_radius else 0.0
    info["cost"] = cost

    # Also store continuous distance for logging
    info["min_obstacle_distance"] = min_distance

    return info
```

**`_compute_truncated()` — add obstacle collision:**

```python
def _compute_truncated(self):
    base_truncated = super()._compute_truncated()
    obstacle_collision = self._check_obstacle_collision()
    return base_truncated or obstacle_collision
```

Where `_check_obstacle_collision()` checks `self.data.contact` for contacts between drone geoms and obstacle geoms.

**`_compute_reward()` — keep as-is:**

Don't add obstacle penalty to the reward. The reward stays purely about trajectory tracking. Safety is handled by the separate cost signal through the Lagrangian. This separation is the whole point.

### 2.4 Config: TrajObstacleAviaryv3Config

Add to `environment_cfg.py`:

```python
@dataclass
class TrajObstacleAviaryv3Config:
    eval: bool
    scale: bool
    scale_lengths: list
    pos_xy: list
    pos_z: float
    linear_vel: list
    angular_vel: list
    roll_pitch: list
    wind_speed: list
    max_wind: float

    env_id: str = "traj_obs_v3"
    episode_length: int = 6
    agent_name: tuple = ("RMA_DATT_Safe",)
    wind_bool: bool = True
    trajectory_window: int = 100

    # Obstacle settings
    n_obstacles: int = 5
    obstacle_radius_range: list = None  # [0.1, 0.3]
    danger_radius: float = 0.15
    safe_radius: float = 0.5

    # Lagrangian settings
    cost_limit: float = 0.0  # zero tolerance for constraint violations

    def __init__(self, eval, scale, wind_bool):
        # Same as TrajAviaryv3Config for all base fields
        self.eval = eval
        self.scale = scale
        self.pos_xy = [-0.10, 0.10] if not eval else [-0.10, 0.10]
        self.pos_z = [-0.1, 0.1] if not eval else [-0.1, 0.1]
        self.linear_vel = [-0.1, 0.1] if not eval else [-0.125, 0.125]
        self.angular_vel = [-0.05, 0.05] if not eval else [-0.05, 0.05]
        self.roll_pitch = [-0.15, 0.15] if not eval else [-0.15, 0.15]
        self.scale_lengths = [0.05, 0.16] if self.scale else [0.05, 0.05]

        self.wind_bool = wind_bool
        self.wind_speed = [0.0, 1.5] if not eval else [0.0, 1.75]
        self.max_wind = 2.0
        self.wind_speed = self.wind_speed if self.wind_bool else [0.0, 0.0]
        self.max_wind = self.max_wind if self.wind_bool else 0.0

        self.obstacle_radius_range = [0.1, 0.3]

        trajectory_path = pkg_resources.resource_filename(
            "adapt_drones", "assets/slow_pi_tcn_train.npy"
        )
        self.trajectory_dataset = np.load(trajectory_path, allow_pickle=True)
        self.eval_trajectory_path = pkg_resources.resource_filename(
            "adapt_drones", "assets/slow_pi_tcn_eval.npy"
        )
```

### 2.5 Registration

In `adapt_drones/__init__.py`, add:

```python
register(
    id="traj_obs_v3",
    entry_point="adapt_drones.envs:TrajObstacleAviaryv3",
    max_episode_steps=600,
    kwargs={
        "mj_freq": 100,
        "ctrl_freq": 100,
    },
)
```

In `adapt_drones/envs/__init__.py`, add:

```python
from adapt_drones.envs.TrajObstacleAviaryv3 import TrajObstacleAviaryv3
```

In `config.py` `env_maps`, add:

```python
"traj_obs_v3": TrajObstacleAviaryv3Config,
```

---

## 3. Agent: RMA_DATT_Safe

A new agent class in `agents.py` that extends `RMA_DATT` with a cost critic.

```python
class RMA_DATT_Safe(RMA_DATT):
    """RMA_DATT with an additional cost value head for constrained PPO."""

    def __init__(self, priv_info_shape, state_shape, traj_shape, action_shape):
        super().__init__(priv_info_shape, state_shape, traj_shape, action_shape)

        # Cost critic: same architecture as reward critic, separate weights
        cost_critic_input = (
            self.network.env_encoder_output
            + state_shape
            + self.network.traj_encoder_output
        )
        # Actually reuse the base_policy_input_size computation from parent
        cost_critic_input = (4 + state_shape + 16)  # env_enc + state + traj_enc
        self.cost_critic = Critic(cost_critic_input, 16, 16, output_size=1)

    def get_cost_value(self, x):
        """Get cost value estimate. Same input processing as get_value."""
        env_obs = x[:, :self.priv_info_shape]
        state_obs = x[:, self.priv_info_shape:self.priv_info_shape + self.state_obs_shape]
        traj_obs = x[:, self.priv_info_shape + self.state_obs_shape:]

        env_encoder = self.env_encoder(env_obs)
        traj_encoder = self.traj_encoder(traj_obs)

        x = torch.cat((state_obs, env_encoder, traj_encoder), dim=-1)
        return self.cost_critic(x)

    def get_action_value_cost(self, x, action=None, predicted_enc=None):
        """Returns action, logprob, entropy, value, cost_value."""
        action, logprob, entropy, value = self.get_action_and_value(
            x, action, predicted_enc
        )
        cost_value = self.get_cost_value(x)
        return action, logprob, entropy, value, cost_value
```

---

## 4. Constrained PPO: `ppo_constrained.py`

Structure mirrors `ppo.py` exactly. Differences are marked with `# COST:` comments below.

### 4.1 Additional storage buffers

```python
# Same as ppo.py, plus:
costs = torch.zeros((num_steps, num_envs)).to(device)          # COST: per-step cost
cost_values = torch.zeros((num_steps, num_envs)).to(device)    # COST: cost value estimates
```

### 4.2 Lagrange multiplier

```python
# COST: Lagrangian multiplier — raw dual gradient descent
# Simple and sufficient for binary cost signals (collision = 1, else 0).
# If lambda oscillates in wandb logs, upgrade to PID variant (Stooke et al. 2020).
log_lagrange = torch.tensor(np.log(0.1), requires_grad=False, device=device)
lagrange_lr = 0.01  # single hyperparameter for constraint tuning
cost_limit = cfg.environment.cost_limit  # 0.0 for zero tolerance
```

### 4.3 Rollout collection — additions to the step loop

Inside the rollout loop (the `for step in range(num_steps)` loop), after collecting reward:

```python
# Existing: get action and value
with torch.no_grad():
    action, logprob, _, value, cost_value = agent.get_action_value_cost(ob)

# Existing: step environment
next_ob, reward, next_termination, next_truncation, info = envs.step(...)

# COST: extract cost from info
# SyncVectorEnv returns info as a dict of arrays
cost = torch.zeros(num_envs, device=device)
for env_idx in range(num_envs):
    if "cost" in info:
        cost[env_idx] = info["cost"][env_idx]

costs[step] = cost
cost_values[step] = cost_value.flatten()
```

### 4.4 GAE computation for cost advantages

After the reward GAE block, add a parallel block for costs:

```python
# COST: GAE for cost signal (same structure as reward GAE)
with torch.no_grad():
    cost_advantages = torch.zeros_like(costs).to(device)
    cost_lastgaelam = 0
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_cost_values = agent.get_cost_value(next_obs[t]).flatten()
        else:
            cost_value_mask = next_dones[t].bool()
            next_cost_values = torch.zeros_like(cost_values[0])
            next_cost_values[cost_value_mask] = agent.get_cost_value(
                next_obs[t][cost_value_mask]
            ).flatten()
            next_cost_values[~cost_value_mask] = cost_values[t + 1][~cost_value_mask]

        cost_delta = (
            costs[t]
            + gamma * next_cost_values * (1 - next_terminations[t])
            - cost_values[t]
        )
        cost_advantages[t] = cost_lastgaelam = (
            cost_delta
            + gamma * gae_lambda * (1 - next_dones[t]) * cost_lastgaelam
        )
    cost_returns = cost_advantages + cost_values
```

### 4.5 Lagrange multiplier update

After rollout collection, before the optimization loop:

```python
# COST: Update Lagrange multiplier — raw dual gradient descent
# lambda increases when cost > limit, decreases when cost < limit
episode_cost = costs.sum(0).mean()  # mean total cost per episode across envs
constraint_error = (episode_cost - cost_limit).item()

log_lagrange = log_lagrange + lagrange_lr * constraint_error
lagrange = torch.clamp(log_lagrange.exp(), min=0.0, max=100.0)

# NOTE: If lambda oscillates (visible in wandb costs/lagrange_multiplier),
# switch to PID variant: add Ki * integral(error) + Kd * d(error)/dt terms.
# See Stooke et al. 2020 "Responsive Safety in RL by PID Lagrangian Methods".
# For binary cost (collision=1, else=0), raw dual GD usually suffices.
```

### 4.6 Modified PPO loss

In the minibatch optimization loop:

```python
# Existing: policy loss (unchanged)
pg_loss1 = -mb_advantages * ratio
pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
pg_loss = torch.max(pg_loss1, pg_loss2).mean()

# COST: cost policy loss (same clipping, uses cost advantages)
mb_cost_advantages = b_cost_advantages[mb_inds]
if norm_adv:
    mb_cost_advantages = (mb_cost_advantages - mb_cost_advantages.mean()) / (
        mb_cost_advantages.std() + 1e-8
    )

cost_pg_loss1 = mb_cost_advantages * ratio
cost_pg_loss2 = mb_cost_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
cost_pg_loss = torch.max(cost_pg_loss1, cost_pg_loss2).mean()

# COST: cost value loss
new_cost_value = agent.get_cost_value(b_obs[mb_inds]).view(-1)
cost_v_loss = 0.5 * ((new_cost_value - b_cost_returns[mb_inds]) ** 2).mean()

# Existing: reward value loss (unchanged)
v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

# COST: combined loss
entropy_loss = entropy.mean()
loss = (
    pg_loss
    + lagrange * cost_pg_loss           # COST: Lagrangian penalty
    - ent_coef * entropy_loss
    + vf_coef * v_loss
    + vf_coef * cost_v_loss             # COST: cost critic loss
)
```

### 4.7 Additional logging

```python
writer.add_scalar("costs/episode_cost", episode_cost.item(), global_step)
writer.add_scalar("costs/lagrange_multiplier", lagrange.item(), global_step)
writer.add_scalar("costs/cost_value_loss", cost_v_loss.item(), global_step)
writer.add_scalar("costs/cost_policy_loss", cost_pg_loss.item(), global_step)
writer.add_scalar("costs/constraint_error", constraint_error, global_step)

# Also log safety metrics from final_info
if "final_info" in info:
    for fi in info["final_info"]:
        if fi and "min_obstacle_distance" in fi:
            writer.add_scalar(
                "safety/min_obstacle_distance",
                fi["min_obstacle_distance"],
                global_step,
            )
```

### 4.8 Full function signature

```python
def ppo_constrained_train(args: Config, envs):
    """Constrained PPO-Lagrangian training loop.

    Same structure as ppo_train, with additions for:
    - Cost signal collection from environment info["cost"]
    - Cost value head (separate critic)
    - Cost GAE computation
    - PID Lagrangian multiplier update
    - Modified loss: pg_loss + λ * cost_pg_loss + value losses + entropy
    """
```

---

## 5. Training Script: `train_obs.py`

Mirrors `train.py`. Key differences:

```python
from adapt_drones.networks.ppo_constrained import ppo_constrained_train
from adapt_drones.networks.adapt_net import adapt_train_datt_rma

@dataclass
class Args:
    env_id: str = "traj_obs_v3"        # default to obstacle env
    scale: bool = True
    wind_bool: bool = True
    seed: int = 15092024
    agent: str = "RMA_DATT_Safe"        # use safe agent

args = tyro.cli(Args)
cfg = Config(
    env_id=args.env_id,
    seed=args.seed,
    scale=args.scale,
    agent=args.agent,
    wind_bool=args.wind_bool,
    wandb_project="adapt-safe",          # SEPARATE wandb project
    learning=Learning(total_timesteps=100_000_000),
)
```

The wandb project name `"adapt-safe"` keeps runs separate. The grp_name auto-generates as `"traj_obs_v3-RMA_DATT_Safe"` from the existing logic in `config.py:75`.

Run storage: `runs/adapt-safe/traj_obs_v3-RMA_DATT_Safe/<run_name>/`

Phase 2 adaptation training follows the same pattern as `train.py`:

```python
# Phase 1: constrained PPO
ppo_constrained_train(args=cfg, envs=envs)

# Phase 2: adaptation (same as before, but on obstacle env)
adapt_cfg = Config(
    env_id=cfg.env_id,
    seed=cfg.seed,
    run_name=cfg.run_name,
    scale=cfg.scale,
    agent=cfg.agent,
    wandb_project="adapt-safe",
    **{"learning": Learning(init_lr=2e-4, anneal_lr=False, num_envs=128, total_timesteps=5_000_000)},
)
envs = gym.vector.SyncVectorEnv(...)
adapt_train_datt_rma(adapt_cfg, envs, best_model=True)
```

Note: The adaptation network training (`adapt_train_datt_rma`) doesn't need modification. It only trains the env_encoder predictor, which is the same 10D → 4D mapping regardless of obstacles. The obstacle observations are part of the state, which the adaptation network doesn't touch.

---

## 6. Evaluation Script: `eval_obs.py`

Mirrors `eval.py`. Loads the safe agent and evaluates with obstacles.

Additional metrics to report:
- Position tracking error (same as existing)
- Collision count (number of episodes with obstacle contact)
- Minimum obstacle clearance (how close the drone got)
- Constraint violation rate (% of steps where cost > 0)

```python
from adapt_drones.utils.eval_obs import phase1_obs_eval, RMA_DATT_obs_eval

@dataclass
class Args:
    env_id: str = "traj_obs_v3"
    run_name: str = ""
    seed: int = 15092024
    agent: str = "RMA_DATT_Safe"
    scale: bool = True
    idx: Union[int, None] = None
    wind_bool: bool = True

args = tyro.cli(Args)
cfg = Config(
    env_id=args.env_id,
    seed=args.seed,
    eval=True,
    run_name=args.run_name,
    agent=args.agent,
    scale=args.scale,
    wind_bool=args.wind_bool,
    wandb_project="adapt-safe",
)

phase1_obs_eval(cfg=cfg, best_model=True, idx=args.idx)
RMA_DATT_obs_eval(cfg=cfg, best_model=True, idx=args.idx)
```

The eval functions in `utils/eval_obs.py` are copies of `phase1_eval` and `RMA_DATT_eval` from `utils/eval.py`, with these additions:
- Track and report collision events
- Track minimum obstacle distance per episode
- Record obstacle positions for visualization
- Compute safety metrics alongside tracking metrics

---

## 7. Key Design Decisions

### 7.1 Obstacle info in state, not a separate obs key

Appending obstacle relative positions to the `"state"` dict entry rather than a new key avoids breaking the existing flattening/slicing logic. The agent just sees a larger state vector. This means:

- `state_obs_shape` = 12 + (n_obstacles * 3) = 27 (for 5 obstacles)
- The actor/critic input size grows by 15 dims
- The adaptation network input also grows (state_action per step = 27 + 4 = 31 instead of 16)
- Adaptation time horizon buffer: 31 * 50 = 1550 input dims (was 800)

### 7.2 Reward stays pure, cost is separate

The reward function is NOT modified. Obstacle avoidance is entirely through the cost → Lagrangian channel. This is cleaner than mixing safety penalties into the reward:
- No reward-shaping hyperparameter tuning
- The Lagrangian auto-tunes the tradeoff
- Clean ablation: compare constrained PPO vs. reward-penalty baseline

### 7.3 wandb and run storage isolation

- wandb project: `"adapt-safe"` (vs existing `"adapt-ICRA"`)
- grp_name: auto-generated as `"traj_obs_v3-RMA_DATT_Safe"`
- Run folder: `runs/adapt-safe/traj_obs_v3-RMA_DATT_Safe/<run_name>/`

Completely separate from existing runs. No collision risk.

### 7.4 Agent naming

`RMA_DATT_Safe` — extends `RMA_DATT` with cost critic. Registered in `environment_cfg.py` via `agent_name: tuple = ("RMA_DATT_Safe",)`.

### 7.5 Handling the adaptation network with larger state

The adaptation network in `adapt_net.py` uses `state_action_shape = state_shape + action_shape`. Since `state_shape` now includes obstacle positions (27 vs 12), the adaptation network automatically gets the right input size — no changes needed to `adapt_net.py`. The only difference is the buffer is larger.

However, consider: should the adaptation network see obstacle positions? The env_encoder encodes dynamics (mass, inertia, etc.) — obstacle positions are not dynamics. The adaptation network's job is to predict the dynamics encoding, not to be aware of obstacles.

**Option A**: Pass full state (including obstacles) to adaptation network. Simple, no code changes. The network learns to ignore obstacle dims since they don't correlate with dynamics encoding.

**Option B (chosen)**: Only pass the first 12 state dims (no obstacle info) to adaptation network. Requires a small modification to `adapt_net.py` to slice out obstacle dims. Cleaner semantically — obstacle positions have nothing to do with dynamics identification, and including them would just add noise to the adaptation signal.

In `adapt_net.py`, the state-action concatenation (line ~118) becomes:

```python
# Slice only the core state (12 dims), excluding obstacle obs
core_state_shape = 12  # delta_pos(3) + delta_ori(3) + delta_vel(3) + delta_ang_vel(3)
core_state_ob = state_ob[:, :core_state_shape]
state_action = torch.cat((core_state_ob, action), dim=-1)  # (batch, 12+4=16)
```

This keeps the adaptation network input at 16 * 50 = 800 dims, identical to the original. The `core_state_shape` can be passed via config or hardcoded since the base state is always 12D.

Same slicing applies in the eval loop (`utils/eval_obs.py`) when building the state-action buffer for the adaptation network at test time.

---

## 8. Experiment Configurations

### 8.1 Baseline experiments to run

| Experiment | Agent | Env | Safety | Purpose |
|---|---|---|---|---|
| No obstacles | RMA_DATT | traj_v3 | None | Existing baseline |
| Obstacles + reward penalty | RMA_DATT | traj_obs_v3 | Reward penalty | Baseline: add -10 to reward on collision |
| Obstacles + constrained PPO | RMA_DATT_Safe | traj_obs_v3 | Lagrangian | Our approach |

### 8.2 Ablations

| Ablation | What changes | Tests |
|---|---|---|
| cost_limit sweep | 0.0, 0.1, 1.0 | How strict the safety constraint is |
| n_obstacles sweep | 3, 5, 8 | Scaling with obstacle density |
| obstacle_radius | [0.1, 0.2], [0.1, 0.3], [0.2, 0.4] | Different obstacle sizes |
| PID vs raw dual GD | PID off (ki=kd=0) | Stability of Lagrangian |

---

## 9. Implementation Order

**Week 1: Environment + Config**
1. Create `quad_obstacles.xml` (copy quad.xml, add obstacle bodies)
2. Add `xml_file` kwarg to `BaseAviary.__init__` (2-line change)
3. Implement `TrajObstacleAviaryv3` (inherit TrajAviaryv3, add obstacles)
4. Add `TrajObstacleAviaryv3Config` to configs
5. Register environment
6. Test: `gym.make("traj_obs_v3", cfg=cfg)` works, obstacles visible, obs shape correct

**Week 1-2: Agent + Training**
7. Add `RMA_DATT_Safe` to `agents.py`
8. Implement `ppo_constrained.py` (copy ppo.py, add cost logic)
9. Create `train_obs.py`
10. Test: short training run (test mode), verify cost logging, Lagrange updates

**Week 2: Evaluation + Experiments**
11. Create `utils/eval_obs.py` and `eval_obs.py`
12. Run full training with 5 obstacles
13. Compare constrained PPO vs reward-penalty baseline
14. Collect safety metrics across drone scales
