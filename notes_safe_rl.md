# Safe RL for Dynamics-Invariant Quadrotor Control

## Motivation

The current system (Phase 1: PPO with privileged info, Phase 2: adaptation via state-action history) achieves dynamics-invariant *control*. The natural extension is dynamics-invariant *safety* — obstacle avoidance, speed limits, attitude constraints — that also transfers across drone scales using the same adaptation embedding `z`.

---

## Core Idea

Train a neural Control Barrier Function `h_θ(x_rel, z)` that serves as a learned, dynamics-aware safety metric. Use it in two complementary ways:

1. **Constrained PPO (training time)** — Lagrangian cost derived from `h_θ` teaches the policy to stay safe
2. **CBF-QP filter (test time, optional)** — online safety filter catches any residual unsafe actions

`h_θ` is the single core artifact. What you do with it is a deployment decision.

---

## Part 1: Neural CBF — `h_θ(x_rel, z)`

### What it is

A neural network that outputs a scalar "safety score":
- `h(x) > 0` → safe
- `h(x) = 0` → safety boundary
- `h(x) < 0` → unsafe

### Inputs

- `x_rel`: relative obstacle state (relative position, relative velocity, obstacle radius). For multiple obstacles, use the closest one or run per-obstacle and take the minimum.
- `z`: the 4D dynamics embedding from `env_encoder(priv_info)` during training, or from `adapt_net(history)` at test time.

The z conditioning is what makes this dynamics-invariant. A heavy drone with low thrust-to-weight needs a larger safety margin than an agile micro-drone at the same distance from an obstacle. The CBF learns this relationship.

### Architecture

Small MLP. Input: `[x_rel, z]` → 2-3 hidden layers (64-128 units) → scalar output. Nothing fancy — the function is smooth and low-dimensional.

### Training Loss

Trained from rollout data collected during Phase 1 PPO. Each transition gives a tuple `(x_rel_t, u_t, x_rel_{t+1}, z, safe_label)`.

Three loss terms:

```
L = λ_1 · L_safe + λ_2 · L_unsafe + λ_3 · L_decrease
```

**L_safe** — push h positive for safe states:
```
L_safe = mean(max(0, -h(x_rel, z) + margin_safe))
```
over states where `||p_drone - p_obs|| > r_safe`

**L_unsafe** — push h negative for unsafe states:
```
L_unsafe = mean(max(0, h(x_rel, z) + margin_unsafe))
```
over states where `||p_drone - p_obs|| < r_danger`

**L_decrease** — the CBF condition, enforced on observed transitions near the boundary:
```
L_decrease = mean(max(0, (1 - α·dt)·h(x_t, z) - h(x_{t+1}, z)))
```
This says: along observed transitions, h should not decrease faster than the rate `α`. No dynamics model needed — just `(x_t, x_{t+1})` pairs from the rollout buffer.

### What counts as safe / unsafe

Define two radii around each obstacle:
- `r_danger` (e.g., 0.15m): definitely unsafe, hard collision zone
- `r_safe` (e.g., 0.5m): definitely safe, comfortable margin

States between `r_danger` and `r_safe` are the "boundary region" — the decrease loss is most important here. The CBF learns the transition from safe to unsafe.

For speed limits: `safe = (||v|| < v_max - margin)`, `unsafe = (||v|| > v_max)`.

For attitude limits: `safe = (tilt < θ_max - margin)`, `unsafe = (tilt > θ_max)`.

Multiple constraints can be handled by training separate `h` per constraint and taking the minimum, or by training a single multi-output network.

### When to train

Two options:
- **Concurrent with Phase 1 PPO**: collect transitions into a replay buffer, train h_θ every N PPO updates. Advantage: h_θ is ready when PPO finishes, and can be used as the Lagrangian cost mid-training.
- **After Phase 1**: train h_θ on a frozen dataset of rollouts. Simpler but can't use h as a cost during PPO — so you'd need a second round of PPO with the learned cost.

Concurrent is better if feasible.

### Reference

Qin et al. 2021 — "Learning Safe Multi-Agent Control with Decentralized Neural Barrier Certificates" (RA-L). Same three-term loss, multi-agent collision avoidance, no dynamics model.

---

## Part 2: Constrained PPO via Lagrangian

### What it is

Standard PPO, but with an additional cost constraint enforced via a learned Lagrange multiplier. The policy optimizes reward while keeping safety cost below a threshold.

### Formulation

The unconstrained PPO objective:
```
max_π  E[Σ r_t]
```

Becomes a constrained optimization:
```
max_π  E[Σ r_t]
s.t.   E[Σ c_t] ≤ d
```

where `c_t` is the per-step safety cost and `d` is the allowed violation budget (e.g., `d = 0` for zero tolerance).

Solved via Lagrangian relaxation:
```
max_π  min_{λ≥0}  E[Σ r_t] - λ · (E[Σ c_t] - d)
```

### The cost signal `c_t`

This is where the learned CBF comes in. Two options:

**Option A — CBF-derived cost (preferred):**
```
c_t = max(0, (1 - α·dt) · h(x_t, z) - h(x_{t+1}, z))
```
Cost is nonzero only when the CBF condition is violated — i.e., the safety score decreased too fast. This directly enforces the CBF invariance through the policy.

**Option B — Simple indicator cost (simpler, use if CBF not yet trained):**
```
c_t = 1  if ||p_drone - p_obs|| < r_danger
c_t = 0  otherwise
```
Just counts collisions. Works with Lagrangian but less principled than using the CBF.

Start with Option B to get things running, switch to Option A once `h_θ` is trained.

### Implementation changes to existing PPO

The changes to `ppo.py` are small:

1. **Cost critic**: Add a second value head `V_c(s)` that estimates expected future cost (same architecture as reward critic).

2. **Cost advantage**: Compute cost advantages using GAE, same as reward advantages but on the cost signal.

3. **Lagrange multiplier**: A single learnable parameter `log_λ`, updated by:
   ```
   λ_loss = -λ · (mean_episode_cost - d)
   ```
   If cost > budget → λ increases → policy penalized more for unsafe behavior.
   If cost < budget → λ decreases → policy gets more freedom.

4. **Modified PPO loss**:
   ```
   L = L_clip_reward - λ · L_clip_cost
   ```
   where `L_clip_cost` uses the cost advantages with the same clipping as standard PPO.

5. **PID Lagrangian (recommended)**: Instead of raw dual gradient descent on λ, use a PID controller on the constraint error `(mean_cost - d)`. This is more stable and avoids oscillation. From Stooke et al. 2020.

### What stays the same

- env_encoder, trajectory encoder, adaptation network — all unchanged
- Phase 2 adaptation training — unchanged (the policy is already trained with safety awareness)
- Action space, observation space for the policy — unchanged (obstacle info enters through `x_rel` in the CBF, but the policy may also benefit from seeing obstacle positions — see environment changes below)

---

## Part 3: CBF-QP Filter (Optional, Test Time)

### When to use

If empirical safety from constrained PPO alone isn't sufficient, layer on a runtime filter. This is optional and independent of the training procedure.

### How it works without a learned dynamics model

At test time, for each action `u_proposed`:

**Finite-difference approach:**
1. The policy proposes `u_proposed`
2. Perturb each action dimension: `u_i+ = u + ε·e_i`, `u_i- = u - ε·e_i`
3. Step the simulator (or a copy) with each perturbation to get `x'_i+`, `x'_i-`
4. Estimate `∂h(x')/∂u_i ≈ (h(x'_i+, z) - h(x'_i-, z)) / (2ε)`
5. Now you have a linear approximation: `h(x') ≈ h(x'_0, z) + g^T · (u - u_proposed)` where `g` is the gradient
6. If `h(x'_0, z) ≥ (1-α·dt) · h(x, z)`: action is safe, pass through
7. Otherwise: project onto the half-space `h(x'_0) + g^T·(u - u_0) ≥ (1-α·dt)·h(x,z)`, closed form:
   ```
   u_safe = u_proposed + ((threshold - h(x'_0)) / ||g||²) · g
   ```

This requires 2·action_dim simulator queries per step (8 queries for 4D action). In MuJoCo this is microseconds.

**Alternative — sampling approach:**
1. Sample K actions near `u_proposed` (e.g., K=32)
2. Step the sim with each, evaluate `h(x', z)` for each
3. Pick the one closest to `u_proposed` that satisfies the CBF condition
4. Simpler but less precise

### When this matters

For sim-to-sim transfer experiments: the CBF filter provides an extra safety layer when the adaptation embedding ẑ is imperfect (early in an episode before the history buffer fills up). Good for the paper's ablation table.

---

## Part 4: Environment Changes Needed

### Obstacles

Add obstacles to the MuJoCo environment:
- Static spheres/boxes with randomized positions each episode
- Start with 3-5 obstacles in the trajectory corridor
- Randomize radii (0.1m - 0.3m) and positions

### Observation space additions

The policy may benefit from obstacle information directly (not just through the CBF):
- Relative positions to K nearest obstacles
- Relative velocities (zero for static, nonzero for moving obstacles later)
- This goes into the state observation, not the privileged info

The CBF gets `x_rel` separately — it doesn't need the full state, just the safety-relevant part.

### Cost signal

The environment should return a cost alongside the reward. Gymnasium supports this via the `info` dict:
```python
info["cost"] = 1.0 if collision else 0.0
```

Or for CBF-derived cost, compute it in the training loop from h values.

---

## Part 5: How z Ties Everything Together

The adaptation embedding z is the linchpin:

```
                    ┌──→ Actor (dynamics-invariant control)
                    │
env_encoder(priv) ──┤──→ h_θ(x_rel, z) (dynamics-invariant safety)
  or adapt_net(h)   │
                    └──→ Critic (dynamics-aware value estimation)
```

During training: z = env_encoder(privileged_info) — ground truth dynamics encoding.
During Phase 2: z = adapt_net(state_action_history) — inferred from behavior.
At test time: same adapted z feeds both the policy AND the CBF.

The CBF paper contribution: a single adaptation mechanism provides both dynamics-invariant control AND dynamics-invariant safety. The safety margins automatically adjust to the drone's capabilities.

### Key experiment

Same CBF trained across drone scales (Phase 1 randomization), deployed on unseen drones via adaptation. Compare:
- Fixed safety margin (baseline): conservative for agile drones, dangerous for heavy drones
- CBF without z: one-size-fits-all learned boundary
- CBF with z (ours): dynamics-aware safety that adapts per-drone

---

## Part 6: Implementation Order

1. Add obstacles to the environment, add relative obstacle positions to observations
2. Add simple indicator cost (collision = 1) to env info dict
3. Implement PPO-Lagrangian (cost critic + Lagrange multiplier) — get this working with the simple cost
4. Train h_θ(x_rel, z) from rollout data concurrently with PPO
5. Switch cost signal from indicator to CBF-derived cost
6. Evaluate: safety rate across drone scales, with and without z conditioning
7. (Optional) Add CBF-QP filter at test time, compare to policy-only safety

Steps 1-3 can be done in ~1 week. Steps 4-6 add another 1-2 weeks. Step 7 is independent and can be done anytime after step 4.

---

## References

- Qin et al. 2021 — "Learning Safe Multi-Agent Control with Decentralized Neural Barrier Certificates" (RA-L) — model-free neural barrier, three-term loss
- Stooke et al. 2020 — "Responsive Safety in RL by PID Lagrangian Methods" — stable Lagrangian optimization
- Dawson et al. 2023 — "Safe Control With Learned Certificates" — survey of neural CBF methods
- Achiam et al. 2017 — "Constrained Policy Optimization" — foundational constrained RL
- Ray et al. 2019 — "Benchmarking Safe Exploration in Deep RL" — Safety Gym, PPO-Lagrangian baseline
- Cheng et al. 2019 — "End-to-End Safe RL through Barrier Functions" — CBF + RL integration
