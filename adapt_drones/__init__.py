from gymnasium.envs.registration import register

register(
    id="hover_v0",
    entry_point="adapt_drones.envs:HoverAviaryv0",
    max_episode_steps=600,
    kwargs={
        "mj_freq": 100,
        "ctrl_freq": 100,
    },
)

register(
    id="hover_v1",
    entry_point="adapt_drones.envs:HoverAviaryv1",
    max_episode_steps=600,
    kwargs={
        "mj_freq": 100,
        "ctrl_freq": 100,
    },
)

register(
    id="traj_v2",
    entry_point="adapt_drones.envs:TrajAviaryv2",
    max_episode_steps=600,
    kwargs={
        "mj_freq": 100,
        "ctrl_freq": 100,
    },
)

register(
    id="traj_v3",
    entry_point="adapt_drones.envs:TrajAviaryv3",
    max_episode_steps=600,
    kwargs={
        "mj_freq": 100,
        "ctrl_freq": 100,
    },
)

register(
    id="traj_v2_ctbr",
    entry_point="adapt_drones.envs:TrajAviaryv2CTBR",
    max_episode_steps=600,
    kwargs={
        "mj_freq": 100,
        "ctrl_freq": 100,
    },
)

register(
    id="traj_v3_ctbr",
    entry_point="adapt_drones.envs:TrajAviaryv3CTBR",
    max_episode_steps=600,
    kwargs={
        "mj_freq": 100,
        "ctrl_freq": 100,
    },
)

register(
    id="traj_pay_v3",
    entry_point="adapt_drones.envs:TrajAviaryPayv3",
    max_episode_steps=600,
    kwargs={
        "mj_freq": 100,
        "ctrl_freq": 100,
    },
)
