OBSTACLES = [
    {"center": [-0.6, 0.6], "r1": 0.5, "r2": 0.6, "p": 4.0, "angle_deg": 30.0},
    {"center": [1.0, 0.3], "r1": 0.5, "r2": 0.6, "p": 8.0, "angle_deg": 60.0},
    {"center": [-1.8, 1.2], "r1": 0.2, "r2": 0.3, "p": 2.0, "scale": [1.6, 0.8]},
    {"center": [0.0, 1.6], "r1": 0.2, "r2": 0.3, "p": 4.0, "angle_deg": 90.0},
    {"center": [-1.4, -0.6], "r1": 0.2, "r2": 0.3, "p": 2.0},
    {"center": [-1.6, -1.4], "r1": 0.2, "r2": 0.3, "p": 2.0},
    {"center": [0.4, -1.6], "r1": 0.2, "r2": 0.3, "p": 2.0, "scale": [0.7, 1.5]},
    {"center": [1.8, -1.2], "r1": 0.2, "r2": 0.3, "p": 2.0},
    {"center": [-0.2, -0.8], "r1": 0.2, "r2": 0.3, "p": 2.0, "scale": [1.4, 1.0]},
    {"center": [1.0, 1.6], "r1": 0.2, "r2": 0.3, "p": 2.0},
]


CONFIG = {
    "simulation": {
        "dt": 0.05,
        "density_dt": 0.1,
        "steps": 12000,
        "stop_tol": 0.1,
        "stop_steps": 500,
        "print_interval": 100,
    },
    "scenario": {
        "agent_radius": 0.1,
        "start": [-2.0, -2.1],
        "goal": [1.8, 2.1],
        "obstacles": OBSTACLES,
    },
    "density": {
        "alpha": 0.4,
        "ctrl_multiplier": 4.0,
        "rad_from_goal": 0.35,
        "q_lqr": 4.0,
        "r_lqr": 1.0,
        "slack_weight": 1e4,
        "max_filter_obstacles": 4,
    },
    "control": {
        "v_max": 2.0,
        "omega_max": 4.0,
        "k_heading": 4.0,
    },
    "animation": {
        "path": "animations/unicycle_static_multi_filter.gif",
        "stride": 5,
        "fps": 15,
    },
}
