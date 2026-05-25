CONFIG = {
    "simulation": {
        "dt": 0.01,
        "density_dt": 0.05,
        "steps": 5000,
        "stop_tol": 0.01,
        "stop_steps": 500,
        "stop_when_stable": True,
        "print_interval": 500,
    },
    "scenario": {
        "agent_radius": 0.1,
        "start": [-2.0, -1.0],
        "goal": [2.0, 1.1],
        "obstacles": [
            {
                "center": [0.0, 0.0],
                "r1": 0.6,
                "r2": 1.0,
                "p": 2.0,
            },
        ],
    },
    "density": {
        "alpha": 0.4,
        "ctrl_multiplier": 4.0,
        "rad_from_goal": 0.35,
        "q_lqr": 4.0,
        "r_lqr": 1.0,
        "slack_weight": 1e4,
    },
    "control": {
        "v_max": 2.0,
        "omega_max": 3.0,
        "k_heading": 4.0,
    },
    "animation": {
        "path": "animations/unicycle_static_filter.gif",
        "stride": 5,
        "fps": 15,
    },
}
