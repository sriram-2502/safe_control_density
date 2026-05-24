from pathlib import Path

import numpy as np

from density_utils.racing import BicycleParams, SystemLimits


EXAMPLE_DIR = Path(__file__).resolve().parent
RACING_DIR = EXAMPLE_DIR.parent
TRACK_FILE = RACING_DIR / "data" / "track_layout" / "l_shape.csv"

TRACK_WIDTH = 1.0
TRACK_MARGIN = 0.12
TRACK_TRANSITION = 0.22

DT = 0.1
NUM_STEPS = 500

INITIAL_CURVILINEAR_STATE = np.array(
    [
        0.0,  # vx
        0.0,  # vy
        0.0,  # yaw rate
        0.0,  # heading error
        0.0,  # progress along track
        0.0,  # lateral error
    ],
    dtype=float,
)

BICYCLE_PARAMS = BicycleParams()
SYSTEM_LIMITS = SystemLimits(delta_max=0.5, a_max=1.0, v_min=0.0, v_max=10.0)

USE_LTI_MODEL = True

MATRIX_A = np.array(
    [
        [
            9.671290682499817937e-01,
            4.242128062809466527e-02,
            -1.334296593039446290e-02,
            -3.371777265649055985e-03,
            3.227927831414771170e-06,
            1.686059132152231384e-03,
        ],
        [
            -4.741648280908570059e-03,
            -2.468400046155550531e-01,
            3.245228399640302103e-02,
            -6.761389137462380265e-04,
            1.409084040214530391e-05,
            5.146180819166828319e-03,
        ],
        [
            -4.687236221283337667e-02,
            -2.359785858136580039e00,
            3.057730658931238632e-01,
            -6.291019348164512102e-03,
            1.407919154285720311e-04,
            5.161261275626113226e-02,
        ],
        [
            1.491341619436537154e-02,
            -7.246004296693270286e-01,
            9.814447747880826467e-02,
            1.067946803205831907e00,
            3.685185147970903791e-06,
            7.050901769705017648e-03,
        ],
        [
            9.697109942460338528e-02,
            -1.868558231814464871e-02,
            1.741062754045796974e-03,
            -4.872250833025211156e-03,
            1.000000846363945595e00,
            -1.023352728807257542e-02,
        ],
        [
            6.714003570349933153e-04,
            -8.179829686470431460e-02,
            1.271581430918338092e-02,
            4.104114675570305626e-02,
            -1.886060580647413303e-06,
            1.000653657321968648e00,
        ],
    ],
    dtype=float,
)

MATRIX_B = np.array(
    [
        [1.487289292077178041e-02, 9.770355093588267703e-02],
        [1.823272351013625059e-01, -9.009215991985008781e-04],
        [1.575741357076274385e00, -9.783864169291941332e-03],
        [1.258236266910766066e-01, 7.018480442549679963e-04],
        [5.201009225593251689e-04, 4.856148476338860952e-03],
        [1.639395364567048513e-02, 1.979138102590245085e-07],
    ],
    dtype=float,
)

TARGET_SPEED = 0.8
TARGET_LATERAL_ERROR = 0.0

MPC_HORIZON = 10
MAX_SOLVER_ITER = 80

Q_SPEED = 10.0
Q_HEADING = 4.0
Q_LATERAL = 40.0
R_STEER = 0.1
R_ACCEL = 0.1
R_STEER_RATE = 3.0
R_ACCEL_RATE = 0.4
DENSITY_COST_WEIGHT = 2.0
DENSITY_SLACK_WEIGHT = 10000.0

DENSITY_MIN = 0.2
DENSITY_TRANSPORT_DIVERGENCE = float(np.trace(MATRIX_A))
ENFORCE_HARD_SUPERELLIPSE = False
OBSTACLE_MARGIN = 0.35
OBSTACLE_TRANSITION = 1.5
OBSTACLE_SUPERELLIPSE_DEGREE = 6
OBSTACLE_DENSITY_SHARPNESS = 8.0
OBSTACLE_DENSITY_MODE = "sigmoid"  # "sigmoid" or "bump"

OBSTACLES = [
    {
        "name": "car1",
        "initial_s": 4.0,
        "initial_ey": 0.1,
        "speed": 0.2,
        "length": 0.45,
        "width": 0.25,
    },
    {
        "name": "car2",
        "initial_s": 10.0,
        "initial_ey": -0.1,
        "speed": 0.2,
        "length": 0.45,
        "width": 0.25,
    },
]

INITIAL_CONTROL = np.array([0.0, 0.0], dtype=float)
