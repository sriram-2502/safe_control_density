from pathlib import Path

import numpy as np

from density_utils.racing import BicycleParams, SystemLimits


EXAMPLE_DIR = Path(__file__).resolve().parent
RACING_DIR = EXAMPLE_DIR.parent
TRACK_FILE = RACING_DIR / "data" / "track_layout" / "l_shape.csv"

TRACK_WIDTH = 0.8

DT = 0.1
INTEGRATION_DT = 0.01
NUM_STEPS = 250

TARGET_SPEED = 0.8
TARGET_LATERAL_ERROR = 0.0

INITIAL_CURVILINEAR_STATE = np.array(
    [
        TARGET_SPEED,
        0.0,
        0.0,
        0.0,
        0.1,
        0.0,
    ],
    dtype=float,
)

BICYCLE_PARAMS = BicycleParams()
SYSTEM_LIMITS = SystemLimits(delta_max=0.5, a_max=1.0, v_min=0.0, v_max=2.0)
