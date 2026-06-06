from pathlib import Path
import sys

EXAMPLE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLE_DIR))

from _common import main


if __name__ == "__main__":
    main(
        default_controller="density_filter",
        expose_controller=False,
        default_waypoint_mode="delayed",
        default_smooth_path=False,
        default_waypoint_spacing=0.0,
        default_waypoint_clearance=0.09,
        default_density_goal_mode="path",
    )
