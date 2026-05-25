from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np

from _sensing_radius_sweep_common import _parse_radii, _run_controller


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--radii",
        default="0.95,1.10,1.25,1.40,1.50",
        help="Comma-separated obstacle sensing radii r2 to sweep.",
    )
    parser.add_argument("--steps", type=int, default=4000, help="Maximum simulation steps per radius.")
    parser.add_argument(
        "--u-nom",
        default="density",
        choices=("goal", "lqr", "density", "density_blend", "pure_pursuit"),
        help="Nominal controller used by the Density filter sweep.",
    )
    parser.add_argument("--stride", type=int, default=12, help="Animation frame stride.")
    parser.add_argument("--fps", type=int, default=18, help="GIF playback frame rate.")
    parser.add_argument("--no-gif", action="store_true", help="Skip saving the dashboard GIF.")
    parser.add_argument("--no-show", action="store_true", help="Save outputs without opening matplotlib windows.")
    args = parser.parse_args()

    animations_to_show = _run_controller(
        "filter",
        radii=_parse_radii(args.radii),
        start=np.array([-2.0, -1.0]),
        goal=np.array([2.0, 1.1]),
        agent_radius=0.1,
        steps_feedback=args.steps,
        steps_filter=args.steps,
        u_nom_mode=args.u_nom,
        output_dir=Path(__file__).resolve().parent / "comparison_results",
        no_gif=args.no_gif,
        stride=args.stride,
        fps=args.fps,
        no_show=args.no_show,
    )

    if not args.no_show:
        plt.show()
    _ = animations_to_show


if __name__ == "__main__":
    main()
