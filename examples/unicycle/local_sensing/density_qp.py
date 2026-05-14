import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _density_qp_common import run_unicycle_density_qp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-gif", action="store_true", help="Save animation as GIF.")
    parser.add_argument("--no-plot", action="store_true", help="Run without opening plots.")
    parser.add_argument("--steps", type=int, default=None, help="Maximum simulation steps.")
    args = parser.parse_args()
    run_unicycle_density_qp("local_sensing", args)


if __name__ == "__main__":
    main()
