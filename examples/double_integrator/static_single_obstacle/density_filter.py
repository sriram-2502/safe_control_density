"""Density filter placeholder for the double-integrator static-obstacle example.

The current Density filter controller is intentionally scoped to the 2D
single-integrator model.  A double-integrator Density filter needs a relative-degree
two formulation or a separate velocity/acceleration reference layer before this
example should be enabled.
"""


def main():
    raise NotImplementedError(
        "Double-integrator Density filter is not implemented yet. "
        "Use the single-integrator Density filter examples for the current clean implementation."
    )


if __name__ == "__main__":
    main()
