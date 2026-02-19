# Safe Control Density (Python)

This folder contains a lightweight Python port of the time-varying density
feedback controller with robotics-style examples inspired by the
`safe_control` repo.

## Quick start

Create a virtual environment, install, and run an example:

```bash
cd safe_control_density
python -m venv .venv
.venv\Scripts\activate
pip install -e .
python examples/single_integrator/static_single_obstacle/density_feedback.py

Timing note: examples include a `log_timing` toggle inside each script. When enabled, logs per-iteration control compute time (mean/std). The summary line prints `sim_time` (total runtime) and `avg_iteration` (average per-iteration time).

Stop note: examples include a `stop_when_stable` toggle (default True) along with `stop_tol` and `stop_steps` to end the simulation once the agent stays within the goal tolerance for a sustained window.
```

## Structure

- `density_utils/`: core density, dynamics, controllers, and simulation utilities
- `examples/`: runnable scripts similar to safe_control examples

## Notes

- Density gradients use finite differences for speed and simplicity.
- The first example targets a 2D single-integrator with a static obstacle.

