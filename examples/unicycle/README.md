# Unicycle Density Controllers

These examples show density-based navigation for a planar unicycle moving from a start pose to a goal while avoiding p-norm obstacles. The runnable examples compare two controller styles:

- `density_feedback.py`: direct density-gradient feedback, converted into unicycle commands.
- `density_filter.py`: a discrete density safety filter wrapped around a nominal unicycle command.

The `density_mpc.py` files in these folders are placeholders and currently raise `NotImplementedError`.

## Setup

From the repository root:

```bash
python -m pip install -e .
python -m pip install pillow
```

Run examples from this directory so GIFs are written to `examples/unicycle/animations`:

```bash
cd examples/unicycle
```

Use `--save-gif` to save an animation. Do not combine it with `--no-plot`, because the animation is created inside the plotting path.

## Dynamics

All examples simulate the unicycle state

```text
z = [x, y, theta]
```

with forward speed `v` and yaw rate `omega`:

```text
x_next     = x + dt * v * cos(theta)
y_next     = y + dt * v * sin(theta)
theta_next = theta + dt * omega
```

The implementation is in `density_utils/dynamics/unicycle.py`.

## Obstacles

Obstacles are p-norm balls with optional rotation and anisotropic scaling. For
obstacle \(i\), define the normalized obstacle distance

$$
d_i(x) =
\left\|
S_i^{-1} R(-\theta_i)(x-c_i)
\right\|_{p_i},
$$

where \(c_i\) is the obstacle center, \(R(-\theta_i)\) rotates into the obstacle
frame, and \(S_i\) is an optional diagonal scaling matrix.

Each obstacle has an inner radius \(r_{1,i}\) and outer radius \(r_{2,i}\). The
smooth bump function is

$$
b_i(x)=
\begin{cases}
0, & d_i(x) \leq r_{1,i},\\[4pt]
1, & d_i(x) \geq r_{2,i},\\[4pt]
\dfrac{\exp(-1/m_i)}
{\exp(-1/m_i)+\exp(-1/(1-m_i))},
& r_{1,i}<d_i(x)<r_{2,i},
\end{cases}
$$

with transition coordinate

$$
m_i(x)=
\frac{d_i(x)^{p_i}-r_{1,i}^{p_i}}
{r_{2,i}^{p_i}-r_{1,i}^{p_i}}.
$$

The agent radius is handled by inflating each obstacle before control is computed.

## Density Function

The basic position density used by direct feedback is

$$
\rho(x) =
\frac{\Phi(x)}
{\lVert x-x_g\rVert^{2\alpha}},
\qquad
\Phi(x)=\prod_i b_i(x).
$$

Here \(x_g\) is the goal and \(\alpha > 0\) controls how strongly the goal term
pulls the density upward near the goal.  The product \(\Phi(x)\) suppresses
density inside obstacles and approaches one in free space.  The density-gradient
controller follows \(\nabla \rho(x)\), which naturally moves toward the goal
while bending around obstacle transition regions.

The filter examples use a pose density:

$$
\rho(z) =
\frac{\Phi(p)}
{V(z)^\alpha},
\qquad
p =
\begin{bmatrix}
x & y
\end{bmatrix}^\top,
$$

where

$$
V(z)=
(x-x_g)^2
+
(y-y_g)^2
+
w_\theta\,\mathrm{wrap}(\theta-\theta_g)^2,
\qquad
w_\theta = 0.05.
$$

This lets the safety filter reason about both position and heading.

## Density Feedback Controller

`density_feedback.py` computes a planar reference vector:

```text
u_xy = ctrl_multiplier * grad rho(position)
```

Near the goal it switches to a small discrete LQR stabilizer. The planar vector is then converted to unicycle commands:

```text
v = ||u_xy||
desired_heading = atan2(u_y, u_x)
omega = desired_heading_rate - k_heading * wrap(theta - desired_heading)
```

Both `v` and `omega` are saturated by the configured limits.

## Density filter Controller

`density_filter.py` first builds a nominal unicycle command from a density-blended single-integrator reference. It then solves one constrained optimization at each control step:

```text
minimize    0.5 * ||u - u_nom||_W^2 + 0.5 * slack_weight * ||s||^2
subject to  rho(z_next) - rho(z) + dt * div(F_d)(z) * rho(z) + s >= 0
            u_min <= u <= u_max
            s >= 0
```

For these unicycle examples, `div(F_d)` is set to zero and

```text
z_next = unicycle_step(z, v, omega, density_dt)
```

The filter is solved with SciPy SLSQP. Slack is logged and plotted; nonzero slack means the solver relaxed a density condition to keep the problem feasible.

## Examples

### Static Single Obstacle

Start: `[-2.0, -1.0]`. The feedback script uses goal `[2.0, 1.0]`; the filter script uses goal `[2.0, 1.1]`. One elliptical p-norm obstacle is centered at the origin.

Density feedback:

```bash
python static_single_obstacle/density_feedback.py --save-gif
```

![Static single obstacle density feedback](animations/unicycle_static.gif)

Density filter:

```bash
python static_single_obstacle/density_filter.py --save-gif
```

![Static single obstacle density filter](animations/unicycle_static_filter.gif)

### Static Multi Obstacle

The feedback script starts at `[-2.1, -2.1]` and goals to `[2.0, 2.0]`. The filter script starts at `[-2.0, -2.1]` and goals to `[1.8, 2.1]`. The map contains ten p-norm obstacles with mixed radii, rotations, and scaling.

Density feedback:

```bash
python static_multi_obstacles/density_feedback.py --save-gif
```

![Static multi obstacle density feedback](animations/unicycle_multi.gif)

Density filter:

```bash
python static_multi_obstacles/density_filter.py --save-gif
```

![Static multi obstacle density filter](animations/unicycle_static_multi_filter.gif)

The multi-obstacle filter keeps the optimization small by selecting the nearest configured obstacles at each step using `max_filter_obstacles`.

### Local Sensing

The local-sensing examples use the same obstacle field as the multi-obstacle map, but the controller only sees obstacles inside a forward camera cone:

```text
cam_range = 1.0
fov_angle = 80 deg
max_sensed = 5
```

Sensed obstacles are kept in a short buffer for `linger_steps`, which prevents the active obstacle set from flickering immediately when an obstacle leaves the field of view.

Density feedback:

```bash
python local_sensing/density_feedback.py --save-gif
```

![Local sensing density feedback](animations/unicycle_multi_local.gif)

Density filter:

```bash
python local_sensing/density_filter.py --save-gif
```

![Local sensing density filter](animations/unicycle_local_sensing_filter.gif)

Reactive FOV filter visualization:

```bash
python local_sensing/density_filter_reactive_fov.py --save-gif
```

![Local sensing reactive FOV density filter](animations/unicycle_local_sensing_filter_reactive_fov.gif)

`density_filter_reactive_fov.py` calls the same filter controller as `density_filter.py`, but highlights the camera cone when an obstacle is actively sensed.

## Common Options

```bash
--save-gif       save a GIF animation
--no-plot        skip plotting and animation
--steps N        override max simulation steps, filter examples only
```

For quick non-visual checks:

```bash
python static_single_obstacle/density_filter.py --no-plot --steps 100
python static_multi_obstacles/density_filter.py --no-plot --steps 100
python local_sensing/density_filter.py --no-plot --steps 100
```
