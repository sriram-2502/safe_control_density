# Racing Density MPC-CDF Example

This example studies density-constrained model predictive control for a racing
vehicle on a closed track with moving obstacle cars.  The setup follows the
structure of the `mpccbf_test.py` example from the car-racing reference code:
the ego vehicle tracks a desired speed on an L-shaped track while avoiding two
scripted moving cars.  The main difference is that the CBF safety constraint is
replaced by a density/CDF-style safety condition.

## Problem Formulation

The ego vehicle state is represented in curvilinear track coordinates as

$$
x =
\begin{bmatrix}
v_x & v_y & \omega_z & e_\psi & s & e_y
\end{bmatrix}^\top,
$$

where \(v_x\) and \(v_y\) are body-frame velocities, \(\omega_z\) is yaw rate,
\(e_\psi\) is heading error relative to the track tangent, \(s\) is progress
along the track, and \(e_y\) is lateral deviation from the centerline.

The control input is

$$
u =
\begin{bmatrix}
\delta & a
\end{bmatrix}^\top,
$$

where \(\delta\) is steering and \(a\) is longitudinal acceleration.

The control objective is to track a target speed and remain near the centerline,

$$
v_x \rightarrow v_{\mathrm{ref}}, \qquad e_y \rightarrow 0,
$$

while avoiding moving obstacle cars and staying inside the track.

## Prediction Dynamics

To match the reference MPC-CBF racing setup, the MPC prediction model uses the
same discrete-time LTI dynamics:

$$
x_{k+1} = A x_k + B u_k.
$$

The matrices \(A\) and \(B\) are copied into `config.py` from the reference
car-racing LTI model.  This keeps the density-MPC comparison focused on the
safety constraint rather than on differences in the prediction model.

The script still keeps the nonlinear dynamic bicycle model available in the
code path, but the default setting is:

```python
USE_LTI_MODEL = True
```

## Moving Obstacles

Each obstacle car follows a scripted trajectory in curvilinear coordinates:

$$
s_i(t) = s_{i,0} + v_i t, \qquad e_{y,i}(t) = \bar e_{y,i}.
$$

In the default setup,

$$
s_1(t) = 4.0 + 0.2t, \qquad e_{y,1} = 0.1,
$$

and

$$
s_2(t) = 10.0 + 0.2t, \qquad e_{y,2} = -0.1.
$$

These match the obstacle cars used in the reference MPC-CBF test.

## Density Function

The obstacle geometry is based on the same superellipse used by the CBF
formulation.  For an ego state \(x\) and obstacle \(i\), define

$$
\Delta s_i = s - s_i,
\qquad
\Delta e_{y,i} = e_y - e_{y,i}.
$$

The signed longitudinal difference \(\Delta s_i\) is wrapped around the closed
track so that nearby cars are treated correctly across the start/finish line.

The obstacle superellipse value is

$$
z_i(x) =
\left(\frac{|\Delta s_i|}{L_i}\right)^d
+
\left(\frac{|\Delta e_{y,i}|}{W_i}\right)^d,
$$

where \(d=6\), and

$$
L_i = \frac{\ell_{\mathrm{ego}}+\ell_i}{2},
\qquad
W_i = \frac{w_{\mathrm{ego}}+w_i}{2}.
$$

The unsafe obstacle region is described by

$$
z_i(x) < 1 + m,
$$

where \(m\) is the configured obstacle safety margin.

We convert this superellipse into a smooth density term using a bump function:

$$
\rho_i(x) =
\operatorname{bump}\bigl(z_i(x);\, 1+m,\, 1+m+\Delta\bigr),
$$

where \(\Delta\) is the obstacle transition width.  Thus,

$$
\rho_i(x) \approx 0 \quad \text{near the obstacle},
\qquad
\rho_i(x) \approx 1 \quad \text{far from the obstacle}.
$$

The track density is defined in the same way from the lateral track clearance:

$$
c_{\mathrm{track}}(x) = w_{\mathrm{track}} - \lvert e_y \rvert.
$$

Using track margin \(m_{\mathrm{track}}\) and transition width
\(\Delta_{\mathrm{track}}\), the track density is

$$
\rho_{\mathrm{track}}(x) =
\operatorname{bump}
\bigl(
c_{\mathrm{track}}(x);\,
m_{\mathrm{track}},\,
m_{\mathrm{track}}+\Delta_{\mathrm{track}}
\bigr).
$$

Finally, the total safety density used by the MPC is the product of the track
density and all obstacle density terms:

$$
\rho(x) = \rho_{\mathrm{track}}(x)\prod_i \rho_i(x).
$$

In the implementation, this product is used to encourage early, smooth
avoidance, while the hard superellipse constraint prevents the ego vehicle from
cutting through obstacle cars.

## MPC-CDF Constraint

At each time step, the controller solves an MPC problem over horizon \(N\):

$$
\min_{u_0,\dots,u_{N-1}}
\sum_{k=0}^{N-1}
\ell(x_k,u_k)
$$

subject to the prediction dynamics

$$
x_{k+1} = A x_k + B u_k,
$$

input bounds

$$
-\delta_{\max} \leq \delta_k \leq \delta_{\max},
\qquad
-a_{\max} \leq a_k \leq a_{\max},
$$

speed bounds

$$
v_{\min} \leq v_{x,k} \leq v_{\max},
$$

and density-CDF constraints

$$
\rho(x_{k+1}) + \sigma_k^{(1)}
\geq
\rho_{\min},
$$

$$
\rho(x_{k+1}) + \sigma_k^{(2)}
\geq
(1-\gamma)\rho(x_k).
$$

The slack variables satisfy

$$
\sigma_k^{(1)} \geq 0,
\qquad
\sigma_k^{(2)} \geq 0,
$$

and are heavily penalized in the objective.  This mirrors the reference CBF-MPC
idea of keeping the optimization problem feasible while making safety
violations expensive.

The hard superellipse safety condition is also enforced:

$$
z_i(x_{k+1}) \geq 1 + m
\qquad \text{for each obstacle } i.
$$

This is important because the bump density is flat near zero inside the unsafe
region; the explicit superellipse constraint gives the optimizer a clear
geometric safety boundary.

## Running The Example

Run the example with:

```bash
MPLCONFIGDIR=/tmp/matplotlib python examples/racing/density_mpc/density_mpc.py
```

By default, the script shows the animation and saves a GIF to:

```text
animations/density_mpc.gif
```

For a headless run:

```bash
MPLCONFIGDIR=/tmp/matplotlib python examples/racing/density_mpc/density_mpc.py --no-animation
```

For quick debugging without saving:

```bash
MPLCONFIGDIR=/tmp/matplotlib python examples/racing/density_mpc/density_mpc.py \
  --steps 40 \
  --no-animation \
  --no-save-animation
```

## Result

The density-MPC controller tracks the L-shaped racing track while avoiding two
moving cars.  The density term causes early response, while the hard
superellipse constraint prevents collision with either moving obstacle.

![Density MPC racing result](../../../animations/density_mpc_Ltrack.gif)
