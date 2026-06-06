# Maze Navigation Prototype

This example is a small, pure-Python testbed for hierarchical maze navigation:

1. Parse an ASCII occupancy map.
2. Plan a global path with A*.
3. Build sparse strategic waypoints with clearance-aware shortcut checks.
4. Track the waypoints with a unicycle controller.
5. Represent maze walls as merged rectangular obstacles.
6. Wrap rectangle-to-point clearance in a smooth bump density.
7. Compare nominal tracking, density feedback, one-step density filtering, and density MPC.

Run from the repository root:

```bash
python3 examples/maze/density_feedback.py
python3 examples/maze/density_filter.py
python3 examples/maze/density_mpc.py
```

Choose a scenario:

```bash
python3 examples/maze/density_feedback.py --map wide
python3 examples/maze/density_feedback.py --map wide_s
python3 examples/maze/density_feedback.py --map multi_room
python3 examples/maze/density_feedback.py --map rooms
python3 examples/maze/density_feedback.py --map narrow_s
python3 examples/maze/density_feedback.py --map clutter_s
python3 examples/maze/density_feedback.py --map clutter_rooms
python3 examples/maze/density_feedback.py --map clutter_zigzag
```

Scenarios use wider openings than the first prototype: the intended routes and
doorway gaps are kept at least two grid cells wide so the unicycle has room to
turn under the rectangular wall-density model.

Scenarios:

- `wide`: roomy baseline; all three controllers reach the goal with defaults.
- `wide_s`: longer S-style route with open turns.
- `multi_room`: room-to-room route with several rectangular wall blocks.
- `rooms`: denser multi-room layout.
- `narrow_s`: narrow stress case retained from the first prototype.
- `clutter_s`: S-style route with extra interior clutter blocks.
- `clutter_rooms`: room-like route with cluttered interior partitions.
- `clutter_zigzag`: zig-zag route with staggered wall blocks.

The example opens the maze animation by default. For headless metric-only runs:

```bash
python3 examples/maze/density_feedback.py --no-plot
```

Runs report `status=success` only when the agent reaches the final target/goal
within `--goal-tolerance`; the summary prints both `dist_to_target` and
`dist_to_goal` so timeout animations are not mistaken for completed navigation.

All three density scripts use the same sparse delayed waypoint profile by
default: the controller holds the current waypoint until the robot is within
`--waypoint-switch-radius`. The default waypoint shortcut clearance is chosen
to avoid low-clearance diagonal corner cuts in the maze turns.

Every run also builds a nominal baseline trajectory from the planned path. It
is shown in green in the animation and can be used as the tracking reference
with `--reference-mode baseline`; direct path/waypoint tracking is selected
with `--reference-mode path`.

For waypoint and reference experiments, use `--smooth-path`, `--no-smooth-path`,
`--smooth-iterations`, `--smooth-clearance`, `--waypoint-spacing`,
`--waypoint-clearance`, `--preferred-clearance`, `--clearance-cost`,
`--baseline-lookahead`, and `--baseline-tracking-lookahead`.

Save animations:

```bash
python3 examples/maze/density_feedback.py --save-gif
python3 examples/maze/density_filter.py --save-gif
python3 examples/maze/density_mpc.py --save-gif
```

Save MP4s as well:

```bash
python3 examples/maze/density_feedback.py --save-gif --save-mp4
```

Animations are written to `examples/maze/animations/`. The generated title uses
the friendly maze label and controller name, for example `Maze 2 - Density MPC`.

Current animation set:

- 8 maze scenarios.
- 3 density controllers per scenario.
- 24 GIFs and 24 MP4s.

Current default-run status:

| Scenario | Feedback | Filter | MPC |
|---|---:|---:|---:|
| `wide` | success | success | success |
| `wide_s` | success | success | success |
| `multi_room` | success | success | success |
| `rooms` | success | collision | success |
| `narrow_s` | success | success | success |
| `clutter_s` | success | success | success |
| `clutter_rooms` | success | success | success |
| `clutter_zigzag` | success | success | success |

This is intentionally a prototype. The default `wide` map is a roomy maze with
space around turns. The `narrow_s` map is kept as a stress
case. The current density obstacle model uses greedily merged wall rectangles,
and each rectangle uses a smooth clearance bump. For rectangles,
`r1 = --robot-radius` and `r2 = --robot-radius + --transition-width`; the
default transition width is intentionally small so the density influence band
does not fill the two-cell corridors.

Most maze/controller pairs reach the goal with the current defaults. The
`rooms` map remains a useful failure case for the one-step density filter: its
animation is saved as a collision and is intentionally kept in the result set.
