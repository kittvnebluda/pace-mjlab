# pace-mjlab

PACE joint and actuator identification on [mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp), ported
from `pace-sim2real` (Isaac Lab) so both stacks can fit each other's data.

- Stage 1 (PACE): armature, viscous friction, Coulomb friction and encoder bias per joint, and one torque delay,
  fitted with CMA-ES on chirp data.
- Stage 2: motor constants c_omega and c_tau per joint group (the back-EMF torque-speed envelope), fitted on step
  data with the stage-1 parameters frozen.

Robot: Unitree Aliengo, fixed base, the same MJCF as pace-sim2real (`src/pace_mjlab/assets/unitree_aliengo`).

## Setup

```sh
make sync
```

## Run

```sh
uv run pace-data-collection --excitation chirp   # data/aliengo_sim/chirp_data.pt
uv run pace-data-collection --excitation step    # data/aliengo_sim/step_data.pt
uv run pace-fit --stage 1                        # logs/pace/aliengo_sim/<time>/
uv run pace-fit --stage 2 --stage1-run logs/pace/aliengo_sim/<time>
```

Data files have the same keys and joint order as pace-sim2real, so either repo can fit either repo's data.
