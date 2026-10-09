# ⚙️ PACE — Sim-to-Real Transfer for Legged Robots in mjlab

PACE joint and actuator identification on [mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp), ported
from [`pace-sim2real`](https://github.com/leggedrobotics/pace-sim2real) (Isaac Lab).

The algorithm here is modified with torque-speed curve identification:

- Stage 1 (original PACE): armature, viscous friction, Coulomb friction and encoder bias per joint, and one torque delay,
  fitted with CMA-ES on chirp data.
- Stage 2: motor constants $c_\tau$, $c_\omega$ per joint group (hip, thigh, knee) of torque-speed curve, fitted on step
  data with the stage-1 parameters frozen.

Robot: Unitree Aliengo.

## 📦 Installation

### 1. Install uv

An extremely fast Python package and project manager, written in Rust.

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone this repository

```sh
git clone https://github.com/kittvnebluda/pace-mjlab.git
cd pace-mjlab
```

### 3. Sync packages

```sh
make sync      # CUDA 13.0
# or
make sync-cpu  # CPU
```

## 🐾 Run

### 1. Collect excitation data

(Alternatively, place your own real-world data in data/)

```sh
uv run collect-data --excitation chirp           # data/aliengo_sim/chirp_data.pt
uv run collect-data --excitation step            # data/aliengo_sim/step_data.pt
```

### 2. Run PACE parameter fitting

```sh
uv run fit --stage 1                             # logs/pace/aliengo_sim/<time>/
uv run fit --stage 2 --stage1-run logs/pace/aliengo_sim/<time>
uv run show-ident [--stage 2] [--run <dir>]      # latest mean_*.pt of the newest (or given) run
```
