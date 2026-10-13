# Experiments

## Plan: PACE on Aliengo, Isaac Lab (Newton/MuJoCo Warp) vs mjlab

2×2 cross-fit with known ground truth: generate synthetic data in each stack, fit each dataset in both.

| | fit in Isaac Lab / Newton (here) | fit in mjlab (new repo) |
| ------------------- | -------------------------------- | ----------------------- |
| data from Isaac Lab | E1, E2, E4: all recovered, stage 1 < 2e-5, stage 2 < 7e-7 | M2: all recovered, stage 1 < 4e-6, stage 2 < 6e-7 |
| data from mjlab | M3: all recovered, stage 1 < 2e-5, stage 2 < 6e-7 | M1: all recovered, stage 1 < 5e-6, stage 2 < 1e-6 |

Diagonal: does each fitter recover known parameters? Off-diagonal: how far apart are the two stacks?

## E1: Aliengo, synthetic data and fit on Newton/MuJoCo Warp

**Setup**

- Branch `aliengo-mjwarp` (`7b9a7d5` + true delay 7, uncommitted), task `Isaac-Pace-Aliengo-v0`, `physics=newton_mjwarp` (integrator `implicitfast`).
- Model: bundled `lorl_mjlab` Aliengo MJCF, fixed base, `dt = 0.002` (500 Hz).
- Actuator: `PaceIdealPDActuator`, Kp 40, Kd 2, effort limit 44.4 Nm, `max_delay` 10.
- Data: 20 s linear chirp 0.1→10 Hz, per joint hip 0 ± 0.3, thigh 0.8 ± 0.5, calf −1.5 ± 0.6 rad; `data/aliengo_sim/chirp_data.pt` (10000 steps).
- Ground truth vs CMA-ES initial guess (bounds midpoint):

  | Parameter        | Bounds      | Initial guess | Truth |
  | ---------------- | ----------- | ------------- | ----- |
  | Armature         | 1e-4 – 0.1  | 0.050         | 0.02  |
  | Viscous friction | 0 – 2       | 1.0           | 0.5   |
  | Coulomb friction | 0 – 1       | 0.5           | 0.2   |
  | Encoder bias     | ±0.1        | 0.0           | 0.05  |
  | Delay (steps)    | 0 – 10      | 5.0           | 7     |

- Fit: CMA-ES, population = 4096 envs, `sigma` 0.5, `max_iteration` 200, `epsilon` 1e-2.
- Run: `logs/pace/aliengo_sim/26_10_09_02-30-08/`, started 2026-10-09 02:30; ~21 s per iteration; CPU package ~98–101 °C.

**Iteration 70** (`mean_070.pt`), best score 9.66e-6 (iteration 0: 5.06e-2)

| Parameter        | Truth | Fitted mean | Min    | Max    | Worst joint error |
| ---------------- | ----- | ----------- | ------ | ------ | ----------------- |
| Armature         | 0.020 | 0.0200      | 0.0200 | 0.0200 | 0.1%              |
| Viscous friction | 0.500 | 0.4996      | 0.4974 | 0.5007 | 0.5%              |
| Coulomb friction | 0.200 | 0.2030      | 0.1959 | 0.2114 | 5.7%              |
| Encoder bias     | 0.050 | 0.0474      | 0.0408 | 0.0504 | 18.3%             |
| Delay (steps)    | 7     | 7.5177      | –      | –      | exact (truncates to 7) |

Delay moved from the initial 5 to the true 7. Bias is still converging at iteration 70.

**Final result** (`mean_199.pt`): stopped at the 200-iteration cap after ~71 min; best score 1.63e-13

| Parameter        | Truth | Fitted mean | Min    | Max    | Worst joint error |
| ---------------- | ----- | ----------- | ------ | ------ | ----------------- |
| Armature         | 0.020 | 0.0200      | 0.0200 | 0.0200 | < 0.05%           |
| Viscous friction | 0.500 | 0.5000      | 0.5000 | 0.5000 | < 0.05%           |
| Coulomb friction | 0.200 | 0.2000      | 0.2000 | 0.2000 | < 0.05%           |
| Encoder bias     | 0.050 | 0.0500      | 0.0500 | 0.0500 | < 0.05%           |
| Delay (steps)    | 7     | 7.5126      | –      | –      | exact (truncates to 7) |

All parameters recovered on all 12 joints. The bias spread seen at iteration 70 (0.041–0.050) closed by the end.

The `epsilon` stop never triggers on noise-free synthetic data: it needs (max − min)/min < 1% over the population, and the best score goes to ~1e-13. Runs on synthetic data always hit `max_iteration`.

## E2: Aliengo, two-stage fit with the back-EMF envelope (Newton/MuJoCo Warp)

**Setup**

- Branch `aliengo-mjwarp` at `97e57a3`, task `Isaac-Pace-Aliengo-v0`, `physics=newton_mjwarp`, `dt = 0.002`.
- Actuator: `PaceDCMotorEnvelope`. PD → envelope clip with q̇(t) → delay, V_bus = 27 V, τ_peak hip/thigh 35.278 N·m, knee 44.4 N·m (URDF).
- Envelope truth (c_ω [V·s/rad] / c_τ [V/(N·m)]), fixed in data collection and in stage 1:

  | Group | Bounds c_ω | Start c_ω | Truth c_ω | Bounds c_τ | Start c_τ | Truth c_τ | → q̇_max / τ_stall | Corner speed |
  | ----- | ---------- | --------- | --------- | ---------- | --------- | --------- | ----------------- | ------------ |
  | hip   | 0.60–1.35  | 0.975     | 1.15      | 0.09–0.77  | 0.43      | 0.60      | 23.5 rad/s / 45 N·m  | 5.1 rad/s  |
  | thigh | 0.60–1.35  | 0.975     | 0.80      | 0.09–0.77  | 0.43      | 0.55      | 33.8 rad/s / 49 N·m  | 9.5 rad/s  |
  | knee  | 0.75–1.70  | 1.225     | 1.45      | 0.07–0.61  | 0.34      | 0.25      | 18.6 rad/s / 108 N·m | 11.0 rad/s |

- PACE truth as in E1: armature 0.02, viscous 0.5, Coulomb 0.2, bias 0.05, delay 7 steps (initial guess 5).
- Data (20 s, 10000 samples each):
  - `data/aliengo_sim/chirp_data.pt`: chirp 0.1→10 Hz, Kp 40 / Kd 2; hip 0 ± 0.3, thigh 0.8 ± 0.3, calf −1.5 ± 0.6 rad. Clip active on 0% of samples for every joint. With thigh ± 0.5 (as in E1), 9–11% of thigh samples were clipped, so the thigh amplitude was reduced. E1's chirp file is backed up in the session scratchpad.
  - `data/aliengo_sim/step_data.pt`: steps held 0.5 s, scales 1.0/0.4/0.8/0.6 with alternating sign, Kp 60 / Kd 2; hip 0 ± 0.5, thigh 0.8 ± 0.6, calf −1.5 ± 0.6 rad.

    | Group | Peak \|q̇\| [rad/s] | Corner speed [rad/s] | Clip-active samples |
    | ----- | ------------------- | -------------------- | ------------------- |
    | hip | 15.6–19.4 | 5.1 | 3.5–3.8% |
    | thigh | 25.3–26.1 | 9.5 | 7.6–8.3% |
    | knee | 27.3–29.0 | 11.0 | 4.8–5.6% |

- Stage 1: PACE loss on chirp data, 4096 envs, 200 iterations.
- Stage 2: J2 = 1/N Σ‖q − q_sim‖² + λ/N Σ‖τ_est − τ_sim‖², λ = 1, on step data, stage-1 mean frozen, 256 envs, 200 iterations.
- Runs: stage 1 `logs/pace/aliengo_sim/26_10_09_10-22-32/` (~71 min); stage 2 `logs/pace/aliengo_sim/stage2/26_10_09_11-35-36/`.

**Stage 1 result** (`mean_199.pt`, 200-iteration cap, best score 1.33e-13): all recovered on all 12 joints. Armature, viscous, Coulomb and bias have < 0.05% error; delay 7.5466 truncates to 7 (exact).

**Stage 2 result**: stopped by the `epsilon` rule after 17 of 200 iterations, best J2 3.50e-3. J2 at the truth is 2.4e-9, so this is not the optimum.

| Group | Truth c_ω / c_τ | Fitted c_ω / c_τ | Recovered |
| ----- | --------------- | ---------------- | --------- |
| hip   | 1.15 / 0.60     | 1.0703 / 0.2499  | no        |
| thigh | 0.80 / 0.55     | 1.0108 / 0.3490  | no        |
| knee  | 1.45 / 0.25     | 1.4502 / 0.2502  | yes       |

Diagnosis. Samples where the truth envelope binds on the speed line rather than at ±τ_peak, counted offline on the step data (PD torque from the recorded states, all 12 joints):

| Group | Speed-line samples per joint     | Speeds        | At ±τ_peak |
| ----- | -------------------------------- | ------------- | ---------- |
| hip   | FL, RL 18 each; FR, RR 0         | 5.6–6.9 rad/s | 353–360    |
| thigh | 0 on every leg                   | –             | 764–833    |
| knee  | 173–235                          | 15.6–25.6 rad/s | 307–325  |

- **thigh: not identifiable from this step data.** The line never binds, and J2 is flat at the noise floor (~1e-9) over a wide region (sweep: τ_stall 50–82 N·m, q̇_max 25–30 rad/s). Above the 9.5 rad/s corner the D term subtracts 19–52 N·m (Kd 2 × q̇). The closest the PD torque gets to the line is 27.2 N·m against 34.9 N·m, at 9.8 rad/s (FL thigh). All thigh clip-active samples are the flat ±τ_peak limit.
- **hip: weakly identifiable.** 36 line samples, from FL and RL only, at 5.6–6.9 rad/s just above the 5.07 rad/s corner. The sweep minimum is at the truth (grid argmin 1.1625 / 0.600), but over a narrow speed band c_ω and c_τ can trade off (e.g. 0.975 / 0.634 has J2 2.1e-4). The E2 fit stopped early on a plateau. At c_τ 0.25 (τ_stall 108 N·m) the line does not bind at those 36 samples, J2 is flat around it (3.5e-3, almost all from the hip joints), and the population spread fell below the 1% `epsilon`.
- **knee: identifiable**, with a sharp minimum at the truth (grid argmin 1.4625 / 0.259). It was recovered in E2.
- The early stop is the `epsilon` rule: it ends the run when the population's relative score spread is below 1%. That is wrong for stage 2, where a plateau with a nonzero score is common.
- Float differences between MuJoCo Warp worlds with identical parameters are ~1e-8 N·m (J2 at the truth 1.1e-9 to 2.4e-9), far below these effects.
- In every group the torque term is > 99.98% of J2 at the argmin.

## E3: stage 2 on steps with Kd 0.5 (Newton/MuJoCo Warp)

**Setup**: as E2 stage 2, but the steps use Kd 0.5 instead of 2 (`aliengo_pace_env_cfg.py`, uncommitted), so the PD torque reaches the torque-speed line on every group. Stage 1 frozen from E2 (`logs/pace/aliengo_sim/26_10_09_10-22-32/mean_199.pt`). 4096 envs, `epsilon` stop on. Run `logs/pace/aliengo_sim/stage2/26_10_09_12-21-19/`, ~23 s per iteration.

Step-data coverage (samples on the speed line per joint): hip 375–490 (5.1–27 rad/s), thigh 637–665 (9.5–27 rad/s), knee 347–448 (11–26 rad/s). With Kd 2 (E2) these were hip 0–18, thigh 0, knee 173–235.

**Result** (`mean_199.pt`, 200-iteration cap; the `epsilon` stop did not trigger, best J2 8.4e-9): all 6 motor constants recovered.

| Group | Truth c_ω / c_τ | Fitted c_ω / c_τ | Max relative error |
| ----- | --------------- | ---------------- | ------------------ |
| hip   | 1.15 / 0.60     | 1.150000 / 0.600000 | 2.3e-7          |
| thigh | 0.80 / 0.55     | 0.800000 / 0.550000 | 3.6e-7          |
| knee  | 1.45 / 0.25     | 1.450000 / 0.250000 | 4.8e-7          |

Caveat: these steps overshoot the hip and calf joint limits (see Stack parity below). That is consistent within one stack, so the sim-to-sim result holds, but the cross-fits need steps that stay inside the limits.

## M1: mjlab data, mjlab fit, stage 1

**Setup**: `pace-mjlab` at `ccb4053`, mjlab 1.6.0, the same truth, bounds, chirp, actuator and solver settings as E2 stage 1. Data `pace-mjlab/data/aliengo_sim/chirp_data.pt`, which matches E2's chirp to 4.8e-7 rad. 4096 envs, 200 iterations, `epsilon` stop on. Run `pace-mjlab/logs/pace/aliengo_sim/26_10_09_13-56-29/`, ~22 s per iteration (CPU-bound), ~75 min.

**Result** (`mean_199.pt`, 200-iteration cap, best score 6.5e-14): all recovered on all 12 joints.

| Parameter        | Truth | Max relative error, mjlab (M1) | Same, Isaac (E2) |
| ---------------- | ----- | ------------------------------ | ---------------- |
| Armature         | 0.02  | 1.6e-7                         | 5.8e-7           |
| Viscous friction | 0.5   | 2.4e-7                         | 7.2e-7           |
| Coulomb friction | 0.2   | 3.0e-6                         | 6.8e-6           |
| Encoder bias     | 0.05  | 4.4e-6                         | 1.7e-5           |
| Delay (steps)    | 7     | 7.5141 → 7 (exact)             | 7.5466 → 7       |

The score trace follows E2's closely (iteration 1: 0.037470 in both; iteration 107: 1.0e-8 vs 9.6e-9).

## M1: mjlab data, mjlab fit, stage 2

**Setup**: `pace-mjlab` at `252c661`. Steps inside the joint limits (hip center −0.2, calf center −1.7 / amplitude 0.5, thigh 0.8 ± 0.6; Kp 60 / Kd 0.5), the same truth and bounds as E3, MJCF with the URDF thigh range. Stage 1 frozen from M1 stage 1 (`26_10_09_13-56-29/mean_199.pt`). 4096 envs, `epsilon` stop on. Run `pace-mjlab/logs/pace/aliengo_sim/stage2/26_10_09_15-32-25/`, ~23 s per iteration, ~77 min. First stage-2 fit on steps that stay inside the limits, in either stack.

**Result** (`mean_199.pt`, 200-iteration cap; best J2 6.4e-9, at that floor from iteration ~30): all 6 motor constants recovered.

| Group | Truth c_ω / c_τ | Fitted c_ω / c_τ    | Max relative error |
| ----- | --------------- | ------------------- | ------------------ |
| hip   | 1.15 / 0.60     | 1.149999 / 0.600001 | 8.3e-7             |
| thigh | 0.80 / 0.55     | 0.800000 / 0.550000 | 2.4e-7             |
| knee  | 1.45 / 0.25     | 1.450000 / 0.250000 | 2.4e-7             |

## E4: Isaac Lab stage 2 on the steps inside the joint limits

**Setup**: `pace-sim2real` at `7dd6780`. As E3, but on the steps inside the joint limits (hip center −0.2, calf −1.7 / 0.5, thigh 0.8 ± 0.6; Kp 60 / Kd 0.5) and the MJCF with the URDF thigh range, so it is the Isaac Lab counterpart of M1 stage 2. Stage 1 frozen from E2 (`26_10_09_10-22-32/mean_199.pt`). 4096 envs, `epsilon` stop on. Run `logs/pace/aliengo_sim/stage2/26_10_09_21-34-45/`, ~24 s per iteration, ~80 min.

**Result** (`mean_199.pt`, 200-iteration cap, best J2 9.0e-9): all 6 motor constants recovered.

| Group | Truth c_ω / c_τ | Fitted c_ω / c_τ    | Max relative error |
| ----- | --------------- | ------------------- | ------------------ |
| hip   | 1.15 / 0.60     | 1.150000 / 0.600000 | 6.4e-7             |
| thigh | 0.80 / 0.55     | 0.800000 / 0.550000 | 5.1e-7             |
| knee  | 1.45 / 0.25     | 1.450000 / 0.250000 | 2.4e-7             |

## M2 and M3: cross-fits (data from one stack, fit in the other)

**Setup**: `pace-sim2real` at `7dd6780`, `pace-mjlab` at `252c661`. The same truth, bounds, chirp and steps (inside the joint limits) as M1. Each stage 2 freezes its own stage 1. M2 = Isaac Lab data fitted by mjlab (`pace-fit --data`), M3 = mjlab data fitted by Isaac Lab (Hydra overrides `env.sim2real.data_dir` and `env.sim2real.envelope.data_dir`). 4096 envs, `epsilon` stop on; all four runs went to the 200-iteration cap. Two runs at a time on the one GPU: ~38–43 s per iteration each, against ~29 s alone, so ~1.45× the throughput.

Runs:

- M2 stage 1: `pace-mjlab/logs/pace/aliengo_sim/26_10_09_17-11-17/` (best score 7.0e-14)
- M2 stage 2: `pace-mjlab/logs/pace/aliengo_sim/stage2/26_10_09_18-59-34/` (best J2 6.6e-9)
- M3 stage 1: `pace-sim2real/logs/pace/aliengo_sim/26_10_09_17-14-29/` (best score 2.1e-12 at iteration 155)
- M3 stage 2: `pace-sim2real/logs/pace/aliengo_sim/stage2/26_10_09_19-25-24/` (best J2 8.9e-9)

**Stage 1** (max relative error over the 12 joints, `mean_199.pt`):

| Parameter        | M2 (Isaac data → mjlab) | M3 (mjlab data → Isaac) |
| ---------------- | ----------------------- | ----------------------- |
| Armature         | 3.0e-7                  | 4.4e-7                  |
| Viscous friction | 2.4e-7                  | 3.6e-7                  |
| Coulomb friction | 3.0e-6                  | 5.9e-6                  |
| Encoder bias     | 4.0e-6                  | 1.5e-5                  |
| Delay (steps)    | 7.5168 → 7 (exact)      | 7.4514 → 7 (exact)      |

**Stage 2** (relative error, `mean_199.pt`):

| Constant      | Truth | M2      | M3      |
| ------------- | ----- | ------- | ------- |
| c_ω hip       | 1.15  | 8.3e-8  | 2.3e-7  |
| c_ω thigh     | 0.80  | 3.6e-7  | 5.4e-7  |
| c_ω knee      | 1.45  | 1.3e-7  | 2.0e-7  |
| c_τ hip       | 0.60  | 4.0e-8  | 5.6e-7  |
| c_τ thigh     | 0.55  | 5.6e-7  | 5.2e-7  |
| c_τ knee      | 0.25  | 2.4e-7  | 2.4e-7  |

**Conclusion**: with the same model and settings, each stack recovers the other's parameters as well as its own. This follows from the recordings matching (5e-7 rad on chirps, 3e-6 rad on in-limit steps). Each fitter does equally well on either stack's data (bias error 1.7e-5 / 1.5e-5 for Isaac on its own / mjlab data, 4.4e-6 / 4.0e-6 for mjlab). For sim-to-sim the two stacks are interchangeable; the open differences are outside the PACE model (joint-limit softening).

## Stack parity: same truth, data recorded in both stacks

`pace-mjlab` (mjlab 1.6.0) against this repo (Newton), both MuJoCo Warp 3.11.0, `dt = 0.002`, `implicitfast`, Newton solver, 100 / 50 iterations, tolerance 1e-6, pyramidal cone. Same MJCF, truth, excitation and actuator model.

- **Chirp (Kp 40 / Kd 2):** identical up to float noise over all 20 s. Joint position RMS difference 2e-8 to 8e-8 rad (max 4.8e-7), torque RMS difference 1e-6 to 4e-6 N·m, on all 12 joints.
- **Steps (Kp 60 / Kd 0.5):** identical (< 1e-4 rad) until 2.09 s, then they diverge (position RMS 0.02–0.13 rad, torque RMS 1–7 N·m). The cause is the joint limits. With Kd 0.5 the steps overshoot: the FR hip reaches 1.413 rad against its ±1.222 rad limit, and the calf reaches −0.592 against −0.646. The divergence starts as the hip crosses 1.2217 rad. Joint limits are soft constraints, and the two stacks set their parameters differently (Newton rescales the limit solref from `dof_invweight0`), so the trajectories split there. E3's step data has the same overshoot.
- **Steps inside the limits (hip center −0.2, calf center −1.7 / amplitude 0.5, thigh unchanged; Kp 60 / Kd 0.5):** identical in both stacks, max position difference 2.7e-6 rad, max torque difference 8.4e-4 N·m. Margins to the URDF limits (hip ±1.222, thigh −2.094..4.189, calf −2.775..−0.646): hip ≥ 0.098 rad, thigh ≥ 1.71 rad, calf ≥ 0.174 rad. Speed-line samples per joint: hip 384–476 (5.1–25.5 rad/s), thigh 734–747 (9.5–31.1 rad/s), knee 391–420 (12.1–26.3 rad/s). The overshooting E3 step data is backed up in the session scratchpad.
- **Thigh limits added to the MJCF** (URDF −120°..240°, both repos; previously the thigh was unlimited): chirp and step recordings are bit-identical before and after in both stacks (the thigh never comes near its limits), so E1–E3 and M1 stand. Both simulators report the new limit (Isaac Lab reconverted the MJCF). Data collection now prints each joint's range against the URDF limits and warns on a violation; the current steps pass.

## Before each synthetic run

- [ ] Ground truth is clearly different from the CMA-ES initial guess. The initial guess is the midpoint of `bounds_params` (hard-coded in `CMAESOptimizer.__init__`); a parameter whose truth sits at the midpoint is only shown to be kept, not found.
- [ ] For delay, also differ by at least one whole step: the fitted value is truncated to an integer, so the whole interval [d, d+1) applies the same lag.
- [ ] Envelope stages: the data-collection report shows 0% clip-active samples on chirps and clearly more than 0% on steps.
