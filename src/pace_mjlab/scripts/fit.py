# Author: Grigorii Sizikov
# Licensed under the Apache License 2.0

"""Fit PACE parameters with CMA-ES: stage 1 on chirp data, stage 2 (motor constants) on step data."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
import tyro
from mjlab.entity import Entity

from pace_mjlab import PROJECT_ROOT
from pace_mjlab.actuators import read_delayed_effort, write_pd_gains
from pace_mjlab.config import CMAESOptimizerCfg, aliengo_pace_cfg
from pace_mjlab.env_cfg import action_columns, make_env
from pace_mjlab.optim import CMAESOptimizer
from pace_mjlab.optim.parameterization import EnvelopeParams, PaceJointParams


@dataclass
class Args:
    stage: Literal[1, 2] = 1
    """1: PACE parameters on chirp data, 2: motor constants on step data."""
    stage1_run: Path | None = None
    """Stage 2: stage-1 run directory whose latest mean_*.pt is frozen."""
    num_envs: int = 4096
    """CMA-ES population size, one candidate per env."""
    data: Path | None = None
    """Data file to fit; defaults to data/aliengo_sim/chirp_data.pt (stage 1) or step_data.pt (stage 2)."""
    cmaes: CMAESOptimizerCfg = field(default_factory=CMAESOptimizerCfg)
    device: str = "cuda:0"


def main() -> None:
    args = tyro.cli(Args)
    cfg = aliengo_pace_cfg()
    env = make_env(args.num_envs, args.device)
    device = env.device
    robot: Entity = env.scene["robot"]

    joint_order = cfg.joint_order
    sim_joint_ids = torch.tensor(robot.find_joints(joint_order, preserve_order=True)[0], device=device)
    columns = action_columns(env, joint_order)

    if args.stage == 1:
        params = PaceJointParams(cfg.bounds_params.to(device), joint_order)
        data_file = PROJECT_ROOT / "data" / cfg.data_dir
        log_dir = PROJECT_ROOT / "logs" / "pace" / cfg.robot_name
        tau_weight = 0.0
    else:
        if args.stage1_run is None:
            raise ValueError("Stage 2 needs --stage1-run.")
        means = sorted(args.stage1_run.glob("mean_*.pt"), key=lambda p: int(re.findall(r"\d+", p.stem)[0]))
        print(f"[INFO]: Stage 1 parameters frozen from {means[-1]}")
        params = EnvelopeParams(
            cfg.envelope.bounds_params.to(device),
            cfg.envelope.groups,
            joint_order,
            torch.load(means[-1]).float(),
        )
        data_file = PROJECT_ROOT / "data" / cfg.envelope.data_dir
        log_dir = PROJECT_ROOT / "logs" / "pace" / cfg.robot_name / "stage2"
        tau_weight = cfg.envelope.lam
    if args.data is not None:
        data_file = args.data
    print(f"[INFO]: Fitting {data_file}")

    data = torch.load(data_file)
    if list(data.get("joint_order", joint_order)) != joint_order:
        raise ValueError(f"{data_file} has joint order {data['joint_order']}, expected {joint_order}")
    time_data = data["time"].to(device)
    target_dof_pos = data["des_dof_pos"].to(device)
    measured_dof_pos = data["dof_pos"].to(device)
    # the torque term of the loss (stage 2) needs the torque estimate
    measured_tau = data["tau_est"].to(device) if tau_weight else None
    # PD gains used during the recording, if stored (older chirp files keep the actuator config gains)
    if "kp" in data:
        write_pd_gains(robot, data["kp"].to(device), data["kd"].to(device), sim_joint_ids)

    initial_dof_pos = measured_dof_pos[0, :].unsqueeze(0).repeat(env.num_envs, 1)

    time_steps = time_data.shape[0]
    sim_dt = env.physics_dt

    opt = CMAESOptimizer(
        params=params,
        population_size=env.num_envs,
        log_dir=log_dir,
        joint_order=joint_order,
        max_iteration=args.cmaes.max_iteration,
        data=data,
        device=device,
        epsilon=args.cmaes.epsilon,
        sigma=args.cmaes.sigma,
        save_interval=args.cmaes.save_interval,
        save_optimization_process=args.cmaes.save_optimization_process,
        tau_weight=tau_weight,
    )

    env.reset()
    opt.update_simulator(env, robot, sim_joint_ids, initial_dof_pos)

    actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=device)
    counter = 0
    while True:
        with torch.inference_mode():
            opt.tell(
                robot.data.joint_pos[:, sim_joint_ids],
                measured_dof_pos[counter, :].unsqueeze(0).repeat(env.num_envs, 1),
            )
            actions[:, columns] = target_dof_pos[counter, :]
            env.step(actions)
            if measured_tau is not None:
                opt.tell_torque(read_delayed_effort(robot, sim_joint_ids), measured_tau[counter, :].unsqueeze(0))
            counter += 1
            if counter % 400 == 0:
                print(
                    f"[INFO]: Step {counter * sim_dt:.1f} / {time_data[-1]:.1f} seconds"
                    f" ({counter / time_steps * 100:.1f} %)"
                )
            if counter >= time_steps:
                print("[INFO]: Reached the end of the trajectory, exiting.")
                counter = 0
                opt.evolve()
                if opt.finished():
                    break
                env.reset()
                opt.update_simulator(env, robot, sim_joint_ids, initial_dof_pos)
    opt.close()
    env.close()


if __name__ == "__main__":
    main()
