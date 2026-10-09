# Author: Grigorii Sizikov
# Licensed under the Apache License 2.0

"""Collect synthetic PACE data (chirp or steps) with the ground-truth parameters, as pace-sim2real does."""

from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import torch
import tyro
from mjlab.entity import Entity

from pace_mjlab import PROJECT_ROOT
from pace_mjlab.actuators import (
    actuator_groups,
    read_clip_active,
    read_clip_at_effort_limit,
    read_delayed_effort,
    write_pd_gains,
)
from pace_mjlab.config import aliengo_pace_cfg
from pace_mjlab.env_cfg import action_columns, make_env
from pace_mjlab.excitation import chirp, steps
from pace_mjlab.joint_properties import write_joint_properties_to_sim, write_joint_state_to_sim


@dataclass
class Args:
    excitation: Literal["chirp", "step"] = "chirp"
    """Chirp (stage 1) or steps (stage 2)."""
    num_envs: int = 1
    min_frequency: float = 0.1
    """Minimum frequency of the chirp [Hz]."""
    max_frequency: float = 10.0
    """Maximum frequency of the chirp [Hz]."""
    duration: float = 20.0
    """Duration of the excitation signal [s]."""
    device: str = "cuda:0"


def main() -> None:
    args = tyro.cli(Args)
    cfg = aliengo_pace_cfg()
    env = make_env(args.num_envs, args.device)
    device = env.device
    robot: Entity = env.scene["robot"]

    joint_order = cfg.joint_order
    joint_ids = torch.tensor(robot.find_joints(joint_order, preserve_order=True)[0], device=device)
    columns = action_columns(env, joint_order)
    num_joints = len(joint_order)

    truth = cfg.synthetic
    ones = torch.ones((env.num_envs, num_joints), device=device)
    bias = truth.encoder_bias * ones
    env.reset()

    write_joint_properties_to_sim(
        env,
        robot,
        armature=truth.armature * ones,
        viscous_friction=truth.viscous_friction * ones,
        coulomb_friction=truth.coulomb_friction * ones,
        joint_ids=joint_ids,
    )
    for actuator, drive_joint_idx in actuator_groups(robot, joint_ids):
        actuator.update_time_lags(truth.delay)
        actuator.update_encoder_bias(bias[:, drive_joint_idx])
        actuator.reset()

    # PD gains of the recording, in joint_order; the steps use their own gains
    if args.excitation == "step":
        kp = torch.full((num_joints,), cfg.step.kp, device=device)
        kd = torch.full((num_joints,), cfg.step.kd, device=device)
        write_pd_gains(robot, kp, kd, joint_ids)
    kp = torch.zeros(num_joints, device=device)
    kd = torch.zeros(num_joints, device=device)
    for actuator, drive_joint_idx in actuator_groups(robot, joint_ids):
        assert actuator.stiffness is not None and actuator.damping is not None
        kp[drive_joint_idx] = actuator.stiffness[0]
        kd[drive_joint_idx] = actuator.damping[0]

    # Create the excitation signal for each joint, in joint_order
    sample_rate = 1 / env.physics_dt  # Hz
    num_steps = int(args.duration * sample_rate)
    t = torch.linspace(0, args.duration, steps=num_steps, device=device)
    if args.excitation == "chirp":
        signal = chirp(t, args.min_frequency, args.max_frequency, args.duration)
        shape = cfg.chirp
    else:
        signal = steps(t, cfg.step.hold, cfg.step.scales)
        shape = cfg.step
    center = torch.tensor(shape.center, device=device)
    amplitude = torch.tensor(shape.amplitude, device=device)
    trajectory = center.unsqueeze(0) + amplitude.unsqueeze(0) * signal.unsqueeze(-1)

    write_joint_state_to_sim(robot, trajectory[0].unsqueeze(0) + bias, joint_ids)

    dof_pos_buffer = torch.zeros(num_steps, num_joints, device=device)
    dof_target_pos_buffer = torch.zeros(num_steps, num_joints, device=device)
    dof_vel_buffer = torch.zeros(num_steps, num_joints, device=device)
    tau_buffer = torch.zeros(num_steps, num_joints, device=device)
    clip_active_buffer = torch.zeros(num_steps, num_joints, dtype=torch.bool, device=device)
    clip_at_limit_buffer = torch.zeros_like(clip_active_buffer)
    actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=device)
    with torch.inference_mode():
        for counter in range(num_steps):
            dof_pos_buffer[counter] = robot.data.joint_pos[0, joint_ids] - bias[0]
            dof_vel_buffer[counter] = robot.data.joint_vel[0, joint_ids]
            actions[:, columns] = trajectory[counter]
            env.step(actions)
            dof_target_pos_buffer[counter] = robot.data.joint_pos_target[0, joint_ids]
            # torque applied during this step (synthetic tau_est), and whether the envelope clipped the PD torque
            tau_buffer[counter] = read_delayed_effort(robot, joint_ids)[0]
            clip_active_buffer[counter] = read_clip_active(robot, joint_ids)[0]
            clip_at_limit_buffer[counter] = read_clip_at_effort_limit(robot, joint_ids)[0]
            if (counter + 1) % 400 == 0:
                print(f"[INFO]: Step {(counter + 1) / sample_rate} seconds")
    env.close()

    # envelope coverage per joint: the clip must stay inactive on chirps; on steps, only the samples on the
    # torque-speed line carry information about the motor constants, the ones at the effort limit do not
    on_line_buffer = clip_active_buffer & ~clip_at_limit_buffer
    print(f"[INFO]: {args.excitation}: envelope coverage per joint")
    header = f"{'joint':<16} {'peak |qd|':>9} {'corner':>7} {'on speed line':>14} {'|qd| on line':>13}"
    print(f"  {header} {'at effort limit':>16}")
    for actuator, drive_joint_idx in actuator_groups(robot, joint_ids):
        assert actuator.velocity_limit_motor is not None and actuator.saturation_effort is not None
        assert actuator.force_limit is not None
        corner = actuator.velocity_limit_motor[0] * (1 - actuator.force_limit[0] / actuator.saturation_effort[0])
        for j, col in enumerate(drive_joint_idx.tolist()):
            on_line = on_line_buffer[:, col]
            speeds = dof_vel_buffer[on_line, col].abs()
            speed_range = f"{speeds.min().item():.1f}-{speeds.max().item():.1f}" if on_line.any() else "-"
            at_limit = clip_at_limit_buffer[:, col]
            print(
                f"  {joint_order[col]:<16} {dof_vel_buffer[:, col].abs().max().item():9.2f} {corner[j].item():7.2f}"
                f" {int(on_line.sum()):6d} ({on_line.float().mean().item():5.1%}) {speed_range:>13}"
                f" {int(at_limit.sum()):7d} ({at_limit.float().mean().item():5.1%})"
            )
    print("  speeds in rad/s; corner: speed above which the torque-speed line is below the effort limit")

    data_dir = PROJECT_ROOT / "data" / cfg.robot_name
    data_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "time": t.cpu(),
            "dof_pos": dof_pos_buffer.cpu(),
            "des_dof_pos": dof_target_pos_buffer.cpu(),
            "dof_vel": dof_vel_buffer.cpu(),
            "tau_est": tau_buffer.cpu(),
            "clip_active": clip_active_buffer.cpu(),
            "clip_at_effort_limit": clip_at_limit_buffer.cpu(),
            "kp": kp.cpu(),
            "kd": kd.cpu(),
            "excitation": args.excitation,
            "joint_order": joint_order,
        },
        data_dir / f"{args.excitation}_data.pt",
    )
    print(f"[INFO]: Saved {data_dir / f'{args.excitation}_data.pt'}")

    for i in range(num_joints):
        plt.figure()
        plt.plot(t.cpu().numpy(), dof_pos_buffer[:, i].cpu().numpy(), label=f"{joint_order[i]} pos")
        plt.plot(
            t.cpu().numpy(),
            dof_target_pos_buffer[:, i].cpu().numpy(),
            label=f"{joint_order[i]} target",
            linestyle="dashed",
        )
        plt.title(f"Joint {joint_order[i]} Trajectory")
        plt.xlabel("Time [s]")
        plt.ylabel("Joint position [rad]")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
