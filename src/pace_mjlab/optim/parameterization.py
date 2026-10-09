# Author: Grigorii Sizikov
# Licensed under the Apache License 2.0

"""Parameter layouts searched by :class:`CMAESOptimizer`, one per identification stage.

A layout owns the bounds, writes a population of physical parameter values to the simulator,
reports the encoder bias used in the position loss, and logs its parameters.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

from pace_mjlab.actuators import actuator_groups, write_envelope
from pace_mjlab.joint_properties import write_joint_properties_to_sim, write_joint_state_to_sim

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv
    from torch.utils.tensorboard import SummaryWriter


class PaceJointParams:
    """Stage 1, PACE: armature, viscous friction, Coulomb friction and encoder bias per joint, and one delay."""

    def __init__(self, bounds: torch.Tensor | None, joint_order: list[str]):
        self.bounds = bounds
        self.joint_order = joint_order
        num_joints = len(joint_order)
        self.armature_idx = slice(0, num_joints)
        self.damping_idx = slice(num_joints, 2 * num_joints)
        self.friction_idx = slice(2 * num_joints, 3 * num_joints)
        self.bias_idx = slice(3 * num_joints, 4 * num_joints)
        self.delay_idx = 4 * num_joints

    def encoder_bias(self, sim_params: torch.Tensor) -> torch.Tensor:
        return sim_params[:, self.bias_idx]

    def apply(
        self,
        env: ManagerBasedRlEnv,
        entity: Entity,
        joint_ids: torch.Tensor,
        initial_position: torch.Tensor,
        sim_params: torch.Tensor,
    ):
        env_ids = torch.arange(len(sim_params[:, self.armature_idx]), device=sim_params.device)
        write_joint_properties_to_sim(
            env,
            entity,
            armature=sim_params[:, self.armature_idx],
            viscous_friction=sim_params[:, self.damping_idx],
            coulomb_friction=sim_params[:, self.friction_idx],
            joint_ids=joint_ids,
        )
        write_joint_state_to_sim(entity, initial_position + sim_params[:, self.bias_idx], joint_ids)
        for actuator, drive_joint_idx in actuator_groups(entity, joint_ids):
            actuator.update_encoder_bias(sim_params[:, self.bias_idx][:, drive_joint_idx])
            actuator.update_time_lags(sim_params[:, self.delay_idx].to(torch.int))
            actuator.reset(env_ids)

    def print_best(self, sim_params: torch.Tensor, index: int):
        print("Armature: ", sim_params[index, self.armature_idx].tolist())
        print("Viscous Friction: ", sim_params[index, self.damping_idx].tolist())
        print("Static/Dynamic Friction: ", sim_params[index, self.friction_idx].tolist())
        print("Bias: ", sim_params[index, self.bias_idx].tolist())
        print("Delay: ", sim_params[index, self.delay_idx].tolist())

    def log(self, writer: SummaryWriter, sim_params: torch.Tensor, best_index: int, iteration: int):
        for i in range(len(self.joint_order)):
            writer.add_histogram(
                "4_Bias/distribution_" + self.joint_order[i], sim_params[:, self.bias_idx][:, i], iteration
            )
            writer.add_histogram(
                "3_Static_Dynamic_Friction/distribution_" + self.joint_order[i],
                sim_params[:, self.friction_idx][:, i],
                iteration,
            )
            writer.add_histogram(
                "2_Viscous_Friction/distribution_" + self.joint_order[i],
                sim_params[:, self.damping_idx][:, i],
                iteration,
            )
            writer.add_histogram(
                "1_Armature/distribution_" + self.joint_order[i], sim_params[:, self.armature_idx][:, i], iteration
            )

            writer.add_scalar(
                "4_Bias/best_" + self.joint_order[i], sim_params[best_index, self.bias_idx][i].item(), iteration
            )
            writer.add_scalar(
                "3_Static_Dynamic_Friction/best_" + self.joint_order[i],
                sim_params[best_index, self.friction_idx][i].item(),
                iteration,
            )
            writer.add_scalar(
                "2_Viscous_Friction/best_" + self.joint_order[i],
                sim_params[best_index, self.damping_idx][i].item(),
                iteration,
            )
            writer.add_scalar(
                "1_Armature/best_" + self.joint_order[i], sim_params[best_index, self.armature_idx][i].item(), iteration
            )
        writer.add_histogram("0_Delay/distribution", sim_params[:, self.delay_idx], iteration)
        writer.add_scalar("0_Delay/best", sim_params[best_index, self.delay_idx].item(), iteration)


class EnvelopeParams:
    """Stage 2: back-EMF constant c_omega and torque constant c_tau per joint group, PACE parameters frozen.

    Layout: c_omega for each group, then c_tau for each group, in the order of ``groups``.
    """

    def __init__(
        self, bounds: torch.Tensor, groups: dict[str, str], joint_order: list[str], stage1_params: torch.Tensor
    ):
        """``groups`` maps a group name to a joint-name regex; ``stage1_params`` are stage 1's fitted values (4n+1)."""
        if bounds.shape[0] != 2 * len(groups):
            raise ValueError(
                f"bounds must have {2 * len(groups)} rows (c_omega, then c_tau per group), got {bounds.shape[0]}"
            )
        self.bounds = bounds
        self.groups = list(groups)
        self.joint_group = torch.tensor(
            [_group_of(name, groups) for name in joint_order], dtype=torch.long, device=bounds.device
        )
        self.omega_idx = slice(0, len(groups))
        self.tau_idx = slice(len(groups), 2 * len(groups))
        self.stage1 = PaceJointParams(None, joint_order)
        self.stage1_params = stage1_params.to(bounds.device)

    def encoder_bias(self, sim_params: torch.Tensor) -> torch.Tensor:
        return self.stage1.encoder_bias(self.stage1_params.unsqueeze(0)).expand(sim_params.shape[0], -1)

    def apply(
        self,
        env: ManagerBasedRlEnv,
        entity: Entity,
        joint_ids: torch.Tensor,
        initial_position: torch.Tensor,
        sim_params: torch.Tensor,
    ):
        frozen = self.stage1_params.unsqueeze(0).repeat(sim_params.shape[0], 1)
        self.stage1.apply(env, entity, joint_ids, initial_position, frozen)
        c_omega = sim_params[:, self.omega_idx][:, self.joint_group]
        c_tau = sim_params[:, self.tau_idx][:, self.joint_group]
        write_envelope(entity, c_tau=c_tau, c_omega=c_omega, joint_ids=joint_ids)

    def print_best(self, sim_params: torch.Tensor, index: int):
        for g, name in enumerate(self.groups):
            c_omega = sim_params[index, self.omega_idx][g].item()
            c_tau = sim_params[index, self.tau_idx][g].item()
            print(f"{name}: c_omega {c_omega:.4f} V s/rad, c_tau {c_tau:.4f} V/(N m)")

    def log(self, writer: SummaryWriter, sim_params: torch.Tensor, best_index: int, iteration: int):
        for g, name in enumerate(self.groups):
            writer.add_histogram("1_c_omega/distribution_" + name, sim_params[:, self.omega_idx][:, g], iteration)
            writer.add_histogram("2_c_tau/distribution_" + name, sim_params[:, self.tau_idx][:, g], iteration)
            writer.add_scalar("1_c_omega/best_" + name, sim_params[best_index, self.omega_idx][g].item(), iteration)
            writer.add_scalar("2_c_tau/best_" + name, sim_params[best_index, self.tau_idx][g].item(), iteration)


def _group_of(joint_name: str, groups: dict[str, str]) -> int:
    matches = [g for g, pattern in enumerate(groups.values()) if re.fullmatch(pattern, joint_name)]
    if len(matches) != 1:
        raise ValueError(f"joint {joint_name} must match exactly one envelope group, matches {matches}")
    return matches[0]
