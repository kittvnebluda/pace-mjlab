# Author: Grigorii Sizikov
# Licensed under the Apache License 2.0

"""Access to the PACE actuators of an entity, with tensors in PACE's ``joint_order``."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch

from pace_mjlab.actuator import PaceDcMotorEnvelope

if TYPE_CHECKING:
    from mjlab.entity import Entity


def actuator_groups(entity: Entity, joint_ids: torch.Tensor) -> Iterator[tuple[PaceDcMotorEnvelope, torch.Tensor]]:
    """Yield each PACE actuator with the ``joint_order`` column of each of its joints.

    ``joint_ids`` are the entity joint indices of ``joint_order``.

    Example:
        >>> joint_order = ["FR_hip", "FR_knee", "FL_hip", "FL_knee"]
        >>> joint_ids = torch.tensor([entity.joint_names.index(name) for name in joint_order])
        >>> joint_ids
        tensor([1, 3, 0, 2])
        >>> for actuator, columns in actuator_groups(entity, joint_ids):
        ...     print(actuator.target_names, columns)
        ['FL_hip', 'FR_hip'] tensor([2, 0])
        ['FL_knee', 'FR_knee'] tensor([3, 1])
    """
    for actuator in entity.actuators:
        if not isinstance(actuator, PaceDcMotorEnvelope):
            raise TypeError(f"PACE needs PaceDcMotorEnvelope actuators, got {type(actuator).__name__}")
        comparison_matrix = joint_ids.unsqueeze(1) == actuator.target_ids.unsqueeze(0)
        yield actuator, torch.argmax(comparison_matrix.int(), dim=0)


def write_pd_gains(entity: Entity, kp: torch.Tensor, kd: torch.Tensor, joint_ids: torch.Tensor) -> None:
    """Set the actuator PD gains, one value per joint in ``joint_order``, for all envs."""
    for actuator, columns in actuator_groups(entity, joint_ids):
        assert actuator.stiffness is not None and actuator.damping is not None
        actuator.stiffness[:] = kp[columns]
        actuator.damping[:] = kd[columns]


def write_envelope(entity: Entity, c_tau: torch.Tensor, c_omega: torch.Tensor, joint_ids: torch.Tensor) -> None:
    """Set the motor constants, shape (num_envs, len(joint_order))."""
    for actuator, columns in actuator_groups(entity, joint_ids):
        actuator.update_envelope(c_tau[:, columns], c_omega[:, columns])


def read_delayed_effort(entity: Entity, joint_ids: torch.Tensor) -> torch.Tensor:
    """Torque applied to the joints in the last step (after clip and delay), shape (num_envs, len(joint_order))."""
    effort = torch.zeros((entity.data.joint_pos.shape[0], joint_ids.shape[0]), device=joint_ids.device)
    for actuator, columns in actuator_groups(entity, joint_ids):
        assert actuator.delayed_effort is not None
        effort[:, columns] = actuator.delayed_effort
    return effort


def read_clip_active(entity: Entity, joint_ids: torch.Tensor) -> torch.Tensor:
    """Whether the actuator clip changed the PD torque in the last step, shape (num_envs, len(joint_order))."""
    active = torch.zeros(
        (entity.data.joint_pos.shape[0], joint_ids.shape[0]), dtype=torch.bool, device=joint_ids.device
    )
    for actuator, columns in actuator_groups(entity, joint_ids):
        active[:, columns] = actuator.computed_effort != actuator.applied_effort
    return active


def read_clip_at_effort_limit(entity: Entity, joint_ids: torch.Tensor) -> torch.Tensor:
    """Whether the actuator clip held the torque at its effort limit in the last step.

    With :func:`read_clip_active`, this splits the clipped samples into those at the flat effort limit and those on
    the torque-speed line. Shape (num_envs, len(joint_order)).
    """
    at_limit = torch.zeros(
        (entity.data.joint_pos.shape[0], joint_ids.shape[0]), dtype=torch.bool, device=joint_ids.device
    )
    for actuator, columns in actuator_groups(entity, joint_ids):
        assert actuator.applied_effort is not None and actuator.computed_effort is not None
        clipped = actuator.computed_effort != actuator.applied_effort
        at_limit[:, columns] = clipped & (actuator.applied_effort.abs() == actuator.force_limit)
    return at_limit
