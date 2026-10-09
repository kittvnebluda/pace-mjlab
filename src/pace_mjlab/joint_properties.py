# Author: Grigorii Sizikov
# Licensed under the Apache License 2.0

"""Per-env joint properties and joint state, the mjlab counterpart of pace-sim2real's ``joint_properties.py``.

MuJoCo's ``dof_armature``, ``dof_damping`` and ``dof_frictionloss`` are the same fields Isaac Lab's Newton backend
writes to (``joint_armature``, ``joint_viscous_friction_coeff``, ``joint_friction_coeff``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.event_manager import RecomputeLevel

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv

JOINT_PROPERTY_FIELDS = ("dof_armature", "dof_damping", "dof_frictionloss")
"""Model fields that need one value per env (``Simulation.expand_model_fields``)."""


def write_joint_properties_to_sim(
    env: ManagerBasedRlEnv,
    entity: Entity,
    armature: torch.Tensor,
    viscous_friction: torch.Tensor,
    coulomb_friction: torch.Tensor,
    joint_ids: torch.Tensor,
) -> None:
    """Write armature, viscous friction and Coulomb friction (friction loss) for all envs.

    Values have shape (num_envs, len(joint_ids)); ``joint_ids`` are entity joint indices. Armature changes the
    constants MuJoCo derives at qpos0, so they are recomputed (``set_const_0``), as Newton does.
    """
    model = env.sim.model
    envs = torch.arange(env.num_envs, device=env.device).unsqueeze(1)
    dofs = entity.indexing.joint_v_adr[joint_ids].unsqueeze(0)
    model.dof_armature[envs, dofs] = armature
    model.dof_damping[envs, dofs] = viscous_friction
    model.dof_frictionloss[envs, dofs] = coulomb_friction
    env.sim.recompute_constants(RecomputeLevel.set_const_0)


def write_joint_state_to_sim(entity: Entity, position: torch.Tensor, joint_ids: torch.Tensor | None) -> None:
    """Write joint positions and zero joint velocities for all envs."""
    entity.write_joint_state_to_sim(position, torch.zeros_like(position), joint_ids=joint_ids)
