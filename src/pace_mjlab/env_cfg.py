# Author: Grigorii Sizikov
# Licensed under the Apache License 2.0

"""Minimal manager-based env for PACE: joint position targets in, no rewards, terminations or observations.

``ManagerBasedRlEnv.step`` keeps the same order as Isaac Lab's: action -> actuator -> physics step -> state.
"""

import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg

from pace_mjlab.joint_properties import JOINT_PROPERTY_FIELDS
from pace_mjlab.robot import get_aliengo_robot_cfg


def make_sim_cfg() -> SimulationCfg:
    """MuJoCo Warp settings of pace-sim2real's Newton backend (``NewtonCfg(MJWarpSolverCfg(integrator="implicitfast"))``)."""
    return SimulationCfg(
        mujoco=MujocoCfg(
            timestep=0.002,  # 500 Hz, the MJCF timestep and the robot's hardware loop
            integrator="implicitfast",
            solver="newton",
            iterations=100,
            tolerance=1e-6,  # Newton's default; mjlab's is 1e-8
            ls_iterations=50,
            ccd_iterations=35,  # Newton's default; mjlab's is 50 (unused, no contacts)
            cone="pyramidal",
            impratio=1.0,
        ),
    )


def make_env_cfg(num_envs: int) -> ManagerBasedRlEnvCfg:
    return ManagerBasedRlEnvCfg(
        decimation=1,
        scene=SceneCfg(num_envs=num_envs, env_spacing=2.5, entities={"robot": get_aliengo_robot_cfg()}),
        # actions = absolute joint position targets
        actions={
            "joint_pos": JointPositionActionCfg(
                entity_name="robot", actuator_names=(".*",), scale=1.0, use_default_offset=False
            )
        },
        sim=make_sim_cfg(),
        episode_length_s=99999.0,  # long episodes
    )


def make_env(num_envs: int, device: str) -> ManagerBasedRlEnv:
    """The PACE env, with per-env armature, damping and friction loss."""
    env = ManagerBasedRlEnv(cfg=make_env_cfg(num_envs), device=device)
    env.sim.expand_model_fields(JOINT_PROPERTY_FIELDS)
    return env


def action_columns(env: ManagerBasedRlEnv, joint_order: list[str]) -> torch.Tensor:
    """Action column of each joint in ``joint_order``: ``actions[:, columns] = targets`` sets the targets."""
    term = env.action_manager.get_term("joint_pos")
    assert isinstance(term, JointPositionAction)
    names = list(term.target_names)
    return torch.tensor([names.index(name) for name in joint_order], device=env.device)
