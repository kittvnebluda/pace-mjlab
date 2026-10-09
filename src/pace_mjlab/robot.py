# Author: Grigorii Sizikov
# Licensed under the Apache License 2.0

"""Fixed-base Unitree Aliengo with PACE DC motor actuators, matching pace-sim2real's ``Isaac-Pace-Aliengo-v0``."""

import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from pace_mjlab import ASSETS_DIR
from pace_mjlab.actuator import PaceDcMotorEnvelopeCfg

ALIENGO_XML = ASSETS_DIR / "unitree_aliengo" / "aliengo.xml"
"""Byte-identical to pace-sim2real's bundled MJCF (and lorl_mjlab's)."""

JOINT_ORDER = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]

# Peak torque per group from the URDF [N m].
PEAK_TORQUE = {"hip": 35.278, "thigh": 35.278, "calf": 44.4}
# Motor constants at 27 V: the synthetic ground truth, fixed in stage 1 and fitted in stage 2.
C_OMEGA = {"hip": 1.15, "thigh": 0.80, "calf": 1.45}  # [V s/rad]
C_TAU = {"hip": 0.60, "thigh": 0.55, "calf": 0.25}  # [V/(N m)]


def get_spec() -> mujoco.MjSpec:
    """The Aliengo MJCF without its freejoint, so mjlab welds the trunk to the world."""
    spec = mujoco.MjSpec.from_file(str(ALIENGO_XML))
    spec.delete(spec.joint("floating_base"))
    return spec


# One actuator per group, since mjlab actuator configs take scalar limits; Kp 40 / Kd 2 as for the chirp.
ALIENGO_PACE_ACTUATORS = tuple(
    PaceDcMotorEnvelopeCfg(
        target_names_expr=(f".*_{group}_joint",),
        stiffness=40.0,
        damping=2.0,
        effort_limit=PEAK_TORQUE[group],
        v_bus=27.0,
        c_tau=C_TAU[group],
        c_omega=C_OMEGA[group],
        max_delay=10,
    )
    for group in ("hip", "thigh", "calf")
)

INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 1.0),
    joint_pos={
        ".*R_hip_joint": -0.1,
        ".*L_hip_joint": 0.1,
        "F[LR]_thigh_joint": 0.8,
        "R[LR]_thigh_joint": 1.0,
        ".*_calf_joint": -1.5,
    },
    joint_vel={".*": 0.0},
)


def get_aliengo_robot_cfg() -> EntityCfg:
    """A fresh fixed-base Aliengo config. The MJCF's collision geoms have contype 0, so there are no self-contacts."""
    return EntityCfg(
        init_state=INIT_STATE,
        spec_fn=get_spec,
        articulation=EntityArticulationInfoCfg(actuators=ALIENGO_PACE_ACTUATORS),
    )
