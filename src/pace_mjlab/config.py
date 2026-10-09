# Author: Grigorii Sizikov
# Licensed under the Apache License 2.0

"""PACE settings for the Aliengo, the same values as pace-sim2real's ``Isaac-Pace-Aliengo-v0``."""

import math
from dataclasses import dataclass, field

import torch

from pace_mjlab.robot import JOINT_ORDER


@dataclass
class CMAESOptimizerCfg:
    max_iteration: int = 200
    epsilon: float | None = 1e-2
    sigma: float = 0.5
    save_interval: int = 10
    # consume more disk space if True, saves optimization process after finishing
    save_optimization_process: bool = False


@dataclass
class ChirpCfg:
    """Excitation trajectory per joint (in ``joint_order``): target = center + amplitude * chirp, chirp in [-1, 1]."""

    center: list[float]  # [rad]
    amplitude: list[float]  # [rad], sign sets the direction


@dataclass
class StepCfg:
    """Step excitation per joint (in ``joint_order``), for identifying the torque-speed envelope.

    Every ``hold`` seconds the target jumps to center + sign * scale * amplitude; the sign alternates and the scale
    cycles through ``scales``, so the steps have several sizes and reach several speeds.
    """

    center: list[float]  # [rad]
    amplitude: list[float]  # [rad]
    scales: list[float] = field(default_factory=lambda: [1.0, 0.4, 0.8, 0.6])
    hold: float = 0.5  # [s]
    kp: float = 60.0  # P gain during the steps [N m/rad]
    kd: float = 2.0  # D gain during the steps [N m s/rad]


@dataclass
class EnvelopeCfg:
    """Stage 2: motor constants per joint group, fitted on step data with the PACE parameters frozen.

    Loss: J2 = 1/N sum_t ||q_t - q_t^sim||^2 + lam/N sum_t ||tau_t^est - tau_t^sim||^2.
    """

    data_dir: str  # step data, relative to data/
    groups: dict[str, str]  # group name -> joint-name regex
    bounds_params: torch.Tensor  # (2 * groups, 2): c_omega per group [V s/rad], then c_tau per group [V/(N m)]
    lam: float = 1.0  # weight of the torque term [rad^2/(N m)^2]


@dataclass
class SyntheticParamsCfg:
    """Ground-truth parameters for synthetic data collection, applied to every joint."""

    armature: float = 0.1  # [kg m^2]
    viscous_friction: float = 4.5  # [N m s/rad]
    coulomb_friction: float = 0.05  # [N m]
    encoder_bias: float = 0.05  # [rad]
    delay: int = 5  # [sim steps]


@dataclass
class PaceCfg:
    robot_name: str
    data_dir: str  # chirp data, relative to data/
    joint_order: list[str]
    bounds_params: torch.Tensor  # (4 * joints + 1, 2)
    chirp: ChirpCfg
    step: StepCfg
    envelope: EnvelopeCfg
    joint_limits: dict[str, tuple[float, float]]  # hardware position limits per joint regex [rad]
    synthetic: SyntheticParamsCfg = field(default_factory=SyntheticParamsCfg)
    cmaes: CMAESOptimizerCfg = field(default_factory=CMAESOptimizerCfg)


def aliengo_pace_cfg() -> PaceCfg:
    bounds = torch.zeros((49, 2))  # 12 + 12 + 12 + 12 + 1 = 49 parameters to optimize
    bounds[:12, 0] = 1e-4
    bounds[:12, 1] = 0.1  # armature between 1e-4 - 0.1 [kg m^2]
    bounds[12:24, 1] = 2.0  # dof_damping between 0.0 - 2.0 [N m s/rad]
    bounds[24:36, 1] = 1.0  # friction between 0.0 - 1.0 [N m]
    bounds[36:48, 0] = -0.1
    bounds[36:48, 1] = 0.1  # bias between -0.1 - 0.1 [rad]
    bounds[48, 1] = 10.0  # delay between 0.0 - 10.0 [sim steps]
    return PaceCfg(
        robot_name="aliengo_sim",
        data_dir="aliengo_sim/chirp_data.pt",
        joint_order=list(JOINT_ORDER),
        bounds_params=bounds,
        # position limits from the URDF (Unitree aliengo const.xacro), the same values as the bundled MJCF
        joint_limits={
            ".*_hip_joint": (math.radians(-70.0), math.radians(70.0)),
            ".*_thigh_joint": (math.radians(-120.0), math.radians(240.0)),
            ".*_calf_joint": (math.radians(-159.0), math.radians(-37.0)),
        },
        # hip +-0.3, thigh 0.5..1.1, calf -2.1..-0.9 rad: inside the MJCF limits (hip +-1.22, calf -2.78..-0.65);
        # thigh amplitude 0.3, not 0.5: at 0.5 the 10 Hz end clipped ~10% of thigh samples on the envelope
        chirp=ChirpCfg(center=[0.0, 0.8, -1.5] * 4, amplitude=[0.3, 0.3, 0.6] * 4),
        # Kd 0.5, not 2: with Kd 2 the D term kept the thigh PD torque off the torque-speed line. With Kd 0.5
        # the joints overshoot far past the targets, so the centers keep the swings inside the URDF limits
        # (hip +-1.222, thigh -2.094..4.189, calf -2.775..-0.646): hip -0.2 balances the larger positive steps,
        # calf -1.7 is the middle of its range
        step=StepCfg(center=[-0.2, 0.8, -1.7] * 4, amplitude=[0.5, 0.6, 0.5] * 4, kd=0.5),
        envelope=EnvelopeCfg(
            data_dir="aliengo_sim/step_data.pt",
            groups={"hip": ".*_hip_joint", "thigh": ".*_thigh_joint", "knee": ".*_calf_joint"},
            bounds_params=torch.tensor(
                [
                    [0.60, 1.35],  # c_omega hip [V s/rad]
                    [0.60, 1.35],  # c_omega thigh
                    [0.75, 1.70],  # c_omega knee
                    [0.09, 0.77],  # c_tau hip [V/(N m)]
                    [0.09, 0.77],  # c_tau thigh
                    [0.07, 0.61],  # c_tau knee
                ]
            ),
        ),
        # ground truth for synthetic data, different from the MJCF defaults (armature 0.01, frictionloss 0.1)
        synthetic=SyntheticParamsCfg(
            armature=0.02,
            viscous_friction=0.5,
            coulomb_friction=0.2,
            encoder_bias=0.05,
            delay=7,  # bounds midpoint (CMA-ES initial guess) is 5
        ),
    )
