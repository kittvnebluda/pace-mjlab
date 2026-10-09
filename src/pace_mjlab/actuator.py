# Author: Grigorii Sizikov
# Licensed under the Apache License 2.0

"""PACE DC motor actuator for mjlab: encoder bias, back-EMF torque-speed envelope and torque delay.

Same model, order of operations and delay semantics as pace-sim2real's ``PaceDCMotorEnvelope`` (Isaac Lab):

1. PD on the encoder-frame position ``q - encoder_bias``.
2. Clip to the torque-speed envelope at the current joint velocity, with saturation_effort = V / c_tau and
   velocity_limit = V / c_omega, further bounded by the peak torque ``effort_limit``.
3. Delay the clipped torque by a fixed number of physics steps per env.

mjlab's own delay (``delay_*_lag``) acts on the commands before the PD, a different model, so it must stay off.
mjlab's ``EntityData.encoder_bias`` (``q + b``, observations only) is not used.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjwarp
import torch
from mjlab.actuator.actuator import ActuatorCmd
from mjlab.actuator.dc_actuator import DcMotorActuator, DcMotorActuatorCfg, dc_motor_clip
from mjlab.actuator.pd_actuator import pd_torque

if TYPE_CHECKING:
    from mjlab.entity import Entity


class TorqueDelayBuffer:
    """Fixed per-env delay with Isaac Lab's ``DelayBuffer`` semantics.

    The output is the input from ``lag`` steps ago. After a reset, the first input fills the whole history, so
    the output holds that first input until ``lag`` newer inputs have arrived. Lags survive resets.
    """

    def __init__(self, max_delay: int, num_envs: int, num_joints: int, device: str):
        self.max_delay = max_delay
        self._history = torch.zeros(max_delay + 1, num_envs, num_joints, device=device)
        self._lags = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._fresh = torch.ones(num_envs, dtype=torch.bool, device=device)
        self._env_ids = torch.arange(num_envs, device=device)
        self._head = 0

    @property
    def lags(self) -> torch.Tensor:
        return self._lags

    def set_lags(self, lags: int | torch.Tensor, env_ids: torch.Tensor | slice | None = None) -> None:
        lags = torch.as_tensor(lags, device=self._lags.device).to(torch.long)
        if lags.min() < 0 or lags.max() > self.max_delay:
            raise ValueError(f"lags must be in [0, {self.max_delay}], got {lags.min().item()}..{lags.max().item()}")
        self._lags[slice(None) if env_ids is None else env_ids] = lags

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._fresh[slice(None) if env_ids is None else env_ids] = True

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        self._history[self._head] = data
        self._history[:] = torch.where(self._fresh.view(1, -1, 1), data.unsqueeze(0), self._history)
        self._fresh[:] = False
        delayed = self._history[(self._head - self._lags) % self._history.shape[0], self._env_ids]
        self._head = (self._head + 1) % self._history.shape[0]
        return delayed


@dataclass(kw_only=True)
class PaceDcMotorEnvelopeCfg(DcMotorActuatorCfg):
    """PACE DC motor with the envelope set by the motor constants.

    ``saturation_effort`` and ``velocity_limit`` are derived from ``v_bus``, ``c_tau`` and ``c_omega`` and must not
    be set; ``effort_limit`` is the peak torque.
    """

    saturation_effort: float = 0.0  # derived: v_bus / c_tau
    velocity_limit: float = 0.0  # derived: v_bus / c_omega
    v_bus: float = 27.0
    """Bus voltage [V]."""
    c_tau: float
    """Torque constant at the joint [V/(N m)]."""
    c_omega: float
    """Back-EMF constant at the joint [V s/rad]."""
    encoder_bias: float = 0.0
    """Encoder bias [rad]; the controller sees q - encoder_bias."""
    max_delay: int = 0
    """Maximum torque delay [physics steps]."""

    def __post_init__(self) -> None:
        self.saturation_effort = self.v_bus / self.c_tau
        self.velocity_limit = self.v_bus / self.c_omega
        super().__post_init__()
        if self.delay_max_lag > 0:
            raise ValueError("PACE delays the torque (max_delay); mjlab's command delay (delay_max_lag) must be 0.")

    def build(self, entity: Entity, target_ids: list[int], target_names: list[str]) -> PaceDcMotorEnvelope:
        return PaceDcMotorEnvelope(self, entity, target_ids, target_names)


class PaceDcMotorEnvelope(DcMotorActuator[PaceDcMotorEnvelopeCfg]):
    """PACE DC motor: PD on the biased position, envelope clip at the current velocity, then torque delay.

    Overriding ``compute`` keeps it out of mjlab's fused actuator groups. Per step it keeps the PD torque
    (``computed_effort``), the clipped torque (``applied_effort``) and the torque that reaches the joint
    (``delayed_effort``).
    """

    def __init__(self, cfg: PaceDcMotorEnvelopeCfg, entity: Entity, target_ids: list[int], target_names: list[str]):
        super().__init__(cfg, entity, target_ids, target_names)
        self.encoder_bias: torch.Tensor | None = None
        self.computed_effort: torch.Tensor | None = None
        self.applied_effort: torch.Tensor | None = None
        self.delayed_effort: torch.Tensor | None = None
        self._torque_delay: TorqueDelayBuffer | None = None

    def initialize(self, mj_model: mujoco.MjModel, model: mjwarp.Model, data: mjwarp.Data, device: str) -> None:
        super().initialize(mj_model, model, data, device)
        shape = (data.nworld, len(self._target_names))
        self.encoder_bias = torch.full(shape, self.cfg.encoder_bias, dtype=torch.float, device=device)
        self.computed_effort = torch.zeros(shape, device=device)
        self.applied_effort = torch.zeros(shape, device=device)
        self.delayed_effort = torch.zeros(shape, device=device)
        self._torque_delay = TorqueDelayBuffer(self.cfg.max_delay, *shape, device=device)

    def update_encoder_bias(self, encoder_bias: torch.Tensor) -> None:
        """Set the encoder bias, shape (num_envs, num_joints)."""
        assert self.encoder_bias is not None
        self.encoder_bias[:] = encoder_bias

    def update_time_lags(self, delay: int | torch.Tensor, env_ids: torch.Tensor | slice | None = None) -> None:
        """Set the torque delay in physics steps, an int or shape (num_envs,)."""
        assert self._torque_delay is not None
        self._torque_delay.set_lags(delay, env_ids)

    def update_envelope(self, c_tau: torch.Tensor, c_omega: torch.Tensor) -> None:
        """Set the motor constants, shape (num_envs, num_joints)."""
        assert self.saturation_effort is not None and self.velocity_limit_motor is not None
        self.saturation_effort[:] = self.cfg.v_bus / c_tau
        self.velocity_limit_motor[:] = self.cfg.v_bus / c_omega

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        if self._torque_delay is not None:
            self._torque_delay.reset(env_ids)

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        assert self.stiffness is not None and self.damping is not None and self.force_limit is not None
        assert self.saturation_effort is not None and self.velocity_limit_motor is not None
        assert self._torque_delay is not None and self.encoder_bias is not None
        # the controller works in the encoder frame
        cmd = dataclasses.replace(cmd, pos=cmd.pos - self.encoder_bias)
        self.computed_effort = pd_torque(self.stiffness, self.damping, cmd)
        self.applied_effort = dc_motor_clip(
            self.computed_effort, self.saturation_effort, self.velocity_limit_motor, self.force_limit, cmd.vel
        )
        self.delayed_effort = self._torque_delay.compute(self.applied_effort)
        return self.delayed_effort
