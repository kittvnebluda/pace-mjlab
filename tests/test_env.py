import pytest
import torch

from pace_mjlab.actuator import PaceDcMotorEnvelope
from pace_mjlab.actuators import actuator_groups, read_delayed_effort, write_envelope
from pace_mjlab.env_cfg import action_columns, make_env
from pace_mjlab.robot import JOINT_ORDER

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

V_BUS = 27.0


@pytest.fixture(scope="module")
def env():
    env = make_env(num_envs=2, device="cuda:0")
    env.reset()
    yield env
    env.close()


def _reference_torque(q, qd, target, bias, kp, kd, c_tau, c_omega, peak):
    """PD on q - bias, clipped as pace-sim2real's PaceDCMotorEnvelope (Isaac Lab DCMotor)."""
    tau_stall, qd_max = V_BUS / c_tau, V_BUS / c_omega
    qd = qd.clamp(-qd_max * (1 + peak / tau_stall), qd_max * (1 + peak / tau_stall))
    upper = (tau_stall * (1 - qd / qd_max)).clamp(max=peak)
    lower = (tau_stall * (-1 - qd / qd_max)).clamp(min=-peak)
    return (kp * (target - (q - bias)) - kd * qd).clamp(lower, upper)


def test_fixed_base_and_custom_actuators(env):
    robot = env.scene["robot"]
    assert robot.is_fixed_base
    assert all(isinstance(a, PaceDcMotorEnvelope) for a in robot.actuators)
    # overriding compute keeps the PACE actuators out of mjlab's fused groups
    assert len(robot._custom_actuators) == len(robot.actuators) == 3
    assert sorted(robot.joint_names) == sorted(JOINT_ORDER)


def test_applied_torque_matches_reference(env):
    robot = env.scene["robot"]
    joint_ids = torch.tensor(robot.find_joints(JOINT_ORDER, preserve_order=True)[0], device=env.device)
    columns = action_columns(env, JOINT_ORDER)
    bias = torch.tensor([[0.05], [-0.03]], device=env.device).expand(2, len(JOINT_ORDER))
    c_tau = torch.tensor([[0.6], [0.3]], device=env.device).expand(2, len(JOINT_ORDER))
    c_omega = torch.tensor([[1.15], [1.4]], device=env.device).expand(2, len(JOINT_ORDER))
    write_envelope(robot, c_tau.contiguous(), c_omega.contiguous(), joint_ids)
    for actuator, cols in actuator_groups(robot, joint_ids):
        actuator.update_encoder_bias(bias[:, cols])
        actuator.update_time_lags(0)
        actuator.reset()
    target = robot.data.joint_pos[:, joint_ids] + 1.5  # large steps, so the envelope binds
    actions = torch.zeros(2, env.action_manager.total_action_dim, device=env.device)
    peak = torch.zeros(2, len(JOINT_ORDER), device=env.device)
    kp = torch.zeros_like(peak)
    kd = torch.zeros_like(peak)
    for actuator, cols in actuator_groups(robot, joint_ids):
        assert actuator.force_limit is not None and actuator.stiffness is not None and actuator.damping is not None
        peak[:, cols] = actuator.force_limit
        kp[:, cols] = actuator.stiffness
        kd[:, cols] = actuator.damping
    for _ in range(20):
        q, qd = robot.data.joint_pos[:, joint_ids].clone(), robot.data.joint_vel[:, joint_ids].clone()
        actions[:, columns] = target
        env.step(actions)
        expected = _reference_torque(q, qd, target, bias, kp, kd, c_tau, c_omega, peak)
        torch.testing.assert_close(read_delayed_effort(robot, joint_ids), expected, rtol=1e-5, atol=1e-4)
