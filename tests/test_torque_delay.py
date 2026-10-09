import pytest
import torch

from pace_mjlab.actuator import TorqueDelayBuffer


def _run(buffer: TorqueDelayBuffer, values: list[float]) -> list[float]:
    return [buffer.compute(torch.full((1, 1), v)).item() for v in values]


def test_delay_backfills_with_first_input_after_reset():
    buffer = TorqueDelayBuffer(max_delay=3, num_envs=1, num_joints=1, device="cpu")
    buffer.set_lags(2)
    assert _run(buffer, [1, 2, 3, 4, 5]) == [1, 1, 1, 2, 3]


def test_lags_survive_reset_and_history_restarts():
    buffer = TorqueDelayBuffer(max_delay=3, num_envs=1, num_joints=1, device="cpu")
    buffer.set_lags(2)
    _run(buffer, [1, 2, 3, 4])
    buffer.reset()
    assert _run(buffer, [10, 11, 12]) == [10, 10, 10]


def test_per_env_lags():
    buffer = TorqueDelayBuffer(max_delay=4, num_envs=2, num_joints=1, device="cpu")
    buffer.set_lags(torch.tensor([0, 3]))
    out = [buffer.compute(torch.tensor([[float(v)], [float(v)]])).squeeze(1).tolist() for v in range(1, 7)]
    assert [o[0] for o in out] == [1, 2, 3, 4, 5, 6]
    assert [o[1] for o in out] == [1, 1, 1, 1, 2, 3]


def test_lag_out_of_range():
    buffer = TorqueDelayBuffer(max_delay=3, num_envs=1, num_joints=1, device="cpu")
    with pytest.raises(ValueError):
        buffer.set_lags(4)
