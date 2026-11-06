"""robot/sport_mode.py, tested against stub clients.

Two behavioural rules worth pinning down. First: `status` and `restore` must
never wait for `LowState`. Before this module existed, `tools/release_sport_mode.py`
constructed a full `DDSBackend` (which blocks in `wait_for_state`) before
dispatching on the action, so `status`/`restore` paid for a wait they had no
use for, exactly the wrong place for an unrelated hang, since `restore` is
the "something is wrong, give control back" path. `connect_switcher` is what
the tool calls first now, and it must not touch `DDSBackend` at all.

Second: `refuse_if_sport_mode_active` must be a true no-op in simulation
(domain 1 has no sport mode) and must fail loudly on hardware when the
built-in controller still owns the robot, since a trial run against an
unreleased robot completes and reports numbers for a robot that never moved.
"""

from __future__ import annotations

import numpy as np
import pytest

from go2_toolkit.go2.conventions import JOINT_NAMES
from go2_toolkit.robot import sport_mode
from go2_toolkit.robot.interface import RobotState

IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


class _StubSwitcher:
    def __init__(self, mode: str = ""):
        self._mode = mode

    def SetTimeout(self, _):
        pass

    def Init(self):
        pass

    def CheckMode(self):
        return 0, {"name": self._mode}


def _fails_if_called(name: str):
    def _boom(*_a, **_k):
        raise AssertionError(f"{name} must not be constructed here")

    return _boom


def test_thighs_is_derived_not_hand_copied() -> None:
    """Regression guard: THIGHS must track JOINT_NAMES, not repeat it by hand."""
    expected = tuple(i for i, name in enumerate(JOINT_NAMES) if name.endswith("_thigh"))
    assert sport_mode.THIGHS == expected == (1, 4, 7, 10)


def test_connect_switcher_never_constructs_a_backend(monkeypatch) -> None:
    monkeypatch.setattr(sport_mode, "init_channel_factory", lambda *a: None)
    monkeypatch.setattr(sport_mode, "MotionSwitcherClient", _StubSwitcher)
    monkeypatch.setattr(sport_mode, "DDSBackend", _fails_if_called("DDSBackend"))

    msc = sport_mode.connect_switcher(domain_id=0, interface="eth0")
    assert isinstance(msc, _StubSwitcher)


def test_release_no_ops_without_touching_sport_client_or_backend(monkeypatch) -> None:
    """Already released: must return before any confirmation prompt or connection."""
    monkeypatch.setattr(sport_mode, "SportClient", _fails_if_called("SportClient"))
    monkeypatch.setattr(sport_mode, "DDSBackend", _fails_if_called("DDSBackend"))
    monkeypatch.setattr("builtins.input", _fails_if_called("input"))

    sport_mode.release(_StubSwitcher(mode=""), domain_id=0, interface="eth0")


@pytest.mark.parametrize("q_offset,expected", [(0.0, True), (-0.5, False)])
def test_thighs_lowered_checks_all_four(q_offset: float, expected: bool) -> None:
    q = np.zeros(12)
    q[list(sport_mode.THIGHS)] = sport_mode.LOWERED_THIGH
    q[sport_mode.THIGHS[0]] += q_offset  # nudge exactly one thigh

    class _Backend:
        def read(self):
            state = RobotState(
                q=q,
                dq=np.zeros(12),
                tau_est=np.zeros(12),
                imu_quat=IDENTITY_QUAT,
                imu_gyro=np.zeros(3),
                imu_accel=np.zeros(3),
            )
            return state, None

    assert sport_mode._thighs_lowered(_Backend()) is expected


# -- refuse_if_sport_mode_active: the "did anything actually move" guard ------


def test_sport_mode_check_is_a_no_op_in_simulation(monkeypatch) -> None:
    """Domain 1 has no sport mode; must not touch DDS at all to know that."""
    boom = _fails_if_called("connect_switcher")
    monkeypatch.setattr(sport_mode, "connect_switcher", boom)

    sport_mode.refuse_if_sport_mode_active(domain_id=1, interface="lo")


def test_sport_mode_check_passes_when_already_released(monkeypatch) -> None:
    monkeypatch.setattr(sport_mode, "connect_switcher", lambda *a: _StubSwitcher(mode=""))

    sport_mode.refuse_if_sport_mode_active(domain_id=0, interface="eth0")


def test_sport_mode_check_refuses_when_active(monkeypatch) -> None:
    monkeypatch.setattr(
        sport_mode, "connect_switcher", lambda *a: _StubSwitcher(mode="normal")
    )

    with pytest.raises(SystemExit, match="sport mode is active"):
        sport_mode.refuse_if_sport_mode_active(domain_id=0, interface="eth0")
