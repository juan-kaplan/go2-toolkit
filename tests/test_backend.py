"""Checks on the DDS path.

The pure tests run anywhere. The integration test needs a peer on the other end
and skips when there is none:

    uv run python tools/run_bridge.py --duration 120 &
    uv run pytest tests/test_backend.py

There is deliberately no reproducibility test. The peer free-runs against its
own clock, so identical inputs do not give identical trajectories and asserting
otherwise would only produce a flaky test. What is checked instead is that the
instrumentation which makes that visible actually works.
"""

from __future__ import annotations

import numpy as np
import pytest

from go2_toolkit import NUM_JOINTS, STAND_POSE, JointCommand

# -- pure: no peer required ---------------------------------------------------


@pytest.mark.parametrize("release_kd", [0.0, 2.5])
def test_release_zeroes_kp_and_applies_release_kd(release_kd: float) -> None:
    from go2_toolkit import DDSBackend

    # bypass __init__, which needs a peer; release() only reads release_kd and send()
    backend = object.__new__(DDSBackend)
    backend.release_kd = release_kd
    sent: list[JointCommand] = []
    backend.send = sent.append

    backend.release()

    (cmd,) = sent
    assert np.all(cmd.kp == 0.0)
    assert np.all(cmd.kd == release_kd)
    assert np.all(cmd.tau_ff == 0.0)


# -- integration: needs a running peer ----------------------------------------


@pytest.fixture
def backend():
    """A connected backend, or a skip if nothing is publishing LowState."""
    from go2_toolkit import DDSBackend

    try:
        connected = DDSBackend(state_timeout=4.0)
    except TimeoutError:
        pytest.skip("no peer on rt/lowstate; start tools/run_bridge.py")
    yield connected
    connected.release()


def test_reads_state_and_responds_to_commands(backend) -> None:
    state, _ = backend.read()
    assert state.q.shape == (NUM_JOINTS,)
    assert state.imu_quat.shape == (4,)

    # drive to a stand and confirm the robot actually moved there. The ramp
    # finishes at 60% of the run so the remainder is spent holding, which is
    # what makes the final position a settled reading rather than a moving one.
    backend.reset_timing()
    steps = int(2.5 / backend.dt)
    for i in range(steps):
        alpha = min(1.0, (i + 1) / (steps * 0.6))
        target = (1 - alpha) * state.q + alpha * STAND_POSE
        backend.send(JointCommand.position_hold(target))
        backend.step()

    final, truth = backend.read()
    # 0.35 rad, not tighter: the default gains are soft and the legs carry the
    # robot's weight, so a settled stand still sits measurably off its setpoint.
    assert np.abs(final.q - STAND_POSE).max() < 0.35, "did not reach the stand pose"
    assert truth is not None, "expected SportModeState pose from the simulator"
    assert truth.base_pos[2] > 0.20, f"robot is down at {truth.base_pos[2]:.3f} m"

    # the timing instrumentation is the safeguard against silent staleness,
    # so it has to report even when nothing went wrong
    assert "overran" in backend.timing_report(steps)
