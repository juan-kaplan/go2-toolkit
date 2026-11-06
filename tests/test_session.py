"""The trial sequence, driven against a stub backend so no simulator is needed.

What is pinned down here is the design of a trial (L2):

* the session owns the stride amplitude, and the command function is called exactly
  once per control tick and never before the loop to find a target, so a stateful
  phase source cannot be stepped twice;
* the stand is the gait's own zero-amplitude pose, and every stage starts from where
  the last one left the legs, so nothing in the whole trial steps a joint;
* the stand carries the gait's own gains and gravity load, so none of them steps at the
  join with the loop;
* every tick of every stage is recorded, and a trial log says what was run;
* every exit is designed: a normal end and the first Ctrl+C run the end sequence,
  and the second Ctrl+C, a tilt and a dead LowState release at once.

The rest is arithmetic on a recorded trial: the IMU site offset, the projection of
displacement onto the body axis, and the report's column layout.
"""

from __future__ import annotations

import json
import re
import signal

import numpy as np
import pytest

from go2_toolkit import control
from go2_toolkit.go2.command import JointCommand
from go2_toolkit.go2.model import IMU_SITE_OFFSET, LYING_POSE, STAND_POSE
from go2_toolkit.robot import session as sess
from go2_toolkit.robot import trial_log
from go2_toolkit.robot.bridge import SCENE_PATH
from go2_toolkit.robot.interface import GroundTruth, RobotState

IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
# a stand that is not the keyframe, as a gait's own stand is not
GAIT_STAND = STAND_POSE + np.array([0.0, -0.1, 0.2] * 4)


class _StubBackend:
    """Records what it was sent. Reports the commanded pose back as measured."""

    def __init__(self, **kw):
        self.kw = kw
        self.sent: list[np.ndarray] = []
        self.sent_cmds: list[JointCommand] = []
        self._last = STAND_POSE.copy()
        self.height = 0.32
        self.overruns = 0
        self.released_after: int | None = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.released_after = len(self.sent)
        return False

    def send(self, cmd):
        self._last = np.asarray(cmd.q, dtype=float).copy()
        self.sent.append(self._last)
        self.sent_cmds.append(cmd)

    def step(self):
        pass

    def state_age(self):
        return 0.0

    def read(self):
        state = RobotState(
            q=self._last.copy(),
            dq=np.zeros(12),
            tau_est=np.zeros(12),
            imu_quat=IDENTITY_QUAT,
            imu_gyro=np.zeros(3),
            imu_accel=np.zeros(3),
        )
        truth = GroundTruth(
            base_pos=np.array([0.0, 0.0, self.height]), base_vel=np.zeros(3)
        )
        return state, truth

    def reset_timing(self):
        pass

    def timing_report(self, steps):
        return f"{0}/{steps} steps overran (0.0%), worst 0.00 ms"


class _NoTruth(_StubBackend):
    def read(self):
        state, _ = super().read()
        return state, None


class _IntermittentTruth(_StubBackend):
    def __init__(self, **kw):
        super().__init__(**kw)
        self._reads = 0

    def read(self):
        state, truth = super().read()
        self._reads += 1
        return state, truth if (self._reads % 2 == 1) else None


@pytest.fixture
def install(monkeypatch):
    """Put a backend class in place of the real one; returns where it is kept."""
    holder: dict = {}

    def _install(cls=_StubBackend, **attrs):
        def factory(**kw):
            holder["backend"] = backend = cls(**kw)
            for name, value in attrs.items():
                setattr(backend, name, value)
            return backend

        monkeypatch.setattr(sess, "DDSBackend", factory)
        return holder

    monkeypatch.setattr(sess, "check_single_peer", lambda: None)
    monkeypatch.setattr(sess, "reset_bridge", lambda: True)
    return _install


@pytest.fixture
def stub(install) -> dict:
    return install()


def _setup(**kw) -> sess.TrialSetup:
    base = dict(
        rate=100.0,
        stand_ramp=0.0,
        settle=0.1,
        ramp=0.1,
        hold=0.05,
        lower=0.1,
        reset=False,
    )
    base.update(kw)
    return sess.TrialSetup(**base)


def _tilted(deg: float) -> np.ndarray:
    """Scalar first quaternion for a roll of `deg` about body x."""
    half = np.radians(deg) / 2
    return np.array([np.cos(half), np.sin(half), 0.0, 0.0])


def _stand_at(q):
    """A command function that stands in `q` whatever the amplitude."""
    return lambda k, t, scale, scale_dot: q


def _swinging(k, t, scale, scale_dot):
    """A command that moves with the amplitude and with the step, like a gait."""
    return GAIT_STAND + scale * 0.3 + 0.01 * np.sin(k)


# -- the start -----------------------------------------------------------------


def test_the_session_owns_the_stride_ramp(stub: dict) -> None:
    seen = []

    def command(k, t, scale, scale_dot):
        seen.append((k, t, scale, scale_dot))
        return GAIT_STAND

    sess.run_trial(command, steps=15, setup=_setup(end_sequence=False), stand=GAIT_STAND)
    # a 0.1 s ramp at 100 Hz: ten steps up, then full amplitude and no rate
    assert [s[2] for s in seen[:11]] == pytest.approx([k / 10 for k in range(11)])
    assert all(s[3] == pytest.approx(10.0) for s in seen[:10])
    assert all(s[2] == 1.0 and s[3] == 0.0 for s in seen[11:])
    # and it is the same ramp the clock path warms up with
    for _, t, scale, scale_dot in seen:
        assert scale == pytest.approx(control.warmup_scale(t, 0.1))
        assert scale_dot == pytest.approx(control.warmup_scale_rate(t, 0.1))


def test_the_command_is_called_once_per_tick_and_never_before_the_loop(
    stub: dict,
) -> None:
    """A network behind the command would be stepped twice by a lookup before the loop.

    The old session called `command(0, 0.0)` to find its ramp target, and again as the
    loop's first step. The stand is the gait's own zero-amplitude pose now, so it needs
    no such call.
    """
    seen = []

    def command(k, t, scale, scale_dot):
        seen.append((k, t, len(stub["backend"].sent)))
        return STAND_POSE

    sess.run_trial(
        command, steps=4, setup=_setup(rate=100.0, end_sequence=False), stand=STAND_POSE
    )
    assert [(k, t) for k, t, _ in seen] == [(0, 0.0), (1, 0.01), (2, 0.02), (3, 0.03)]
    # the first call comes after the whole settle, and each call sends exactly one command
    assert [sent for _, _, sent in seen] == [10, 11, 12, 13]


def test_the_end_sequence_keeps_counting_ticks_for_the_command(stub: dict) -> None:
    seen = []

    def command(k, t, scale, scale_dot):
        seen.append(k)
        return GAIT_STAND

    sess.run_trial(command, steps=20, setup=_setup(), stand=GAIT_STAND)
    assert seen == list(range(len(seen))), "a tick was skipped or repeated"
    # 20 loop steps, a 0.1 s wind down from full amplitude, a 0.05 s hold
    assert len(seen) == 20 + 10 + 5


def test_scipy_is_loaded_before_any_command_is_sent(stub: dict, monkeypatch) -> None:
    """The first rotation imports scipy, and a pause between commands moves the robot.

    The backend is built before anything is sent, so "before the backend exists" is
    before every command.

    Measured in the simulator: a 270 ms gap between the ramp and the loop made the
    body jump 2 cm. Warming the import up front puts that cost where no commands flow.
    """
    events: list[str] = []
    monkeypatch.setattr(
        sess,
        "rotation_matrices",
        lambda q: events.append("late" if "backend" in stub else "warm"),
    )
    monkeypatch.setattr(sess, "body_forward_axis", lambda q: np.array([1.0, 0, 0]))
    result = sess.run_trial(_stand_at(STAND_POSE), steps=2, setup=_setup())
    assert events == ["warm"]
    assert result.steps == 2


def test_the_stand_comes_first_and_is_the_one_the_driver_gave(stub: dict) -> None:
    sess.run_trial(_swinging, steps=2, setup=_setup(), stand=GAIT_STAND)
    sent = stub["backend"].sent
    assert all(np.array_equal(q, GAIT_STAND) for q in sent[:10])


def test_the_stand_defaults_to_the_model_keyframe(stub: dict) -> None:
    sess.run_trial(_stand_at(STAND_POSE), steps=2, setup=_setup())
    assert np.array_equal(stub["backend"].sent[0], STAND_POSE)


def test_nothing_in_a_whole_trial_steps_a_joint(stub: dict) -> None:
    """The start, the loop, the wind down and the lowering all join without a step.

    The loop's first command is the stand (amplitude zero), so there is no ramp to
    absorb a difference between them; the same holds at the other end. The largest
    jump between consecutive commands is the gait's own motion.
    """
    sess.run_trial(
        _swinging, steps=30, setup=_setup(stand_ramp=0.2, lower=1.0), stand=GAIT_STAND
    )
    q = np.array(stub["backend"].sent)
    jumps = np.abs(np.diff(q, axis=0)).max(axis=1)
    assert jumps.max() < 0.05, f"a joint stepped {jumps.max():.3f} rad in one tick"


def test_the_loaded_stand_joins_the_first_loop_command_in_every_field(stub: dict) -> None:
    """A stand held with `tau_ff = 0` hands the legs the whole gravity load in one tick.

    That was up to 11 N m on a calf, and the stand sagged 0.03 rad meanwhile. The stand
    a driver hands over is the gait's command at zero amplitude, so the loop's first
    command (amplitude zero) is the same command, field for field.
    """
    params = control.default_gait_params("trot")
    offsets = control.GAITS["trot"]
    kp, kd = 180.0, 6.0

    def command(k, t, scale, scale_dot):
        return control.clock_joint_command(
            t, offsets, params, scale=scale, scale_dot=scale_dot, kp=kp, kd=kd
        )

    stand = control.stand_command(params, kp=kp, kd=kd)
    assert np.abs(stand.tau_ff).max() > 1.0, "the stand carries no load"
    sess.run_trial(command, steps=30, setup=_setup(kp=kp, kd=kd), stand=stand)

    sent = stub["backend"].sent_cmds
    last_stand, first_loop = sent[9], sent[10]  # a 0.1 s settle at 100 Hz, no ramp in
    for field in ("q", "kp", "kd", "tau_ff"):
        assert np.array_equal(getattr(last_stand, field), getattr(first_loop, field))
    # dq steps by the ramp's own growth term: about 0.2 rad/s for each 1/s of ramp rate
    assert np.abs(first_loop.dq - last_stand.dq).max() < 0.25 * 10.0


def test_the_stand_ramp_blends_the_load_in_from_zero(stub: dict) -> None:
    params = control.default_gait_params("trot")
    stand = control.stand_command(params, kp=180.0, kd=6.0)
    sess.run_trial(
        _stand_at(stand.q),
        steps=1,
        setup=_setup(stand_ramp=0.2, end_sequence=False),
        stand=stand,
    )
    load = np.array([np.abs(c.tau_ff).max() for c in stub["backend"].sent_cmds[:20]])
    assert 0.0 < load[0] < 0.1 * load[-1] and np.all(np.diff(load) >= 0)
    assert np.abs(load[-1] - np.abs(stand.tau_ff).max()) < 1e-9


def test_stand_ramp_eases_in_from_the_measured_pose(install) -> None:
    """From the ground the first command must be near where the robot is.

    Sending the stand at full gain from a pose most of a radian away is the
    step this option exists to remove.
    """
    start = STAND_POSE + np.array([0.0, 0.45, -0.85] * 4)
    holder = install(_last=start.copy())

    sess.run_trial(
        _stand_at(STAND_POSE),
        steps=1,
        setup=_setup(stand_ramp=0.5, settle=0.1),
        stand=STAND_POSE,
    )
    sent = holder["backend"].sent
    n_ramp = 50

    one_step = np.abs(start - STAND_POSE).max() / n_ramp
    assert np.abs(sent[0] - start).max() < one_step + 1e-9
    steps = np.abs(np.diff(np.array(sent[:n_ramp]), axis=0)).max()
    assert steps < 0.02, "the ramp stepped instead of easing"
    assert np.allclose(sent[n_ramp - 1], STAND_POSE), "ramp did not end on the stand"


def test_the_stale_pose_after_a_reset_does_not_trip_the_tilt_abort(install) -> None:
    """The last trial's heap is tipped over, and the bridge reports it at first."""

    class _TippedUntilReset(_StubBackend):
        def read(self):
            state, truth = super().read()
            if len(self.sent) < 5:
                state.q, state.imu_quat = LYING_POSE, _tilted(60.0)
            return state, truth

    install(_TippedUntilReset)
    result = sess.run_trial(
        _stand_at(STAND_POSE), steps=5, setup=_setup(reset=True, max_tilt=45.0)
    )
    assert result.aborted is None


def test_the_simulator_stand_ramp_starts_after_the_reset_has_landed(install) -> None:
    """Right after the reset signal the bridge still reports the last trial's heap.

    A stand ramp that read its start pose then eased out of the collapsed pose, a
    violent motion that the simulator never meant. The session holds the keyframe
    until the readings are the keyframe's.
    """
    heap = LYING_POSE.copy()

    class _ResetsAfterFiveTicks(_StubBackend):
        def read(self):
            state, truth = super().read()
            if len(self.sent) < 5:
                state.q = heap
            return state, truth

    holder = install(_ResetsAfterFiveTicks)
    sess.run_trial(
        _stand_at(GAIT_STAND),
        steps=1,
        setup=_setup(reset=True, stand_ramp=0.2),
        stand=GAIT_STAND,
    )
    sent = holder["backend"].sent
    assert all(np.array_equal(q, STAND_POSE) for q in sent[:5]), "held the keyframe"
    # the ramp eased from the keyframe, not from the heap
    assert np.abs(sent[5] - STAND_POSE).max() < 0.05


# -- the end -------------------------------------------------------------------


def _stage_names(result) -> list[str]:
    return [name for name, _ in result.stages]


def test_a_normal_end_lowers_to_the_lying_pose_then_releases(stub: dict) -> None:
    result = sess.run_trial(_swinging, steps=20, setup=_setup(), stand=GAIT_STAND)
    backend = stub["backend"]
    assert _stage_names(result) == [
        "stand ramp",
        "settle",
        "loop",
        "wind down",
        "hold",
        "lower",
    ]
    assert result.aborted is None
    assert np.allclose(backend.sent[-1], LYING_POSE)
    assert backend.released_after == len(backend.sent), "released before it was down"
    # the loop's own record is the loop only
    assert result.steps == 20


def test_the_wind_down_shrinks_the_stride_from_where_the_gait_is(stub: dict) -> None:
    scales = []

    def command(k, t, scale, scale_dot):
        scales.append((scale, scale_dot))
        return GAIT_STAND

    # stop in the middle of the stride ramp: the last loop step is at amplitude 0.4
    sess.run_trial(command, steps=5, setup=_setup(), stand=GAIT_STAND)
    down = [s for s in scales[5:] if s[1] < 0]
    assert [d[0] for d in down] == pytest.approx([0.3, 0.2, 0.1, 0.0])
    assert all(d[1] == pytest.approx(-10.0) for d in down)
    holds = scales[5 + len(down) :]
    assert holds and all(h == (0.0, 0.0) for h in holds)


def test_the_end_sequence_can_be_turned_off(stub: dict) -> None:
    result = sess.run_trial(
        _swinging, steps=20, setup=_setup(end_sequence=False), stand=GAIT_STAND
    )
    assert _stage_names(result) == ["stand ramp", "settle", "loop"]
    assert len(stub["backend"].sent) == 10 + 20


def _ramped(**kw) -> sess.TrialSetup:
    return _setup(**{"release_ramp": 0.1, "release_kd": 2.0, "lower": 0.2, **kw})


def test_the_release_ramp_runs_after_the_lowering_and_lets_go_gradually(
    stub: dict,
) -> None:
    result = sess.run_trial(_swinging, steps=20, setup=_ramped(), stand=GAIT_STAND)
    backend = stub["backend"]
    assert _stage_names(result)[-2:] == ["lower", "release ramp"]
    start = dict(result.stages)["release ramp"]
    cmds = backend.sent_cmds[start:]
    assert len(cmds) == 10
    kp = np.array([c.kp[0] for c in cmds])
    kd = np.array([c.kd[0] for c in cmds])
    assert np.all(np.diff(kp) < 0) and kp[0] < 100.0 and kp[-1] == 0.0
    assert kd[0] < 6.0 and kd[-1] == pytest.approx(2.0), "damping goes to release_kd"
    assert all(np.allclose(c.q, LYING_POSE) and not c.tau_ff.any() for c in cmds)
    assert backend.released_after == len(backend.sent), "released before it was down"


def test_the_release_ramp_runs_on_the_first_ctrl_c_too(stub: dict) -> None:
    result = sess.run_trial(
        _press_ctrl_c_at(30), steps=100, setup=_ramped(), stand=GAIT_STAND
    )
    assert result.aborted and "Ctrl+C" in result.aborted
    assert _stage_names(result)[-1] == "release ramp"


def test_the_release_ramp_never_runs_on_an_immediate_exit(install) -> None:
    install()
    second = sess.run_trial(
        _press_ctrl_c_at(30, every_call_after=True),
        steps=100,
        setup=_ramped(),
        stand=GAIT_STAND,
    )
    assert "release ramp" not in _stage_names(second)
    assert "lower" not in _stage_names(second)

    install(_GoesOver)
    tilted = sess.run_trial(
        _swinging, steps=30, setup=_ramped(max_tilt=10.0), stand=GAIT_STAND
    )
    assert tilted.aborted and "tilt" in tilted.aborted
    assert "release ramp" not in _stage_names(tilted)

    class _GoesQuiet(_StubBackend):
        def state_age(self):
            return 0.5 if len(self.sent) > 30 else 0.001

    install(_GoesQuiet)
    stale = sess.run_trial(_swinging, steps=30, setup=_ramped(), stand=GAIT_STAND)
    assert stale.aborted and "LowState" in stale.aborted
    assert "release ramp" not in _stage_names(stale)


def test_nothing_steps_a_field_into_or_through_the_release_ramp(stub: dict) -> None:
    sess.run_trial(_swinging, steps=30, setup=_ramped(release_ramp=0.5), stand=GAIT_STAND)
    cmds = stub["backend"].sent_cmds
    for field in ("q", "dq", "kp", "kd", "tau_ff"):
        series = np.array([getattr(c, field) for c in cmds])
        # the loop's own motion is the largest legitimate step in any field
        steps = np.abs(np.diff(series[-80:], axis=0)).max(axis=1)
        assert steps.max() < 0.05 * max(1.0, np.abs(series).max()), field


def test_without_the_release_ramp_the_end_is_what_it_was(stub: dict) -> None:
    result = sess.run_trial(
        _swinging, steps=20, setup=_setup(lower=0.5), stand=GAIT_STAND
    )
    cmds = stub["backend"].sent_cmds[dict(result.stages)["lower"] :]
    assert "release ramp" not in _stage_names(result)
    q = np.array([c.q for c in cmds])
    assert np.allclose(np.diff(q, axis=0), np.diff(q, axis=0)[0]), "not a straight line"
    assert cmds[-1].kp[0] == 100.0 and cmds[-1].kd[0] == 6.0


def _press_ctrl_c_at(step: int, every_call_after: bool = False):
    def command(k, t, scale, scale_dot):
        if k == step or (every_call_after and k > step):
            signal.raise_signal(signal.SIGINT)
        return GAIT_STAND

    return command


def test_the_first_ctrl_c_runs_the_whole_end_sequence(stub: dict) -> None:
    result = sess.run_trial(
        _press_ctrl_c_at(30), steps=100, setup=_setup(), stand=GAIT_STAND
    )
    backend = stub["backend"]
    assert result.steps == 31, "the loop did not stop at the next tick"
    assert result.aborted and "Ctrl+C" in result.aborted
    assert "lower" in _stage_names(result)
    assert np.allclose(backend.sent[-1], LYING_POSE)
    assert backend.released_after == len(backend.sent)


def test_the_second_ctrl_c_releases_at_once(stub: dict) -> None:
    result = sess.run_trial(
        _press_ctrl_c_at(30, every_call_after=True),
        steps=100,
        setup=_setup(),
        stand=GAIT_STAND,
    )
    backend = stub["backend"]
    assert result.aborted and "second Ctrl+C" in result.aborted
    assert not np.allclose(backend.sent[-1], LYING_POSE), "it lowered anyway"
    assert "lower" not in _stage_names(result)
    # released during the wind down, not after the sequence
    assert backend.released_after < 10 + 31 + 10 + 5 + 10


def test_ctrl_c_is_the_default_again_after_the_trial(stub: dict) -> None:
    before = signal.getsignal(signal.SIGINT)
    sess.run_trial(_stand_at(STAND_POSE), steps=3, setup=_setup())
    assert signal.getsignal(signal.SIGINT) is before
    with pytest.raises(KeyboardInterrupt):
        signal.raise_signal(signal.SIGINT)


class _GoesOver(_StubBackend):
    """Tips over once it has been sent `after` commands."""

    after = 30
    deg = 60.0

    def read(self):
        state, truth = super().read()
        if len(self.sent) > self.after:
            state.imu_quat = _tilted(self.deg)
        return state, truth


def test_tilt_past_the_limit_releases_at_once_and_says_so(install) -> None:
    holder = install(_GoesOver)
    result = sess.run_trial(_stand_at(STAND_POSE), steps=200, setup=_setup(max_tilt=45.0))
    backend = holder["backend"]
    assert result.aborted and "60" in result.aborted and "loop step" in result.aborted
    assert result.steps < 200
    assert len(result.tracking_error) == result.steps
    assert "ABORTED" in result.report()
    assert "wind down" not in _stage_names(result), "a falling robot cannot wait"
    assert backend.released_after == len(backend.sent)


def test_without_max_tilt_a_tilted_robot_still_runs_every_step(install) -> None:
    install(_GoesOver, after=0, deg=80.0)
    result = sess.run_trial(_stand_at(STAND_POSE), steps=40, setup=_setup())
    assert result.aborted is None and result.steps == 40


def test_a_dead_lowstate_releases_at_once_and_says_so(install) -> None:
    class _GoesQuiet(_StubBackend):
        def state_age(self):
            return 0.5 if len(self.sent) > 30 else 0.001

    holder = install(_GoesQuiet)
    result = sess.run_trial(_stand_at(STAND_POSE), steps=200, setup=_setup())
    assert result.aborted and "LowState" in result.aborted
    assert result.steps < 200
    assert "wind down" not in _stage_names(result)
    assert holder["backend"].released_after == len(holder["backend"].sent)


def test_a_fresh_lowstate_does_not_trip_the_watchdog(stub: dict) -> None:
    result = sess.run_trial(_stand_at(STAND_POSE), steps=20, setup=_setup())
    assert result.aborted is None


def test_the_watchdog_can_be_turned_off(install) -> None:
    class _AlwaysQuiet(_StubBackend):
        def state_age(self):
            return 10.0

    install(_AlwaysQuiet)
    result = sess.run_trial(
        _stand_at(STAND_POSE), steps=5, setup=_setup(max_state_age=None)
    )
    assert result.aborted is None


# -- what a command may be -----------------------------------------------------


def test_session_accepts_joint_command(stub: dict) -> None:
    def command(k, t, scale, scale_dot) -> JointCommand:
        return JointCommand(
            q=STAND_POSE + 0.01 * k,
            dq=np.zeros(12),
            kp=np.full(12, 80.0),
            kd=np.full(12, 4.0),
            tau_ff=np.full(12, 1.5),
        )

    r = sess.run_trial(command, steps=4, setup=_setup(end_sequence=False))
    assert r.commanded_q is not None
    assert len(r.tracking_error) == 4
    assert np.allclose(r.commanded_q[0], STAND_POSE)
    assert np.allclose(r.commanded_q[1], STAND_POSE + 0.01)
    sent = stub["backend"].sent_cmds[-1]  # the command passes through untouched
    assert np.all(sent.kp == 80.0) and np.all(sent.tau_ff == 1.5)


def test_an_array_command_is_held_with_the_trials_gains(stub: dict) -> None:
    sess.run_trial(
        _stand_at(STAND_POSE + 0.1),
        steps=2,
        setup=_setup(kp=120.0, kd=7.0, end_sequence=False),
    )
    cmd = stub["backend"].sent_cmds[-1]
    assert np.all(cmd.kp == 120.0) and np.all(cmd.kd == 7.0)
    assert np.all(cmd.dq == 0.0) and np.all(cmd.tau_ff == 0.0)


def test_trial_records_joint_trajectories(stub: dict) -> None:
    steps = 4
    r = sess.run_trial(
        lambda k, t, scale, scale_dot: STAND_POSE + 0.05 * k,
        steps=steps,
        setup=_setup(),
    )
    assert r.measured_q is not None
    assert r.commanded_q is not None
    assert r.measured_q.shape == (steps, 12)
    assert r.commanded_q.shape == (steps, 12)
    assert np.allclose(r.commanded_q[0], STAND_POSE)
    assert np.allclose(r.commanded_q[1], STAND_POSE + 0.05)


# -- the record ----------------------------------------------------------------


def test_body_height_corrects_the_imu_site_offset(stub: dict) -> None:
    r = sess.run_trial(_stand_at(STAND_POSE), steps=3, setup=_setup())
    assert r.base_pos is not None
    expected = stub["backend"].height - IMU_SITE_OFFSET[2]
    assert np.allclose(r.body_height, expected)
    # and the raw reading is still available, for comparisons made in that frame
    assert np.allclose(r.base_pos[:, 2], stub["backend"].height)


def test_forward_distance_projects_onto_the_body_axis(stub: dict) -> None:
    r = sess.run_trial(_stand_at(STAND_POSE), steps=2, setup=_setup())
    # identity orientation, so the body x axis is world x
    assert np.allclose(r.forward_axis, [1.0, 0.0, 0.0])
    r.start_pos = np.array([0.0, 0.0, 0.3])
    r.end_pos = np.array([1.5, 9.0, 0.3])  # y must not count
    assert r.forward_distance() == pytest.approx(1.5)


def test_a_trial_without_ground_truth_still_reports(install) -> None:
    """On hardware SportModeState stops updating once low level control starts.

    The old drivers disagreed about this: one guarded every dereference and the
    other would have raised AttributeError. The session has to be the careful one.
    """
    install(_NoTruth)
    r = sess.run_trial(_stand_at(STAND_POSE), steps=3, setup=_setup())
    assert r.base_pos is None
    assert r.body_height is None
    assert r.forward_distance() is None
    text = r.report()
    assert "tracking error" in text and "torso attitude" in text
    assert "height" not in text, "printed a height line with no height data"


def _value_column(line: str) -> int:
    """Where a report line's value starts: past the label and its padding."""
    m = re.match(r"^([A-Za-z][A-Za-z ]*?)( {2,})", line)
    return len(m.group(1)) + len(m.group(2)) if m else -1


def test_report_columns_line_up(stub: dict) -> None:
    """The old replay put `torso attitude` one column out from its neighbours."""
    r = sess.run_trial(_stand_at(STAND_POSE), steps=3, setup=_setup())
    text = r.report(
        height_label="body height", extra=("travelled       +0.1 m along body x",)
    )
    columns = {
        ln.split("  ")[0]: _value_column(ln)
        for ln in text.splitlines()
        if ln[:1].isalpha() and "overran" not in ln
    }
    assert columns, "report produced no labelled lines"
    assert set(columns.values()) == {sess._LABEL}, f"misaligned: {columns}"


def test_the_column_check_would_notice_a_misalignment() -> None:
    """Guard the guard: the alignment test must be able to fail."""
    assert _value_column("torso attitude   roll +0.0") == 17
    assert _value_column("torso attitude  roll +0.0") == 16


def _record_video(r, tmp_path, name: str) -> None:
    out_video = tmp_path / name
    saved = r.record_video(out_video, fps=10, width=320, height=240)
    assert saved.exists()
    assert saved.stat().st_size > 0


@pytest.fixture
def video_ready() -> None:
    pytest.importorskip("cv2", reason="opencv-python not installed; extra plots needed")
    if not SCENE_PATH.is_file():
        pytest.skip(f"MuJoCo scene XML not found; expected at {SCENE_PATH}")


def test_record_video_creates_valid_mp4(stub: dict, video_ready, tmp_path) -> None:
    r = sess.run_trial(_stand_at(STAND_POSE), steps=4, setup=_setup())
    _record_video(r, tmp_path, "test_trial.mp4")


def test_record_video_handles_intermittent_ground_truth(
    install, video_ready, tmp_path
) -> None:
    install(_IntermittentTruth)
    steps = 6
    r = sess.run_trial(_stand_at(STAND_POSE), steps=steps, setup=_setup())
    assert r.base_pos is not None
    assert len(r.base_pos) == steps
    assert np.isnan(r.base_pos).any()
    _record_video(r, tmp_path, "test_intermittent.mp4")


def test_record_video_handles_absent_ground_truth(install, video_ready, tmp_path) -> None:
    install(_NoTruth)
    r = sess.run_trial(_stand_at(STAND_POSE), steps=4, setup=_setup())
    assert r.base_pos is None
    _record_video(r, tmp_path, "test_no_truth.mp4")


# -- the options and the checks before the trial -------------------------------


def test_release_kd_reaches_the_backend(stub: dict) -> None:
    sess.run_trial(_stand_at(STAND_POSE), steps=1, setup=_setup(release_kd=2.0))
    assert stub["backend"].kw["release_kd"] == 2.0


def test_overruns_is_a_field_not_just_prose_in_timing(install) -> None:
    """A caller scoring trials needs the count, not a sentence to re-parse."""
    install(overruns=3)
    result = sess.run_trial(_stand_at(STAND_POSE), steps=1, setup=_setup())
    assert result.overruns == 3


def test_run_trial_checks_sport_mode_before_connecting(monkeypatch) -> None:
    """A trial against an unreleased robot must not run at all, not run and no-op."""
    calls = []

    def spy(domain_id, interface):
        calls.append((domain_id, interface))
        raise SystemExit("sport mode is active (stub)")

    def _no_backend(**_kw):
        raise AssertionError(
            "DDSBackend must not be constructed while sport mode is refused"
        )

    monkeypatch.setattr(sess, "refuse_if_sport_mode_active", spy)
    monkeypatch.setattr(sess, "DDSBackend", _no_backend)
    monkeypatch.setattr(sess, "check_single_peer", lambda: None)

    with pytest.raises(SystemExit, match="sport mode is active"):
        sess.run_trial(
            _stand_at(STAND_POSE),
            steps=1,
            setup=_setup(domain_id=0, interface="eth0"),
        )
    assert calls == [(0, "eth0")]


# -- the log -------------------------------------------------------------------


def _run_and_log(tmp_path, steps=20, channels=None, **kw):
    result = sess.run_trial(_swinging, steps=steps, setup=_setup(**kw), stand=GAIT_STAND)
    path = trial_log.write_trial_log(
        result, "test", "swinging", {"steps": steps}, channels, directory=tmp_path
    )
    return result, *trial_log.read_trial_log(path), path


def test_the_log_covers_every_tick_of_every_stage(stub: dict, tmp_path) -> None:
    result, arrays, meta, _ = _run_and_log(tmp_path)
    sent = stub["backend"].sent_cmds
    assert meta["ticks"] == len(sent) > 0
    for name in trial_log._RECORD_ARRAYS + ["command_tick"]:
        assert len(arrays[name]) == len(sent), name
    # the command is what was sent, in order, release not included
    assert np.array_equal(arrays["q_cmd"], np.array([c.q for c in sent]))
    assert np.array_equal(arrays["tau_ff"], np.array([c.tau_ff for c in sent]))
    names = [st["stage"] for st in meta["stages"]]
    assert names == [n for n, _ in result.stages]
    assert meta["stages"][0]["first_tick"] == 0
    assert np.all(np.diff(arrays["wall"]) >= 0) and arrays["wall"][0] == 0.0


def test_the_reading_in_a_record_is_the_one_before_its_command(
    stub: dict, tmp_path
) -> None:
    """The stub reports the last command back, so a record's reading is the one before."""
    _, arrays, _, _ = _run_and_log(tmp_path)
    assert np.array_equal(arrays["q_meas"][1:], arrays["q_cmd"][:-1])


def test_the_log_has_no_ground_truth_on_hardware_so_it_is_nan(install, tmp_path) -> None:
    install(_NoTruth)
    _, arrays, _, _ = _run_and_log(tmp_path)
    assert np.isnan(arrays["pos"]).all() and np.isnan(arrays["vel"]).all()
    assert np.isfinite(arrays["quat"]).all()


def test_the_metadata_says_what_was_run(stub: dict, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sess, "source_version", lambda: ("abc123", True))
    result, _, meta, path = _run_and_log(tmp_path)
    assert path.with_suffix(".json").exists()
    json.loads(path.with_suffix(".json").read_text())  # plain JSON
    assert meta["commit"] == "abc123" and meta["dirty"] is True
    assert meta["driver"] == "test" and meta["config"] == {"steps": 20}
    assert meta["setup"]["rate"] == 100.0
    assert meta["condition"] == {"where": "simulator", "support": None}
    assert meta["aborted"] is None
    assert meta["overruns"] == 0 and "overran" in meta["timing"]
    assert path.name.endswith("_swinging_sim.npz")
    assert result.commit == "abc123"


def test_the_commit_is_read_before_the_backend_connects(install, monkeypatch) -> None:
    """A subprocess inside a trial is a gap in the command stream."""
    holder = install()
    seen = []
    monkeypatch.setattr(
        sess,
        "source_version",
        lambda: (seen.append("backend" in holder), ("x", False))[1],
    )
    sess.run_trial(_stand_at(STAND_POSE), steps=1, setup=_setup())
    assert seen == [False]


def test_an_aborted_trial_still_writes_its_log_and_says_why(install, tmp_path) -> None:
    install(_GoesOver)
    result = sess.run_trial(_stand_at(STAND_POSE), steps=200, setup=_setup(max_tilt=45.0))
    path = trial_log.write_trial_log(result, "test", "tipped", directory=tmp_path)
    arrays, meta = trial_log.read_trial_log(path)
    assert "tilt" in meta["aborted"] and "loop step" in meta["aborted"]
    # the tick that tripped the exit is in the log, and nothing after it
    assert meta["ticks"] == len(result.record["wall"]) == 10 + result.steps + 1
    assert [st["stage"] for st in meta["stages"]][-1] == "loop"


def test_a_crashing_command_releases_the_legs_and_still_yields_a_log(
    stub: dict, tmp_path
) -> None:
    channels = {"k": []}

    def command(k, t, scale, scale_dot):
        if k == 7:
            raise ZeroDivisionError("a bug in a driver")
        channels["k"].append(k)
        return GAIT_STAND

    with pytest.raises(sess.TrialCrashed) as crash:
        sess.run_trial(command, steps=20, setup=_setup(), stand=GAIT_STAND)
    assert isinstance(crash.value.__cause__, ZeroDivisionError)
    backend = stub["backend"]
    assert backend.released_after is not None  # the legs were released
    result = crash.value.result
    assert "crashed during loop step 8/20" in result.aborted
    assert "ZeroDivisionError" in result.aborted
    path = trial_log.write_trial_log(
        result, "test", "crashed", channels=channels, directory=tmp_path
    )
    arrays, meta = trial_log.read_trial_log(path)
    assert meta["aborted"] == result.aborted
    # seven calls answered and an eighth raised, which sent nothing
    assert meta["command_ticks"]["count"] == 7
    assert meta["ticks"] == len(backend.sent)
    assert np.isfinite(arrays["extra_k"][arrays["command_tick"]][:7]).all()


def test_extra_channels_line_up_with_the_ticks_of_the_command_calls(
    stub: dict, tmp_path
) -> None:
    calls: list[int] = []

    def command(k, t, scale, scale_dot):
        calls.append(k)
        return GAIT_STAND

    channels = {"k_squared": []}

    def recording(k, t, scale, scale_dot):
        channels["k_squared"].append([k * k, -k])
        return command(k, t, scale, scale_dot)

    result = sess.run_trial(recording, steps=20, setup=_setup(), stand=GAIT_STAND)
    path = trial_log.write_trial_log(
        result, "test", "channel", channels=channels, directory=tmp_path
    )
    arrays, meta = trial_log.read_trial_log(path)
    first = dict(result.stages)["loop"]
    x, called = arrays["extra_k_squared"], arrays["command_tick"]
    assert x.shape == (meta["ticks"], 2)
    assert meta["command_ticks"] == {"first": first, "count": len(calls)}
    assert np.flatnonzero(called).tolist() == list(range(first, first + len(calls)))
    # tick first + k holds the value of call k, and the ticks before and after are NaN
    assert np.array_equal(x[first : first + len(calls), 0], np.array(calls) ** 2)
    assert np.isnan(x[:first]).all() and np.isnan(x[first + len(calls) :]).all()


def test_the_command_ticks_are_only_the_ticks_it_was_called_on(
    stub: dict, tmp_path
) -> None:
    """With the end sequence off, nothing after the loop calls it."""
    result = sess.run_trial(
        _swinging, steps=7, setup=_setup(end_sequence=False), stand=GAIT_STAND
    )
    assert result.command_ticks.tolist() == list(range(10, 17))


def test_a_hardware_trial_must_say_lifted_or_floor(monkeypatch) -> None:
    def _no_backend(**_kw):
        raise AssertionError("refused trials must not connect")

    monkeypatch.setattr(sess, "DDSBackend", _no_backend)
    monkeypatch.setattr(sess, "check_single_peer", lambda: None)
    monkeypatch.setattr(sess, "refuse_if_sport_mode_active", lambda *_: None)
    hardware = _setup(domain_id=0, interface="eth0")
    with pytest.raises(SystemExit, match="lifted"):
        sess.run_trial(_stand_at(STAND_POSE), steps=1, setup=hardware)
    hardware.support = "in the air"
    with pytest.raises(SystemExit, match="support must be"):
        sess.run_trial(_stand_at(STAND_POSE), steps=1, setup=hardware)


@pytest.mark.parametrize("support", ["lifted", "floor"])
def test_a_hardware_log_records_what_the_robot_stood_on(
    install, monkeypatch, tmp_path, support: str
) -> None:
    install(_NoTruth)
    monkeypatch.setattr(sess, "refuse_if_sport_mode_active", lambda *_: None)
    result = sess.run_trial(
        _stand_at(STAND_POSE),
        steps=3,
        setup=_setup(domain_id=0, interface="eth0", support=support),
    )
    path = trial_log.write_trial_log(result, "test", "hw", directory=tmp_path)
    _, meta = trial_log.read_trial_log(path)
    assert meta["condition"] == {"where": "hardware", "support": support}
    assert path.name.endswith(f"_hw_{support}.npz")


def test_two_logs_in_one_second_do_not_overwrite_each_other(stub: dict, tmp_path) -> None:
    paths = {_run_and_log(tmp_path, steps=2)[-1] for _ in range(3)}
    assert len(paths) == 3
