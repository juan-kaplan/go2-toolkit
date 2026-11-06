"""One trial on the robot: connect, stand, ramp up, run, ramp down, lie down, release.

Every driver in this repo (the generated gait, the tuner, the startup trace) needs the
same sequence in the same order, and `run_trial` is that sequence. The command function
is the only thing a driver supplies, plus the pose its gait stands in:

    result = run_trial(command, steps=1600, setup=TrialSetup(), stand=stand)
    print(result.report())

**The sequence.** Each stage starts from where the last one left the legs, so no
stage steps a joint:

    wait for reset   simulator only: hold the keyframe until the reset has landed
    stand ramp       ease from the measured pose into the stand
    settle           hold the stand
    loop             `steps` control periods; the stride ramps from zero to one
    wind down        the stride ramps back to zero, from wherever the gait is
    hold             stand again, briefly
    lower            ease to Unitree's lying pose
    release ramp     optional: `kp` eases to zero at the lying pose, damping kept
    release          damped, see `DDSBackend.release`

**The session owns the stride amplitude.** The command function is called as
`command(k, t, scale, scale_dot)`: the step index, the elapsed time, the stride
amplitude in [0, 1] and its rate of change per second. At amplitude zero a gait is
a standing pose whatever its phase (roadmap D8), so the stand is that pose and
the ramp has nothing to bridge: the start needs no look at the first command, and
the end can stop a gait anywhere in its cycle. A driver whose command has no
amplitude blends in joint space itself, `(1 - scale) * stand + scale * q`.

**The command function is called exactly once per control tick**, during the
loop and the wind down and hold of the end, and never before the loop to find a
target. A stateful phase source (a network) is stepped once per tick and cannot be
stepped twice by the session.

**Every exit is one of three kinds.** A normal end and the first Ctrl+C run the
whole end sequence. The second Ctrl+C, a tilt past `max_tilt` and a LowState that
has stopped arriving (`max_state_age`) release at once: a falling robot, a second
press and a dead link cannot wait for a sequence. The reason is in
`TrialResult.aborted`. An unexpected exception from the command function is a
fourth, undesigned one: the legs are released like the immediate exits, and the
`TrialCrashed` that follows carries the result, so a driver can still write its log.

**Every tick is recorded.** Each command the session sends, in every stage, is kept
with the state read just before it (`TrialResult.record`), and `trial_log` writes it
to disk after the release. The stand the driver gives is a command, not only a pose:
it carries the gait's own `tau_ff`, `kp` and `kd` at zero amplitude, so the first
loop command adds nothing the stand was not already holding.

**Never leave a gap in the command stream.** The simulator's bridge computes
torque only when a command arrives, so a pause freezes the legs on their last
command. Anything slow on its first call goes before the backend connects.
"""

from __future__ import annotations

import signal
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from ..go2.command import JointCommand
from ..go2.conventions import (
    body_forward_axis,
    rotation_matrices,
    rpy_degrees,
)
from ..go2.model import IMU_SITE_OFFSET, LYING_POSE, STAND_POSE
from .bridge import check_single_peer, reset_bridge
from .dds import DDSBackend
from .sport_mode import refuse_if_sport_mode_active
from .trial_log import refuse_without_support, source_version, stack_records

# (step index, elapsed s, stride amplitude, its rate per s) -> (12,) joint angles in
# Unitree order, or a JointCommand
CommandFn = Callable[[int, float, float, float], np.ndarray | JointCommand]

_LABEL = 16  # report label column, so the numbers line up
_RESET_TOLERANCE = 0.15  # rad from the keyframe that counts as "the reset has landed"
_RESET_TIMEOUT = 0.5  # s to wait for it


@dataclass
class TrialSetup:
    """Everything about how a trial is driven, and nothing about what it does."""

    # fmt: off
    rate: float = 200.0         # control loop rate, Hz
    kp: float = 100.0
    kd: float = 6.0
    stand_ramp: float = 1.0     # s easing from the measured pose into the stand
    settle: float = 1.0         # s held on the stand before the stride starts
    ramp: float = 1.0           # s the stride takes to grow from zero, and to shrink back
    domain_id: int = 1
    interface: str = "lo"
    reset: bool = True          # SIGUSR1 the bridge, so trials are independent
    # The end of a trial. A tuner that scores hundreds of trials and never looks at
    # the robot afterwards turns it off, and the legs are released where they are.
    end_sequence: bool = True
    hold: float = 0.5           # s on the stand after the stride has shrunk to zero
    lower: float = 2.0          # s easing from the stand to the lying pose
    # What the physical robot needs on top. The defaults cost the simulator nothing.
    max_tilt: float | None = None   # deg of roll or pitch that ends the trial at once
    max_state_age: float | None = 0.2   # s without a LowState that ends it at once
    release_kd: float = 0.0     # damping at the release, see DDSBackend.release
    # s over which `kp` eases to zero at the lying pose, before the release. Zero
    # skips the stage: the release then takes the legs' whole grip in one tick.
    release_ramp: float = 0.0
    # What the robot stands on, "lifted" or "floor". Required on hardware, where the
    # log has to say it; ignored in the simulator.
    support: str | None = None
    # fmt: on


@dataclass
class TrialResult:
    """What a trial measured. Formatting is the driver's business, mostly."""

    # fmt: off
    tracking_error: np.ndarray          # (steps,) max |q_measured - q_commanded|
    quats: np.ndarray                   # (steps, 4) body orientation, scalar first
    base_pos: np.ndarray | None         # (steps, 3) IMU site in world, if known
    start_pos: np.ndarray | None        # position when the loop began
    end_pos: np.ndarray | None
    forward_axis: np.ndarray            # body x in world when the loop began
    stand_height: float | None          # IMU site height after the settle
    stand_error: float                  # max |q - stand| after the settle
    steps: int
    rate: float
    timing: str
    overruns: int                          # deadline overruns; `timing` is this in prose
    measured_q: np.ndarray | None = None   # (steps, 12) measured joint angles
    commanded_q: np.ndarray | None = None  # (steps, 12) commanded joint angles
    aborted: str | None = None             # why the trial ended early, if it did
    stages: tuple[tuple[str, int], ...] = ()   # (stage, index of its first command)
    # Every command of every stage with the state read before it, see `trial_log`.
    record: dict[str, np.ndarray] | None = None
    command_ticks: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    commit: str = "unknown"
    dirty: bool = True
    setup: TrialSetup | None = None
    log_path: Path | None = None   # where the trial log went, once a driver wrote one
    # fmt: on

    @property
    def elapsed(self) -> float:
        return self.steps / self.rate

    @property
    def body_height(self) -> np.ndarray | None:
        """Heights of base_link, (steps,).

        `base_pos` is a framepos on the model's `imu` site, about 4 cm above the
        body origin, so comparing it against a kinematic hip height overstates
        the robot by that much. Use this when the comparison is kinematic and
        `base_pos` when it is against something measured in the same frame.
        """
        if self.base_pos is None:
            return None
        return self.base_pos[:, 2] - IMU_SITE_OFFSET[2]

    @property
    def rpy(self) -> np.ndarray:
        return rpy_degrees(self.quats)

    def forward_distance(self) -> float | None:
        """Displacement projected onto the body's own forward axis, m.

        Projected rather than taken as world x, so a robot that drifts in yaw is
        still measured along the direction it was facing.
        """
        if self.start_pos is None or self.end_pos is None:
            return None
        if not (
            np.all(np.isfinite(self.start_pos)) and np.all(np.isfinite(self.end_pos))
        ):
            return None
        return float((self.end_pos - self.start_pos) @ self.forward_axis)

    def report(
        self,
        *,
        height_label: str = "body height",
        height_note: str = "",
        attitude_note: str = "",
        use_body_height: bool = True,
        extra: tuple[str, ...] = (),
    ) -> str:
        """The block both drivers print, with the two annotations they differ in."""
        e = self.tracking_error
        lines = [
            f"\n{'tracking error':<{_LABEL}}mean {e.mean():.4f}  "
            f"p95 {np.percentile(e, 95):.4f}  max {e.max():.4f} rad"
        ]
        if self.aborted:
            lines.insert(0, f"\n{'ABORTED':<{_LABEL}}{self.aborted}")

        h = (
            self.body_height
            if use_body_height
            else (None if self.base_pos is None else self.base_pos[:, 2])
        )
        if h is not None and len(h) and np.isfinite(h).any():
            lines.append(
                f"{height_label:<{_LABEL}}mean {np.nanmean(h):.3f}  "
                f"min {np.nanmin(h):.3f}  max {np.nanmax(h):.3f} m{height_note}"
            )

        r = self.rpy
        lines.append(
            f"{'torso attitude':<{_LABEL}}"
            f"roll {r[:, 0].mean():+.1f} +-{r[:, 0].std():.1f}   "
            f"pitch {r[:, 1].mean():+.1f} +-{r[:, 1].std():.1f} deg"
            f"{attitude_note}"
        )
        lines.extend(extra)
        lines.append(self.timing)
        return "\n".join(lines)

    def record_video(self, path: Path | str, **kwargs) -> Path:
        """Render an MP4 video of the trial using MuJoCo offscreen rendering."""
        from .video import render_trial_video

        return render_trial_video(self, path, **kwargs)


def _ramp_scale(t: float, ramp: float) -> tuple[float, float]:
    """Stride amplitude and its rate at time `t` into a trial of ramp length `ramp`."""
    if ramp <= 0.0:
        return 1.0, 0.0
    return min(1.0, t / ramp), (1.0 / ramp if t < ramp else 0.0)


def _blend(c0: JointCommand, c1: JointCommand, a: float) -> JointCommand:
    """Every field of two commands, interpolated: `c0` at a = 0, `c1` at a = 1."""
    return JointCommand(
        q=(1 - a) * c0.q + a * c1.q,
        dq=(1 - a) * c0.dq + a * c1.dq,
        kp=(1 - a) * c0.kp + a * c1.kp,
        kd=(1 - a) * c0.kd + a * c1.kd,
        tau_ff=(1 - a) * c0.tau_ff + a * c1.tau_ff,
    )


class _Release(Exception):
    """Leave the trial at once and release the legs. The message is the reason."""


class TrialCrashed(RuntimeError):
    """The trial died from an unexpected exception, after the legs were released.

    `result` is everything recorded up to the crash, so a driver can still write its
    log (`aborted` says "crashed" and what raised); the original exception is the
    `__cause__`.
    """

    def __init__(self, result: TrialResult) -> None:
        super().__init__(result.aborted)
        self.result = result


class _CtrlC:
    """Counts Ctrl+C presses while a trial runs, in place of KeyboardInterrupt.

    The first press asks for the end sequence and the second for the immediate
    release; the loop reads `presses` once per tick. Only the main thread can
    install a signal handler, so anywhere else Ctrl+C keeps its default.
    """

    def __init__(self) -> None:
        self.presses = 0
        self._previous = None

    def __enter__(self) -> _CtrlC:
        if threading.current_thread() is threading.main_thread():
            self._previous = signal.signal(signal.SIGINT, self._count)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._previous is not None:
            signal.signal(signal.SIGINT, self._previous)

    def _count(self, *_: object) -> None:
        self.presses += 1


class _Ticks:
    """One control period at a time: send, wait out the period, read, and guard.

    Every stage sends its commands through here, so the three immediate exits
    (second Ctrl+C, tilt, a dead LowState) hold in every stage alike, and each
    stage's first command is logged for `TrialResult.stages`.
    """

    def __init__(self, backend: DDSBackend, setup: TrialSetup, ctrl_c: _CtrlC) -> None:
        self.backend, self.setup, self.ctrl_c = backend, setup, ctrl_c
        self.count = 0
        self.stages: list[tuple[str, int]] = []
        self.where = ""  # where the trial is, for the reason an exit gives
        self.check_tilt = True
        self.last: JointCommand | None = None
        # the recording: what was sent, the reading it was sent against, and when
        self.commands: list[JointCommand] = []
        self.states: list = []
        self.truths: list = []
        self.walls: list[float] = []
        self.state, self.truth = backend.read()

    def begin(self, stage: str) -> None:
        self.stages.append((stage, self.count))
        self.where = stage

    def send(self, cmd: JointCommand):
        self.commands.append(cmd)
        self.states.append(self.state)
        self.truths.append(self.truth)
        self.walls.append(time.perf_counter())
        self.backend.send(cmd)
        self.backend.step()
        self.count += 1
        self.last = cmd
        self.state, self.truth = self.backend.read()
        reason = self._reason(self.state)
        if reason:
            raise _Release(f"{reason} during {self.where}; legs released")
        return self.state, self.truth

    def record(self) -> dict[str, np.ndarray]:
        return stack_records(self.commands, self.states, self.truths, self.walls)

    def _reason(self, state) -> str | None:
        if self.ctrl_c.presses >= 2:
            return "second Ctrl+C"
        age = self.backend.state_age()
        if self.setup.max_state_age is not None and age > self.setup.max_state_age:
            return f"no LowState for {age:.2f} s (limit {self.setup.max_state_age:g} s)"
        if self.check_tilt and self.setup.max_tilt is not None:
            roll, pitch, _ = rpy_degrees(state.imu_quat)[0]
            tilt = max(abs(roll), abs(pitch))
            if tilt > self.setup.max_tilt:
                return f"tilt {tilt:.1f} deg exceeds {self.setup.max_tilt:g}"
        return None

    def ease(self, target: JointCommand, seconds: float, *, smooth: bool = False) -> None:
        """Move from the last command to `target` over `seconds`, linearly.

        `smooth` bends the line into a smoothstep, which starts and stops at zero speed.
        """
        start, n = self.last, round(seconds * self.setup.rate)
        for i in range(n):
            a = (i + 1) / n
            self.send(_blend(start, target, a * a * (3 - 2 * a) if smooth else a))

    def hold(self, cmd: JointCommand, seconds: float) -> None:
        for _ in range(round(seconds * self.setup.rate)):
            self.send(cmd)


def _wait_for_reset(ticks: _Ticks) -> None:
    """Hold the keyframe until the simulator has actually reset to it.

    `reset_bridge` only signals the bridge, which acts a few steps later, and until
    it does the readings are still the last trial's collapsed pose. A stand ramp
    that started from one of those would ease from a heap. The keyframe is what the
    reset puts the robot on, so it is also what to hold meanwhile.
    """
    ticks.begin("wait for reset")
    ticks.check_tilt = False  # the readings are still last trial's, tipped over maybe
    keyframe = JointCommand.position_hold(
        STAND_POSE, kp=ticks.setup.kp, kd=ticks.setup.kd
    )
    for _ in range(round(_RESET_TIMEOUT * ticks.setup.rate)):
        state, _ = ticks.send(keyframe)
        if np.abs(state.q - STAND_POSE).max() < _RESET_TOLERANCE:
            break
    else:
        print("  the simulator did not reset to its keyframe in time; carrying on")
    ticks.check_tilt = True


def _end_sequence(
    ticks: _Ticks,
    call: Callable[[int, float, float], JointCommand],
    k: int,
    scale: float,
    setup: TrialSetup,
) -> None:
    """Wind the stride down to zero, hold the stand, lower to the lying pose.

    With `setup.release_ramp` the legs then let go gradually: the pose and the damping
    are held while `kp` eases to zero, so the weight they still carry settles onto the
    ground instead of dropping in the release's one tick.

    `call(k, scale, scale_dot)` is the command function with its time and
    normalisation filled in. The step index keeps counting from where the loop
    stopped, so a clock or a network sees an unbroken run of ticks.
    """
    ticks.begin("wind down")
    n = round(np.ceil(scale * setup.ramp * setup.rate)) if setup.ramp > 0 else 0
    for i in range(n):
        ticks.send(
            call(
                k,
                max(0.0, scale - (i + 1) / (setup.ramp * setup.rate)),
                -1.0 / setup.ramp,
            )
        )
        k += 1

    ticks.begin("hold")
    for _ in range(round(setup.hold * setup.rate)):
        ticks.send(call(k, 0.0, 0.0))
        k += 1

    ticks.begin("lower")
    ticks.ease(
        JointCommand.position_hold(LYING_POSE, kp=setup.kp, kd=setup.kd),
        setup.lower,
        smooth=setup.release_ramp > 0,
    )

    if setup.release_ramp > 0:
        ticks.begin("release ramp")
        limp = replace(
            ticks.last,
            kp=np.zeros_like(ticks.last.kp),
            kd=np.full_like(ticks.last.kd, setup.release_kd),
        )
        ticks.ease(limp, setup.release_ramp)


def run_trial(
    command: CommandFn,
    steps: int,
    setup: TrialSetup,
    stand: np.ndarray | JointCommand | None = None,
) -> TrialResult:
    """Run the sequence in the module docstring, with `steps` control periods of loop.

    `stand` is what the gait stands in: its zero-amplitude pose, (12,), held with the
    trial's gains, or its zero-amplitude command (`control.stand_command`) with its
    own gains and gravity load, which is what a gait that loads its stand gives. It
    defaults to the model's own keyframe, which is right for a command with no gait
    behind it. The loop's arrays and the report cover the loop only; the start and
    the end are in `TrialResult.stages` and, for a recording backend, its records.

    Refuses to run against two simulators, since they share rt/lowcmd and
    rt/lowstate: readings would interleave between two unrelated robots while
    both receive the commands, and every number below would be meaningless.
    """
    stand = STAND_POSE if stand is None else stand
    if isinstance(stand, JointCommand):
        stand_cmd = stand
    else:
        stand_cmd = JointCommand.position_hold(
            np.asarray(stand, dtype=float), kp=setup.kp, kd=setup.kd
        )
    check_single_peer()
    refuse_if_sport_mode_active(setup.domain_id, setup.interface)
    refuse_without_support(setup.domain_id, setup.support)
    dt = 1.0 / setup.rate
    # git is a subprocess, so it goes before the backend, where a pause costs nothing
    commit, dirty = source_version()

    # The first rotation imports scipy, about 250 ms, and it used to land inside the
    # trial, where the silence froze the legs and the body jumped 2 cm.
    rotation_matrices(np.array([1.0, 0.0, 0.0, 0.0]))

    command_ticks: list[int] = []  # the tick each call of the command function lands on

    def call(k: int, scale: float, scale_dot: float) -> JointCommand:
        out = command(k, k * dt, scale, scale_dot)
        command_ticks.append(ticks.count)  # after, so a call that raises is not a tick
        if isinstance(out, JointCommand):
            return out
        return JointCommand.position_hold(out, kp=setup.kp, kd=setup.kd)

    # connect BEFORE resetting. Constructing the backend blocks waiting for the
    # first LowState, and a reset issued before that would leave the robot
    # untorqued and collapsing for the whole of that wait.
    with (
        DDSBackend(
            domain_id=setup.domain_id,
            interface=setup.interface,
            control_dt=dt,
            release_kd=setup.release_kd,
        ) as backend,
        _CtrlC() as ctrl_c,
    ):
        ticks = _Ticks(backend, setup, ctrl_c)
        errors, quats, positions, measured_qs, commanded_qs = [], [], [], [], []
        aborted, crash = None, None
        stand_error, stand_height = float("nan"), None
        start_pos, forward_axis = None, np.array([1.0, 0.0, 0.0])
        scale = 0.0

        try:
            if setup.reset and reset_bridge():
                _wait_for_reset(ticks)

            # On hardware the robot starts wherever it was left, often on the ground,
            # and the simulator starts on the keyframe, which is not the gait's stand.
            ticks.begin("stand ramp")
            state, _ = backend.read()
            ticks.last = JointCommand.position_hold(state.q, kp=setup.kp, kd=setup.kd)
            ticks.ease(stand_cmd, setup.stand_ramp)

            ticks.begin("settle")
            ticks.hold(
                stand_cmd, max(setup.settle, dt)
            )  # at least one tick, to read from
            state, truth = backend.read()
            stand_error = float(np.abs(state.q - stand_cmd.q).max())
            stand_height = float(truth.base_pos[2]) if truth is not None else None
            if stand_height is not None:
                print(
                    f"\nstanding at {stand_height:.3f} m "
                    f"(max |q - stand| {stand_error:.3f} rad)"
                )
            else:
                print(f"\nstanding (max |q - stand| {stand_error:.3f} rad)")

            start_pos = truth.base_pos.copy() if truth is not None else None
            forward_axis = body_forward_axis(state.imu_quat)

            backend.reset_timing()
            ticks.begin("loop")
            for k in range(steps):
                if ctrl_c.presses:
                    aborted = f"Ctrl+C at step {k}/{steps}; ending the trial"
                    print(f"\n{aborted} (press again to release at once)")
                    break
                ticks.where = f"loop step {k + 1}/{steps}"
                scale, scale_dot = _ramp_scale(k * dt, setup.ramp)
                cmd = call(k, scale, scale_dot)
                st, tr = ticks.send(cmd)
                errors.append(float(np.abs(st.q - cmd.q).max()))
                quats.append(st.imu_quat)
                measured_qs.append(st.q.copy())
                commanded_qs.append(cmd.q.copy())
                positions.append(
                    tr.base_pos.copy() if tr is not None else np.full(3, np.nan)
                )
        except _Release as why:
            aborted = str(why)
            run_end_sequence = False
        except Exception as exc:
            aborted = f"crashed during {ticks.where}: {exc!r}; legs released"
            crash, run_end_sequence = exc, False
        else:
            run_end_sequence = setup.end_sequence

        _, tr = backend.read()
        end_pos = tr.base_pos.copy() if tr is not None else None
        if run_end_sequence:
            try:
                _end_sequence(ticks, call, len(errors), scale, setup)
            except _Release as why:
                aborted = str(why)
            except Exception as exc:
                aborted = f"crashed during {ticks.where}: {exc!r}; legs released"
                crash = exc

        positions_arr = np.asarray(positions)
        has_truth = len(positions) > 0 and np.isfinite(positions_arr).any()
        base_pos = positions_arr if has_truth else None
        finite = np.isfinite(positions_arr).all(axis=1) if has_truth else None

        if start_pos is None and base_pos is not None and finite.any():
            start_pos = base_pos[finite][0].copy()
        if end_pos is None and base_pos is not None and finite.any():
            end_pos = base_pos[finite][-1].copy()

        done = len(errors)
        result = TrialResult(
            tracking_error=np.asarray(errors),
            quats=np.asarray(quats),
            base_pos=base_pos,
            start_pos=start_pos,
            end_pos=end_pos,
            forward_axis=forward_axis,
            stand_height=stand_height,
            stand_error=stand_error,
            steps=done,
            rate=setup.rate,
            timing=backend.timing_report(done),
            overruns=backend.overruns,
            measured_q=np.asarray(measured_qs) if measured_qs else None,
            commanded_q=np.asarray(commanded_qs) if commanded_qs else None,
            aborted=aborted,
            stages=tuple(ticks.stages),
            record=ticks.record() if ticks.commands else None,
            command_ticks=np.asarray(command_ticks, dtype=int),
            commit=commit,
            dirty=dirty,
            setup=setup,
        )
        if crash is not None:
            raise TrialCrashed(result) from crash
        return result
