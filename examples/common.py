"""What the gait examples share: one config, one command function, the command line.

`GaitTrialConfig` holds everything that defines one trial of a generated gait and
turns it into a `TrialSetup` and a command function for `robot.session.run_trial`.
`add_arguments` and `add_hardware_arguments` declare the command line once.

The defaults are the ones from `control` (`control.DEFAULT_KP` and friends), with each
gait's own duty factor and period. The hardware options (`stand_ramp`, `max_tilt`,
`release_kd`, ...) are not part of the config: `hardware_options` turns them into the
keywords `run` and `trial_setup` pass to `TrialSetup`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from go2_toolkit import control
from go2_toolkit.robot.session import CommandFn, TrialResult, TrialSetup, run_trial
from go2_toolkit.robot.trial_log import SUPPORTS, write_trial_log


@dataclass
class GaitTrialConfig:
    """Everything that defines one trial of a generated gait."""

    # fmt: off
    gait: str = "walk"
    step_length: float = control.DEFAULT_STEP_LENGTH    # m per stride
    swing_height: float = control.DEFAULT_SWING_HEIGHT  # m of foot clearance
    duty: float | None = None                           # None: the gait's own
    height: float = control.DEFAULT_HEIGHT              # m, hip above the foot
    period: float | None = None                         # None: the gait's own
    kp: float = control.DEFAULT_KP
    kd: float = control.DEFAULT_KD
    warmup: float = control.DEFAULT_WARMUP              # s growing the stride from zero
    settle: float = 1.0                                 # s held on the stand first
    cycles: float = 8.0
    rate: float = 200.0                                 # control rate, Hz
    gravity_comp: bool = True
    velocity_feedforward: bool = True
    domain_id: int = 1
    interface: str = "lo"
    # fmt: on

    def __post_init__(self) -> None:
        if self.gait not in control.GAITS:
            raise ValueError(
                f"unknown gait {self.gait!r}; have {', '.join(control.GAITS)}"
            )
        if self.duty is None:
            self.duty = control.DEFAULT_DUTY[self.gait]
        if self.period is None:
            self.period = control.DEFAULT_PERIOD[self.gait]

    @property
    def offsets(self) -> np.ndarray:
        """Per-leg phase offsets of this gait, in Unitree leg order."""
        return control.GAITS[self.gait]

    def gait_params(self) -> control.GaitParams:
        return control.GaitParams(
            step_length=self.step_length,
            swing_height=self.swing_height,
            stance_duration=self.duty,
            nominal_height=self.height,
            cycle_period=self.period,
        )

    @property
    def steps(self) -> int:
        """Control steps in `cycles` full cycles at this period and rate."""
        return int(self.cycles * self.period * self.rate)

    def trial_setup(self, reset: bool = True, **options) -> TrialSetup:
        """How the session drives this trial. `options` are `TrialSetup`'s own fields
        (the hardware ones, `end_sequence`, ...), left to it to default."""
        return TrialSetup(
            rate=self.rate,
            kp=self.kp,
            kd=self.kd,
            settle=self.settle,
            ramp=self.warmup,
            domain_id=self.domain_id,
            interface=self.interface,
            reset=reset,
            **options,
        )

    def stand(self) -> control.JointCommand:
        """The gait's own stand, what the session stands in and ramps out of.

        A command, not only a pose: it carries the gravity load and the gains the
        loop's first command has, so the stand does not sag and nothing steps at the
        join.
        """
        return control.stand_command(
            self.gait_params(), kp=self.kp, kd=self.kd, gravity_comp=self.gravity_comp
        )

    def run(
        self, reset: bool = True, log: bool = True, label: str | None = None, **options
    ) -> TrialResult:
        """One trial of this gait; the session owns the stride ramp and the end.

        The trial log is written after the release and its path is on the result
        (`log_path`); `log=False` writes none, for the tuner, which runs hundreds of
        headless trials and keeps its own record. The gait's leg state each tick goes
        into the log as extra channels. `label` names the log (default: the gait).
        """
        channels: dict[str, list] = {"is_stance": [], "progress": [], "rate": []}
        result = run_trial(
            self.command_fn(channels),
            self.steps,
            self.trial_setup(reset, **options),
            self.stand(),
        )
        if log:
            result.log_path = write_trial_log(
                result, "run_gait", label or self.gait, self, channels
            )
        return result

    def command_fn(self, channels: dict[str, list] | None = None) -> CommandFn:
        """The function the session calls each step: time and amplitude in, command out.

        Phase advances on the wall clock here (`control.clock_leg_state`). Another
        phase source plugs in by giving the same leg state; the session does not
        know the difference. The session passes the stride amplitude, because it
        also owns the ramp down at the end.

        `channels` is how a driver adds to the trial log: a list per channel, and
        each call appends one value. Here that is the clock's leg state, the same
        three values any phase source gives (`is_stance`, `progress`, `rate`, one per
        leg in Unitree order); the log puts them on the ticks of the calls.
        """
        offsets, params = self.offsets, self.gait_params()

        def command(
            k: int, t: float, scale: float, scale_dot: float
        ) -> control.JointCommand:
            stance, progress, rate = control.clock_leg_state(t, offsets, params)
            if channels is not None:
                channels["is_stance"].append(stance)
                channels["progress"].append(progress)
                channels["rate"].append(rate)
            return control.state_joint_command(
                stance,
                progress,
                rate,
                params,
                scale=scale,
                scale_dot=scale_dot,
                kp=self.kp,
                kd=self.kd,
                gravity_comp=self.gravity_comp,
                velocity_feedforward=self.velocity_feedforward,
            )

        return command

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> GaitTrialConfig:
        """Build from a namespace `add_arguments` filled in, for the fields it sets."""
        return cls(**{name: getattr(args, name) for name in _CLI_FIELDS})


# The fields the shared command line options set, by dataclass field name.
# fmt: off
_CLI_FIELDS = (
    "gait", "step_length", "swing_height", "duty", "height", "period", "kp", "kd",
    "warmup", "settle", "cycles", "rate", "gravity_comp", "velocity_feedforward",
    "domain_id", "interface",
)
# fmt: on


def add_gait_arguments(p: argparse.ArgumentParser) -> None:
    """Add the stride and the gains, with `GaitTrialConfig`'s defaults."""
    d = GaitTrialConfig()
    p.add_argument("--step-length", type=float, default=d.step_length)
    p.add_argument("--swing-height", type=float, default=d.swing_height)
    p.add_argument("--kp", type=float, default=d.kp)
    p.add_argument("--kd", type=float, default=d.kd)


def add_arguments(p: argparse.ArgumentParser) -> None:
    """Add the options the examples share, with `GaitTrialConfig`'s defaults."""
    d = GaitTrialConfig()
    p.add_argument("--gait", default=d.gait, choices=sorted(control.GAITS))
    add_gait_arguments(p)
    p.add_argument(
        "--duty",
        type=float,
        default=None,
        help="stance fraction (default: canonical for the gait, e.g. "
        "0.78 for walk, 0.5 for trot)",
    )
    p.add_argument("--height", type=float, default=d.height, help="hip above the foot")
    p.add_argument(
        "--period",
        type=float,
        default=None,
        help="seconds per cycle (default: canonical for the gait, e.g. "
        "1.0 for walk, 0.5 for trot)",
    )
    p.add_argument("--cycles", type=float, default=d.cycles)
    p.add_argument(
        "--warmup",
        type=float,
        default=d.warmup,
        help="seconds spent growing the stride from zero",
    )
    p.add_argument("--settle", type=float, default=d.settle)
    p.add_argument("--rate", type=float, default=d.rate, help="control rate, Hz")
    p.add_argument("--domain-id", type=int, default=d.domain_id)
    p.add_argument("--interface", default=d.interface)
    p.add_argument(
        "--gravity-comp",
        action=argparse.BooleanOptionalAction,
        default=d.gravity_comp,
        help="open-loop gravity feedforward torques (tau_ff)",
    )
    p.add_argument(
        "--velocity-feedforward",
        action=argparse.BooleanOptionalAction,
        default=d.velocity_feedforward,
        help="analytical velocity feedforward (dq)",
    )


# fmt: off
_HARDWARE_FIELDS = ("stand_ramp", "max_tilt", "release_kd", "release_ramp", "support")
# fmt: on


def add_hardware_arguments(p: argparse.ArgumentParser, max_tilt=None) -> None:
    """Add the options for the physical robot, with `TrialSetup`'s defaults.

    The defaults leave the simulator's behaviour unchanged. `max_tilt` replaces the
    default of no limit, for a driver that wants one in the simulator too.
    """
    d = TrialSetup()
    p.add_argument("--no-reset", action="store_true", help="hardware: no bridge to reset")
    p.add_argument(
        "--stand-ramp",
        type=float,
        default=d.stand_ramp,
        help="seconds to ease from the robot's current pose into the "
        "stand before starting (hardware: use 3 or more)",
    )
    p.add_argument(
        "--max-tilt",
        type=float,
        default=d.max_tilt if max_tilt is None else max_tilt,
        help="release the legs at once past this roll/pitch in degrees",
    )
    p.add_argument(
        "--release-kd",
        type=float,
        default=d.release_kd,
        help="damping at the release; 0 is no torque, which on hardware is free fall",
    )
    p.add_argument(
        "--release-ramp",
        type=float,
        default=d.release_ramp,
        help="seconds over which the legs let go at the lying pose, before the "
        "release: kp and kd ease to zero and --release-kd, and the lowering eases "
        "in and out. 0 releases in one tick (hardware: try 1)",
    )
    p.add_argument(
        "--support",
        choices=SUPPORTS,
        default=d.support,
        help="hardware only, and required there: whether the robot is lifted off "
        "the ground or stands on the floor. The trial log records it",
    )


def hardware_options(args: argparse.Namespace) -> dict:
    """The `TrialSetup` keywords the hardware options set, from parsed arguments."""
    return {name: getattr(args, name) for name in _HARDWARE_FIELDS}
