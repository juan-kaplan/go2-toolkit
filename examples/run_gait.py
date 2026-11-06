"""Drive the simulated Go2 on a generated gait, from a clock.

The foot path comes from `go2_toolkit.control`, so stance contact is non-slip
by construction and swing clearance is whatever we ask for.

    python tools/run_bridge.py --viewer &
    uv run python examples/run_gait.py --gait walk

Phase advances on wall clock here, one cycle per `--period` seconds. Any other
source can supply it instead, which is the point of taking phase as the input
rather than time: nothing in the generator knows where phase comes from.

The number to watch is the speed ratio. The generator derives a body speed from
the stride it sweeps, and a foot that stays planted moves the robot at exactly
that speed. A ratio well under 1 means the feet are slipping, which is the
failure to look for.
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import numpy as np

from examples.common import (
    GaitTrialConfig,
    add_arguments,
    add_hardware_arguments,
    hardware_options,
)
from go2_toolkit import control
from go2_toolkit.robot.metrics import (
    FALLEN_FRACTION,
    MIN_EXPECTED_DISTANCE,
    final_height,
    speed_metrics,
)
from go2_toolkit.robot.session import TrialResult

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# How far the ground sits below the foot point the IK targets: the foot geom is a
# 0.022 m sphere centred on that point (go2.xml, class foot), plus roughly 2 mm of
# contact softness in the simulator's stance. Body height is the asked hip height
# plus this.
_GROUND_BELOW_FOOT_POINT = 0.024


def _print_setup(c: GaitTrialConfig, params: control.GaitParams) -> None:
    """The gait's parameters and what they imply, before the trial starts."""
    offsets = c.offsets
    print(
        f"{c.gait}: {control.phase_count(offsets)} phase, "
        f"offsets {np.round(offsets, 3)} (FR FL RR RL)"
    )
    print(
        f"  stride {params.step_length:g} m, clearance {params.swing_height:g} m, "
        f"duty {params.stance_duration:g}, hip height {params.nominal_height:g} m"
    )
    print(f"  cycle {params.cycle_period:g} s -> body speed {params.body_speed:.3f} m/s")
    if c.gravity_comp:
        print("  gravity compensation: active (open-loop tau_ff)")
    if c.velocity_feedforward:
        print("  velocity feedforward: active (analytical dq)")

    feet_down = control.support_count(offsets, params.stance_duration)
    print(
        f"  feet down: {feet_down.min()} to {feet_down.max()}"
        + (
            "  (statically stable throughout)"
            if feet_down.min() >= 3
            else "  (needs dynamic balance)"
        )
    )


def _outcome_lines(
    result: TrialResult, params: control.GaitParams, warmup: float
) -> list[str]:
    """Distance, speed against the stride's promise, and whether the robot fell."""
    lines = []
    speed = speed_metrics(result, params.body_speed, warmup)
    if speed.has_truth:
        elapsed, along = speed.elapsed_time, speed.distance_along_x
        lines.append(
            f"{'travelled':<16}{along:+.2f} m along body x in "
            f"{elapsed:.1f} s ({along / elapsed:+.3f} m/s)"
        )
        if speed.expected_distance > MIN_EXPECTED_DISTANCE:
            lines.append(
                f"{'total speed':<16}{speed.total_speed_ratio:.2f} of the "
                f"{speed.expected_distance:.2f} m the stride should have "
                "covered"
            )
        steady_expected = speed.steady_expected_distance
        if speed.steady_is_measured and steady_expected > MIN_EXPECTED_DISTANCE:
            lines.append(
                f"{'steady speed':<16}{speed.steady_speed_ratio:.2f} of the "
                f"{steady_expected:.2f} m expected after "
                f"{warmup:.1f} s warmup "
                f"({speed.steady_distance / speed.steady_time:+.3f} m/s)"
            )

    # the post-loop reading, not the last in-loop sample, matching what this
    # printed before the session existed
    final = final_height(result)
    if final is not None and final < FALLEN_FRACTION * params.nominal_height:
        lines.append(" " * 16 + "robot is down")
    return lines


def run(
    config: GaitTrialConfig,
    reset: bool = True,
    record_video: str | bool = False,
    **hardware,
) -> None:
    """Run one trial of `config`, print its report, and optionally record a video.

    `record_video` is a path, or True for `results/<gait>_<time>.mp4`. `hardware`
    are the `TrialSetup` options for the physical robot, plus `log=False` to skip
    the trial log, which is written to `results/trials/` otherwise.
    """
    params = config.gait_params()
    _print_setup(config, params)

    result = config.run(reset, **hardware)

    print(
        result.report(
            height_label="body height",
            height_note=(
                f"   (asked {params.nominal_height:g} hip to foot, so expect "
                f"{params.nominal_height + _GROUND_BELOW_FOOT_POINT:.3f} at the "
                "body)"
            ),
            extra=tuple(_outcome_lines(result, params, config.warmup)),
        )
    )

    if result.log_path:
        print(f"\ntrial log {result.log_path}")

    if record_video:
        if record_video is True:
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            record_video = RESULTS_DIR / f"{config.gait}_{stamp}.mp4"
        print(f"\nrecording trial video to {record_video} ...", flush=True)
        out_path = result.record_video(record_video)
        print(f"video saved to {out_path}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    add_arguments(p)
    add_hardware_arguments(p)
    p.add_argument(
        "--no-log",
        action="store_true",
        help="do not write the trial log (default: results/trials/)",
    )
    p.add_argument(
        "--record-video",
        "--record",
        dest="record_video",
        nargs="?",
        const=True,
        default=False,
        help="record MP4 video of the experiment (default: results/<gait>_<time>.mp4)",
    )
    a = p.parse_args()
    run(
        GaitTrialConfig.from_args(a),
        reset=not a.no_reset,
        record_video=a.record_video,
        log=not a.no_log,
        **hardware_options(a),
    )
