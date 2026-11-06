"""Ramp the Go2 into a stand and hold it. Smoke test for the DDS path.

Nothing here knows whether it is talking to the simulator or the robot, which
is the property the whole setup exists to provide. Start a peer first:

    python tools/run_bridge.py &
    python tools/stand_demo.py

Deliberately does NOT use `robot.session`. This is the minimal example of the
layer below it: one backend, one loop, no reset, no peer check, ramping from
wherever the robot currently is rather than from a keyframe. Read it to see
what `DDSBackend` does; read `robot/session.py` for how a measured trial is
actually run.

On hardware run `check_connection.py` first: it only reads, so it is safe from
any pose, whereas this commands the motors. The default gains (kp 60, kd 6) are
soft on purpose, enough for a smoke test; trials use stiffer ones.

One check survives the minimalism: on hardware, sport mode being active means
this ramps into nothing. That is not hypothetical, it is what the first
hardware session actually hit, so it fails loudly instead of running to
completion and reporting numbers for a robot that never moved.
"""

from __future__ import annotations

import argparse

import numpy as np

from go2_toolkit import STAND_POSE, DDSBackend, JointCommand
from go2_toolkit.robot.sport_mode import refuse_if_sport_mode_active


def run(
    ramp_s: float = 1.5,
    hold_s: float = 2.0,
    kp: float = 60.0,
    kd: float = 6.0,
    domain_id: int = 1,
    interface: str = "lo",
    release_kd: float = 0.0,
) -> None:
    refuse_if_sport_mode_active(domain_id, interface)
    with DDSBackend(
        domain_id=domain_id, interface=interface, release_kd=release_kd
    ) as backend:
        start, _ = backend.read()
        start_q = start.q.copy()
        print(f"connected; starting from q[0:3]={np.round(start_q[:3], 3)}")

        backend.reset_timing()
        ramp_steps = int(ramp_s / backend.dt)
        hold_steps = int(hold_s / backend.dt)

        for i in range(ramp_steps):
            alpha = (i + 1) / ramp_steps
            target = (1 - alpha) * start_q + alpha * STAND_POSE
            backend.send(JointCommand.position_hold(target, kp=kp, kd=kd))
            backend.step()

        hold_cmd = JointCommand.position_hold(STAND_POSE, kp=kp, kd=kd)
        for _ in range(hold_steps):
            backend.send(hold_cmd)
            backend.step()

        state, truth = backend.read()
        err = np.abs(STAND_POSE - state.q).max()
        print(f"max |joint error| {err:.4f} rad")
        if truth is not None:
            print(f"base height {truth.base_pos[2]:.3f} m")
        print(backend.timing_report(ramp_steps + hold_steps))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ramp", type=float, default=1.5)
    p.add_argument("--hold", type=float, default=2.0)
    p.add_argument("--kp", type=float, default=60.0)
    p.add_argument("--kd", type=float, default=6.0)
    p.add_argument("--domain-id", type=int, default=1)
    p.add_argument("--interface", type=str, default="lo")
    p.add_argument(
        "--release-kd",
        type=float,
        default=0.0,
        help="damping applied when the run ends; 0 means no torque, "
        "which on hardware is free fall",
    )
    a = p.parse_args()
    run(
        ramp_s=a.ramp,
        hold_s=a.hold,
        kp=a.kp,
        kd=a.kd,
        domain_id=a.domain_id,
        interface=a.interface,
        release_kd=a.release_kd,
    )
