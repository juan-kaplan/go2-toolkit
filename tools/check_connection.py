"""Read-only DDS connectivity smoke test. Never publishes anything.

Confirms LowState (and, if sport mode hasn't been released yet, SportModeState)
arrive on the given domain/interface. Safe to run regardless of the robot's
physical position: unlike `stand_demo.py`, this never calls `backend.send()`,
so it cannot command the motors. Run this first, then `stand_demo.py`.

    python tools/check_connection.py --domain-id 0 --interface <your-wired-nic>

`--interface` has no default on purpose: Wi-Fi does not carry DDS to the Go2 (see
docs/hardware_bringup.md), so a default would silently point at an interface
that can never work. `ip -br addr` names your wired adapter.
"""

from __future__ import annotations

import argparse

import numpy as np

from go2_toolkit import DDSBackend
from go2_toolkit.robot.dds import HARDWARE_DOMAIN_ID


def run(domain_id: int, interface: str, timeout: float) -> None:
    print(
        f"connecting: domain={domain_id} interface={interface} "
        f"(waiting up to {timeout:.0f}s for rt/lowstate)"
    )
    # Never call backend.release(): that publishes a zero-gain LowCmd to
    # rt/lowcmd, and this tool's entire point is to touch nothing. There is
    # nothing else to clean up, so it is just left to be garbage collected.
    backend = DDSBackend(domain_id=domain_id, interface=interface, state_timeout=timeout)
    state, truth = backend.read()
    print("connected: LowState received")
    print(f"  joint q [0:3]     {np.round(state.q[:3], 3)}")
    print(f"  imu quat (wxyz)   {np.round(state.imu_quat, 3)}")
    print(f"  imu gyro          {np.round(state.imu_gyro, 3)}")
    if truth is not None:
        print(f"  base pos (SportModeState)  {np.round(truth.base_pos, 3)}")
    else:
        print(
            "  SportModeState: none received "
            "(expected if the robot is already in low-level mode)"
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--domain-id",
        type=int,
        default=HARDWARE_DOMAIN_ID,
        help="0 for physical hardware, 1 for the simulator",
    )
    p.add_argument(
        "--interface",
        type=str,
        required=True,
        help="wired network interface facing the robot, from `ip -br addr`",
    )
    p.add_argument("--timeout", type=float, default=10.0)
    a = p.parse_args()
    run(domain_id=a.domain_id, interface=a.interface, timeout=a.timeout)
