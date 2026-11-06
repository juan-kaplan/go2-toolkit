"""Hand the physical Go2's motors to LowCmd, or give them back to the built-in mode.

While the built-in motion service (sport mode) is active it holds the robot and
LowCmd is ignored, so `stand_demo.py` and every measured trial do nothing on
real hardware until it is released. Simulation needs none of this.

    python tools/release_sport_mode.py status  --interface <your-wired-nic>
    python tools/release_sport_mode.py release --interface <your-wired-nic>
    python tools/release_sport_mode.py restore --interface <your-wired-nic> --mode normal

`status` is a read-only query. `release` lowers the robot to the ground first,
then releases; the SDK's own example calls StandDown and ReleaseMode back to
back, which would release a robot that is still standing. Here the release is
refused unless all four thighs read as lowered. `restore` selects a mode again;
`release` prints the mode that was active so you know what to pass.

Releasing leaves the motors with nothing holding them until LowCmd arrives. Keep
`DDSBackend(release_kd=...)` in mind for the end of a run: its default of zero
gains means no torque at all.

The logic lives in `go2_toolkit.robot.sport_mode`; this is a thin argparse
wrapper over it.
"""

from __future__ import annotations

import argparse

from go2_toolkit.robot import sport_mode
from go2_toolkit.robot.dds import HARDWARE_DOMAIN_ID

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("action", choices=("status", "release", "restore"))
    p.add_argument(
        "--interface", required=True, help="network interface facing the robot"
    )
    p.add_argument("--domain-id", type=int, default=HARDWARE_DOMAIN_ID)
    p.add_argument(
        "--mode", default="normal", help="mode for `restore` (normal, ai, advanced)"
    )
    a = p.parse_args()

    # Only sets up the DDS channel factory; does not wait for LowState, which
    # only `release` below needs.
    msc = sport_mode.connect_switcher(a.domain_id, a.interface)

    if a.action == "status":
        sport_mode.status(msc)
    elif a.action == "release":
        sport_mode.release(msc, a.domain_id, a.interface)
    else:
        sport_mode.restore(msc, a.mode)
