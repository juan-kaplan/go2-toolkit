"""Talking to the robot: the wire types, the transport, and the trial loop.

    interface.py   RobotState, GroundTruth: what a reading is. (`JointCommand`,
                   what is sent, is in `go2/command.py`.)
    dds.py         DDSBackend. The only place DDS appears.
    bridge.py      finding and resetting the simulator process.
    session.py     the connect / stand / ramp / loop / report sequence.
    trial_log.py   the file a trial leaves behind: arrays and metadata.
    sport_mode.py  hardware only: handing the Go2's motors to LowCmd, and back.

Everything here may import `go2`, and nothing here may import `control`. A
trial sends whatever joint angles it is handed; where they come from is not the
transport's business. That is what lets the same `Session` drive a generated
gait, a recorded clip, and later a network, without knowing the difference.

Importing this package needs no SDK: it loads only `interface.py`. The SDK
arrives with `robot.dds`, `session` and `sport_mode`, which is why the top level
resolves `DDSBackend` lazily. See the note in `go2_toolkit/__init__.py`.
"""

from __future__ import annotations

from .interface import GroundTruth, RobotState

__all__ = ["GroundTruth", "RobotState"]
