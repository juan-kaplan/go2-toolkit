"""Types read back from the robot.

Everything here is in Unitree's wire order, because that is what LowCmd and
LowState actually use and the project talks to the robot over DDS in every
mode. The ordering itself, and the permutations to and from the other two
conventions this project meets, live in `go2.conventions`; the robot's
dimensions and sensor frames live in `go2.model`, and the command sent to it,
`JointCommand`, in `go2.command`. This module is only what a reading is.

See the README for why the low-level layer is LowCmd plus a joint PD.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["GroundTruth", "RobotState"]


@dataclass
class RobotState:
    """Decoded LowState. Identical fields in simulation and on hardware."""

    # fmt: off
    q: np.ndarray          # joint positions [rad]
    dq: np.ndarray         # joint velocities [rad/s]
    tau_est: np.ndarray    # estimated joint torques [N*m]
    imu_quat: np.ndarray   # body orientation, (w, x, y, z)
    imu_gyro: np.ndarray   # body-frame angular velocity [rad/s]
    imu_accel: np.ndarray  # body-frame linear acceleration [m/s^2]
    # fmt: on


@dataclass
class GroundTruth:
    """Pose from SportModeState, for evaluation only.

    In simulation the bridge fills this from the simulator, so it is exact.
    On hardware the same topic carries the onboard state estimator's output,
    which is an estimate, and it stops updating once sport mode is released for
    low-level control. Use it for metrics (cost of transport, distance trotted
    without falling, pitch/roll variation) and never as a controller input, or
    the controller will not transfer.
    """

    # fmt: off
    base_pos: np.ndarray  # world position [m]
    base_vel: np.ndarray  # world linear velocity [m/s]
    # fmt: on
