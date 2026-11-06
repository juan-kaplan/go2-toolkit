"""Index orders and frame conventions, and the permutations between them.

This project reads from three sources that each order the same twelve joints
differently, and a wrong permutation does not raise. It mirrors the robot, or
swaps front for rear, and produces motion that looks deliberate. Every ordering
fact lives here so there is exactly one place to check when a leg moves wrong.

The quaternion helpers are here for the same reason: scalar-first versus
scalar-last is a convention, the two differ only by a roll, and getting it wrong
gives a rotation that is plausible rather than obviously broken. That shim was
written out by hand in four separate files before this module existed.

numpy only, and scipy is imported inside the functions that need it, so that
importing a convention costs nothing.
"""

from __future__ import annotations

import numpy as np

# Unitree LowCmd/LowState motor indexing: leg0=FR, leg1=FL, leg2=RR, leg3=RL,
# and within a leg hip, thigh, calf. Confirmed against the SDK's own LegID table
# (unitree_sdk2_python/example/go2/low_level/unitree_legged_const.py). The
# unitree_mujoco model declares its sensors in this same order, so indices are
# consistent from the simulator through to hardware.
# fmt: off
JOINT_NAMES = (
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
)
# fmt: on
NUM_JOINTS = len(JOINT_NAMES)

# Leg order for any per-leg array in this project, matching the above.
LEG_NAMES = ("FR", "FL", "RR", "RL")
NUM_LEGS = len(LEG_NAMES)

# The MJCF models list legs FL, FR, RL, RR, differing from the wire order by a
# pairwise leg swap. That is the order of `qpos` in unitree_mujoco's own go2.xml
# (its kinematic tree starts at FL) as well as in mujoco_menagerie, even though
# unitree_mujoco's actuators and sensors run FR, FL, RR, RL like the wire.
# `robot/video.py` uses this to write measured joints into `qpos`; arrays taken
# from either model must be permuted or the robot is mirrored left/right with no
# error raised. The permutation is its own inverse, so it converts both ways.
#
#     unitree_ordered = menagerie_ordered[MENAGERIE_SWAP]
MENAGERIE_SWAP = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])


def wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    """Scalar first to scalar last, the convention scipy expects.

    Unitree's IMU and the MuJoCo model both report (w, x, y, z). scipy's
    Rotation takes (x, y, z, w). The two differ only by rolling the array one
    place, which is exactly why mixing them up produces a rotation that looks
    reasonable instead of one that obviously fails.
    """
    q = np.asarray(quat_wxyz, dtype=float)
    return np.roll(q, -1, axis=-1)


def rotation_matrices(quat_wxyz: np.ndarray) -> np.ndarray:
    """Scalar first quaternion(s) to rotation matrix/matrices, (..., 3, 3)."""
    from scipy.spatial.transform import Rotation as Rot

    q = np.atleast_2d(wxyz_to_xyzw(quat_wxyz))
    m = Rot.from_quat(q).as_matrix()
    return m[0] if np.ndim(quat_wxyz) == 1 else m


def rpy_degrees(quat_wxyz: np.ndarray) -> np.ndarray:
    """Scalar first quaternion(s) to roll, pitch, yaw in degrees."""
    from scipy.spatial.transform import Rotation as Rot

    q = np.atleast_2d(wxyz_to_xyzw(quat_wxyz))
    return Rot.from_quat(q).as_euler("xyz", degrees=True)


def body_forward_axis(quat_wxyz: np.ndarray) -> np.ndarray:
    """The body's own x axis expressed in world coordinates, (3,).

    Distance travelled is projected onto this rather than onto world x, so a
    robot that drifts in yaw is still measured along the direction it is facing.
    """
    return np.asarray(rotation_matrices(np.asarray(quat_wxyz).reshape(4)))[:, 0]
