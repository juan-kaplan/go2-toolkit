"""The physical Go2: dimensions, travel, torque, and its rest pose.

Every number here is read from the simulator's own model file rather than from a
datasheet, and each one cites the line it came from:

    third_party/unitree_mujoco/unitree_robots/go2/go2.xml

That matters because the tests compare this package's inverse kinematics against
MuJoCo's forward kinematics on that exact model. A datasheet value that differs
in the fourth decimal would make an exact test fail for a reason that looks like
a bug in the maths.

These constants used to live in three places: the leg geometry in gait.py, the
IMU offset in interface.py, and the stand keyframe copied byte for byte into
three separate files. A robot has one set of dimensions, so it gets one module.

numpy only. Nothing here knows about DDS, clips, or control.
"""

from __future__ import annotations

import numpy as np

from .conventions import NUM_JOINTS, NUM_LEGS

# -- link geometry ------------------------------------------------------------
# go2.xml:95   <body name="FL_calf" pos="0 0 -0.213">   hip pitch axis to knee
THIGH_LENGTH = 0.213
# knee to the foot geom's centre, same offset down the calf
CALF_LENGTH = 0.213
# go2.xml:86   <body name="FL_thigh" pos="0 0.0955 0">
# lateral gap between the hip roll axis and the leg's own sagittal plane
ABDUCTION_OFFSET = 0.0955
# go2.xml:78   <body name="FL_hip" pos="0.1934 0.0465 0">
# fmt: off
HIP_X = 0.1934      # forward of base centre
HIP_Y = 0.0465      # lateral of base centre
# fmt: on

# Longest straight-line distance a foot can be from its hip pitch axis. Reaching
# it means a fully straight knee, which is a singularity, so targets must stay
# strictly inside.
MAX_REACH = THIGH_LENGTH + CALF_LENGTH

# Left legs carry a positive abduction offset and right legs a negative one.
# Getting this wrong mirrors the robot without raising anything.
LEG_SIGN_Y = np.array([-1.0, 1.0, -1.0, 1.0])  # FR, FL, RR, RL

# fmt: off
HIP_ORIGINS = np.array([
    [HIP_X, -HIP_Y, 0.0],       # FR
    [HIP_X, HIP_Y, 0.0],        # FL
    [-HIP_X, -HIP_Y, 0.0],      # RR
    [-HIP_X, HIP_Y, 0.0],       # RL
])
# fmt: on

# -- travel -------------------------------------------------------------------
# From go2.xml's default classes, which is why front and rear thighs differ:
#   go2.xml:11  class="abduction"   range="-1.0472 1.0472"   every hip
#   go2.xml:15  class="front_hip"   range="-1.5708 3.4907"   FR/FL thigh
#   go2.xml:18  class="back_hip"    range="-0.5236 4.5379"   RR/RL thigh
#   go2.xml:22  class="knee"        range="-2.7227 -0.83776" every calf
# The rear thigh can swing an extra radian forward and a radian less back. A
# trajectory that is legal at the front can therefore be illegal at the rear,
# which is not a symmetry you would guess from looking at the robot.
_HIP = (-1.0472, 1.0472)
_FRONT_THIGH = (-1.5708, 3.4907)
_REAR_THIGH = (-0.5236, 4.5379)
_KNEE = (-2.7227, -0.83776)

# fmt: off
JOINT_LIMITS = np.array([
    _HIP, _FRONT_THIGH, _KNEE,      # FR
    _HIP, _FRONT_THIGH, _KNEE,      # FL
    _HIP, _REAR_THIGH, _KNEE,       # RR
    _HIP, _REAR_THIGH, _KNEE,       # RL
])
# fmt: on

# go2.xml:10  <motor ctrlrange="-23.7 23.7"/>     hip and thigh
# go2.xml:24  <motor ctrlrange="-45.43 45.43"/>   calf, inside class="knee"
# Worth knowing when choosing kp: a position error of e rad asks for kp * e Nm,
# and anything past these saturates. At kp=100 a 0.24 rad error is already at
# the hip and thigh limit, which is how the startup snap became a real impact.
TORQUE_LIMITS = np.array([[23.7, 23.7, 45.43]] * NUM_LEGS).reshape(NUM_JOINTS)

# Sum of every body's mass in go2.xml, 15.206 kg, rounded: what the gravity
# feedforward has to hold up.
TOTAL_MASS = 15.2

# -- poses --------------------------------------------------------------------
# go2.xml keyframe "home", its `ctrl` field:
#     ctrl="0 0.9 -1.8  0 0.9 -1.8  0 0.9 -1.8  0 0.9 -1.8"
# The pose the simulator resets to, and the one every trial starts from. Used to
# be copied into simctl.py, stand_demo.py and test_backend.py independently.
STAND_POSE = np.array([0.0, 0.9, -1.8] * NUM_LEGS)

# Unitree's own lying pose, from their low-level example (unitree_sdk2_python/
# example/go2/low_level/go2_stand_example.py): thigh 1.36, calf -2.65, hips
# straight. It is where the robot's built-in StandDown ends up, and where a trial
# lowers the robot to before it lets go (G3).
LYING_POSE = np.array([0.0, 1.36, -2.65] * NUM_LEGS)

# -- sensor frames ------------------------------------------------------------
# go2.xml  <site name="imu" pos="-0.02557 0 0.04232" />
# The simulator's pose sensor is a framepos on this site, not on base_link, so
# GroundTruth.base_pos reads about 4 cm higher than the body origin. Comparing
# it against a kinematic hip height silently overstates the robot by that much.
# Subtract this before comparing the two; relative comparisons between trials
# are unaffected.
IMU_SITE_OFFSET = np.array([-0.02557, 0.0, 0.04232])


def within_limits(q: np.ndarray) -> np.ndarray:
    """Per joint mask of whether an angle is inside this robot's travel."""
    q = np.asarray(q, dtype=float).reshape(NUM_JOINTS)
    return (q >= JOINT_LIMITS[:, 0]) & (q <= JOINT_LIMITS[:, 1])
