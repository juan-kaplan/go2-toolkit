"""Foot position to joint angles, and back, and what the leg Jacobian gives.

The Go2's leg is a hip roll about x, a fixed lateral offset, then two parallel
pitch joints. That structure has a closed form solution in both directions, so
neither of these is iterative and neither can fail to converge.

    q = inverse_kinematics(feet_body)      # (4, 3) m -> (12,) rad
    feet_body = forward_kinematics(q)      # (12,) rad -> (4, 3) m

The two are exact inverses on the reachable set, and the tests check that round
trip as well as checking `forward_kinematics` against MuJoCo's own solver on the
same model file the dimensions came from.

Built on the same leg Jacobian, and no longer purely geometry: `joint_velocities`
turns foot velocities into joint velocities, and `gravity_feedforward_torques`
turns the robot's weight into the joint torques that hold it up. Both are
statics and kinematics with no clock in them.
"""

from __future__ import annotations

import numpy as np

from ..go2.conventions import NUM_JOINTS, NUM_LEGS
from ..go2.model import (
    ABDUCTION_OFFSET,
    CALF_LENGTH,
    HIP_ORIGINS,
    LEG_SIGN_Y,
    MAX_REACH,
    THIGH_LENGTH,
    TOTAL_MASS,
)

# Numerical margin on the reachable shell. A foot exactly at full extension is a
# singularity: the knee is straight, the Jacobian drops rank, and the IK's
# arccos sits on its endpoint where small input noise swings the output.
_REACH_MARGIN = 1e-9

STANDARD_GRAVITY = 9.80665  # m/s^2


def _hip_roll(p: np.ndarray, d: float, leg: int) -> tuple[float, float]:
    """Hip roll angle for foot `p` in the hip frame, and the leg's planar length.

    The frontal plane check: the foot must be farther from the roll axis than the
    lateral abduction offset `d`, or no hip angle reaches it. Returns the roll and
    the leg length projected into its own sagittal plane, always a real extension.
    """
    frontal = p[1] ** 2 + p[2] ** 2
    if frontal < d * d:
        raise ValueError(
            f"leg {leg}: foot is {np.sqrt(frontal):.4f} m from the hip roll "
            f"axis in the frontal plane, inside the {abs(d):.4f} m abduction "
            "offset, so no hip angle reaches it"
        )
    planar = np.sqrt(frontal - d * d)
    q0 = np.arctan2(p[2], p[1]) - np.arctan2(-planar, d)
    return q0, planar


def _planar_two_link(fwd: float, planar: float, leg: int) -> tuple[float, float]:
    """Thigh and calf angles for a foot `fwd` ahead of the hip and `planar` below it.

    The reachability check is on the shell between the fully folded and fully
    straight knee, with `_REACH_MARGIN` kept off both ends.
    """
    reach = np.hypot(fwd, planar)
    floor = abs(THIGH_LENGTH - CALF_LENGTH) + _REACH_MARGIN
    if reach > MAX_REACH - _REACH_MARGIN or reach < floor:
        raise ValueError(
            f"leg {leg}: foot is {reach:.4f} m from the hip, outside the "
            f"reachable shell (0, {MAX_REACH:.4f}) m"
        )
    cos_knee = (reach**2 - THIGH_LENGTH**2 - CALF_LENGTH**2) / (
        2.0 * THIGH_LENGTH * CALF_LENGTH
    )
    # negative root: the Go2's knee folds backward, and its calf limit is
    # entirely negative, so the other branch is unreachable on this robot
    q2 = -np.arccos(np.clip(cos_knee, -1.0, 1.0))
    q1 = np.arctan2(-fwd, planar) - np.arctan2(
        CALF_LENGTH * np.sin(q2), THIGH_LENGTH + CALF_LENGTH * np.cos(q2)
    )
    return q1, q2


def inverse_kinematics(feet_body: np.ndarray) -> np.ndarray:
    """Body frame foot positions, (4, 3), to joint angles, (12,) Unitree order.

    Solved per leg as a reachability check in the frontal plane followed by
    planar two link inverse kinematics in the leg's own sagittal plane.

    Raises if a target is out of reach rather than silently clamping, because a
    clamped foot is a trajectory that quietly stops being the one requested.
    """
    feet_body = np.asarray(feet_body, dtype=float).reshape(NUM_LEGS, 3)
    q = np.zeros(NUM_JOINTS)
    for leg in range(NUM_LEGS):
        p = feet_body[leg] - HIP_ORIGINS[leg]  # into the hip frame
        q0, planar = _hip_roll(p, LEG_SIGN_Y[leg] * ABDUCTION_OFFSET, leg)
        q1, q2 = _planar_two_link(p[0], planar, leg)
        q[3 * leg : 3 * leg + 3] = (np.arctan2(np.sin(q0), np.cos(q0)), q1, q2)
    return q


def _roll(q0: float, fwd: float, down: float, lateral: float) -> np.ndarray:
    """Place a sagittal (forward, down) point given hip roll and lateral offset.

    The hip roll axis is the body x axis, so it mixes y and z and leaves x
    alone. `down` is positive downward, hence the sign flip into z.
    """
    c, s = np.cos(q0), np.sin(q0)
    return np.array([fwd, c * lateral + s * down, s * lateral - c * down])


def link_positions(q: np.ndarray) -> np.ndarray:
    """Joint angles, (12,), to the three points of each leg, (4, 3, 3).

    Per leg, in the body frame: the hip origin, the knee, and the foot. Walks
    the same chain the MJCF declares: hip body at the hip origin, thigh offset
    laterally by the abduction offset, calf a thigh length down, foot a calf
    length below that. Positive thigh and calf angles pitch the leg backward,
    which is why the link vectors carry a minus sign in z.

    Returns the whole chain rather than only the foot because drawing a leg
    needs the knee, and a plotting tool that recomputes the chain by hand is a
    picture that can disagree with the robot.
    """
    q = np.asarray(q, dtype=float).reshape(NUM_LEGS, 3)
    out = np.empty((NUM_LEGS, 3, 3))
    for leg in range(NUM_LEGS):
        q0, q1, q2 = q[leg]
        lateral = LEG_SIGN_Y[leg] * ABDUCTION_OFFSET

        knee_f = -THIGH_LENGTH * np.sin(q1)
        knee_d = THIGH_LENGTH * np.cos(q1)
        foot_f = knee_f - CALF_LENGTH * np.sin(q1 + q2)
        foot_d = knee_d + CALF_LENGTH * np.cos(q1 + q2)

        out[leg, 0] = HIP_ORIGINS[leg]
        out[leg, 1] = HIP_ORIGINS[leg] + _roll(q0, knee_f, knee_d, lateral)
        out[leg, 2] = HIP_ORIGINS[leg] + _roll(q0, foot_f, foot_d, lateral)
    return out


def forward_kinematics(q: np.ndarray) -> np.ndarray:
    """Joint angles, (12,) Unitree order, to body frame foot positions, (4, 3).

    The exact inverse of `inverse_kinematics` on the reachable set.
    """
    return link_positions(q)[:, 2]


def leg_jacobian(q_leg: np.ndarray, leg: int) -> np.ndarray:
    """Analytical 3x3 leg Jacobian d([x, y, z]^T) / d([q0, q1, q2]^T) in body frame.

    q_leg: (3,) angles for [hip_roll, thigh_pitch, calf_pitch].
    leg: leg index in [0, 3] (FR, FL, RR, RL).
    """
    q_leg = np.asarray(q_leg, dtype=float).reshape(3)
    q0, q1, q2 = q_leg
    l1 = THIGH_LENGTH
    l2 = CALF_LENGTH
    d = LEG_SIGN_Y[leg] * ABDUCTION_OFFSET

    f = -l1 * np.sin(q1) - l2 * np.sin(q1 + q2)
    z_down = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)

    s0, c0 = np.sin(q0), np.cos(q0)
    s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)

    return np.array(
        [
            [0.0, -z_down, -l2 * c12],
            [-d * s0 + z_down * c0, f * s0, -l2 * s12 * s0],
            [d * c0 + z_down * s0, -f * c0, l2 * s12 * c0],
        ]
    )


def gravity_feedforward_torques(
    q: np.ndarray,
    stance_mask: np.ndarray,
    total_mass: float = TOTAL_MASS,
    swing_load: float = 0.0,
) -> np.ndarray:
    """Open-loop feedforward torques (tau_ff) supporting robot weight under gravity.

    Computes tau_ff = J^T * F_contact per leg, with the total weight M*g shared out
    by load: a stance leg carries 1 and a swing leg carries `swing_load`. At the
    default of 0 only stance legs hold the robot up and swing legs get nothing.

    A leg that is "swinging" at a tiny stride amplitude has not left the ground,
    so a caller ramping the amplitude passes `swing_load = 1 - amplitude`: all four
    legs share the weight at amplitude zero and the swing legs hand it over as the
    stride grows.

    q: (12,) joint angles in Unitree order.
    stance_mask: (4,) boolean stance indicator (FR, FL, RR, RL).
    total_mass: total robot mass [kg], default `go2.model.TOTAL_MASS`.
    swing_load: load of a swing leg relative to a stance leg, in [0, 1].
    """
    q = np.asarray(q, dtype=float).reshape(NUM_JOINTS)
    mask = np.asarray(stance_mask, dtype=bool).reshape(NUM_LEGS)
    load = np.where(mask, 1.0, swing_load)
    if load.sum() == 0.0:
        return np.zeros(NUM_JOINTS)

    weight = total_mass * STANDARD_GRAVITY

    tau_ff = np.zeros(NUM_JOINTS)
    for leg in range(NUM_LEGS):
        if load[leg] > 0.0:
            f_contact = np.array([0.0, 0.0, -weight * load[leg] / load.sum()])
            j_leg = leg_jacobian(q[3 * leg : 3 * leg + 3], leg)
            tau_ff[3 * leg : 3 * leg + 3] = j_leg.T @ f_contact
    return tau_ff


def joint_velocities(q: np.ndarray, v_feet: np.ndarray) -> np.ndarray:
    """Joint angular velocities, (12,) rad/s, from body-frame foot velocities, (4, 3) m/s.

    Solves dq_leg = J^{-1} * v_foot for each leg via the analytical 3x3 leg Jacobian.
    """
    q = np.asarray(q, dtype=float).reshape(NUM_JOINTS)
    v_feet = np.asarray(v_feet, dtype=float).reshape(NUM_LEGS, 3)
    dq = np.zeros(NUM_JOINTS)
    for leg in range(NUM_LEGS):
        j_leg = leg_jacobian(q[3 * leg : 3 * leg + 3], leg)
        dq[3 * leg : 3 * leg + 3] = np.linalg.solve(j_leg, v_feet[leg])
    return dq
