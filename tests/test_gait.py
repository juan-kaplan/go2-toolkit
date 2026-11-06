"""Checks on the generated gait.

Two of these carry real weight. The kinematics is verified against MuJoCo's own
solver rather than against a second derivation of itself, and the non-slip
property is measured rather than asserted, since non-slip contact is the entire
reason this module exists instead of a replay.

    uv run pytest tests/test_gait.py
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import MujocoFK

from go2_toolkit import control
from go2_toolkit.control import GaitParams
from go2_toolkit.control.kinematics import STANDARD_GRAVITY
from go2_toolkit.go2 import model

LEGS = ("FR", "FL", "RR", "RL")


# -- kinematics against an independent solver ---------------------------------


def test_ik_round_trips_against_mujoco_fk(mj: MujocoFK) -> None:
    """Ask for a foot position, solve, and let the model say where it went.

    This is the only check that can catch a wrong abduction sign or a mirrored
    leg, because both produce a self consistent but wrong solver.
    """
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(300):
        feet = model.HIP_ORIGINS.copy()
        feet[:, 1] += model.LEG_SIGN_Y * model.ABDUCTION_OFFSET
        feet[:, 0] += rng.uniform(-0.12, 0.12, 4)
        feet[:, 1] += rng.uniform(-0.05, 0.05, 4)
        feet[:, 2] += -rng.uniform(0.20, 0.36, 4)
        q = control.inverse_kinematics(feet)
        worst = max(worst, float(np.abs(mj.feet(q) - feet).max()))
    assert worst < 1e-9, f"IK and MuJoCo FK disagree by {worst:.2e} m"


def test_gait_cycle_round_trips_for_every_gait(mj: MujocoFK) -> None:
    p = GaitParams()
    worst = 0.0
    for offsets in control.GAITS.values():
        for phi in np.linspace(0, 1, 60, endpoint=False):
            feet = control.foot_positions(offsets + phi, p)
            q = control.inverse_kinematics(feet)
            worst = max(worst, float(np.abs(mj.feet(q) - feet).max()))
    assert worst < 1e-9, f"cycle round trip off by {worst:.2e} m"


def test_forward_kinematics_matches_mujoco(mj: MujocoFK) -> None:
    """The other direction, checked against the same authority as the IK.

    Checked independently so the pair cannot agree with each other while both
    being wrong about the robot.
    """
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(200):
        q = control.joint_trajectory(rng.random(4), GaitParams())
        got = control.forward_kinematics(q)
        worst = max(worst, float(np.abs(got - mj.feet(q)).max()))
    assert worst < 1e-12, f"analytic FK disagrees with MuJoCo by {worst:.2e} m"


def test_link_positions_puts_the_knee_where_mujoco_does(mj: MujocoFK) -> None:
    """The chain, not just its endpoint.

    A wrong knee still gives a right foot if the two link lengths happen to
    compensate, so the middle of the chain is checked too.
    """
    rng = np.random.default_rng(12)
    worst = 0.0
    for _ in range(100):
        q = control.joint_trajectory(rng.random(4), GaitParams())
        chain = control.link_positions(q)
        for i, leg in enumerate(LEGS):
            knee = mj.body_offset(f"{leg}_calf", q)
            worst = max(worst, float(np.abs(chain[i, 1] - knee).max()))
    assert worst < 1e-12, f"knee off by {worst:.2e} m"


def test_ik_and_fk_are_inverses() -> None:
    """Pure geometry, so no model needed."""
    rng = np.random.default_rng(13)
    worst = 0.0
    for _ in range(300):
        q = control.joint_trajectory(rng.random(4), GaitParams())
        back = control.inverse_kinematics(control.forward_kinematics(q))
        worst = max(worst, float(np.abs(back - q).max()))
    assert worst < 1e-12, f"IK(FK(q)) drifted by {worst:.2e} rad"


def test_leg_jacobian_matches_finite_differences() -> None:
    """Verify analytical leg Jacobian against central finite differences of FK.

    Matches to < 1e-8 m/rad across random reachable postures for all 4 legs.
    """
    rng = np.random.default_rng(42)
    eps = 1e-7
    worst = 0.0
    for _ in range(100):
        q = control.joint_trajectory(rng.random(4), GaitParams())
        for leg in range(4):
            q_leg = q[3 * leg : 3 * leg + 3]
            j_analytic = control.leg_jacobian(q_leg, leg)
            j_fd = np.zeros((3, 3))
            for j in range(3):
                qp = q.copy()
                qm = q.copy()
                qp[3 * leg + j] += eps
                qm[3 * leg + j] -= eps
                pos_p = control.forward_kinematics(qp)[leg]
                pos_m = control.forward_kinematics(qm)[leg]
                j_fd[:, j] = (pos_p - pos_m) / (2 * eps)
            worst = max(worst, float(np.max(np.abs(j_analytic - j_fd))))
    assert worst < 1e-8, (
        f"Jacobian disagrees with finite differences by {worst:.2e} m/rad"
    )


def test_unreachable_targets_raise_rather_than_clamp() -> None:
    feet = control.foot_positions(np.zeros(4), GaitParams())
    feet[0, 2] -= 0.30  # well past the leg's reach
    with pytest.raises(ValueError, match="reachable shell"):
        control.inverse_kinematics(feet)


# -- the foot path -------------------------------------------------------------


def test_stance_foot_is_stationary_in_the_world() -> None:
    """The property the Kine2Go clips all failed, measured the same way.

    Best clip in that set slid 0.086 m/s while loaded, median 0.27. A stance
    foot here must not move at all once the body's own advance is added back.
    """
    p = GaitParams(step_length=0.15, stance_duration=0.75, cycle_period=1.0)
    t = np.linspace(0.0, p.stance_duration * p.cycle_period, 400, endpoint=False)
    x_body = control.foot_offsets(t / p.cycle_period, p)[:, 0]
    world = x_body + p.body_speed * t  # body carries the foot forward
    slip = np.abs(np.diff(world)).max() / (t[1] - t[0])
    assert slip < 1e-9, f"stance foot slides at {slip:.4f} m/s"


def test_swing_returns_the_foot_without_scrubbing() -> None:
    """Touchdown and liftoff must be C1 with stance, and vertical velocity zero.

    A foot that lands with residual horizontal velocity scrubs, and one that
    lands with vertical velocity impacts. Both are what make a low duty factor
    gait hard to transfer, so neither should be built in at the start.
    """
    p = GaitParams()
    eps = 1e-6
    stance_rate = -p.step_length / p.stance_duration  # d(x)/d(phase)
    for edge in (p.stance_duration, 1.0):
        lo = control.foot_offsets(np.array([edge - eps]), p)[0]
        hi = control.foot_offsets(np.array([(edge + eps) % 1.0]), p)[0]
        assert abs(lo[0] - hi[0]) < 1e-4, f"x jumps at phase {edge}"
        assert abs(lo[1] - hi[1]) < 1e-4, f"z jumps at phase {edge}"
    for probe in (p.stance_duration + eps, 1.0 - eps):
        a = control.foot_offsets(np.array([probe - eps]), p)[0]
        b = control.foot_offsets(np.array([probe + eps]), p)[0]
        dx = (b[0] - a[0]) / (2 * eps)
        dz = (b[1] - a[1]) / (2 * eps)
        assert abs(dx - stance_rate) < 1e-2, (
            f"at phase {probe:.3f} the foot moves at {dx:.4f} rad^-1, "
            f"stance moves at {stance_rate:.4f}; it would scrub on contact"
        )
        assert abs(dz) < 1e-2, f"vertical velocity {dz:.4f} at phase {probe:.3f}"


@pytest.mark.parametrize("height", [0.04, 0.08, 0.12])
def test_swing_peaks_at_the_requested_clearance(height: float) -> None:
    p = GaitParams(swing_height=height)
    z = control.foot_offsets(np.linspace(0, 1, 2001, endpoint=False), p)[:, 1]
    assert abs(z.max() - height) < 1e-4, f"asked {height}, peaked at {z.max():.4f}"
    assert z.min() > -1e-12, "foot dips below the ground plane"


def test_body_speed_is_consistent_with_the_stride() -> None:
    p = GaitParams(step_length=0.18, stance_duration=0.8, cycle_period=1.25)
    assert abs(p.body_speed - 0.18 / (0.8 * 1.25)) < 1e-12


# -- the phase table -----------------------------------------------------------


def test_walk_has_four_distinct_phases_and_trot_two() -> None:
    """A k-phase gait needs k distinct offsets: a walk has four, a trot two."""
    assert control.phase_count(control.GAITS["walk"]) == 4
    for two_phase in ("trot", "pace", "bound"):
        assert control.phase_count(control.GAITS[two_phase]) == 2, two_phase


def test_walk_always_keeps_three_feet_down() -> None:
    n_down = control.support_count(control.GAITS["walk"], 0.75)
    assert n_down.min() >= 3, f"only {n_down.min()} feet down at worst"


def test_generated_walk_stays_within_joint_limits() -> None:
    p = GaitParams()
    for phi in np.linspace(0, 1, 200, endpoint=False):
        q = control.joint_trajectory(control.GAITS["walk"] + phi, p)
        assert model.within_limits(q).all(), f"phase {phi:.3f}: {np.round(q, 3)}"


# -- entering and leaving a gait -----------------------------------------------


@pytest.mark.parametrize("name", sorted(control.GAITS))
@pytest.mark.parametrize("warmup", [0.0, 2.0])
def test_commanded_joints_never_step(name: str, warmup: float) -> None:
    """Regression: a 0.276 rad jump snapped two legs into the body.

    The driver ramped into the full amplitude phase zero pose and then started
    its loop at zero amplitude. Under the walk offsets FL sits at -L/2 and RR at
    +L/2 at phase zero, the two extremes of the stance sweep, so those two legs
    stepped roughly three times as far as the others in a single control period.
    At kp=100 a 0.276 rad error asks for 27.6 Nm against a joint limit near 23.7.

    Any per step change here should be swing motion, never a discontinuity.
    """
    rate = 200.0
    p = GaitParams()
    offsets = control.GAITS[name]
    t = np.arange(0.0, 6.0, 1.0 / rate)
    q = np.array([control.commanded_joints(x, offsets, p, warmup) for x in t])
    step = np.abs(np.diff(q, axis=0)).max()
    assert step < 0.05, (
        f"{name} warmup={warmup}: {step:.4f} rad in one period at "
        f"{rate:g} Hz ({step * rate:.1f} rad/s)"
    )


def test_zero_amplitude_is_a_standing_pose() -> None:
    """Warmup must start from a pose that does not depend on phase.

    At scale 0 every foot sits at its rest position, so a gait can be entered
    from a stand at any phase without a step.
    """
    p = control.amplitude_scaled(GaitParams(), 0.0)
    rest = control.foot_positions(np.zeros(4), p)
    for phi in np.linspace(0, 1, 17):
        feet = control.foot_positions(control.GAITS["walk"] + phi, p)
        assert np.abs(feet - rest).max() < 1e-12, f"phase {phi:.2f} moved the feet"


# -- the seam ------------------------------------------------------------------


def test_the_pipeline_accepts_any_phase_source() -> None:
    """Nothing downstream of `phase` knows where phase came from.

    Another phase source can replace `clock_phases` and the foot path and IK
    are untouched. The test stands in a
    deliberately un-clocklike source, phases that are not evenly spaced, not
    monotonic in time, and not derived from a period, and asserts the pipeline
    is indifferent to all of that.
    """
    params = GaitParams()
    rng = np.random.default_rng(14)
    for _ in range(50):
        phases = rng.random(4)  # no period, no offsets, no clock
        q = control.joint_trajectory(phases, params)
        assert q.shape == (12,)
        assert model.within_limits(q).all()

    # and the clock source is just one such producer, with no privileged status
    offsets = control.GAITS["walk"]
    for t in (0.0, 0.37, 1.0, 2.5):
        phases = control.clock_phases(t, offsets, params.cycle_period)
        direct = control.joint_trajectory(phases, params)
        assert np.array_equal(direct, control.commanded_joints(t, offsets, params))

    # the state entry point takes what a phase source produces, not a phase: a foot
    # waiting at the end of a sweep, a stance of any length, an arbitrary rate
    for _ in range(50):
        stance = rng.random(4) < 0.5
        progress = rng.random(4)
        progress[rng.random(4) < 0.25] = 1.0  # parked at the end of its half
        rate = np.where(progress == 1.0, 0.0, rng.uniform(0.5, 5.0, 4))
        feet = control.foot_positions_state(stance, progress, params)
        assert model.within_limits(control.inverse_kinematics(feet)).all()
        cmd = control.state_joint_command(stance, progress, rate, params)
        assert np.isfinite(cmd.dq).all()
        assert np.array_equal(cmd.q, control.inverse_kinematics(feet))


# -- gravity feedforward and joint commands -----------------------------------


def _stand_weight_share(stance_legs: int) -> float:
    """Vertical force each stance leg carries when `stance_legs` share the weight, N."""
    return model.TOTAL_MASS * STANDARD_GRAVITY / stance_legs


def test_gravity_feedforward_on_the_stand_is_the_weight_times_each_lever() -> None:
    """On STAND_POSE all four legs carry a quarter of the weight, and the torque at
    each joint is that force times the horizontal lever to the foot.

    Derived rather than recorded: the foot is directly below the thigh joint, so the
    thigh torque is zero; the hip roll lever is the abduction offset, with opposite
    sign left and right; the calf lever is the calf length times the sine of the
    thigh plus calf angle.
    """
    fz = _stand_weight_share(4)
    hip = model.ABDUCTION_OFFSET * fz
    thigh_angle, calf_angle = model.STAND_POSE[1], model.STAND_POSE[2]
    calf = fz * model.CALF_LENGTH * np.sin(-(thigh_angle + calf_angle))

    tau = control.gravity_feedforward_torques(model.STAND_POSE, np.ones(4, dtype=bool))

    for leg, roll_sign in enumerate(model.LEG_SIGN_Y * -1.0):  # FR, FL, RR, RL
        assert tau[3 * leg] == pytest.approx(roll_sign * hip, abs=1e-6)
        assert tau[3 * leg + 1] == pytest.approx(0.0, abs=1e-6)
        assert tau[3 * leg + 2] == pytest.approx(calf, abs=1e-6)


def test_gravity_feedforward_contact_forces_add_up_to_the_weight() -> None:
    """Undo the torques through J^T: each stance foot must be pushed straight up."""
    q = model.STAND_POSE
    tau = control.gravity_feedforward_torques(q, np.ones(4, dtype=bool))
    total_up = 0.0
    for leg in range(4):
        j_leg = control.leg_jacobian(q[3 * leg : 3 * leg + 3], leg)
        force = np.linalg.solve(j_leg.T, tau[3 * leg : 3 * leg + 3])
        # fmt: off
        assert force[:2] == pytest.approx([0.0, 0.0], abs=1e-9)   # no sideways push
        assert force[2] < 0                                       # loads the ground
        # fmt: on
        total_up += -force[2]
    assert total_up == pytest.approx(_stand_weight_share(1), rel=1e-5)


def test_gravity_feedforward_only_loads_the_stance_legs() -> None:
    q = model.STAND_POSE
    none = control.gravity_feedforward_torques(q, np.zeros(4, dtype=bool))
    assert np.all(none == 0.0)

    three = control.gravity_feedforward_torques(q, np.array([True, True, True, False]))
    four = control.gravity_feedforward_torques(q, np.ones(4, dtype=bool))
    assert np.all(three[9:12] == 0.0), "the swing leg must carry nothing"
    assert three[:9] == pytest.approx(four[:9] * 4.0 / 3.0, rel=1e-9)


def test_a_swing_load_shares_the_weight_with_the_swing_legs() -> None:
    """The swing legs take a fraction of a stance leg's share, and the total holds."""
    q = model.STAND_POSE
    mask = np.array([True, True, False, False])
    half = control.gravity_feedforward_torques(q, mask, swing_load=0.5)
    full = control.gravity_feedforward_torques(q, np.ones(4, dtype=bool))
    # loads 1, 1, 0.5, 0.5 of a total 3: stance legs carry 1/3 each, swing 1/6 each
    assert half[:6] == pytest.approx(full[:6] * 4.0 / 3.0, rel=1e-9)
    assert half[6:] == pytest.approx(full[6:] * 2.0 / 3.0, rel=1e-9)
    # swing_load = 1 is every leg a stance leg, and 0 is the old behaviour
    assert control.gravity_feedforward_torques(q, mask, swing_load=1.0) == pytest.approx(
        full, rel=1e-9
    )


@pytest.mark.parametrize("name", sorted(control.GAITS))
def test_at_zero_amplitude_every_leg_is_loaded_equally(name: str) -> None:
    """Trot at t = 0 used to put 11 N m on two calves of a robot standing on four."""
    p = control.default_gait_params(name)
    offsets = control.GAITS[name]
    for t in (0.0, 0.13, 0.31, 0.77):
        cmd = control.clock_joint_command(t, offsets, p, scale=0.0)
        tau = cmd.tau_ff.reshape(4, 3)
        # the hip torque mirrors between the left and right legs, so compare sizes
        assert np.abs(tau) == pytest.approx(np.tile(np.abs(tau[0]), (4, 1)), abs=1e-9)
        assert np.all(tau[:, 2] > 0.0)


def test_the_swing_legs_hand_the_load_over_as_the_amplitude_grows() -> None:
    p = control.default_gait_params("trot")
    offsets = control.GAITS["trot"]
    t = 0.3 * p.cycle_period  # a diagonal pair is in swing
    swing_calf = []
    for scale in (0.0, 0.5, 1.0):
        cmd = control.clock_joint_command(t, offsets, p, scale=scale)
        swing_calf.append(np.abs(cmd.tau_ff.reshape(4, 3)[:, 2]).min())
    assert swing_calf[0] > swing_calf[1] > swing_calf[2] == 0.0


def test_the_stand_is_the_zero_amplitude_pose_at_any_phase() -> None:
    p = control.default_gait_params("walk")
    stand = control.stand_joints(p)
    assert stand.shape == (12,)
    for t in (0.0, 0.2, 0.55):
        cmd = control.clock_joint_command(t, control.GAITS["walk"], p, scale=0.0)
        np.testing.assert_array_equal(cmd.q, stand)


@pytest.mark.parametrize("name", sorted(control.GAITS))
def test_the_warmup_path_is_the_clock_path_at_the_warmup_amplitude(name: str) -> None:
    p = control.default_gait_params(name)
    offsets = control.GAITS[name]
    for t in (0.0, 0.2, 0.9, 1.0, 2.5):
        a = control.commanded_joint_command(t, offsets, p, warmup=1.0)
        b = control.clock_joint_command(
            t,
            offsets,
            p,
            scale=control.warmup_scale(t, 1.0),
            scale_dot=control.warmup_scale_rate(t, 1.0),
        )
        for field in ("q", "dq", "tau_ff", "kp", "kd"):
            np.testing.assert_array_equal(getattr(a, field), getattr(b, field))


def test_commanded_joint_command_returns_valid_feedforward() -> None:
    p = GaitParams()
    offsets = control.GAITS["walk"]
    cmd_with_ff = control.commanded_joint_command(0.5, offsets, p, gravity_comp=True)
    assert isinstance(cmd_with_ff, control.JointCommand)
    assert cmd_with_ff.q.shape == (12,)
    assert cmd_with_ff.tau_ff.shape == (12,)
    assert np.any(cmd_with_ff.tau_ff != 0.0)

    cmd_no_ff = control.commanded_joint_command(0.5, offsets, p, gravity_comp=False)
    assert isinstance(cmd_no_ff, control.JointCommand)
    assert np.all(cmd_no_ff.tau_ff == 0.0)


def test_foot_velocities_analytical_vs_finite_difference() -> None:
    p = GaitParams(
        step_length=0.15, swing_height=0.08, stance_duration=0.75, cycle_period=1.0
    )
    eps = 1e-6
    # Sample a grid of phases avoiding exact boundary discontinuity
    phases = np.linspace(0.05, 0.95, 50)
    for phi in phases:
        phi_vec = np.full(4, phi)
        v_ana = control.foot_velocities(phi_vec, p)

        # Finite difference: (pos(phi + eps) - pos(phi - eps)) / (2 * eps * period)
        p_plus = control.foot_positions(phi_vec + eps, p)
        p_minus = control.foot_positions(phi_vec - eps, p)
        v_num = (p_plus - p_minus) / (2.0 * eps * p.cycle_period)

        np.testing.assert_allclose(v_ana, v_num, rtol=1e-4, atol=1e-4)


def _state_velocity_vs_difference(
    stance: np.ndarray, progress: np.ndarray, rate: np.ndarray, p: GaitParams
) -> None:
    """Commanded foot velocity against a central difference of foot position.

    Time advances each foot's progress by its own `rate`, as a phase source would.
    """
    eps = 1e-6
    v_ana = control.foot_velocities_state(stance, progress, rate, p)
    p_plus = control.foot_positions_state(stance, progress + rate * eps, p)
    p_minus = control.foot_positions_state(stance, progress - rate * eps, p)
    np.testing.assert_allclose(
        v_ana, (p_plus - p_minus) / (2.0 * eps), rtol=1e-4, atol=1e-6
    )


def test_state_velocity_matches_a_finite_difference_of_position() -> None:
    """A sweeping stance foot, a waiting one and a swing foot, with unclocklike rates."""
    p = GaitParams(step_length=0.15, swing_height=0.08, stance_duration=0.6)
    stance = np.array([True, True, False, False])
    progress = np.array([0.3, 0.8, 0.4, 0.7])
    rate = np.array([3.1, 1.7, 2.3, 4.9])
    _state_velocity_vs_difference(stance, progress, rate, p)


def test_a_waiting_foot_is_commanded_no_velocity_and_stays_loaded() -> None:
    p = GaitParams(step_length=0.15, swing_height=0.08, stance_duration=0.6)
    stance = np.array([True, True, False, False])
    progress = np.array([1.0, 0.5, 0.2, 1.0])
    rate = np.array([0.0, 2.0, 2.0, 0.0])
    v = control.foot_velocities_state(stance, progress, rate, p)
    assert not v[0].any() and np.isfinite(v).all()
    assert np.allclose(v[3], 0.0, atol=1e-12)  # rate 0 zeroes the swing slope too
    cmd = control.state_joint_command(stance, progress, rate, p)
    assert not cmd.dq[:3].any()
    # loaded legs carry weight: the mask is the flag, not progress < duty
    unloaded = control.state_joint_command(~stance, progress, rate, p)
    assert not np.array_equal(cmd.tau_ff, unloaded.tau_ff)
    assert np.any(cmd.tau_ff[:3] != 0.0)


def test_the_clock_path_is_the_state_path_for_a_clock_driven_state() -> None:
    p = GaitParams(
        step_length=0.12, swing_height=0.035, stance_duration=0.6, cycle_period=0.5
    )
    offsets = control.GAITS["trot"]
    for t in (0.0, 0.13, 0.31, 0.77):
        phases = control.clock_phases(t, offsets, p.cycle_period)
        stance, progress, rate = control.clock_state(
            phases, p.stance_duration, p.cycle_period
        )
        assert np.array_equal(
            control.foot_offsets_state(stance, progress, p),
            control.foot_offsets(phases, p),
        )
        assert np.array_equal(
            control.foot_velocities_state(stance, progress, rate, p),
            control.foot_velocities(phases, p),
        )
        for warmup in (0.0, 1.0):
            via_state = control.state_joint_command(
                stance,
                progress,
                rate,
                p,
                scale=min(1.0, t / warmup) if warmup else 1.0,
                scale_dot=1.0 / warmup if warmup and t < warmup else 0.0,
            )
            via_clock = control.commanded_joint_command(t, offsets, p, warmup=warmup)
            for field in ("q", "dq", "tau_ff"):
                assert np.array_equal(
                    getattr(via_state, field), getattr(via_clock, field)
                )


def test_joint_velocities_inverts_jacobian() -> None:
    p = GaitParams()
    phases = np.array([0.1, 0.4, 0.6, 0.85])
    q = control.joint_trajectory(phases, p)
    v_feet = control.foot_velocities(phases, p)
    dq = control.joint_velocities(q, v_feet)

    assert dq.shape == (12,)
    # Verify J * dq == v_feet for each leg
    for leg in range(4):
        j_leg = control.leg_jacobian(q[3 * leg : 3 * leg + 3], leg)
        v_reconstructed = j_leg @ dq[3 * leg : 3 * leg + 3]
        np.testing.assert_allclose(v_reconstructed, v_feet[leg], rtol=1e-5, atol=1e-5)


def test_commanded_joint_command_populates_dq() -> None:
    p = GaitParams()
    offsets = control.GAITS["walk"]
    cmd = control.commanded_joint_command(0.5, offsets, p, velocity_feedforward=True)
    assert isinstance(cmd, control.JointCommand)
    assert cmd.dq.shape == (12,)
    assert np.any(np.abs(cmd.dq) > 0.1)

    cmd_no_vf = control.commanded_joint_command(
        0.5, offsets, p, velocity_feedforward=False
    )
    assert np.all(cmd_no_vf.dq == 0.0)


def test_warmup_velocity_feedforward_matches_central_difference() -> None:
    p = GaitParams(
        step_length=0.15, swing_height=0.08, stance_duration=0.75, cycle_period=1.0
    )
    offsets = control.GAITS["walk"]
    warmup = 1.0
    dt = 1e-5

    # Sample times during warmup avoiding exact leg cycle boundaries
    times = [0.15, 0.35, 0.65, 0.85]
    for t in times:
        cmd = control.commanded_joint_command(
            t, offsets, p, warmup=warmup, velocity_feedforward=True
        )
        q_plus = control.commanded_joints(t + dt, offsets, p, warmup=warmup)
        q_minus = control.commanded_joints(t - dt, offsets, p, warmup=warmup)
        dq_num = (q_plus - q_minus) / (2.0 * dt)

        np.testing.assert_allclose(cmd.dq, dq_num, rtol=1e-4, atol=1e-4)
