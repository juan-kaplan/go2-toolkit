"""Phase to foot position. The shape of the step, and nothing else.

Takes a stance flag and a progress per leg, or a phase per leg, and returns where
each foot should be in the body frame. The state entry point is the primary one
and phase is an adapter over it. Knows nothing about where either comes from,
nothing about joints, and nothing about the robot's state.

    params = GaitParams(step_length=0.15, swing_height=0.08)
    feet = foot_positions(phases, params)      # (4, 3) in the body frame

**Touchdown is the part worth understanding.** The swing curve is a quintic
Bezier whose control points are placed so the foot arrives at touchdown already
moving backward at exactly stance speed, and with zero vertical velocity. In the
world frame the foot therefore lands at rest, and the stance-swing joins are C1
in both directions. Naive swing curves that return the foot to the ground with
forward or vertical velocity produce the impact discontinuity that makes a low
duty factor gait hard to transfer, which is the central risk this project is
trying to characterise, so it is not worth importing at the start.

The stance half is the reason this is generated rather than recorded. Every
Kine2Go clip slides its feet while they are loaded, 0.086 m/s at best and 0.27
median against body speeds of 0.2 to 0.8, because the dataset retargets
kinematically and never enforces non-slip contact. Here the stance foot is
*defined* to travel backward at exactly the body's forward speed, so it is
stationary on the ground by construction rather than by tuning.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from ..go2.model import ABDUCTION_OFFSET, HIP_ORIGINS, LEG_SIGN_Y, MAX_REACH
from .phase import clock_state

# A quintic Bezier with its two middle control points at equal height peaks at
# 0.625 of that height: z(u) collapses to 10*h*u^2*(1-u)^2, which at u=0.5 is
# 0.625*h. Dividing the requested clearance by it makes the peak exact.
_QUINTIC_PEAK_FRACTION = 0.625
_BINOMIAL_5 = np.array([1.0, 5.0, 10.0, 10.0, 5.0, 1.0])
_BINOMIAL_4 = np.array([1.0, 4.0, 6.0, 4.0, 1.0])


@dataclass
class GaitParams:
    """Everything the foot path needs, and nothing about the robot's state.

    `stance_duration` is the duty factor, the fraction of a cycle a foot spends
    on the ground. At or above 0.75 a four phase walk always has three feet
    down and never needs dynamic balance. At 0.5 a trot has two, and does.
    """

    # fmt: off
    step_length: float = 0.15       # m of ground covered per stride
    swing_height: float = 0.08      # m of peak foot clearance
    stance_duration: float = 0.75   # duty factor
    nominal_height: float = 0.30    # m, hip origin above the foot at rest
    cycle_period: float = 1.0       # s per full cycle
    # fmt: on

    def __post_init__(self) -> None:
        if not 0.0 < self.stance_duration < 1.0:
            raise ValueError(f"duty factor must be in (0, 1), got {self.stance_duration}")
        if self.nominal_height >= MAX_REACH:
            raise ValueError(
                f"nominal_height {self.nominal_height} m is at or past the leg's "
                f"{MAX_REACH:.3f} m reach; the knee would have to straighten fully"
            )

    @property
    def body_speed(self) -> float:
        """Forward speed implied by the stance sweep, m/s.

        This is not a free parameter. The stance foot crosses `step_length` in
        `stance_duration * cycle_period` seconds, so the body must advance at
        exactly this rate for the foot to stay planted. Driving the body at any
        other speed reintroduces the slip the generator exists to avoid.
        """
        return self.step_length / (self.stance_duration * self.cycle_period)


def swing_control_points(params: GaitParams) -> tuple[np.ndarray, np.ndarray]:
    """The six (x, z) control points of the swing Bezier.

    Chosen for the boundary conditions rather than by eye:

      * `dx/du` at both ends equals -L*(1-b)/b, which after the change of
        variable into cycle phase is the stance velocity -L/b. The foot is
        therefore still moving backward at stance speed when it touches down,
        so its world frame velocity at impact is zero. The `/ 5` in the
        overshoot is the quintic's degree: `dx/du` at an endpoint of a degree-n
        Bezier is `n * (P1 - P0)`, so the offset needed is the target slope
        divided by 5.
      * `dz/du` is zero at both ends, because the first two and last two
        control points share a height. The foot lifts and lands vertically at
        rest instead of slamming or clipping.

    Returned rather than kept private because the plotting tool draws them, and
    a diagram of a curve's control polygon that is reconstructed by hand is a
    diagram that can disagree with the curve.
    """
    half = 0.5 * params.step_length
    ratio = (1.0 - params.stance_duration) / params.stance_duration
    overshoot = params.step_length * ratio / 5.0
    peak = params.swing_height / _QUINTIC_PEAK_FRACTION

    px = np.array([-half, -half - overshoot, -half, half, half + overshoot, half])
    pz = np.array([0.0, 0.0, peak, peak, 0.0, 0.0])
    return px, pz


def _swing_curve(u: np.ndarray, params: GaitParams) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the swing Bezier at u in [0, 1] across the swing."""
    px, pz = swing_control_points(params)
    u = np.asarray(u, dtype=float)[..., None]
    i = np.arange(6)
    basis = _BINOMIAL_5 * u**i * (1.0 - u) ** (5 - i)
    return basis @ px, basis @ pz


def _swing_velocity(u: np.ndarray, params: GaitParams) -> tuple[np.ndarray, np.ndarray]:
    """Analytical derivative d(x, z)/du of the swing Bezier curve."""
    px, pz = swing_control_points(params)
    dpx = np.diff(px)
    dpz = np.diff(pz)
    u = np.asarray(u, dtype=float)[..., None]
    i = np.arange(5)
    basis = 5.0 * _BINOMIAL_4 * u**i * (1.0 - u) ** (4 - i)
    return basis @ dpx, basis @ dpz


def foot_offsets_state(
    is_stance: np.ndarray, progress: np.ndarray, params: GaitParams
) -> np.ndarray:
    """Foot displacement from its rest position, (N, 2) as forward and up.

    The state entry point: the caller says whether each foot is in stance or
    swing and how far through that half it is, and the foot path follows. Nothing
    here assumes a clock, so a foot may wait at the end of its sweep, or a stance
    may last longer than the last one.

    `progress` is in [0, 1] within the current stance or swing, and 1 is the
    end of the sweep where a waiting foot sits. Stance sweeps the foot backward at
    constant speed; swing returns it along the Bezier.
    """
    stance = np.asarray(is_stance, dtype=bool)
    progress = np.asarray(progress, dtype=float)
    half = 0.5 * params.step_length

    x = np.empty_like(progress)
    z = np.zeros_like(progress)

    # stance: a straight constant speed sweep, which is what plants the foot
    x[stance] = half - params.step_length * progress[stance]
    if (~stance).any():
        x[~stance], z[~stance] = _swing_curve(progress[~stance], params)
    return np.stack([x, z], axis=-1)


def foot_offsets(phases: np.ndarray, params: GaitParams) -> np.ndarray:
    """Foot displacement from its rest position, (N, 2) as forward and up.

    Phase runs [0, 1). Stance occupies [0, duty) and swing [duty, 1). The clock
    path: an adapter that turns phase into state and calls `foot_offsets_state`.
    """
    stance, progress, _ = clock_state(phases, params.stance_duration, params.cycle_period)
    return foot_offsets_state(stance, progress, params)


def foot_positions_state(
    is_stance: np.ndarray, progress: np.ndarray, params: GaitParams
) -> np.ndarray:
    """Foot positions in the body frame, (4, 3), Unitree leg order.

    Rest position is directly below each thigh origin, which is the hip origin
    displaced laterally by the abduction offset. That choice makes the rest
    pose have zero hip abduction exactly.
    """
    off = foot_offsets_state(
        np.asarray(is_stance).reshape(4), np.asarray(progress).reshape(4), params
    )
    feet = HIP_ORIGINS.copy()
    feet[:, 1] += LEG_SIGN_Y * ABDUCTION_OFFSET
    feet[:, 0] += off[:, 0]
    feet[:, 2] += -params.nominal_height + off[:, 1]
    return feet


def foot_positions(phases: np.ndarray, params: GaitParams) -> np.ndarray:
    """Foot positions in the body frame, (4, 3), Unitree leg order.

    The clock path: phase turned into state, then `foot_positions_state`.
    """
    stance, progress, _ = clock_state(
        np.asarray(phases, dtype=float).reshape(4),
        params.stance_duration,
        params.cycle_period,
    )
    return foot_positions_state(stance, progress, params)


def foot_velocities_state(
    is_stance: np.ndarray,
    progress: np.ndarray,
    rate: np.ndarray,
    params: GaitParams,
    scale_dot: float = 0.0,
    unscaled_params: GaitParams | None = None,
) -> np.ndarray:
    """Foot velocity in the body frame, (4, 3) m/s, Unitree leg order.

    Analytical time derivative of `foot_offsets_state` laid out as foot positions:
        v = (d(foot_position) / d(progress)) * rate

    `rate` is progress per second within the current stance or swing, and is 0
    for a foot that waits, which then has zero velocity rather than one inferred
    from a clock. Stance sweeps at -step_length * rate; swing returns along the
    quintic Bezier.

    **The touchdown match assumes the rates.** The swing curve's control points
    put the foot at -L / duty per cycle at both ends, which is the stance speed
    only when the swing rate and the next stance rate are in the ratio
    `duty / (1 - duty)`, as a clock's are. A decoder whose rates leave that ratio
    lands with some world frame velocity; the curve is not adjusted for it.

    If scale_dot != 0.0 (during warmup ramp), includes the product rule term:
        d/dt [A(t) * Delta_p] = A(t) * d/dt(Delta_p) + A'(t) * Delta_p

    `params` is then the amplitude-scaled parameters, A(t) * Delta_p above, and
    `unscaled_params` the full-amplitude ones that Delta_p comes from. Leave
    `unscaled_params` out only when `scale_dot` is 0: it falls back to `params`,
    which for a nonzero `scale_dot` applies the amplitude twice.
    """
    stance = np.asarray(is_stance, dtype=bool).reshape(4)
    progress = np.asarray(progress, dtype=float).reshape(4)
    rate = np.asarray(rate, dtype=float).reshape(4)

    vx = np.empty_like(progress)
    vy = np.zeros_like(progress)
    vz = np.zeros_like(progress)

    # Written as a distance over a time, -L / (1 / rate), so a clock's rate gives
    # the bit pattern the golden files were recorded with; a waiting foot's
    # infinite time gives a velocity of zero.
    with np.errstate(divide="ignore"):
        vx[stance] = -params.step_length / (1.0 / rate[stance])

    # Swing: analytical quintic Bezier derivative
    if (~stance).any():
        dx_du, dz_du = _swing_velocity(progress[~stance], params)
        vx[~stance] = dx_du * rate[~stance]
        vz[~stance] = dz_du * rate[~stance]

    if scale_dot != 0.0:
        ref = unscaled_params if unscaled_params is not None else params
        off = foot_offsets_state(stance, progress, ref)
        vx += off[:, 0] * scale_dot
        vz += off[:, 1] * scale_dot

    return np.stack([vx, vy, vz], axis=-1)


def foot_velocities(
    phases: np.ndarray,
    params: GaitParams,
    scale_dot: float = 0.0,
    unscaled_params: GaitParams | None = None,
) -> np.ndarray:
    """Foot velocity in the body frame, (4, 3) m/s, Unitree leg order.

    The clock path: an adapter over `foot_velocities_state` for a phase that
    advances at exactly one cycle per `cycle_period`. A phase from another source
    needs the state entry point and its real rate instead. Stance sweeps backward
    at -body_speed. See `foot_velocities_state` for `scale_dot` and
    `unscaled_params`.
    """
    stance, progress, rate = clock_state(
        np.asarray(phases, dtype=float).reshape(4),
        params.stance_duration,
        params.cycle_period,
    )
    return foot_velocities_state(
        stance, progress, rate, params, scale_dot, unscaled_params
    )


def amplitude_scaled(params: GaitParams, scale: float) -> GaitParams:
    """The same gait at a fraction of its stride and clearance.

    At scale 0 the foot path collapses to a point, so the robot stands still at
    `nominal_height` whatever its phase. That makes the scale a continuous way
    in and out of a gait rather than a switch.
    """
    return dataclasses.replace(
        params,
        step_length=params.step_length * scale,
        swing_height=params.swing_height * scale,
    )
