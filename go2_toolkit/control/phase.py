"""Where each leg is in its cycle. This is the seam a rhythm source plugs into.

A phase is one number per leg in [0, 1). 0 is the start of stance, `duty` is
liftoff, and the wrap back to 0 is touchdown. Everything downstream, the foot
path and the inverse kinematics, is a pure function of that number, so whatever
produces it is interchangeable.

`clock_phases` produces it from the wall clock. Any other source, such as a
central pattern generator or a learned policy, can give a phase per leg instead.
The foot path and the inverse kinematics do not change for that, and that
substitutability is the reason this module is separate rather than three lines
inside the generator. The velocity and gravity feedforward in
`gait.commanded_joint_command` do assume things about the phase source, though:
that it advances at `1 / cycle_period`, and that stance is `[0, duty)`. If your
source does not match, adjust those terms.

`GAITS` is the other half of the seam. A k-phase gait needs k distinct offsets,
which is why a walk needs four distinct phases where a trot needs two.
"""

from __future__ import annotations

import numpy as np

from ..go2.conventions import NUM_LEGS

# Offsets per leg, in Unitree order FR, FL, RR, RL.
# fmt: off
GAITS = {
    "walk": np.array([0.25, 0.75, 0.0, 0.5]),    # one foot at a time, RR FR RL FL
    "trot": np.array([0.0, 0.5, 0.5, 0.0]),      # diagonal pairs
    "pace": np.array([0.0, 0.5, 0.0, 0.5]),      # lateral pairs
    "bound": np.array([0.0, 0.0, 0.5, 0.5]),     # front pair, then rear pair
}
# fmt: on

# Canonical default duty factor for each gait.
# fmt: off
DEFAULT_DUTY = {
    "walk": 0.78,    # empirical optimum with smooth inter-phase support overlap
    "trot": 0.50,    # symmetric alternating diagonal pairs
    "pace": 0.50,    # symmetric alternating lateral pairs
    "bound": 0.50,   # symmetric alternating front/rear pairs
}
# fmt: on

# Canonical default cycle period for each gait (seconds).
# fmt: off
DEFAULT_PERIOD = {
    "walk": 1.0,     # slow statically stable crawl (4 phases)
    "trot": 0.5,     # dynamic 2-beat alternating diagonal trot (2.0 Hz cadence)
    "pace": 0.5,     # dynamic 2-beat alternating lateral pace
    "bound": 0.5,    # dynamic 2-beat alternating sagittal bound
}
# fmt: on

# Sampling density for the support-pattern queries below. Fine enough that a
# transition is never missed at any duty factor we use, cheap enough to call
# once per run for a printed summary.
_SAMPLES = 400


def clock_phases(t: float, offsets: np.ndarray, cycle_period: float) -> np.ndarray:
    """Phase per leg at wall clock time `t`, one cycle per `cycle_period`.

    The placeholder phase source, and where the clock enters the gait. Kept
    deliberately trivial: the contract is `-> (4,) in [0, 1)` and nothing more.
    The joint angles depend on nothing else about the source; the feedforward
    torques and velocities also assume the clock's rate (see the module docstring).
    """
    return np.mod(np.asarray(offsets, dtype=float) + t / cycle_period, 1.0)


def clock_state(
    phases: np.ndarray, duty: float, cycle_period: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The state a clock-driven phase stands for: (is_stance, progress, rate).

    `progress` is the fraction of the current stance or swing already done and
    `rate` is how fast it advances per second, which for a clock is constant in
    each half. This is the adapter between the phase seam and the trajectory's
    state entry point; a decoder produces the same three arrays directly.
    """
    phase = np.mod(np.asarray(phases, dtype=float), 1.0)
    stance = phase < duty
    progress = np.where(stance, phase / duty, (phase - duty) / (1.0 - duty))
    rate = np.where(
        stance, 1.0 / (duty * cycle_period), 1.0 / ((1.0 - duty) * cycle_period)
    )
    return stance, progress, rate


def is_stance(phases: np.ndarray, duty: float) -> np.ndarray:
    """Which legs are loaded at these phases. Stance is [0, duty)."""
    return np.mod(np.asarray(phases, dtype=float), 1.0) < duty


def phase_count(offsets: np.ndarray) -> int:
    """How many distinct phases this gait uses, which is its stage count.

    Rounded before counting, because offsets that differ in the last bit are
    the same stage. This is the number the network's connectivity has to match:
    four for a walk, two for a trot, pace or bound.
    """
    return len(set(np.round(np.asarray(offsets, dtype=float), 6)))


def support_count(
    offsets: np.ndarray, duty: float, phase: np.ndarray | None = None
) -> np.ndarray:
    """Feet on the ground at each sampled phase, (N,).

    `min()` of this is the gait's worst case support. Three or more means the
    centre of mass can always be kept inside a support triangle, so the gait is
    statically stable and never needs dynamic balance. Two means it does.
    """
    phase = np.linspace(0.0, 1.0, _SAMPLES) if phase is None else np.asarray(phase)

    offsets = np.asarray(offsets, dtype=float)
    return (np.mod(offsets[None, :] + phase[:, None], 1.0) < duty).sum(axis=1)


def stance_spans(
    offset: float, duty: float, phase: np.ndarray | None = None
) -> list[tuple[float, float]]:
    """Contiguous [start, end) phase intervals where one leg is loaded.

    A leg whose stance wraps past phase 1 comes back as two spans rather than
    one, which is what a gait diagram needs in order to draw it.
    """
    if phase is None:
        phase = np.linspace(0.0, 1.0, 2001, endpoint=False)
    phase = np.asarray(phase)
    on = is_stance(offset + phase, duty)
    edge = np.diff(on.astype(int))
    starts = list(phase[1:][edge == 1])
    ends = list(phase[1:][edge == -1])
    if on[0]:
        starts = [float(phase[0])] + starts
    if on[-1]:
        ends = ends + [1.0]
    return list(zip(starts, ends, strict=True))


# A gait with the wrong number of offsets would broadcast silently rather than
# raise, so the table is checked once at import.
assert all(len(v) == NUM_LEGS for v in GAITS.values()), (
    "every gait needs exactly one phase offset per leg"
)
