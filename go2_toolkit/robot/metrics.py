"""What a finished trial says about how well the robot walked.

Two drivers report these numbers and the tuner scores with them, and they used to
be written out separately in each, where they had already drifted: one checked
that the ground-truth sample at the end of the warmup was finite and the other
printed `nan` when it was not. There is one definition here. It takes the
expected speed as a plain float, so this module needs nothing from `control`
(`GaitParams.body_speed` is the caller's to pass).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..go2.model import IMU_SITE_OFFSET

if TYPE_CHECKING:
    from .session import TrialResult

# A robot whose body ends the trial below this fraction of its standing height has
# fallen. Well below any gait's normal bob, well above a robot lying down.
FALLEN_FRACTION = 0.6

# An expected distance smaller than this cannot be turned into a ratio.
MIN_EXPECTED_DISTANCE = 1e-6


@dataclass
class SpeedMetrics:
    """Distance covered against the distance the stride should have covered."""

    # fmt: off
    has_truth: bool                # False: no ground truth, distances are 0
    steady_is_measured: bool       # False: the steady_* fields repeat the totals
    distance_along_x: float        # m along the body's own x axis over the whole trial
    expected_distance: float       # m, discounting the warmup ramp
    elapsed_time: float            # s
    total_speed_ratio: float
    steady_distance: float         # m, from the end of the warmup to the end
    steady_time: float             # s
    steady_expected_distance: float
    steady_speed_ratio: float
    # fmt: on


def speed_metrics(result: TrialResult, body_speed: float, warmup: float) -> SpeedMetrics:
    """Total distance covered, and the steady-state figure past the warmup.

    The warmup ramp goes from zero to full stride, so a trial that includes it
    necessarily undershoots `body_speed`: the expectation counts half the warmup
    as time not walked, and the steady ratio excludes that window altogether. Falls
    back to the total figures when ground truth or the warmup boundary is not
    available (`steady_is_measured` says which), so a trial without ground truth is
    still scoreable rather than raising.
    """
    fwd_dist = result.forward_distance()
    has_truth = fwd_dist is not None
    along = fwd_dist if has_truth else 0.0
    elapsed = result.elapsed
    expected_total = body_speed * max(0.0, elapsed - 0.5 * min(warmup, elapsed))
    can_divide = has_truth and expected_total > MIN_EXPECTED_DISTANCE
    total_ratio = along / expected_total if can_divide else 0.0

    k_warmup = int(round(warmup * result.rate))
    steady_is_measured = bool(
        has_truth
        and result.base_pos is not None
        and len(result.base_pos) == result.steps
        and 0 <= k_warmup < result.steps
        and np.isfinite(result.base_pos[k_warmup]).all()
        and (result.end_pos is not None and np.isfinite(result.end_pos).all())
    )
    if steady_is_measured:
        pos_warmup = result.base_pos[k_warmup]
        ss_along = float((result.end_pos - pos_warmup) @ result.forward_axis)
        ss_time = (result.steps - k_warmup) / result.rate
        ss_expected = body_speed * ss_time
        ss_ratio = ss_along / ss_expected if ss_expected > MIN_EXPECTED_DISTANCE else 0.0
    else:
        ss_along, ss_time, ss_expected, ss_ratio = (
            along,
            elapsed,
            expected_total,
            total_ratio,
        )

    return SpeedMetrics(
        has_truth=has_truth,
        steady_is_measured=steady_is_measured,
        distance_along_x=along,
        expected_distance=expected_total,
        elapsed_time=elapsed,
        total_speed_ratio=total_ratio,
        steady_distance=ss_along,
        steady_time=ss_time,
        steady_expected_distance=ss_expected,
        steady_speed_ratio=ss_ratio,
    )


def final_height(result: TrialResult, body_origin: bool = True) -> float | None:
    """Height after the loop, from the reading taken then, or None if there is none.

    `end_pos` is the IMU site, about 4 cm above the body origin; `body_origin`
    subtracts that. Pass False to compare against a height measured in the site's
    own frame, as the clip replay does.
    """
    if result.end_pos is None or not np.isfinite(result.end_pos[2]):
        return None
    height = float(result.end_pos[2])
    return float(height - IMU_SITE_OFFSET[2]) if body_origin else height
