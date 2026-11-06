"""robot/metrics.py: the speed and fall measurements both drivers and the tuner share.

The arithmetic itself is exercised through the tuner's scoring tests
(`test_tune_gait_parameters.py`). What is pinned here is what the drivers rely on
and what the two copies used to disagree about.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import walking_trial

from go2_toolkit.go2.model import IMU_SITE_OFFSET
from go2_toolkit.robot.metrics import final_height, speed_metrics


def test_a_walk_that_matches_its_stride_scores_one() -> None:
    result = walking_trial(steps=400, distance=1.0)  # 1 m in 4 s
    m = speed_metrics(result, body_speed=0.25, warmup=0.0)
    assert m.has_truth and m.steady_is_measured
    assert m.total_speed_ratio == pytest.approx(1.0)
    assert m.steady_speed_ratio == pytest.approx(1.0)


def test_a_non_finite_sample_at_the_warmup_falls_back_instead_of_printing_nan() -> None:
    """The driver used to divide through a NaN here; the tuner already guarded it."""
    result = walking_trial(steps=400, distance=1.0)
    result.base_pos[100] = np.nan  # the sample at warmup = 1 s
    m = speed_metrics(result, body_speed=0.25, warmup=1.0)
    assert not m.steady_is_measured
    assert np.isfinite(m.steady_speed_ratio)
    assert m.steady_speed_ratio == m.total_speed_ratio


def test_no_ground_truth_gives_zero_distance_and_says_so() -> None:
    m = speed_metrics(walking_trial(has_truth=False), body_speed=0.25, warmup=1.0)
    assert not m.has_truth and m.distance_along_x == 0.0


def test_final_height_is_the_body_origin_unless_asked_for_the_site() -> None:
    result = walking_trial(height=0.32)
    assert final_height(result) == pytest.approx(0.32 - IMU_SITE_OFFSET[2])
    assert final_height(result, body_origin=False) == pytest.approx(0.32)
    assert final_height(walking_trial(has_truth=False)) is None
