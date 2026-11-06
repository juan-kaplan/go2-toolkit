"""robot/video.py: the body positions the renderer draws from."""

from __future__ import annotations

import numpy as np
import pytest

from go2_toolkit.robot.video import base_link_positions


def test_body_positions_fill_gaps_and_survive_missing_readings() -> None:
    quats = np.tile([1.0, 0.0, 0.0, 0.0], (5, 1))
    pos = np.c_[np.arange(5.0), np.zeros(5), np.full(5, 0.34)]
    pos[2] = np.nan
    filled = base_link_positions(quats, pos)
    assert np.isfinite(filled).all()
    assert filled[2, 0] == pytest.approx(
        2.0 + (filled[1, 0] - 1.0)
    )  # between its neighbours
    assert base_link_positions(quats, None)[:, 2] == pytest.approx(0.30)
    quats[3] = np.nan
    assert np.isfinite(base_link_positions(quats, pos)).all()
