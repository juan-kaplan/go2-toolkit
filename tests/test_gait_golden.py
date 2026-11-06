"""The gait generator still commands exactly what it commanded before.

This is the tripwire for any refactor of the generator. A refactor is meant to
move code between modules without changing a single number, and this is the test
that makes the claim checkable instead of hopeful. It compares against
`tests/data/gait_golden.npz`, recorded at commit 3fdd17d.

Comparison is EXACT. A pure move of code cannot perturb a float, so any
difference at all is a real change and should be looked at rather than
absorbed by a tolerance. If a later change legitimately reorders arithmetic,
say so in the commit message and loosen `TOL` deliberately; do not regenerate
the file to make the test pass, which would delete the test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from go2_toolkit import control
from go2_toolkit.control import GaitParams

GOLDEN = Path(__file__).resolve().parent / "data" / "gait_golden.npz"
TOL = 0.0

pytestmark = pytest.mark.skipif(
    not GOLDEN.exists(),
    reason=f"golden file missing; see {GOLDEN.parent}/make_gait_golden.py",
)


@pytest.fixture(scope="module")
def golden() -> dict[str, np.ndarray]:
    with np.load(GOLDEN, allow_pickle=False) as fh:
        return {k: fh[k] for k in fh.files}


def _params(name: str) -> GaitParams:
    from tests.data.make_gait_golden import PARAM_SETS

    return PARAM_SETS[name]


def test_the_golden_file_covers_every_gait(golden: dict[str, np.ndarray]) -> None:
    """A gait added without a golden entry would otherwise go unguarded."""
    assert set(golden["_gaits"]) == set(control.GAITS), (
        "GAITS changed since the golden file was recorded; add the new gait to "
        "the file deliberately rather than letting it go untested"
    )


@pytest.mark.parametrize("pname", ["default", "short_stride", "low_duty", "tall_slow"])
def test_foot_offsets_are_unchanged(golden: dict[str, np.ndarray], pname: str) -> None:
    got = control.foot_offsets(golden["_phase"], _params(pname))
    want = golden[f"{pname}/offsets2d"]
    assert got.shape == want.shape
    np.testing.assert_allclose(
        got, want, rtol=TOL, atol=TOL, err_msg=f"{pname}: swing/stance path moved"
    )


@pytest.mark.parametrize("pname", ["default", "short_stride", "low_duty", "tall_slow"])
def test_joint_commands_are_unchanged(golden: dict[str, np.ndarray], pname: str) -> None:
    params = _params(pname)
    phase = golden["_phase"]
    for gname, offs in control.GAITS.items():
        feet = np.stack([control.foot_positions(offs + p, params) for p in phase])
        q = np.stack([control.joint_trajectory(offs + p, params) for p in phase])
        np.testing.assert_allclose(
            feet,
            golden[f"{pname}/{gname}/feet"],
            rtol=TOL,
            atol=TOL,
            err_msg=f"{pname}/{gname}: foot positions moved",
        )
        np.testing.assert_allclose(
            q,
            golden[f"{pname}/{gname}/q"],
            rtol=TOL,
            atol=TOL,
            err_msg=f"{pname}/{gname}: IK output moved",
        )


@pytest.mark.parametrize("pname", ["default", "short_stride", "low_duty", "tall_slow"])
def test_the_commanded_ramp_is_unchanged(
    golden: dict[str, np.ndarray], pname: str
) -> None:
    """Covers the warmup path, which is where the startup snap lived."""
    params = _params(pname)
    for gname, offs in control.GAITS.items():
        t = golden[f"{pname}/{gname}/t"]
        got = np.stack(
            [control.commanded_joints(ti, offs, params, warmup=2.0) for ti in t]
        )
        np.testing.assert_allclose(
            got,
            golden[f"{pname}/{gname}/q_cmd"],
            rtol=TOL,
            atol=TOL,
            err_msg=f"{pname}/{gname}: commanded ramp moved",
        )
