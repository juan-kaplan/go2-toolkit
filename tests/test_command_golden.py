"""The command every run sends still matches what it sent before.

`test_gait_golden.py` pins `commanded_joints`, which the experiments do not
call. They call `commanded_joint_command`, so this file pins that: its joint
angles against `commanded_joints` (the two must never disagree, the startup snap
came from exactly that), and its `q`, `dq` and `tau_ff` against
`tests/data/command_golden.npz`. Comparison is exact, as in the gait golden.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from go2_toolkit import control
from tests.data.make_command_golden import WARMUP, sample_times
from tests.data.make_gait_golden import PARAM_SETS

GOLDEN = Path(__file__).resolve().parent / "data" / "command_golden.npz"

pytestmark = pytest.mark.skipif(
    not GOLDEN.exists(),
    reason=f"golden file missing; see {GOLDEN.parent}/make_command_golden.py",
)


@pytest.fixture(scope="module")
def golden() -> dict[str, np.ndarray]:
    with np.load(GOLDEN, allow_pickle=False) as fh:
        return {k: fh[k] for k in fh.files}


@pytest.mark.parametrize("pname", sorted(PARAM_SETS))
@pytest.mark.parametrize("gname", sorted(control.GAITS))
def test_the_command_is_unchanged(
    golden: dict[str, np.ndarray], pname: str, gname: str
) -> None:
    params, offs = PARAM_SETS[pname], control.GAITS[gname]
    t = golden[f"{pname}/{gname}/t"]
    cmds = [control.commanded_joint_command(ti, offs, params, warmup=WARMUP) for ti in t]
    for field in ("q", "dq", "tau_ff"):
        got = np.stack([getattr(c, field) for c in cmds])
        np.testing.assert_array_equal(
            got,
            golden[f"{pname}/{gname}/{field}"],
            err_msg=f"{pname}/{gname}: {field} moved",
        )


@pytest.mark.parametrize("pname", sorted(PARAM_SETS))
@pytest.mark.parametrize("gname", sorted(control.GAITS))
def test_both_entry_points_send_the_same_angles(pname: str, gname: str) -> None:
    """`commanded_joints` and `commanded_joint_command` must never drift apart."""
    params, offs = PARAM_SETS[pname], control.GAITS[gname]
    for ti in sample_times(params.cycle_period):
        np.testing.assert_array_equal(
            control.commanded_joint_command(ti, offs, params, warmup=WARMUP).q,
            control.commanded_joints(ti, offs, params, warmup=WARMUP),
        )
