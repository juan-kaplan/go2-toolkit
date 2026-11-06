"""Record what `commanded_joint_command` commands today, as a refactor tripwire.

`gait_golden.npz` pins the joint angles from `commanded_joints`, but every run
calls `commanded_joint_command`, which also builds `dq` and `tau_ff`. This file
pins that whole command. Run it ONCE, before the code it guards changes, and
never again without a reason stated in the commit message: regenerating it
silently is the same as deleting the test.

    uv run python tests/data/make_command_golden.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from go2_toolkit import control
from tests.data.make_gait_golden import PARAM_SETS

OUT = Path(__file__).resolve().parent / "command_golden.npz"

# Long enough for two full warmup ramps of the slowest gait, so the samples
# cover the ramp, its end, and steady state.
WARMUP = 2.0


def sample_times(period: float) -> np.ndarray:
    """Times from before zero (the ramp clamps there) to four cycles."""
    return np.linspace(-0.1, 4.0 * period, 201)


def main() -> None:
    rec: dict[str, np.ndarray] = {}
    for pname, params in PARAM_SETS.items():
        for gname, offs in control.GAITS.items():
            t = sample_times(params.cycle_period)
            cmds = [
                control.commanded_joint_command(ti, offs, params, warmup=WARMUP)
                for ti in t
            ]
            rec[f"{pname}/{gname}/t"] = t
            rec[f"{pname}/{gname}/q"] = np.stack([c.q for c in cmds])
            rec[f"{pname}/{gname}/dq"] = np.stack([c.dq for c in cmds])
            rec[f"{pname}/{gname}/tau_ff"] = np.stack([c.tau_ff for c in cmds])

    np.savez_compressed(OUT, **rec)
    total = sum(v.size for v in rec.values())
    print(f"wrote {OUT}")
    print(f"  {len(rec)} arrays, {total} values, {OUT.stat().st_size / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
