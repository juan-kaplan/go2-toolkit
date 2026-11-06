"""Record what the gait generator commands today, as a refactor tripwire.

Run this ONCE before restructuring, never again afterwards without a reason
stated in the commit message. The point of a golden file is that regenerating
it silently is the same as deleting the test.

    uv run python tests/data/make_gait_golden.py

The refactor this guards is meant to move code without changing a single
number. Sampling the generator densely across phase, across time including the
warmup ramp, and across several parameter sets pins down the whole observable
surface: if an arithmetic step is reordered or a constant drifts, the stored
arrays stop matching.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from go2_toolkit import control
from go2_toolkit.control import GaitParams

OUT = Path(__file__).resolve().parent / "gait_golden.npz"

# Off-nominal parameter sets as well as the default, so the golden file
# constrains the formulae and not just one operating point.
PARAM_SETS = {
    "default": GaitParams(),
    "short_stride": GaitParams(step_length=0.06, swing_height=0.03),
    "low_duty": GaitParams(stance_duration=0.5, cycle_period=0.6),
    "tall_slow": GaitParams(
        nominal_height=0.34, cycle_period=1.7, step_length=0.21, swing_height=0.11
    ),
}


def main() -> None:
    rec: dict[str, np.ndarray] = {}
    phase = np.linspace(0.0, 1.0, 401, endpoint=False)

    for pname, params in PARAM_SETS.items():
        rec[f"{pname}/offsets2d"] = control.foot_offsets(phase, params)
        rec[f"{pname}/body_speed"] = np.asarray(params.body_speed)

        for gname, offs in control.GAITS.items():
            # foot positions and joints straight from phase
            feet = np.stack([control.foot_positions(offs + p, params) for p in phase])
            q = np.stack([control.joint_trajectory(offs + p, params) for p in phase])
            rec[f"{pname}/{gname}/feet"] = feet
            rec[f"{pname}/{gname}/q"] = q

            # the commanded path through time, warmup ramp included, which is
            # the surface the snap bug lived on
            t = np.linspace(0.0, 4.0 * params.cycle_period, 401)
            rec[f"{pname}/{gname}/q_cmd"] = np.stack(
                [control.commanded_joints(ti, offs, params, warmup=2.0) for ti in t]
            )
            rec[f"{pname}/{gname}/t"] = t

    rec["_phase"] = phase
    rec["_gaits"] = np.array(sorted(control.GAITS))
    rec["_params"] = np.array(sorted(PARAM_SETS))

    np.savez_compressed(OUT, **rec)
    total = sum(v.size for v in rec.values())
    print(f"wrote {OUT}")
    print(f"  {len(rec)} arrays, {total} values, {OUT.stat().st_size / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
