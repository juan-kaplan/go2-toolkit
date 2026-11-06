"""Fixtures shared across the suite.

Two things lived in three copies before this file existed: the MuJoCo forward
kinematics harness, and the decision about what to do when a fixture's data is
not on disk. The second mattered more than the first. Each test used to print
`skipped: ...` and return, which pytest records as a pass, so a suite reporting
"11 passed" could have run eight tests. Here the unmet preconditions raise
`pytest.skip`, and a skip looks like a skip.

The model is session scoped: it was being parsed from XML once per test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from go2_toolkit.go2.conventions import rotation_matrices

REPO = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO / "third_party/unitree_mujoco/unitree_robots/go2/go2.xml"

# Unitree leg order, FR FL RR RL, matching everything else in this project.
FEET = ("FR_foot", "FL_foot", "RR_foot", "RL_foot")


@dataclass
class MujocoFK:
    """MuJoCo as an independent authority on this robot's kinematics.

    The point of testing against it is that it is a second implementation, not
    a second derivation of ours: it reads the same `go2.xml` we take our
    dimensions from, and solves the chain itself.
    """

    model: object
    data: object
    qadr: np.ndarray

    def feet(self, q: np.ndarray) -> np.ndarray:
        """Foot positions in the base frame for joint angles `q`, (4, 3)."""
        import mujoco

        self.data.qpos[:] = 0.0
        self.data.qpos[3] = 1.0  # identity base quaternion
        self.data.qpos[self.qadr] = q
        mujoco.mj_forward(self.model, self.data)
        base = self.data.body("base_link").xpos
        return np.array([self.data.body(f).xpos - base for f in FEET])

    def body_offset(self, name: str, q: np.ndarray) -> np.ndarray:
        """Any body's position relative to base_link, after placing `q`."""
        import mujoco

        self.data.qpos[:] = 0.0
        self.data.qpos[3] = 1.0
        self.data.qpos[self.qadr] = q
        mujoco.mj_forward(self.model, self.data)
        return self.data.body(name).xpos - self.data.body("base_link").xpos

    def foot_error(self, clip, stride_divisor: int = 12) -> float:
        """Worst disagreement between this model's feet and a clip's own, m.

        Places the clip's joint angles on the model and compares where the feet
        land against where the clip says they were, rotated into the base frame.
        Both describe the same physical robot, so they must agree. This is the
        check that caught the feet permutation: joints run FL, FR, RL, RR but
        feet run FL, RL, FR, RR, and the two agree at positions 0 and 3, so FL
        and RR looked correct under either assumption.
        """
        worst = 0.0
        for k in range(0, len(clip), max(1, len(clip) // stride_divisor)):
            rot = rotation_matrices(clip.base_quat[k])
            ref = (clip.feet_pos[k] - clip.base_pos[k]) @ rot
            worst = max(worst, float(np.abs(self.feet(clip.q[k]) - ref).max()))
        return worst


@pytest.fixture(scope="session")
def mj() -> MujocoFK:
    """The Go2 model, or a skip if it has not been fetched."""
    mujoco = pytest.importorskip("mujoco", reason="mujoco not installed")
    if not MODEL_PATH.is_file():
        pytest.skip(f"go2.xml not fetched; expected at {MODEL_PATH}")
    try:
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    except Exception as exc:  # malformed or partial checkout
        pytest.skip(f"go2.xml did not load: {exc}")
    data = mujoco.MjData(model)
    qadr = np.array(
        [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)]
    )
    return MujocoFK(model=model, data=data, qadr=qadr)


def walking_trial(
    steps: int = 200,
    distance: float = 1.0,
    height: float = 0.32,
    overruns: int = 0,
    has_truth: bool = True,
    rate: float = 100.0,
):
    """A `TrialResult` walking `distance` m along +x at a constant rate, dead level.

    Hand-built, so the arithmetic on a trial can be checked without running one.
    With `has_truth` False the ground truth is absent, as on hardware.
    """
    from go2_toolkit.robot.session import TrialResult

    t = np.arange(steps) / rate
    x = distance * t / t[-1]
    if has_truth:
        pos = np.stack([x, np.zeros(steps), np.full(steps, height)], axis=1)
        start_pos, end_pos = pos[0].copy(), pos[-1].copy()
    else:
        pos = np.full((steps, 3), np.nan)
        start_pos = end_pos = None
    quats = np.tile((1.0, 0.0, 0.0, 0.0), (steps, 1))
    return TrialResult(
        tracking_error=np.full(steps, 0.05),
        quats=quats,
        base_pos=pos,
        start_pos=start_pos,
        end_pos=end_pos,
        forward_axis=np.array([1.0, 0.0, 0.0]),
        stand_height=height,
        stand_error=0.01,
        steps=steps,
        rate=rate,
        timing=f"{overruns}/{steps} steps overran",
        overruns=overruns,
    )
