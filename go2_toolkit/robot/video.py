"""Render the robot of a recorded trial with MuJoCo offscreen rendering.

This renders offline after the control loop completes, ensuring zero impact on
real-time physics and preventing timing overruns. There are two layers:

  * `RobotRenderer` draws one frame from a measured pose: the twelve joint angles
    (Unitree order), the body orientation and, where there is one, the body position.
    `tools/dashboard.py` calls it for every frame it shows.
  * `render_trial_video` writes a whole `TrialResult` as an MP4, a thin wrapper
    around the same renderer.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..go2.conventions import MENAGERIE_SWAP, rotation_matrices
from ..go2.model import IMU_SITE_OFFSET
from .bridge import SCENE_PATH

if TYPE_CHECKING:
    from .session import TrialResult

# Camera framing, degrees: looks down 20 degrees, at azimuth 135 around the robot.
_CAMERA_ELEVATION = -20.0
_CAMERA_AZIMUTH = 135.0

# Where to draw the body when there is no ground truth at all: the hip height
# of a standing Go2, in metres, at the origin.
_FALLBACK_BODY_HEIGHT = 0.30


def base_link_positions(quats: np.ndarray, pos: np.ndarray | None) -> np.ndarray:
    """World position of `base_link` at every tick, (n, 3), from the log's arrays.

    `pos` is the IMU site, not the body origin, so the site offset rotated into the
    world is subtracted. It may be None, or NaN on some or all ticks: ground truth
    stops on hardware once sport mode is released, and it can arrive intermittently.
    Gaps are filled by interpolating between the readings either side; with no
    reading at all the robot is drawn standing at the origin, in place. A NaN
    orientation is drawn upright.
    """
    n = len(quats)
    if pos is None or len(pos) != n or not np.isfinite(pos).any(axis=1).any():
        return _standing_at_origin(n)
    pos = pos.copy()
    valid = np.isfinite(pos).all(axis=1)
    if not valid.all():
        indices = np.arange(n)
        for col in range(3):
            pos[:, col] = np.interp(indices, indices[valid], pos[valid, col])
    offset_world = np.einsum(
        "nij,j->ni", rotation_matrices(upright_where_missing(quats)), IMU_SITE_OFFSET
    )
    return pos - offset_world


def upright_where_missing(quats: np.ndarray) -> np.ndarray:
    """The quaternions, with the identity where a reading is NaN."""
    quats = np.asarray(quats, dtype=float).copy()
    quats[~np.isfinite(quats).all(axis=1)] = (1.0, 0.0, 0.0, 0.0)
    return quats


def _standing_at_origin(steps: int) -> np.ndarray:
    pos = np.zeros((steps, 3))
    pos[:, 2] = _FALLBACK_BODY_HEIGHT
    return pos


class RobotRenderer:
    """Draws the Go2 in a pose, one frame at a time. Close it when done.

    The camera tracks `base_link`, so the robot stays framed as it moves. Rendering is
    headless EGL unless the environment already names a `MUJOCO_GL`: about 3 ms a
    frame here, against 240 ms for osmesa, the pure software fallback. Set
    `MUJOCO_GL=osmesa` on a machine with no EGL.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        camera_distance: float = 1.8,
        scene_path: Path | str | None = None,
    ) -> None:
        os.environ.setdefault("MUJOCO_GL", "egl")
        import mujoco

        scene_file = Path(scene_path if scene_path is not None else SCENE_PATH)
        if not scene_file.exists():
            raise FileNotFoundError(f"MuJoCo scene XML not found at {scene_file}")
        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(scene_file))
        self.data = mujoco.MjData(self.model)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.camera.trackbodyid = self.model.body("base_link").id
        self.camera.distance = camera_distance
        self.camera.elevation = _CAMERA_ELEVATION
        self.camera.azimuth = _CAMERA_AZIMUTH
        # the offscreen buffer defaults to 640x480, which caps the frame size
        self.model.vis.global_.offwidth = max(width, self.model.vis.global_.offwidth)
        self.model.vis.global_.offheight = max(height, self.model.vis.global_.offheight)
        self._renderer = mujoco.Renderer(self.model, height, width)

    def frame(self, q: np.ndarray, quat: np.ndarray, base_link_pos: np.ndarray):
        """One RGB image (height, width, 3); `q` is Unitree order, `quat` scalar first."""
        self.data.qpos[0:3] = base_link_pos
        self.data.qpos[3:7] = quat
        self.data.qpos[7:19] = np.asarray(q)[MENAGERIE_SWAP]
        self._mujoco.mj_forward(self.model, self.data)
        self._renderer.update_scene(self.data, camera=self.camera)
        return self._renderer.render()

    def close(self) -> None:
        self._renderer.close()

    def __enter__(self) -> RobotRenderer:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def render_trial_video(
    result: TrialResult,
    path: Path | str,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
    camera_distance: float = 1.8,
    scene_path: Path | str | None = None,
) -> Path:
    """Render the trial trajectory to an MP4 video file."""
    if result.measured_q is None or len(result.measured_q) == 0:
        raise ValueError("TrialResult contains no measured joint angles to render")

    out_path = Path(path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import cv2

    # Frame sampling stride to match the requested FPS
    step_skip = max(1, int(round(result.rate / fps)))
    actual_fps = result.rate / step_skip
    base_link_pos = base_link_positions(result.quats, result.base_pos)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, actual_fps, (width, height))
    try:
        with RobotRenderer(width, height, camera_distance, scene_path) as robot:
            for idx in range(0, len(result.measured_q), step_skip):
                rgb = robot.frame(
                    result.measured_q[idx], result.quats[idx], base_link_pos[idx]
                )
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    return out_path
