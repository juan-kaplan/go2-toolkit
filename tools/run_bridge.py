"""Run the unitree_mujoco simulator behind the real DDS topics.

The upstream launcher gates its loop on `viewer.is_running()`, so it cannot run
without a GUI window. This does the same job with the viewer optional, which is
what batch work needs: a sweep over hundreds of trials should not open
a window, and CI cannot.

    python tools/run_bridge.py                 # headless, until Ctrl-C
    python tools/run_bridge.py --viewer        # with a window
    python tools/run_bridge.py --duration 30   # stop after 30 s

Start this before anything that constructs a DDSBackend. The scene is the one
`tools/setup.sh` leaves, with the hurdles and stairs removed.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

import mujoco

from go2_toolkit.robot.bridge import SCENE_PATH

SIM_DIR = SCENE_PATH.parents[2] / "simulate_python"

# the bridge imports `config` by name and reads it at import time
sys.path.insert(0, str(SIM_DIR))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", type=Path, default=SCENE_PATH)
    p.add_argument("--domain-id", type=int, default=1)
    p.add_argument("--interface", type=str, default="lo")
    p.add_argument("--timestep", type=float, default=0.002)
    p.add_argument(
        "--duration",
        type=float,
        default=None,
        help="seconds to run; default is until interrupted",
    )
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--keyframe", type=str, default="home")
    args = p.parse_args()

    import config

    config.ROBOT = "go2"
    config.USE_JOYSTICK = 0  # no gamepad attached; pygame would fail
    config.PRINT_SCENE_INFORMATION = False
    config.ENABLE_ELASTIC_BAND = False

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py_bridge import UnitreeSdk2Bridge

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    model.opt.timestep = args.timestep

    try:
        mujoco.mj_resetDataKeyframe(model, data, model.key(args.keyframe).id)
    except KeyError:
        print(f"no keyframe {args.keyframe!r}; starting from model defaults")
    mujoco.mj_forward(model, data)

    ChannelFactoryInitialize(args.domain_id, args.interface)
    bridge = UnitreeSdk2Bridge(model, data)

    viewer = None
    if args.viewer:
        # `from ... import` rather than `import mujoco.viewer`, which would
        # rebind the name `mujoco` as a local and shadow the module above
        from mujoco import viewer as mj_viewer

        viewer = mj_viewer.launch_passive(model, data)

    stop = False
    reset_requested = False

    def _stop(*_):
        nonlocal stop
        stop = True

    def _request_reset(*_):
        # set a flag rather than resetting here: the handler runs between
        # bytecodes of the stepping loop, and mutating mj_data mid step races
        nonlocal reset_requested
        reset_requested = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGUSR1, _request_reset)

    print(
        f"bridge up: domain {args.domain_id} on {args.interface!r}, "
        f"dt {args.timestep}s, {'viewer' if viewer else 'headless'}. "
        f"SIGUSR1 resets to the {args.keyframe!r} keyframe (pid {os.getpid()})",
        flush=True,
    )

    started = time.perf_counter()
    last_sync = started
    try:
        while not stop:
            step_start = time.perf_counter()
            if args.duration is not None and step_start - started >= args.duration:
                break
            if viewer is not None and not viewer.is_running():
                break

            if reset_requested:
                reset_requested = False
                mujoco.mj_resetDataKeyframe(model, data, model.key(args.keyframe).id)
                data.ctrl[:] = 0.0
                mujoco.mj_forward(model, data)
                print("reset", flush=True)

            mujoco.mj_step(model, data)

            if viewer is not None and step_start - last_sync >= 0.02:
                viewer.sync()
                last_sync = step_start

            # free running against the wall clock, exactly like the real robot
            slack = model.opt.timestep - (time.perf_counter() - step_start)
            if slack > 0:
                time.sleep(slack)
    finally:
        if viewer is not None:
            viewer.close()
        del bridge
        print("bridge down", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
