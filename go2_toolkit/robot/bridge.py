"""Finding and steering the simulator process.

Two things every trial needs from the bridge: proof that exactly one simulator
is listening, and a way to put the robot back on its keyframe between runs.

This is simulator-only. On hardware there is no process to find and no keyframe
to reset to, so `Session` treats both as best effort and carries on without
them. It lives in the package rather than beside the drivers because
`robot.session` needs it, and a library reaching up into a scripts directory is
how a checkout in a git worktree once imported the main checkout's copy instead
of its own, and only failed because the two had diverged.

The stance a trial starts from is not here: it is the model's own home keyframe
and lives in `go2.model.STAND_POSE`.
"""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path

# The simulated Go2 in an empty world, shared by the bridge and the video renderer.
# This is the scene as `tools/setup.sh` leaves it: setup removes the hurdle and
# staircase boxes from upstream's scene.xml, so it is not upstream's file.
SCENE_PATH = (
    Path(__file__).resolve().parents[2]
    / "third_party/unitree_mujoco/unitree_robots/go2/scene.xml"
)


def find_bridges() -> list[int]:
    """PIDs of running bridge processes, excluding this one.

    Matches on the executable, not just the command line. Any shell that
    launched a bridge carries "run_bridge.py" in its own command line, so a
    plain substring test counts the wrapper as a second simulator and refuses
    to run against a single healthy peer.
    """
    out = subprocess.run(
        ["ps", "-eo", "pid,comm,cmd"], capture_output=True, text=True
    ).stdout
    pids = []
    for line in out.splitlines()[1:]:
        parts = line.split(maxsplit=2)
        if len(parts) < 3:
            continue
        pid, comm, cmd = parts
        if "run_bridge.py" not in cmd or not comm.startswith("python"):
            continue
        if int(pid) != os.getpid():
            pids.append(int(pid))
    return pids


def check_single_peer() -> None:
    """Refuse to measure anything while two simulators share the topic.

    DDS has no notion of exclusivity. Two bridges on the same domain both
    publish LowState and both subscribe to LowCmd, so a client receives state
    interleaved from two independent simulations while driving both. Readings
    then alternate between two unrelated robots, which looks like a frozen or
    erratic simulation rather than like a configuration error.
    """
    pids = find_bridges()
    if len(pids) > 1:
        raise SystemExit(
            f"\n{len(pids)} bridge processes are running (pids {pids}).\n"
            "They share rt/lowstate and rt/lowcmd, so state readings interleave "
            "between two simulations and both robots receive your commands.\n"
            "Any measurement taken now is meaningless. Stop all but one:\n"
            "    ps -eo pid,cmd | grep '[r]un_bridge'\n"
            "    kill <pid>\n"
        )
    if not pids:
        print("  no bridge process found; expecting a peer elsewhere")


def reset_bridge() -> bool:
    """Put the simulated robot back on the home keyframe before a trial.

    Returns immediately and deliberately does not wait. The reset sets ctrl to
    zero, and these are torque motors with no holding torque, so every
    millisecond spent waiting here is a millisecond the robot spends falling.
    The caller must already be connected and must start commanding at once.
    """
    pids = find_bridges()
    if not pids:
        print("  no bridge process found, cannot reset")
        return False
    for pid in pids:
        os.kill(pid, signal.SIGUSR1)
    return True
