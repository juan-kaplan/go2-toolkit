"""Handing the physical Go2's motors to LowCmd, and back.

Sport mode (the built-in controller) holds the robot and ignores LowCmd until
released, so `stand_demo.py` and any measured trial do nothing on real
hardware until this runs first. Simulation has no sport mode and needs none of
this.

Talking to `MotionSwitcherClient`/`SportClient` and reading `LowState` are two
different prerequisites, and this module keeps them that way. `status` and
`restore` only need the DDS channel factory, which `connect_switcher` sets up
without waiting for anything; `release` additionally needs `LowState`, to
confirm the robot has actually reached the ground before letting go of it, so
it alone opens a full `DDSBackend`. Collapsing the two used to mean `status`
and `restore` paid for a `LowState` wait they never read from, which is
exactly the wrong place for an unrelated hang: `restore` is the "something is
wrong, give control back" path.
"""

from __future__ import annotations

import sys
import time

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
    MotionSwitcherClient,
)
from unitree_sdk2py.go2.sport.sport_client import SportClient

from ..go2.conventions import JOINT_NAMES
from ..go2.model import LYING_POSE
from .dds import HARDWARE_DOMAIN_ID, DDSBackend, init_channel_factory

# "Close enough to call it lowered": 80% of the way to the lying pose's thigh, a
# policy threshold and not the exact pose, so it stays here and is derived from it.
LOWERED_THIGH = 0.8 * float(LYING_POSE[1])
LOWER_TIMEOUT = 10.0  # s for the robot to reach the ground after StandDown

# Timings against the robot's motion service, all in seconds.
# fmt: off
_CLIENT_TIMEOUT = 5.0      # one request to MotionSwitcherClient or SportClient
_STATE_TIMEOUT = 10.0      # wait for the first LowState when release opens a backend
_LOWER_POLL = 0.2          # how often to re-read the thighs while it lowers
_SETTLE = 1.0              # let it rest on the ground before the controller lets go
_RELEASE_TIMEOUT = 5.0     # for ReleaseMode to actually clear the active mode
_RELEASE_POLL = 0.5        # how often to ask while waiting for that
_RESTORE_SETTLE = 1.0      # let a re-selected mode start before reporting it
# fmt: on

# Every leg's thigh index, wherever conventions.py puts it, rather than a
# second hand-copied (1, 4, 7, 10) that could silently drift from JOINT_NAMES.
THIGHS = tuple(i for i, name in enumerate(JOINT_NAMES) if name.endswith("_thigh"))


def _confirm(prompt: str) -> None:
    """Ask before anything that moves the robot; anything but yes exits."""
    answer = input(f"{prompt}\nType 'yes' to continue, anything else aborts: ")
    if answer.strip().lower() != "yes":
        sys.exit("aborted, nothing changed")


def _current_mode(msc: MotionSwitcherClient) -> str:
    code, result = msc.CheckMode()
    if code != 0 or result is None:
        sys.exit(f"CheckMode failed (code {code}); is the robot on this interface?")
    return result.get("name") or ""


def _thighs_lowered(backend: DDSBackend) -> bool:
    state, _ = backend.read()
    return all(state.q[i] >= LOWERED_THIGH for i in THIGHS)


def connect_switcher(domain_id: int, interface: str) -> MotionSwitcherClient:
    """A ready `MotionSwitcherClient`, without waiting for `LowState`.

    `status` and `restore` need nothing more than this. `release` goes on to
    open its own `DDSBackend` besides, since it alone needs to read joint
    state.
    """
    init_channel_factory(domain_id, interface)
    msc = MotionSwitcherClient()
    msc.SetTimeout(_CLIENT_TIMEOUT)
    msc.Init()
    return msc


def refuse_if_sport_mode_active(domain_id: int, interface: str) -> None:
    """Fail loudly if the built-in controller still owns the robot.

    Simulation has no sport mode (domain 1, always released in spirit), so
    this is a no-op there and never touches DDS. On hardware (`HARDWARE_DOMAIN_ID`), a
    trial run while sport mode is still active sends LowCmd that nothing is
    listening to: the built-in controller holds the robot regardless, so the
    loop completes and the report prints as if the trial were a real, if
    uneventful, one. That is what happened in the first hardware session,
    `stand_demo` ran to completion and changed nothing. Silently-wrong data
    is exactly the failure shape this repo checks hardest against elsewhere
    (`check_single_peer` exists for the same reason), so this fails fast
    instead of producing a trial to be misread later.
    """
    if domain_id != HARDWARE_DOMAIN_ID:
        return
    msc = connect_switcher(domain_id, interface)
    mode = _current_mode(msc)
    if mode:
        raise SystemExit(
            f"sport mode is active ({mode!r}); LowCmd has no effect until it is "
            "released. Run first:\n"
            f"    python tools/release_sport_mode.py release "
            f"--interface {interface} --domain-id {domain_id}"
        )


def status(msc: MotionSwitcherClient) -> None:
    """Print which mode, if any, currently owns the robot. Read-only."""
    mode = _current_mode(msc)
    print(f"active mode: {mode!r}" if mode else "no mode active (released)")


def release(msc: MotionSwitcherClient, domain_id: int, interface: str) -> None:
    """Lower the robot to the ground, then release LowCmd.

    Refuses to release while the robot is still standing: the SDK's own
    example calls `StandDown` and `ReleaseMode` back to back with no check,
    which releases a robot that has not actually reached the ground yet.
    """
    mode = _current_mode(msc)
    if not mode:
        print("already released, nothing to do")
        return
    _confirm(
        f"Active mode is {mode!r}. The robot will lower itself to the ground, then "
        "its built-in controller is released.\n"
        "Clear the space around it and be next to it."
    )

    sc = SportClient()
    sc.SetTimeout(_CLIENT_TIMEOUT)
    sc.Init()
    sc.StandDown()

    # The only step in `release` that needs LowState, so the only one that
    # opens a full backend.
    backend = DDSBackend(
        domain_id=domain_id, interface=interface, state_timeout=_STATE_TIMEOUT
    )
    deadline = time.time() + LOWER_TIMEOUT
    while not _thighs_lowered(backend):
        if time.time() > deadline:
            sys.exit(
                f"robot did not reach the ground within {LOWER_TIMEOUT:.0f}s "
                f"(thigh < {LOWERED_THIGH} rad); NOT released"
            )
        time.sleep(_LOWER_POLL)
    time.sleep(_SETTLE)

    code, _ = msc.ReleaseMode()
    if code != 0:
        sys.exit(f"ReleaseMode failed (code {code}); state unchanged")
    deadline = time.time() + _RELEASE_TIMEOUT
    while _current_mode(msc):
        if time.time() > deadline:
            sys.exit("ReleaseMode accepted but a mode is still active; run `status`")
        time.sleep(_RELEASE_POLL)
    print(
        "released. LowCmd is now accepted. To hand back: "
        f"python tools/release_sport_mode.py restore --mode {mode}"
    )


def restore(msc: MotionSwitcherClient, mode: str) -> None:
    """Hand the robot back to a built-in mode (`release` prints which one it was)."""
    _confirm(
        f"Selecting {mode!r}. The robot may move as its built-in controller "
        "starts; clear the space around it."
    )
    code, _ = msc.SelectMode(mode)
    if code != 0:
        sys.exit(f"SelectMode failed (code {code})")
    time.sleep(_RESTORE_SETTLE)
    status(msc)
