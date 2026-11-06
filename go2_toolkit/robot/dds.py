"""The one backend: real LowCmd/LowState over DDS.

The same class drives the unitree_mujoco simulator and the physical Go2; only
the domain id and network interface differ. Control code is therefore identical
in both, which is the point of doing it this way.

The peer on the other end does not wait for us. It steps on its own wall clock
and uses whatever command last arrived, so there is timing jitter and the
occasional missed deadline. That is the intended operating condition rather than
a defect to work around: a controller that only holds together under perfect
timing is a property of the simulator, not of the robot.

Two things follow for anything measuring behaviour through this class.

  * Identical inputs give different trajectories, because the interleaving of
    publish and step depends on scheduling. Measure over repeated trials and
    report dispersion. At fixed parameters the spread is itself a result, since
    a tightly held attractor gives a narrow one and a marginal attractor a wide
    one.
  * A control loop slower than its period runs on stale commands, which is
    silent by nature. `step()` measures it, `overruns` counts it, and
    `timing_report()` formats it. Log these with each trial as part of its
    recorded conditions, so behaviour can be correlated against timing quality.
"""

from __future__ import annotations

import time

import numpy as np
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_, SportModeState_
from unitree_sdk2py.utils.crc import CRC

from ..go2.command import JointCommand
from ..go2.conventions import NUM_JOINTS
from .interface import GroundTruth, RobotState

TOPIC_LOWCMD = "rt/lowcmd"
TOPIC_LOWSTATE = "rt/lowstate"
TOPIC_HIGHSTATE = "rt/sportmodestate"

# The DDS domain of the physical robot. The simulator bridge runs on domain 1, so
# a trial on domain 0 is a trial on hardware. `sport_mode.refuse_if_sport_mode_active`
# keys on this: hardware on any other domain would skip that check silently.
HARDWARE_DOMAIN_ID = 0

# The LowCmd header bytes, level flag and motor mode below are what Unitree's own
# low-level example sets (unitree_sdk2_python/example/go2/low_level/
# go2_stand_example.py): 0xFE 0xEF marks the message, 0xFF selects low-level
# control, and 0x01 is servo mode for the Go2's PMSM motors.
_HEAD = (0xFE, 0xEF)
_LOWLEVEL = 0xFF
_MOTOR_MODE_PMSM = 0x01

# ChannelFactoryInitialize is process-global and throws if called twice.
_factory_domain: int | None = None


def init_channel_factory(domain_id: int, interface: str) -> None:
    """Initialise the process-global DDS channel factory, idempotently.

    Public because it is the only piece of `DDSBackend.__init__` that
    `robot.sport_mode` needs: talking to `MotionSwitcherClient` rides the same
    channel factory, but does not touch `LowState`, so it has no reason to pay
    for the wait the rest of `DDSBackend.__init__` does.
    """
    global _factory_domain
    if _factory_domain is None:
        ChannelFactoryInitialize(domain_id, interface)
        _factory_domain = domain_id
    elif _factory_domain != domain_id:
        raise RuntimeError(
            f"DDS factory already initialised on domain {_factory_domain}; "
            f"cannot switch to {domain_id} in the same process"
        )


class DDSBackend:
    """Drive the Go2, simulated or real, over LowCmd/LowState.

    Defaults target the unitree_mujoco bridge (domain 1, loopback). For the
    physical robot use `HARDWARE_DOMAIN_ID` (0) and the NIC facing it, e.g.
    ``DDSBackend(domain_id=0, interface="eth0")``.
    """

    def __init__(
        self,
        domain_id: int = 1,
        interface: str = "lo",
        control_dt: float = 0.002,
        state_timeout: float = 5.0,
        release_kd: float = 0.0,
    ) -> None:
        init_channel_factory(domain_id, interface)
        self.dt = control_dt
        self.release_kd = release_kd

        self._low_state: LowState_ | None = None
        self._low_state_time = time.perf_counter()
        self._high_state: SportModeState_ | None = None
        self._crc = CRC()
        self._cmd = self._blank_lowcmd()

        self._pub = ChannelPublisher(TOPIC_LOWCMD, LowCmd_)
        self._pub.Init()
        self._sub = ChannelSubscriber(TOPIC_LOWSTATE, LowState_)
        self._sub.Init(self._on_low_state, 10)
        self._high_sub = ChannelSubscriber(TOPIC_HIGHSTATE, SportModeState_)
        self._high_sub.Init(self._on_high_state, 10)

        self.overruns = 0
        self.worst_overrun = 0.0
        self._next_deadline: float | None = None

        self.wait_for_state(state_timeout)

    # -- setup ----------------------------------------------------------------

    @staticmethod
    def _blank_lowcmd() -> LowCmd_:
        cmd = unitree_go_msg_dds__LowCmd_()
        cmd.head[0], cmd.head[1] = _HEAD
        cmd.level_flag = _LOWLEVEL
        cmd.gpio = 0
        for m in cmd.motor_cmd:
            m.mode = _MOTOR_MODE_PMSM
            m.q = m.dq = m.kp = m.kd = m.tau = 0.0
        return cmd

    def wait_for_state(self, timeout: float, pose_grace: float = 0.5) -> None:
        """Block until the first LowState arrives, or explain why none did.

        Then give pose a short grace period. LowState and SportModeState are
        published independently, so reading immediately after the first LowState
        would otherwise report pose as unavailable purely because it had not
        arrived yet. Absence is not an error: hardware in low-level mode may
        genuinely never publish it.
        """
        deadline = time.perf_counter() + timeout
        while self._low_state is None:
            if time.perf_counter() > deadline:
                raise TimeoutError(
                    f"no LowState on {TOPIC_LOWSTATE} within {timeout:.0f}s. "
                    "Is the peer running? For simulation start it with "
                    "`python tools/run_bridge.py`; check that domain id and "
                    "interface match on both sides."
                )
            time.sleep(0.01)

        pose_deadline = time.perf_counter() + pose_grace
        while self._high_state is None and time.perf_counter() < pose_deadline:
            time.sleep(0.01)

    # -- DDS callbacks (other thread) -----------------------------------------

    def _on_low_state(self, msg: LowState_) -> None:
        self._low_state = msg
        self._low_state_time = time.perf_counter()

    def _on_high_state(self, msg: SportModeState_) -> None:
        self._high_state = msg

    # -- control loop ---------------------------------------------------------

    def read(self) -> tuple[RobotState, GroundTruth | None]:
        """Most recent state. GroundTruth is None if nothing publishes pose."""
        s = self._low_state
        if s is None:
            raise RuntimeError("no LowState received yet")

        state = RobotState(
            q=np.array([s.motor_state[i].q for i in range(NUM_JOINTS)]),
            dq=np.array([s.motor_state[i].dq for i in range(NUM_JOINTS)]),
            tau_est=np.array([s.motor_state[i].tau_est for i in range(NUM_JOINTS)]),
            imu_quat=np.array(s.imu_state.quaternion),
            imu_gyro=np.array(s.imu_state.gyroscope),
            imu_accel=np.array(s.imu_state.accelerometer),
        )

        truth = None
        if self._high_state is not None:
            truth = GroundTruth(
                base_pos=np.array(self._high_state.position),
                base_vel=np.array(self._high_state.velocity),
            )
        return state, truth

    def state_age(self) -> float:
        """Seconds since the last LowState arrived: how stale `read()` is."""
        return time.perf_counter() - self._low_state_time

    def send(self, cmd: JointCommand) -> None:
        """Publish a LowCmd. Does not wait for it to take effect."""
        for i in range(NUM_JOINTS):
            m = self._cmd.motor_cmd[i]
            m.q = float(cmd.q[i])
            m.dq = float(cmd.dq[i])
            m.kp = float(cmd.kp[i])
            m.kd = float(cmd.kd[i])
            m.tau = float(cmd.tau_ff[i])
        self._cmd.crc = self._crc.Crc(self._cmd)
        self._pub.Write(self._cmd)

    def step(self) -> None:
        """Wait out the rest of the control period.

        This paces the loop; it does not advance the peer, which is free
        running. Falling behind is recorded rather than corrected, because the
        peer has already moved on and there is nothing to catch up to.
        """
        now = time.perf_counter()
        if self._next_deadline is None:
            self._next_deadline = now + self.dt
            return

        remaining = self._next_deadline - now
        if remaining > 0:
            time.sleep(remaining)
            self._next_deadline += self.dt
        else:
            self.overruns += 1
            self.worst_overrun = max(self.worst_overrun, -remaining)
            # resynchronise instead of accumulating debt we cannot repay
            self._next_deadline = time.perf_counter() + self.dt

    def reset_timing(self) -> None:
        """Zero the overrun count and the pacing deadline, at the start of a loop."""
        self.overruns = 0
        self.worst_overrun = 0.0
        self._next_deadline = None

    def timing_report(self, steps: int) -> str:
        """One line: overruns out of `steps`, as a percentage, and the worst one."""
        pct = 100.0 * self.overruns / steps if steps else 0.0
        return (
            f"{self.overruns}/{steps} steps overran ({pct:.1f}%), "
            f"worst {self.worst_overrun * 1e3:.2f} ms"
        )

    def release(self) -> None:
        """Zero position gain so the peer stops holding a stale command.

        Velocity gain is `release_kd`, zero by default: no torque at all, which is
        right for the simulator. On the physical robot that means free fall, so
        pass a small `release_kd` to damp the legs instead, or lower the robot
        to the ground before the run ends.
        """
        self.send(
            JointCommand(
                q=np.zeros(NUM_JOINTS),
                dq=np.zeros(NUM_JOINTS),
                kp=np.zeros(NUM_JOINTS),
                kd=np.full(NUM_JOINTS, self.release_kd),
                tau_ff=np.zeros(NUM_JOINTS),
            )
        )

    def __enter__(self) -> DDSBackend:
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()
