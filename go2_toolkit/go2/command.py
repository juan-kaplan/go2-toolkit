"""Joint command definitions in Unitree wire order."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .conventions import NUM_JOINTS


@dataclass
class JointCommand:
    """One LowCmd, in Unitree joint order.

    The resulting torque, computed by the joint's motor board on hardware or by
    the bridge in simulation, is

        tau = tau_ff + kp * (q - q_measured) + kd * (dq - dq_measured)
    """

    # fmt: off
    q: np.ndarray       # target position [rad]
    dq: np.ndarray      # target velocity [rad/s]
    kp: np.ndarray      # position gain
    kd: np.ndarray      # velocity gain
    tau_ff: np.ndarray  # feedforward torque [N*m]
    # fmt: on

    @classmethod
    def position_hold(
        cls,
        q: np.ndarray,
        kp: float | np.ndarray = 60.0,
        kd: float | np.ndarray = 5.0,
    ) -> JointCommand:
        """Hold a posture: zero target velocity, zero feedforward torque.

        The default gains are soft, enough to hold a stand for a smoke test.
        Trials pass their own: 100 and 6 for a stand, `control.DEFAULT_KP` and
        `DEFAULT_KD` for a gait.
        """
        q = np.asarray(q, dtype=float)
        if q.shape != (NUM_JOINTS,):
            raise ValueError(f"expected {NUM_JOINTS} joint targets, got {q.shape}")
        return cls(
            q=q,
            dq=np.zeros(NUM_JOINTS),
            kp=np.broadcast_to(np.asarray(kp, dtype=float), (NUM_JOINTS,)).copy(),
            kd=np.broadcast_to(np.asarray(kd, dtype=float), (NUM_JOINTS,)).copy(),
            tau_ff=np.zeros(NUM_JOINTS),
        )
