"""What lives in `go2/`: the joint orderings and `JointCommand`.

Pure, no simulator and no robot.
"""

from __future__ import annotations

import numpy as np
import pytest

from go2_toolkit import (
    JOINT_NAMES,
    MENAGERIE_SWAP,
    NUM_JOINTS,
    STAND_POSE,
    JointCommand,
)


def test_menagerie_swap_is_self_inverse() -> None:
    x = np.arange(NUM_JOINTS)
    assert np.array_equal(x[MENAGERIE_SWAP][MENAGERIE_SWAP], x)


def test_menagerie_swap_maps_to_unitree_order() -> None:
    """menagerie FL,FR,RL,RR -> our canonical FR,FL,RR,RL"""
    # fmt: off
    menagerie = [
        "FL_hip", "FL_thigh", "FL_calf",
        "FR_hip", "FR_thigh", "FR_calf",
        "RL_hip", "RL_thigh", "RL_calf",
        "RR_hip", "RR_thigh", "RR_calf",
    ]
    # fmt: on
    assert [menagerie[i] for i in MENAGERIE_SWAP] == list(JOINT_NAMES)


def test_position_hold_zeroes_velocity_and_feedforward() -> None:
    cmd = JointCommand.position_hold(STAND_POSE, kp=60.0, kd=5.0)
    assert np.array_equal(cmd.q, STAND_POSE)
    assert not cmd.dq.any() and not cmd.tau_ff.any()
    assert np.all(cmd.kp == 60.0) and np.all(cmd.kd == 5.0)


def test_position_hold_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="12 joint targets"):
        JointCommand.position_hold(np.zeros(11))
