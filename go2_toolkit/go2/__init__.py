"""What the robot is, as opposed to what it does.

Three modules, all numpy only, importing nothing else from this package.
`model` holds the physical Go2: link lengths, joint travel, torque ceilings, the
rest pose. `conventions` holds index orders and frame conventions, and the
permutations between the three different joint orderings this project has to
reconcile. `command` holds `JointCommand`, one LowCmd in wire order, here so
that `control` can build one without importing `robot`.

This is the bottom of the dependency graph. Control imports it, the transport
imports it, and it imports nothing of ours.
"""

from __future__ import annotations

from . import command, conventions, model
from .command import JointCommand

__all__ = ["JointCommand", "command", "conventions", "model"]
