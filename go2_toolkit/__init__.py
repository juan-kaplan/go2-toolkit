"""The package's vocabulary, and nothing that costs anything to import.

Importing this package must stay cheap, because the geometry and the transport
are separate concerns and only one of them needs a robot attached. `control` is
pure numpy: a phase table, a Bezier curve and an analytic IK. `dds` binds to the
Unitree SDK and through it to a native DDS library. Eagerly importing the second
from here made the first unusable without the whole robot stack installed, which
surfaced the first time anyone tried to plot a swing curve:

    >>> from go2_toolkit import control
    ModuleNotFoundError: No module named 'unitree_sdk2py'

So `DDSBackend` is resolved on first use instead, via PEP 562. The name is still
importable from the package exactly as before, and nothing that only wants the
geometry pays for the SDK. `GroundTruth` and `RobotState` are resolved lazily
too, though they need no SDK: they live in `robot`, and
`tests/test_imports.py::test_control_does_not_import_robot` asserts that
importing the package loads no `robot` module at all. The rule all this
encodes: the control path depends on numpy and on the robot's dimensions, never
on the transport.

The robot's own facts live one level down in `go2`: `go2.model` for dimensions,
travel, torque and the rest pose, `go2.conventions` for the three joint orderings
this project has to reconcile. Both are numpy only and import nothing of ours,
so they sit at the bottom of the graph and everything else may depend on them.

Submodules (`control`, `robot`, `go2`) resolve through the normal import
machinery, so `from go2_toolkit import control` works without listing them
here, and importing one does not drag in the others.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .go2.command import JointCommand
from .go2.conventions import JOINT_NAMES, MENAGERIE_SWAP, NUM_JOINTS
from .go2.model import IMU_SITE_OFFSET, STAND_POSE

if TYPE_CHECKING:
    # For type checkers and editors only. Never executed at runtime, so it costs
    # nothing, and `DDSBackend` still resolves for autocomplete and mypy.
    from .robot.dds import DDSBackend
    from .robot.interface import GroundTruth, RobotState

__all__ = [
    "DDSBackend",
    "GroundTruth",
    "IMU_SITE_OFFSET",
    "JOINT_NAMES",
    "JointCommand",
    "MENAGERIE_SWAP",
    "NUM_JOINTS",
    "RobotState",
    "STAND_POSE",
    "control",
    "robot",
    "go2",
]


def __getattr__(name: str) -> object:
    """Resolve the transport lazily, so the geometry does not pay for it.

    Raises the SDK's own ImportError, unchanged, if someone actually asks for
    `DDSBackend` without it installed. That is the right moment to complain:
    at the point of use, naming the thing that is missing, rather than at
    `import go2_toolkit` where the message would not say why.
    """
    if name == "DDSBackend":
        from .robot.dds import DDSBackend

        return DDSBackend
    if name in ("GroundTruth", "RobotState"):
        from .robot.interface import GroundTruth, RobotState

        return GroundTruth if name == "GroundTruth" else RobotState
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
