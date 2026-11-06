"""The control path stays importable without the robot stack.

This is a structural test, not a behavioural one. It guards the rule the package
is built around: the geometry depends on numpy and on the robot's dimensions,
never on the transport. The rule is easy to break by accident,
because adding one convenience import to `__init__.py` is enough, and nothing
about the failure points at the cause. It showed up originally as a
ModuleNotFoundError for `unitree_sdk2py` while trying to draw a Bezier curve.

Each check runs in a subprocess with the heavy modules actively blocked, since
they are installed in this environment and a plain import would succeed for the
wrong reason.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import textwrap

# Blocked so that "it imported fine" cannot mean "it imported the SDK fine".
# scipy and matplotlib are here too: the control path must not need a solver or
# a plotting library either.
BLOCKED = ("unitree_sdk2py", "cyclonedds", "torch", "scipy", "matplotlib")

_PREAMBLE = f"""
import sys
from importlib.abc import MetaPathFinder

BLOCKED = {BLOCKED!r}


class _Blocker(MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"{{name}} is blocked by tests/test_imports.py")
        return None


sys.meta_path.insert(0, _Blocker())
"""


def _run(body: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _PREAMBLE + textwrap.dedent(body)],
        capture_output=True,
        text=True,
    )


def test_the_package_imports_without_the_robot_stack() -> None:
    r = _run("""
        import go2_toolkit as pkg
        assert pkg.NUM_JOINTS == 12
        assert pkg.JointCommand.position_hold([0.0] * 12).q.shape == (12,)
        print("ok")
    """)
    assert r.returncode == 0, (
        f"importing the package needed a blocked module:\n{r.stderr}"
    )


def test_the_gait_generator_runs_on_numpy_alone() -> None:
    """The whole control path: phase in, joint angles out, nothing else loaded."""
    r = _run("""
        from go2_toolkit import control
        from go2_toolkit.go2 import model
        q = control.joint_trajectory(control.GAITS["walk"], control.GaitParams())
        assert q.shape == (12,), q.shape
        assert model.within_limits(q).all()
        print("ok")
    """)
    assert r.returncode == 0, f"the gait generator needed a blocked module:\n{r.stderr}"


def test_importing_the_package_does_not_pull_in_the_transport() -> None:
    """Cheap import means dds is absent from sys.modules, not merely importable."""
    r = _run("""
        import sys
        import go2_toolkit  # noqa: F401
        from go2_toolkit import control  # noqa: F401
        loaded = [m for m in sys.modules if m.startswith("go2_toolkit.")]
        assert "go2_toolkit.robot.dds" not in loaded, loaded
        print("ok")
    """)
    assert r.returncode == 0, r.stderr


def test_control_does_not_import_robot() -> None:
    """The dependency rule: control imports only go2, never robot."""
    r = _run("""
        import sys
        from go2_toolkit import control  # noqa: F401
        loaded = [m for m in sys.modules if m.startswith("go2_toolkit.robot")]
        assert not loaded, f"control imported robot modules: {loaded}"
        print("ok")
    """)
    assert r.returncode == 0, r.stderr


def test_asking_for_the_backend_still_raises_the_sdk_error() -> None:
    """Lazy must not mean silent: the failure moves to the point of use."""
    r = _run("""
        import go2_toolkit
        try:
            go2_toolkit.DDSBackend
        except ImportError as exc:
            assert "unitree_sdk2py" in str(exc), exc
            print("ok")
        else:
            raise AssertionError("DDSBackend resolved with the SDK blocked")
    """)
    assert r.returncode == 0, r.stderr


def test_a_typo_on_the_package_still_raises_attribute_error() -> None:
    """__getattr__ must not turn every misspelling into an ImportError."""
    r = _run("""
        import go2_toolkit
        try:
            go2_toolkit.DDSBakcend
        except AttributeError:
            print("ok")
    """)
    assert r.returncode == 0, r.stderr


def test_the_backend_is_importable_when_the_sdk_is_present() -> None:
    """The lazy path must actually work, not just fail cleanly."""
    from go2_toolkit import DDSBackend

    assert DDSBackend.__name__ == "DDSBackend"
    assert "DDSBackend" in dir(__import__("go2_toolkit"))


def test_every_entry_point_imports() -> None:
    """Nothing in tests/ imports the entry points, so a stale import is invisible.

    The gap was real: renaming the gait module left
    `from go2_toolkit.gait import GaitParams` in run_generated_gait.py, the whole
    suite stayed green, and the only thing that caught it was running --help by
    hand.

    This imports each entry point without running its `__main__` block, which
    catches the stale-import class of breakage without executing anything or
    needing a simulator.
    """
    root = pathlib.Path(__file__).resolve().parents[1]
    by_dir = {d: sorted((root / d).glob("*.py")) for d in ("examples", "tools")}
    for directory, found in by_dir.items():
        assert found, f"no entry points under {directory}/; has it moved again?"
    scripts = [p for found in by_dir.values() for p in found]
    broken = []
    for script in scripts:
        code = (
            "import importlib.util, sys\n"
            f"sys.path.insert(0, {str(root)!r})\n"
            f"spec = importlib.util.spec_from_file_location('_ep', {str(script)!r})\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            # register before executing: @dataclass resolves its own module via
            # sys.modules[cls.__module__], and gets None if we skip this
            "sys.modules['_ep'] = mod\n"
            "spec.loader.exec_module(mod)\n"
        )
        r = subprocess.run(
            [sys.executable, "-c", code], cwd=root, capture_output=True, text=True
        )
        if r.returncode != 0:
            last = r.stderr.strip().splitlines()[-1:] or ["(no stderr)"]
            broken.append(f"{script.name}: {last[0]}")
    assert not broken, "entry points failed to import:\n  " + "\n  ".join(broken)
