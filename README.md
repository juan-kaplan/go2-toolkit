# go2-toolkit

Drive a Unitree Go2 from Python at the joint level, in a MuJoCo simulator and on the real robot, with the same code. Control code talks to the robot through Unitree's DDS topics (`rt/lowcmd`, `rt/lowstate`). The simulator sits behind the same topics, so switching between the two is a change of domain id and network interface.

This comes from a thesis project on gait control of the Go2. It holds the parts that are useful to anyone who wants to control the robot: the transport, a trial session with safety stops, a simple gait generator, and the notes on what went wrong.

## What is in it

| Path | What |
|---|---|
| `go2_toolkit/robot/` | `DDSBackend` (the DDS transport), `session.py` (a trial: stand ramp, loop, wind down, lower, release, with tilt and state-timeout aborts), sport-mode release, trial logs, metrics, video |
| `go2_toolkit/go2/` | The robot's facts: dimensions, stand and lying poses, joint orders, `JointCommand` |
| `go2_toolkit/control/` | A phase-driven gait generator: phase, foot trajectory, analytic IK, gravity feedforward. Trot, walk and bound |
| `tools/` | `run_bridge.py` (the simulator), `check_connection.py`, `release_sport_mode.py`, `stand_demo.py`, `plot_gait.py`, `setup.sh` |
| `examples/run_gait.py` | Run a gait for N cycles, in the simulator or on the robot |
| `docs/` | [hardware.md](docs/hardware.md): reaching and running the real robot. [command_book.md](docs/command_book.md): copy-paste commands. [traps.md](docs/traps.md): the mistakes that cost us time |

## Safety

The Go2 weighs about 15 kg. Run everything in the simulator first. On the robot, start with it supported (a harness, or held off the ground), with a second person next to it and the Unitree remote within reach. Read [docs/hardware.md](docs/hardware.md) before you release sport mode. Use this software at your own risk.

## Setup

Linux, Python 3.10, and [uv](https://docs.astral.sh/uv/).

```bash
pip install uv          # once, if you do not have it
bash tools/setup.sh
```

This clones `unitree_sdk2_python` and `unitree_mujoco` into `third_party/` and installs everything from `pyproject.toml`. It takes about a minute and needs no compiler on x86. On the robot's Jetson (aarch64) you have to build CycloneDDS yourself; see [docs/hardware.md](docs/hardware.md).

## Quickstart

Start the simulator in one terminal. Drop `--viewer` to run headless:

```bash
uv run python tools/run_bridge.py --viewer
```

In another terminal:

```bash
uv run python tools/stand_demo.py                        # ramp into a stand and hold it
uv run python examples/run_gait.py --gait trot           # trot for 8 cycles
uv run pytest                                            # the test suite
```

Start the bridge first, or `DDSBackend` raises `TimeoutError`.

## Using it in your own code

```python
from go2_toolkit import DDSBackend, JointCommand, STAND_POSE

backend = DDSBackend(domain_id=1, interface="lo")  # the simulator; hardware is domain 0
```

`examples/run_gait.py` and `go2_toolkit/robot/session.py` show a complete trial. Your own controller can be any function that gives a `JointCommand` each tick; the session handles the stand, the stop conditions and the release. The gait generator takes a phase per leg as input, so you can drive it from a clock (as the example does) or from any other source of rhythm.

## On the real robot

Short version: wired Ethernet, release sport mode with `tools/release_sport_mode.py`, run the loop on the robot's Jetson, use the safety flags. The long version, with what we measured and what we only inferred, is in [docs/hardware.md](docs/hardware.md).

## Credits and licence

MIT, see [LICENSE](LICENSE). It builds on Unitree's [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python) and [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco), which `setup.sh` clones and which carry their own licences.
