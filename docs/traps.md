# Engineering Traps and Caveats

The non-obvious traps and simulator quirks we ran into while controlling the Go2. Each one cost us time; none raises an error by itself.

---

## 1. Robotics & Simulation

### Joint Order Conventions
**Warning:** Do not mix the two orders below. The swap mirrors the robot left to right, and no dimension or limit error reports it.

The Go2 SDK and native firmware use Unitree's wire order:
```
FR (0..2), FL (3..5), RR (6..8), RL (9..11)
```
Each leg has three joints: Hip, Thigh, Calf.

In contrast, `mujoco_menagerie` and many reinforcement learning codebases use:
```
FL, FR, RL, RR
```
This is a pairwise swap that mirrors the robot left-to-right without triggering dimension or limit errors. `go2_toolkit.go2.conventions.MENAGERIE_SWAP` converts between them (it is an involution: applying it twice returns the original ordering).

### Privileged State vs. Controller Inputs
**Warning:** Never use `GroundTruth` as feedback into the low-level controller.

`go2_toolkit.robot.interface.GroundTruth` (base position and orientation from MuJoCo or Unitree `SportModeState`) is deliberately kept in a separate object from `RobotState`.
- In simulation, the bridge populates `GroundTruth` directly from MuJoCo ground truth.
- On physical hardware, the corresponding topic is populated by the onboard state estimator, which ceases updating once sport mode is released to grant low-level control.
- `GroundTruth` must only be used for offline metrics and evaluation, never as feedback into the low-level controller.

### DDS Bridge Exclusivity (Never Run Two Bridges)
**Warning:** Never run two bridges on the same DDS domain. The data you collect is meaningless.

CycloneDDS communicates via multicast/unicast on a domain ID without mutual exclusion. If two instances of `tools/run_bridge.py` run simultaneously on domain 1:
- Both instances publish `LowState` and both subscribe to `LowCmd`.
- Control clients receive interleaved state messages from two independent physics simulations while sending commands to both.
- This manifests as erratic physics, sudden jumps, or apparent freezing, rendering collected data meaningless.
- Check for active simulator processes before running:
  ```bash
  ps -eo pid,cmd | grep "[r]un_bridge"
  ```

### Holding Torque After Simulation Reset
**Warning:** After a simulation reset, send stance commands at once. Do not wait. Otherwise the robot collapses.

Unitree Go2 actuators are torque-controlled motors. Resetting the MuJoCo simulation clears the actuator controls (`ctrl = 0`). Motors with no command produce zero holding torque. If nothing commands them, the robot collapses under gravity to about 0.117 m height within 200 ms.

Operational rule:
1. Connect the client.
2. Trigger the reset (`SIGUSR1`).
3. Start to send stance setpoints immediately. Do not add a sleep between the steps.

### IMU Sensor Site Offset
**Warning:** `GroundTruth.base_pos` is not the center of `base_link`. Account for the offset before you compare heights.

`GroundTruth.base_pos` reflects the MuJoCo model's `imu` site sensor rather than the geometric center of `base_link`:
- The IMU site is offset by `(-0.0256, 0.0, +0.0423)` meters relative to `base_link`.
- Reported z-height is approximately 4.2 cm higher than the hip kinematic origin.
- Use `go2_toolkit.go2.model.IMU_SITE_OFFSET` when comparing state estimator heights to kinematic leg clearances.

### Headless vs. GUI Viewer Timing
**Warning:** Run the bridge headless for any benchmark, tracking error or gait statistic. The viewer causes control loop overruns.

MuJoCo's interactive OpenGL viewer runs on the same process thread as the step loop in the bridge script:
- Display rendering and VSync introduce timing jitter, causing control loop overruns (missing the 5 ms / 200 Hz control deadlines).
- When recording benchmarks, tracking error, or gait statistics, run the bridge headlessly (omit `--viewer`). Headless execution reliably achieves 0.0% deadline overruns.

---

## 2. Metrics

### Pitfalls of Absolute Height Fall Detection
**Warning:** Do not detect falls with a fixed world z cutoff. It marks some gaits as fallen for the whole run.

Using a fixed world z-coordinate (e.g. `base_pos[2] < 0.15 m`) to detect falls fails across varied gaits:
- A gait that rides low by design, such as a crawl, makes a fixed cutoff mark the robot as fallen for the entire run.
- Fall detection should always be evaluated relative to the nominal height of the commanded trajectory.

---

## 3. Environment & Python

### Process Management with `pkill`
**Warning:** Never run `pkill -f run_bridge.py`. It kills the terminal session.

Executing:
```bash
pkill -f run_bridge.py
```
matches the text of the command inside the invoking bash subshell, killing the terminal session. Instead, filter specifically for Python processes:
```bash
ps -eo pid,cmd | grep "[r]un_bridge" | awk '{print $1}' | xargs kill
```
or send `SIGUSR1` to the specific PID to trigger a clean simulator reset.

### Conda User-Site Shadowing
**Warning:** A package in the user site can silently shadow the version that the environment pins.

Conda environments act as path prefixes rather than isolated virtual environments:
- Python by default prepends `~/.local/lib/pythonX.Y/site-packages` to `sys.path`.
- Packages installed globally via `pip install --user` silently shadow versions pinned inside the conda environment without warning.
- Standard Python `venv` environments set `ENABLE_USER_SITE = False`, preventing user-site shadowing. This repository uses `uv` with `.venv/` to ensure deterministic dependency resolution.

### Silent Test Skips
**Warning:** Never end a skipped test with a bare `return`. Pytest records it as a pass.

In pytest test suites:
- A test function that prints `"skipped"` and executes a bare `return` exits with status 0, which pytest records as a **pass**.
- Always call `pytest.skip("reason")` so skipped tests are properly registered in the test runner summary.

### Script Execution and `sys.path`
Executing a script directly (e.g., `python examples/run_gait.py`) places the script's directory (`examples/`) at the head of `sys.path`:
- If scripts in subdirectories import each other via naked imports, they can shadow top-level packages or resolve differently depending on the working directory.
- All entry points in this repository import `go2_toolkit` as an installed package.
