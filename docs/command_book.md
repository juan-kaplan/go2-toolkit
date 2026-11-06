# Go2 command book

Copy-paste commands for the simulator and the physical Go2, with one line on what each does. Replace `<iface>` with the name of your wired adapter (`ip -br addr`), and `<jetson>` with the Jetson's address. For the reasoning and the measurements see `hardware.md`.

Gains and flags are the ones we used on the robot: `--kp 120 --step-length 0.12 --release-kd 6 --release-ramp 1 --max-tilt 35`.

## Simulator (laptop)

**Start the bridge** in the background. Wait for `bridge up:`. Run it headless for anything you measure:

```bash
uv run python tools/run_bridge.py --domain-id 1 --interface lo &
```

**With a window**, for looking only (it causes control-loop overruns):

```bash
uv run python tools/run_bridge.py --viewer
```

**Ramp into a stand and hold it:**

```bash
uv run python tools/stand_demo.py
```

**Trot, 8 cycles:**

```bash
uv run python examples/run_gait.py --gait trot --cycles 8
```

**Walk, with a video** (needs `uv sync --extra plots`):

```bash
uv run python examples/run_gait.py --gait walk --record-video
```

**Stop the bridge by its PID. Never use `pkill -f`:**

```bash
kill <pid>
```

If you lost the PID: `pgrep -af run_bridge.py`. The first trial after a bridge start is a throwaway.

## Connect to the robot

**Log in to the Jetson** (laptop on the same Wi-Fi):

```bash
ssh unitree@<jetson>
```

Run every command below on the Jetson with `--interface eth0`, from the repo, in a shell where `CYCLONEDDS_HOME` and `LD_LIBRARY_PATH` are set. Without them, DDS tools print nothing.

**Link to the robot, read only. Must print `LowState received`:**

```bash
uv run python tools/check_connection.py --domain-id 0 --interface eth0
```

**Sport mode status. `normal` means the built-in controller is on, `released` means yours:**

```bash
uv run python tools/release_sport_mode.py status --interface eth0
```

**Wired fallback, from the laptop only:**

```bash
nmcli con up go2-wired
ssh unitree@192.168.123.18
```

## Sport mode

The robot must be released before any gait runs. A restart of the robot brings the built-in controller back.

**Release.** The robot lowers itself first. Clear the space, be next to it, type `yes`:

```bash
uv run python tools/release_sport_mode.py release --interface eth0
```

**Restore** the built-in controller. Put the robot on the ground first, type `yes`:

```bash
uv run python tools/release_sport_mode.py restore --interface eth0 --mode normal
uv run python tools/release_sport_mode.py status --interface eth0
```

The status must print `active mode: 'normal'`.

## Walk the robot

Sport mode released, the robot supported by a person or a tether, a second person next to it. Use `--support lifted` for tethered runs and `--support floor` for free standing. Do not raise `--kp`, `--step-length` or `--max-tilt` in the middle of a session.

**Trot, 12 cycles:**

```bash
uv run python examples/run_gait.py --gait trot --cycles 12 --step-length 0.12 --kp 120 --domain-id 0 --interface eth0 --no-reset --stand-ramp 3 --max-tilt 35 --release-kd 6 --release-ramp 1 --support lifted
```

**Walk, 12 cycles:**

```bash
uv run python examples/run_gait.py --gait walk --cycles 12 --step-length 0.12 --kp 120 --domain-id 0 --interface eth0 --no-reset --stand-ramp 3 --max-tilt 35 --release-kd 6 --release-ramp 1 --support lifted
```

## Logs

Each run writes `<time>_<what>_<where>.npz` and a `.json` to `results/trials/`. The Jetson's clock may not match your laptop's.

**List the newest logs (Jetson):**

```bash
ls -1t results/trials | head -8
```

**Copy them to the laptop (laptop):**

```bash
mkdir -p results/trials
scp "unitree@<jetson>:<path-to-repo>/results/trials/*" results/trials/
```

## Stops and quick fixes

| Situation | What to do |
| --- | --- |
| A leg buzzes, shakes or hits the body | Ctrl+C once: wind down, hold 0.5 s, lower 2 s, release ramp 1 s. Do not repeat the step |
| Legs behave dangerously | Ctrl+C twice: immediate damped release |
| Tilt past `--max-tilt` | Automatic damped release; the report says `ABORTED`. Stop and write down why |
| No `LowState` for 0.2 s | Automatic damped release |
| Overruns above 1.5% of steps | Run the step once more, then stop |
| Says sport mode is active | Release it again |
| SSH drops during a trial | Reconnect, check the legs, then restore sport mode |
| `release` or `check_connection` print nothing | `CYCLONEDDS_HOME` or `LD_LIBRARY_PATH` is not set in this shell |
| `eth0: does not match an available interface` | You are on the laptop; run on the Jetson, or pass your wired adapter's name |
| `TimeoutError` from `DDSBackend` in the simulator | The bridge is not running, or the domain id or interface differ |

Never raise `--kp`, `--step-length` or `--max-tilt` in a step, and never go to the floor from a step that was not clean.
