# Running on the real Go2

How we reach the robot, take low-level control, run a gait and end a run safely. Simulation needs none of this.

Some statements below are measurements and some are inferences. Where it matters, the text says which.

**Read this first.** A Go2 weighs about 15 kg and its legs are strong. Run in the simulator first. For the first runs on the robot, have it supported (a harness, or held off the ground by a person) and keep a second person next to it. Keep the Unitree remote within reach. Do not raise `--kp`, `--step-length` or `--max-tilt` in the middle of a session.

---

## 1. Summary

- Reach the robot over **wired Ethernet**. Low-level DDS (`rt/lowstate`, `rt/lowcmd`) did not work for us over Wi-Fi or over the robot's own hotspot.
- The built-in controller (**sport mode**) must be released before `LowCmd` has any effect. With it active, `stand_demo` ran and changed nothing.
- Run the 200 Hz control loop on the robot's **Jetson** (the robot's internal wire, `eth0`), and reach the Jetson over Wi-Fi with SSH. Only keystrokes cross Wi-Fi.
- After sport mode is released, the pose topic stops, so the report has no distance or speed on hardware.
- The safety options (`--stand-ramp`, `--max-tilt`, `--release-kd`, `--release-ramp`, `--support`) default to off, which keeps the simulator's behaviour unchanged. Use them on hardware.

---

## 2. Network

### Addresses

These are the Go2's factory defaults.

| Host | Address | Notes |
|---|---|---|
| Your wired NIC | `192.168.123.99/24` (your choice) | Any free address in the /24 outside `.1`, `.18` and `.161` works |
| Main board | `192.168.123.161` | Publishes DDS. No SSH. It also serves the robot's Wi-Fi and hotspot |
| Jetson (Orin/Xavier "PC2") | `192.168.123.18` | SSH open (user `unitree`). Ubuntu 20.04, aarch64, no Wi-Fi hardware of its own |

### Bring the link up

Once, on a machine that has not talked to this robot:

```bash
ip -br addr                                             # name of the new wired adapter
sudo nmcli con add type ethernet ifname <iface> con-name go2-wired \
    ipv4.method manual ipv4.addresses 192.168.123.99/24 connection.autoconnect no
```

At the start of every session:

```bash
nmcli con up go2-wired
ip -br addr show <iface>                                # expect 192.168.123.99/24
ping -c 3 192.168.123.161
uv run python tools/check_connection.py --domain-id 0 --interface <iface>
```

`check_connection.py` is read-only: it never publishes. `--interface` is required and has no default, because a default would be wrong for every machine but one. Wi-Fi for internet can stay up at the same time. The wired link has no gateway, and the DDS tools use the interface you give them.

### Running on the Jetson

The control loop should run on the Jetson, not on your laptop. The Jetson reaches the main board through the robot's internal wire (`eth0`), so the 200 Hz loop never goes through Wi-Fi. DDS from a laptop over Wi-Fi did not work for us (section 5.1), and its round trip does not fit in a 5 ms period anyway.

**Warning:** If the SSH session drops, the process can stop with it. The legs then keep their last command and nobody sends a new one. Stay connected and keep trials short. A terminal multiplexer such as `tmux` would help, but we did not install one.

1. Plug a USB Wi-Fi dongle into the Jetson (the Go2's free USB port). The Jetson then has internet. `eth0` stays on the robot network: the default route goes out through `wlan0`, and the robot's addresses stay on `eth0`.
2. Put your laptop on the same Wi-Fi network.
3. Find the Jetson's Wi-Fi address in your router, or with `ip -br addr show wlan0` from a shell on the Jetson. DHCP gives this address, so it changes. A fixed hostname (mDNS, for example) avoids that.
4. `ssh unitree@<jetson-address>`.
5. Clone this repo on the Jetson and run `bash tools/setup.sh` there.
6. Run the commands of section 3 with `--interface eth0`.

How the stack was built on the Jetson:

- **Dongle:** an RTL8822BU adapter (TP-Link Archer T3U Plus v1). The kernel has no driver for it. We built the out-of-tree driver `morrownr/88x2bu-20210702` on the Jetson with `make`. We loaded it with `modprobe cfg80211`, then `insmod`, and joined the network with `nmtui`. A reboot removes the driver, so repeat the `modprobe` and `insmod` after one.
- **Controller:** `uv` installs Python 3.10. PyPI has no aarch64 wheel for `cyclonedds`, so we built CycloneDDS 0.10.2 from source into `~/cyclonedds-install`, and then ran `uv sync`. We exported `CYCLONEDDS_HOME` and `LD_LIBRARY_PATH` from a small env file that we source in every shell. Without it, the DDS tools print nothing.
- Do not `apt upgrade` the Jetson.

---

## 3. Running a gait on the robot

**Warning:** Put the robot on clear ground with a person next to it. Releasing sport mode lowers the robot first. The script waits until all four thighs read at least 1.09 rad and refuses to release otherwise (section 5.3).

**Warning:** Choose `--support` before you run a gait: `lifted` if someone holds the robot off the ground, `floor` if it stands. Run lifted first, then on the floor. The log records the value, and the two conditions are not comparable.

1. Release sport mode. The script asks for confirmation. Type `yes` at the prompt.

```bash
uv run python tools/release_sport_mode.py release --interface <iface>
```

2. Run a gait.

```bash
uv run python examples/run_gait.py --gait trot --cycles 16 \
    --step-length 0.12 --kp 120 --domain-id 0 --interface <iface> \
    --no-reset --stand-ramp 3 --max-tilt 35 --release-kd 6 --release-ramp 1 \
    --support lifted
```

3. Give the built-in controller back, with the robot on the ground.

```bash
uv run python tools/release_sport_mode.py restore --interface <iface> --mode normal
```

To read the active mode without changing anything, run `tools/release_sport_mode.py status`. A restart of the robot also brings sport mode back.

| Flag | Why it is needed on hardware |
|---|---|
| `--no-reset` | There is no simulator bridge to reset |
| `--stand-ramp 3` | The robot starts on the ground. The session eases from the measured pose into the stand over this many seconds. The default is 1, for the simulator. Without a ramp, the session would send the stand at full gain, a step of most of a radian |
| `--max-tilt 35` | Release the legs at once past 35 degrees of roll or pitch |
| `--release-kd 6` | Damping at the end-of-trial release. The default of 0 is no torque, and on hardware no torque is free fall. We used 6 and have not tuned it |
| `--release-ramp 1` | The legs let go over 1 s at the lying pose, not in one tick. Without the ramp, the body falls the last few centimetres at the end of a trial (section 5.4) |
| `--support` | Required on hardware (domain 0). The trial does not start without it |

A trial is one sequence. Each stage starts where the last stage left the legs, so no stage steps a joint:

1. **Stand ramp:** ease from the measured pose into the gait's own stand: its pose, its gains and its gravity load.
2. **Settle:** hold the stand for 1 s.
3. **Loop:** the stride grows from zero to full over the first second (`--warmup`). Then the gait runs.
4. **Wind down:** the stride shrinks to zero over the same time, from wherever the gait is in its cycle.
5. **Hold:** stand for 0.5 s.
6. **Lower:** ease to Unitree's lying pose (thigh 1.36, calf -2.65) over 2 s.
7. **Release ramp** (only with `--release-ramp`): at the lying pose, `kp` eases to zero and `kd` to `--release-kd`.
8. **End-of-trial release:** damped with `--release-kd`. The robot is already on the ground.

Four events end a trial early:

| Exit | What happens |
|---|---|
| First Ctrl+C | The loop stops at the next tick. The sequence runs from the wind down, release ramp included |
| Second Ctrl+C | Immediate damped release, from any stage |
| Tilt past `--max-tilt` | Immediate damped release. The report says `ABORTED` |
| No `LowState` for 0.2 s | Immediate damped release. The robot's state has stopped arriving, so nothing in the sequence can be trusted |

The tests run these exits against a stub backend, and we exercised them in the simulator. Treat the exits as untested on the robot until you have tried them with the robot lifted.

### The trial log

Every trial writes a log after the release, to `results/trials/`: `<time>_<what>_<where>.npz` and a `.json` with the same stem, for example `20261007_150656_trot_sim` or `20261007_150656_trot_hw_floor`. `--no-log` turns it off. The Jetson has no matplotlib, so copy logs to your laptop with `scp` to plot them (`tools/plot_gait.py` needs `uv sync --extra plots`).

---

## 4. What a run measures

On hardware the report has the tracking error, the torso attitude (roll and pitch), the timing (control-loop overruns) and the commanded stand error. It does **not** have height, distance or speed, because `SportModeState` never arrives after sport mode is released (section 5.3). That the robot moved properly is the operator's observation.

In the simulator the report also has the body height and the speed against what the stride should give. Run the bridge headless for any number you want to trust (`docs/traps.md`).

---

## 5. Findings

### 5.1 DDS did not work over Wi-Fi

Evidence, with the wired link down for the Wi-Fi tests:

- With the robot in station mode on a campus Wi-Fi network, the default DDS configuration received no `rt/lowstate`.
- The robot sent RTPS discovery multicast on its wired side (340 packets in 40 s) and **none** on Wi-Fi in 40 s.
- The network was not the blocker as far as we could tell. The robot answered ping and TCP 9991 (the app) on Wi-Fi, and other hosts' RTPS multicast reached the laptop. We did not test the laptop-to-robot direction of multicast.
- Unicast-only discovery to a named peer fails even on the wire, so it is not a workaround.
- Joining the robot's hotspot gave the laptop an address with no gateway, and default DDS still received nothing. We did not finish ruling the hotspot out.

Most likely the robot's own DDS service is bound to its wired interface. This is **unverified**: the main board offers no SSH to read its configuration.

Timing makes the question mostly moot. The round trip to the robot over Wi-Fi measured about 6 ms with peaks of 14 ms, against 0.9 ms wired. A 5 ms control period cannot tolerate that.

### 5.2 The two boards

The main board (`.161`) owns DDS and the Wi-Fi and cannot be logged into. The Jetson (`.18`) can be logged into but is a separate computer. Out of the box it has Python 3.8, an old numpy, and neither `cyclonedds` nor `unitree_sdk2py`, and over `eth0` it has no route to the internet. That is why the dongle in section 2 is needed.

Switching the robot to hotspot mode over SSH is not possible in a supported way. The supported route is the Unitree Go app.

### 5.3 Sport mode

While the built-in controller is active it holds the robot and `LowCmd` has no effect. The evidence is a before and after with the same demo. Before the release, `stand_demo` ended with the joints still 0.30 rad from the target and the base height unchanged (0.320 m to 0.316 m). After the release, it reached 0.025 rad.

`tools/release_sport_mode.py` follows Unitree's own example (`CheckMode`, `StandDown`, `ReleaseMode`) with one addition: it waits until all four thighs read at least 1.09 rad (80% of the lying pose's thigh, 1.36), and refuses to release otherwise. The SDK example releases immediately, even while the robot is still standing.

After the release, `SportModeState` stops updating, so `GroundTruth` is absent on hardware.

### 5.4 The end of a run

`DDSBackend.release()` sends zero position gain and zero torque so the peer stops holding a stale command. In simulation that is correct. On hardware it lets go of the robot. `DDSBackend(release_kd=...)` adds damping so a drop is soft.

We first ended every trial this way, with the legs damped and the robot dropping. A trial now lowers the robot to the lying pose before the release, so the release happens on the ground. Only the immediate exits (second Ctrl+C, tilt, no `LowState`) still release from standing, and those drop the robot softly by design.

We still saw a drop at the very end on the robot, and the simulator reproduces it. Two causes are likely, and we did not separate them:

- **The release takes the legs' grip in one tick.** At the lying pose the legs still carry the body, about 4.9 cm above where it rests when the legs are limp. `kp` going to zero lets it fall that far.
- **The damping steps too.** `kd` goes from the trial's value to `--release-kd` in the same tick, which lets the fall run faster.

`--release-ramp` addresses both. If you see a drop, try a longer ramp first.
