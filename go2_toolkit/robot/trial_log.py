"""The file a trial leaves behind: what was run, and what happened, tick by tick.

A trial log is two files with the same stem in `results/trials/`:

    20261007_153012_trot_sim.npz     the arrays, one row per control tick
    20261007_153012_trot_sim.json    the metadata, readable without Python

The session records every command it sends (`TrialResult.record`, built by
`stack_records`), and `write_trial_log` puts that on disk after the trial is over,
never inside a stage. Recording is appending to lists; the stacking and all the I/O
happen after the release. `tools/plot_trial.py` opens a log.

**Arrays** (`n` ticks, from the first command of the first stage to the last of the
last, the release not included; every reading is the one taken just before its
command was sent):

    wall          (n,)     s since the first command
    q_cmd, dq_cmd, kp, kd, tau_ff    (n, 12)   the command, Unitree joint order
    q_meas, dq_meas, tau_est         (n, 12)   the reading
    quat          (n, 4)   body orientation, scalar first
    gyro, accel   (n, 3)   IMU, body frame
    pos, vel      (n, 3)   ground truth, world frame; NaN where there is none
    command_tick  (n,)     bool: the command function was called on this tick
    extra_<name>  (n, ...) a driver's own channels, NaN on ticks without a call

**Extra channels.** The command function is called once per tick from the loop's
first tick on, with `k` unbroken, so its `k`-th call is tick `first_tick + k`. A
driver keeps a list per channel and appends one value per call; it hands the lists
to `write_trial_log(channels=...)`, which lines them up with the ticks.
"""

from __future__ import annotations

import datetime
import json
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .dds import HARDWARE_DOMAIN_ID

if TYPE_CHECKING:
    from .session import TrialResult

TRIALS_DIR = Path(__file__).resolve().parents[2] / "results" / "trials"
SUPPORTS = ("lifted", "floor")  # what a hardware trial can say the robot stood on

_RECORD_ARRAYS = (
    "q_cmd dq_cmd kp kd tau_ff q_meas dq_meas tau_est quat gyro accel pos vel wall"
).split()


def source_version() -> tuple[str, bool]:
    """The commit this code is at, and whether the working tree differs from it.

    Run before the backend connects: a subprocess inside a trial is a gap in the
    command stream. ("unknown", True) when git cannot say.
    """
    here = Path(__file__).resolve().parent
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=here, capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()  # fmt: skip
        changed = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=here, capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return "unknown", True
    return head, bool(changed)


def condition(domain_id: int, support: str | None) -> dict[str, str | None]:
    """Where a trial ran and, on hardware, what the robot stood on."""
    if domain_id == HARDWARE_DOMAIN_ID:
        return {"where": "hardware", "support": support}
    return {"where": "simulator", "support": None}


def refuse_without_support(domain_id: int, support: str | None) -> None:
    """A hardware trial must say whether the legs were lifted or on the floor."""
    if support is not None and support not in SUPPORTS:
        raise SystemExit(f"support must be one of {SUPPORTS}, not {support!r}")
    if domain_id == HARDWARE_DOMAIN_ID and support is None:
        raise SystemExit(
            "A hardware trial must say what the robot stands on, because the log "
            "records it and a trial on the floor is not comparable with one in the "
            f"air. Pass one of {SUPPORTS} (run_generated_gait.py: --support)."
        )


def stack_records(
    commands: list, states: list, truths: list, walls: list[float]
) -> dict[str, np.ndarray]:
    """The recorded ticks as arrays. Commands pair with the states read before them."""
    nan3 = np.full(3, np.nan)
    return dict(
        q_cmd=np.array([c.q for c in commands]),
        dq_cmd=np.array([c.dq for c in commands]),
        kp=np.array([c.kp for c in commands]),
        kd=np.array([c.kd for c in commands]),
        tau_ff=np.array([c.tau_ff for c in commands]),
        q_meas=np.array([s.q for s in states]),
        dq_meas=np.array([s.dq for s in states]),
        tau_est=np.array([s.tau_est for s in states]),
        quat=np.array([s.imu_quat for s in states]),
        gyro=np.array([s.imu_gyro for s in states]),
        accel=np.array([s.imu_accel for s in states]),
        pos=np.array([nan3 if t is None else t.base_pos for t in truths]),
        vel=np.array([nan3 if t is None else t.base_vel for t in truths]),
        wall=np.array(walls),
    )


def _align(values, ticks: np.ndarray, n_ticks: int) -> np.ndarray:
    """One channel's per-call values placed on the ticks of the calls, NaN elsewhere."""
    v = np.asarray(values, dtype=float)
    out = np.full((n_ticks, *v.shape[1:]), np.nan)
    out[ticks[: len(v)]] = v
    return out


def write_trial_log(
    result: TrialResult,
    driver: str,
    label: str,
    config: object = None,
    channels: dict[str, list] | None = None,
    extra: dict | None = None,
    directory: Path | str = TRIALS_DIR,
) -> Path:
    """Write the log of a finished trial; returns the path of the `.npz`.

    `driver` is the script's name and `label` says what the trial was (`trot`); the
    stem is the time, the label and the condition, so logs sort by time. `config` is
    the driver's parameters (a dataclass or a dict), `channels` its per-call values
    (see the module docstring) and `extra` anything else worth keeping (a seed, a
    stretch): metadata only.
    """
    rec = result.record
    if rec is None:
        raise ValueError("this trial recorded nothing")
    n = len(rec["wall"])
    ticks = result.command_ticks
    called = np.zeros(n, dtype=bool)
    called[ticks] = True

    arrays = {k: rec[k] for k in _RECORD_ARRAYS}
    arrays["wall"] = arrays["wall"] - arrays["wall"][0]
    arrays["command_tick"] = called
    for name, values in (channels or {}).items():
        arrays[f"extra_{name}"] = _align(values, ticks, n)

    now = datetime.datetime.now().astimezone()
    cond = condition(result.setup.domain_id, result.setup.support)
    where = "sim" if cond["where"] == "simulator" else f"hw_{cond['support']}"
    stem = f"{now:%Y%m%d_%H%M%S}_{label}_{where}"
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}.npz"
    for i in range(2, 100):  # two trials in one second
        if not path.exists():
            break
        path = directory / f"{stem}_{i}.npz"

    meta = dict(
        driver=driver,
        label=label,
        date=now.isoformat(timespec="seconds"),
        commit=result.commit,
        dirty=result.dirty,
        condition=cond,
        config=_plain(config),
        setup=_plain(result.setup),
        rate=result.rate,
        ticks=n,
        stages=[{"stage": s, "first_tick": t} for s, t in result.stages],
        command_ticks=(
            {"first": int(ticks[0]), "count": len(ticks)} if len(ticks) else None
        ),
        aborted=result.aborted,
        overruns=result.overruns,
        timing=result.timing,
        arrays=sorted(arrays),
        extra=_plain(extra),
    )
    np.savez(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    return path


def _plain(obj):
    """A dataclass, dict, numpy value or None as something `json` accepts."""
    if is_dataclass(obj) and not isinstance(obj, type):
        obj = asdict(obj)
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def read_trial_log(path: Path | str) -> tuple[dict[str, np.ndarray], dict]:
    """The arrays and the metadata of a log, from the path of either file."""
    path = Path(path)
    with np.load(path.with_suffix(".npz")) as f:
        arrays = {k: f[k] for k in f.files}
    return arrays, json.loads(path.with_suffix(".json").read_text())


def replay_spikes(
    arrays: dict[str, np.ndarray], meta: dict, recording: Path | str, t: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The spikes of the recording a replay played, as (time, neuron) arrays.

    A spike's time is that of the tick whose network steps contain it, taken from `t`
    (the time of every tick), so the replay's stretch is already in the axis. Only the
    loop is covered: that is where the network advanced after the pre-roll.
    """
    called = np.flatnonzero(arrays["command_tick"])
    first = meta["extra"]["loop_first_step"]
    ends = first + np.cumsum(arrays["extra_steps"][called].astype(int))
    with np.load(recording) as f:
        steps, neurons = f["spike_steps"], f["spike_neurons"]
    inside = (steps >= first) & (steps < ends[-1])
    tick = np.searchsorted(ends, steps[inside], side="right")
    return t[called[tick]], neurons[inside]
