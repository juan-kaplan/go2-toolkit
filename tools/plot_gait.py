"""Draw what the generator commands, by calling the generator.

    uv run python tools/plot_gait.py --out gait.png

Every curve here is produced by `go2_toolkit.control` itself, so the picture
cannot drift from the code the way a hand drawn diagram does. It exists because
the properties that matter in this module are geometric and are quicker to
check by eye than by reading a tolerance: whether the swing curve leaves and
returns to the stance rate, whether clearance is the value that was asked for,
whether a phase table keeps three feet down, and whether the legs actually
reach the path the IK was handed.

Panel (c) is the one to look at after any change to the swing curve. If the
purple trace does not start and end exactly on the dashed line, touchdown is no
longer at zero world frame velocity and the foot is landing with a slam.

Needs matplotlib, which the control path does not, so this is a tool and not
part of the package.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from go2_toolkit import control
from go2_toolkit.control import GaitParams

if TYPE_CHECKING:
    from matplotlib.axes import Axes

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

LEG_COLOURS = {"FR": "#d1495b", "FL": "#2e86ab", "RR": "#e0a458", "RL": "#57996b"}
LEGS = ("FR", "FL", "RR", "RL")


def _panel_path(
    ax: Axes, params: GaitParams, phase: np.ndarray, x: np.ndarray, z: np.ndarray
) -> None:
    """(a) the closed foot path, with the control polygon that shapes it"""
    duty = params.stance_duration
    px, pz = control.swing_control_points(params)
    swing = phase >= duty
    ax.plot(x[~swing], z[~swing], lw=4, color="#333", label="stance (foot planted)")
    ax.plot(x[swing], z[swing], lw=3, color="#2e86ab", label="swing (quintic Bezier)")
    ax.plot(
        px, pz, "o--", color="#d1495b", ms=8, lw=1.2, alpha=0.75, label="control points"
    )
    for k, (a, b) in enumerate(zip(px, pz, strict=True)):
        ax.annotate(
            f"P{k}",
            (a, b),
            textcoords="offset points",
            xytext=(6, 7),
            color="#d1495b",
            fontsize=9,
            weight="bold",
        )
    ax.axhline(0, color="#999", lw=0.8, ls=":")
    ax.axhline(params.swing_height, color="#2e86ab", lw=0.8, ls=":")
    ax.set_title("(a) one foot's path, body frame", weight="bold", loc="left")
    ax.set_xlabel("forward  x [m]")
    ax.set_ylabel("up  z [m], 0 = ground")
    ax.legend(fontsize=8.5, loc="lower center")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)


def _panel_against_phase(
    ax: Axes, params: GaitParams, phase: np.ndarray, x: np.ndarray, z: np.ndarray
) -> None:
    """(b) the same path resolved against phase"""
    duty = params.stance_duration
    ax.plot(phase, x, color="#333", lw=2, label="x  forward")
    ax.plot(phase, z, color="#2e86ab", lw=2, label="z  up")
    ax.axvspan(0, duty, color="#333", alpha=0.07)
    ax.axvline(duty, color="#d1495b", lw=1.3, ls="--")
    span = max(abs(x).max(), z.max()) * 1.35
    ax.set_ylim(-span, span)
    ax.text(
        duty / 2,
        span * 0.86,
        "STANCE  linear sweep",
        ha="center",
        fontsize=9,
        color="#333",
    )
    ax.text(
        (duty + 1) / 2, span * 0.86, "SWING", ha="center", fontsize=9, color="#2e86ab"
    )
    ax.set_title("(b) the same path against phase", weight="bold", loc="left")
    ax.set_xlabel("phase  [0, 1)")
    ax.set_ylabel("[m]")
    ax.legend(fontsize=8.5, loc="lower left")
    ax.grid(alpha=0.25)


def _panel_touchdown(
    ax: Axes, params: GaitParams, phase: np.ndarray, x: np.ndarray
) -> None:
    """(c) the C1 join. This is the panel that catches a broken swing curve."""
    duty = params.stance_duration
    stance_rate = -params.step_length / duty
    dx = np.gradient(x, phase)
    ax.plot(phase[2:-2], dx[2:-2], color="#6a4c93", lw=2)
    ax.axhline(
        stance_rate,
        color="#333",
        ls="--",
        lw=1.2,
        label=f"stance rate  -L/b = {stance_rate:.3f}",
    )
    ax.axvline(duty, color="#d1495b", lw=1.3, ls="--")
    ax.axvspan(0, duty, color="#333", alpha=0.07)
    lo, hi = dx.min(), dx.max()
    ax.set_ylim(lo - 0.45 * (hi - lo), hi + 0.22 * (hi - lo))
    ax.annotate(
        "swing BEGINS and ENDS at the stance rate:\n"
        "the foot lands with zero world-frame velocity",
        xy=(0.998, stance_rate),
        xytext=(0.19, lo - 0.38 * (hi - lo)),
        fontsize=8.5,
        color="#6a4c93",
        arrowprops=dict(
            arrowstyle="->", color="#6a4c93", lw=1.2, connectionstyle="arc3,rad=-0.25"
        ),
    )
    ax.set_title("(c) why touchdown is soft", weight="bold", loc="left")
    ax.set_xlabel("phase")
    ax.set_ylabel("dx/dphase  [m/cycle]")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.25)


def _panel_stance_bars(ax: Axes, params: GaitParams, phase: np.ndarray) -> None:
    """(d) every phase table in GAITS, as stance bars"""
    duty = params.stance_duration
    row, ticks, labels = 0, [], []
    for name in sorted(control.GAITS):
        for i, leg in enumerate(LEGS):
            for s, e in control.stance_spans(control.GAITS[name][i], duty, phase):
                ax.barh(row, e - s, left=s, height=0.62, color=LEG_COLOURS[leg])
            ticks.append(row)
            labels.append(leg)
            row -= 1
        row -= 0.9
        ax.text(1.02, row + 3.4, name, fontsize=10, weight="bold", va="center")
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("phase")
    ax.set_xlim(0, 1)
    ax.set_title(f"(d) stance bars, duty {duty:g}", weight="bold", loc="left")


def _panel_support_count(ax: Axes, params: GaitParams, phase: np.ndarray) -> None:
    """(e) support polygon count: the static stability claim, checked"""
    duty = params.stance_duration
    floors = {}
    for name, colour, style in [("walk", "#57996b", "-"), ("trot", "#d1495b", "--")]:
        n = (np.mod(control.GAITS[name][None, :] + phase[:, None], 1.0) < duty).sum(1)
        floors[name] = int(n.min())
        ax.plot(
            phase,
            n,
            color=colour,
            lw=4.0 if name == "walk" else 2.0,
            ls=style,
            zorder=3 if name == "walk" else 4,
            label=f"{name}  (min {n.min()} feet down)",
        )
    ax.axhline(3, color="#333", ls=":", lw=1.4, zorder=1)
    ax.text(0.02, 3.06, "3 feet = static stability floor", fontsize=8.5, color="#333")
    ax.set_ylim(0.5, 4.5)
    ax.set_yticks([1, 2, 3, 4])
    ax.set_xlabel("phase")
    ax.set_ylabel("feet on the ground")
    ax.set_title(
        "(e) the walk never needs to balance"
        if floors["walk"] >= 3
        else f"(e) at duty {duty:g} even the walk must balance",
        weight="bold",
        loc="left",
    )
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.25)


def _panel_leg_through_cycle(
    ax: Axes, params: GaitParams, x: np.ndarray, z: np.ndarray, gait_name: str
) -> None:
    """(f) the IK solving the real leg, drawn in the sagittal plane"""
    duty = params.stance_duration
    offsets = control.GAITS[gait_name]
    for frac in np.linspace(0, 1, 13, endpoint=False):
        q = control.joint_trajectory(offsets + frac, params)
        # FR leg, projected into the sagittal plane and referred to its own hip
        chain = control.link_positions(q)[0]
        xs = chain[:, 0] - chain[0, 0]
        zs = chain[:, 2] - chain[0, 2]
        airborne = not control.phase.is_stance(offsets[0] + frac, duty)
        ax.plot(
            xs,
            zs,
            "-o",
            color="#2e86ab" if airborne else "#bbb",
            ms=3.4,
            lw=2.2 if airborne else 1.1,
            alpha=0.95 if airborne else 0.6,
            zorder=3 if airborne else 1,
        )
    ax.plot(
        x,
        z - params.nominal_height,
        color="#d1495b",
        lw=2.2,
        zorder=4,
        label="commanded foot path",
    )
    ax.plot(0, 0, "ks", ms=9, zorder=5)
    ax.text(0.012, -0.004, "hip", fontsize=9)
    ax.axhline(-params.nominal_height, color="#999", ls=":", lw=0.9)
    ax.set_title(f"(f) FR leg through one {gait_name} cycle", weight="bold", loc="left")
    ax.set_xlabel("forward  x [m]")
    ax.set_ylabel("z from hip [m]")
    ax.set_aspect("equal")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.25)


def figure(params: GaitParams, gait_name: str = "walk"):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    phase = np.linspace(0.0, 1.0, 2001, endpoint=False)
    off = control.foot_offsets(phase, params)
    x, z = off[:, 0], off[:, 1]

    fig = plt.figure(figsize=(15.5, 9.2))
    gs = GridSpec(2, 3, figure=fig, hspace=0.34, wspace=0.26)

    _panel_path(fig.add_subplot(gs[0, 0]), params, phase, x, z)
    _panel_against_phase(fig.add_subplot(gs[0, 1]), params, phase, x, z)
    _panel_touchdown(fig.add_subplot(gs[0, 2]), params, phase, x)
    _panel_stance_bars(fig.add_subplot(gs[1, 0]), params, phase)
    _panel_support_count(fig.add_subplot(gs[1, 1]), params, phase)
    _panel_leg_through_cycle(fig.add_subplot(gs[1, 2]), params, x, z, gait_name)

    duty = params.stance_duration
    fig.suptitle(
        f"control: stride {params.step_length:g} m, clearance "
        f"{params.swing_height:g} m, duty {duty:g}, "
        f"{params.cycle_period:g} s/cycle -> {params.body_speed:.3f} m/s",
        fontsize=13,
        weight="bold",
        y=0.985,
    )
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "gait.png")
    p.add_argument("--gait", default="walk", choices=sorted(control.GAITS))
    p.add_argument("--step-length", type=float, default=control.DEFAULT_STEP_LENGTH)
    p.add_argument("--swing-height", type=float, default=control.DEFAULT_SWING_HEIGHT)
    p.add_argument(
        "--duty",
        type=float,
        default=None,
        help="stance fraction (default: canonical for the gait)",
    )
    p.add_argument("--height", type=float, default=control.DEFAULT_HEIGHT)
    p.add_argument(
        "--period",
        type=float,
        default=None,
        help="seconds per cycle (default: canonical for the gait)",
    )
    p.add_argument("--dpi", type=int, default=135)
    a = p.parse_args()

    import matplotlib

    matplotlib.use("Agg")

    params = GaitParams(
        step_length=a.step_length,
        swing_height=a.swing_height,
        stance_duration=a.duty if a.duty is not None else control.DEFAULT_DUTY[a.gait],
        nominal_height=a.height,
        cycle_period=a.period if a.period is not None else control.DEFAULT_PERIOD[a.gait],
    )
    fig = figure(params, a.gait)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight", facecolor="white")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
