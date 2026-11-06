"""Generate the tutorial figures.

Run from the repository root:  python docs/make_tutorial_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "tutorial_assets"
OUT.mkdir(exist_ok=True)

BLUE, ORANGE, GREEN, GRAY, RED = "#2b6cb0", "#dd6b20", "#2f855a", "#718096", "#c53030"
NL = chr(10)


def box(ax, x, y, w, h, text, color, fs=9, fc=None):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
            fc=fc or "white", ec=color, lw=1.8,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#1a202c")


def arrow(ax, p, q, color=GRAY, ls="-", lw=1.6):
    ax.add_patch(
        FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, color=color, lw=lw, ls=ls)
    )


def route(ax, points, color, lw=2.0):
    for p, q in zip(points[:-2], points[1:-1]):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw)
    arrow(ax, points[-2], points[-1], color, lw=lw)


def setup_figure():
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.3, 6.2)
    ax.axis("off")

    box(ax, 4.3, 5.4, 3.4, 0.6, "10 MHz reference (shared by all)", GRAY, fc="#edf2f7")
    for x, src in ((1.75, 5.0), (6.0, 6.0), (11.17, 7.0)):
        arrow(ax, (src, 5.4), (x, 4.7), GRAY, ls="--")

    # AWG
    box(ax, 0.3, 2.2, 2.9, 2.5, "", BLUE)
    ax.text(1.75, 4.45, "AWG5208", ha="center", fontsize=10, weight="bold", color=BLUE)
    ax.text(1.75, 4.0, "ch2  qubit drive (IF)", ha="center", fontsize=8.5)
    ax.text(1.75, 3.5, "ch3  reference tone (IF)", ha="center", fontsize=8.5, color=ORANGE)
    ax.text(1.75, 3.0, "ch1  marker M1 (trigger)", ha="center", fontsize=8.5, color=RED)
    ax.text(1.75, 2.5, "same IF for drive, ref, emission", ha="center", fontsize=7.5, color=GRAY)

    # LO and splitter
    box(ax, 4.9, 4.0, 2.0, 0.7, "SGS100A LO" + NL + "(locked to 10 MHz)", GRAY, fs=8.5, fc="#edf2f7")
    box(ax, 5.55, 3.4, 0.7, 0.4, "split", GRAY, fs=8)
    arrow(ax, (5.9, 4.0), (5.9, 3.8), GRAY)

    # up-conversion
    box(ax, 3.7, 2.75, 1.0, 0.6, "mixer" + NL + "(up)", GRAY, fs=8.5)
    arrow(ax, (3.2, 4.0), (3.9, 3.35))
    arrow(ax, (5.55, 3.6), (4.7, 3.2), GRAY, ls="--")

    # fridge
    box(ax, 5.3, 0.95, 3.5, 2.0, "", BLUE, fc="#ebf4ff")
    ax.text(7.05, 2.7, "dilution fridge", ha="center", fontsize=9.5, weight="bold", color=BLUE)
    box(ax, 5.5, 1.2, 1.5, 0.85, "transmon" + NL + "+ waveguide", BLUE, fs=8.5)
    box(ax, 7.2, 1.2, 1.4, 0.85, "TWPA / HEMT" + NL + "(amplifier)", BLUE, fs=8.5)
    arrow(ax, (4.7, 3.0), (5.5, 1.9))
    arrow(ax, (7.0, 1.62), (7.2, 1.62))

    # down-conversion
    box(ax, 9.2, 1.95, 1.0, 0.6, "mixer" + NL + "(down)", GRAY, fs=8.5)
    arrow(ax, (8.6, 1.62), (9.2, 2.1))
    arrow(ax, (6.25, 3.6), (9.6, 2.55), GRAY, ls="--")
    ax.text(8.2, 3.35, "same LO", fontsize=8, color=GRAY, rotation=-13)
    box(ax, 9.2, 1.0, 1.0, 0.55, "LPF / amp", GRAY, fs=8.5)
    arrow(ax, (9.7, 1.95), (9.7, 1.55))

    # ATS
    box(ax, 10.45, 0.0, 1.45, 4.7, "", GREEN)
    ax.text(11.17, 4.45, "ATS9371", ha="center", fontsize=10, weight="bold", color=GREEN)
    ax.text(11.17, 3.5, "CHA" + NL + "signal", ha="center", fontsize=9)
    ax.text(11.17, 2.1, "CHB" + NL + "reference", ha="center", fontsize=9, color=ORANGE)
    ax.text(11.17, 0.3, "TRIG IN", ha="center", fontsize=9, color=RED)
    arrow(ax, (10.2, 1.3), (10.45, 3.4), GREEN)

    # reference tone and marker paths
    route(ax, [(3.2, 3.5), (3.55, 3.5), (3.55, 0.65), (10.32, 0.65), (10.32, 2.0), (10.45, 2.0)], ORANGE)
    ax.text(7.0, 0.42, "reference tone straight to CHB (no mixer, no fridge)",
            fontsize=8.5, color=ORANGE, ha="center")
    route(ax, [(3.2, 3.0), (3.35, 3.0), (3.35, 0.15), (10.45, 0.15)], RED, lw=1.6)
    ax.text(7.0, -0.12, "marker M1 -> TRIG IN", fontsize=8.5, color=RED, ha="center")

    ax.set_title(
        "Recommended waveguide-QED setup (drive out, emission in, CHB phase reference)",
        fontsize=11.5, weight="bold",
    )
    fig.savefig(OUT / "waveguide_setup.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def jitter_figure():
    rng = np.random.default_rng(0)
    n = 3000
    sample_deg = 360 * 50e6 * 1e-9
    shift = rng.uniform(-0.5, 0.5, n) * sample_deg  # measured: uniform +/- 1/2 sample
    alpha = 3.2
    noise = rng.normal(0, 0.03, (2, n))
    raw = alpha * np.exp(1j * np.radians(shift)) + noise[0] + 1j * noise[1]
    fixed = alpha * np.exp(1j * np.radians(rng.normal(0, 0.15, n))) + noise[0] + 1j * noise[1]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for axis, data, title in (
        (axes[0], raw, "without CHB reference" + NL + "(phase std 5.2 deg)"),
        (axes[1], fixed, "with CHB reference" + NL + "(limited by noise)"),
    ):
        axis.scatter(data.real, data.imag, s=3, alpha=0.4, color=BLUE)
        axis.set_aspect("equal")
        axis.set_xlim(2.4, 3.6)
        axis.set_ylim(-0.6, 0.6)
        axis.set_xlabel("I (mV)")
        axis.set_ylabel("Q (mV)")
        axis.set_title(title, fontsize=10)
        axis.grid(alpha=0.25)
    fig.suptitle("Sub-sample trigger jitter at 50 MHz IF (loopback, illustrative)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "reference_correction.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    setup_figure()
    jitter_figure()
    print("wrote", OUT)
