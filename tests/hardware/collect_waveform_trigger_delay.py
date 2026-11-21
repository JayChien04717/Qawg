"""Acquire an amplified readout and visualize integration trigger delays."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from QAWG import AWGAlazar, ns, us
from hardware_marker_validation import (
    AWG_RESOURCE,
    BASE_CFG,
    MarkerValidationProgram,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "trigger_delay_background"
TRIGGER_DELAYS_NS = (0.0, 100.0, 200.0, 400.0)
INTEGRATION_TIME_NS = 100.0
MARKER_LENGTH_NS = 40.0
N_AVERAGE = 1000
WAVEFORM_GAIN = 1.0


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    experiment = AWGAlazar.connect(
        AWG_RESOURCE,
        awg_sample_rate_hz=2.5e9,
        alazar_sample_rate_hz=1e9,
        acquire_window_s=1.5 * us,
        trigger_slope="rising",
        trigger_level=140,
        mask_external_trigger_lsb=True,
    )
    try:
        cfg = {
            **BASE_CFG,
            "marker_length": MARKER_LENGTH_NS * ns,
            "pulse_gain": WAVEFORM_GAIN,
            "trigger_delay": 0.0,
            "integrate_time": INTEGRATION_TIME_NS * ns,
        }
        program = MarkerValidationProgram(
            cfg,
            name="waveform_trigger_delay",
            final_delay_s=1 * us,
        )
        program.REMOVE_DC_OFFSET = False
        result = program.compile(hardware=experiment).acquire(n_average=N_AVERAGE)
    finally:
        experiment.close()

    raw_records = result.raw[:, 0, :]
    iq_records = result.iq_traces[:, 0, :]
    raw_average = np.mean(raw_records, axis=0)
    iq_average = np.mean(iq_records, axis=0)
    time_ns = result.raw_time_s * 1e9
    iq_time_ns = result.iq_time_s * 1e9
    summaries = []
    iq_window_shots = []
    for delay_ns in TRIGGER_DELAYS_NS:
        start = int(round(delay_ns))
        stop = int(round(delay_ns + INTEGRATION_TIME_NS))
        raw_window = raw_records[:, start:stop]
        iq_shots = np.mean(iq_records[:, start:stop], axis=1)
        iq_window_shots.append(iq_shots)
        mean_iq = np.mean(iq_shots)
        summaries.append(
            {
                "trigger_delay_ns": delay_ns,
                "integration_time_ns": INTEGRATION_TIME_NS,
                "waveform_gain": WAVEFORM_GAIN,
                "marker_length_ns": MARKER_LENGTH_NS,
                "n_average": N_AVERAGE,
                "raw_window_mean_mV": float(np.mean(raw_window) * 1e3),
                "raw_window_rms_mV": float(np.sqrt(np.mean(raw_window**2)) * 1e3),
                "integrated_iq_magnitude_mV": float(np.abs(mean_iq) * 1e3),
                "integrated_iq_phase_rad": float(np.angle(mean_iq)),
                "remove_dc_offset": False,
                "external_trigger_lsb_masked": True,
            }
        )
        print(
            f"delay={delay_ns:5.0f} ns: "
            f"raw RMS={summaries[-1]['raw_window_rms_mV']:.6f} mV, "
            f"|IQ|={summaries[-1]['integrated_iq_magnitude_mV']:.6f} mV"
        )

    np.savez_compressed(
        OUTPUT_DIR / "waveform_trigger_delay_data.npz",
        time_ns=time_ns,
        iq_time_ns=iq_time_ns,
        trigger_delays_ns=np.asarray(TRIGGER_DELAYS_NS),
        integration_time_ns=INTEGRATION_TIME_NS,
        marker_length_ns=MARKER_LENGTH_NS,
        waveform_gain=WAVEFORM_GAIN,
        raw_average_volts=raw_average,
        iq_average_volts=iq_average,
        iq_window_shots_volts=np.stack(iq_window_shots),
    )
    write_summary(summaries)
    plot_raw_windows(time_ns, raw_average)
    plot_iq_windows(iq_time_ns, np.abs(iq_average))
    plot_iq_components(iq_time_ns, iq_average)
    plot_iq_constellations(np.stack(iq_window_shots))
    plot_integrated_iq(summaries)


def write_summary(summaries) -> None:
    with (OUTPUT_DIR / "waveform_trigger_delay_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (OUTPUT_DIR / "waveform_trigger_delay_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n",
        encoding="utf-8",
    )


def plot_raw_windows(time_ns: np.ndarray, raw_average: np.ndarray) -> None:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 7),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for axis, delay_ns in zip(axes.flat, TRIGGER_DELAYS_NS):
        axis.plot(time_ns, raw_average * 1e3, color="tab:blue", linewidth=1.0)
        add_windows(axis, delay_ns)
        axis.set_title(f"integration window = {delay_ns:.0f}..{delay_ns + INTEGRATION_TIME_NS:.0f} ns")
        axis.grid(True, alpha=0.25)
    figure.supxlabel("Time after received marker (ns)")
    figure.supylabel("Averaged raw ADC voltage (mV)")
    figure.suptitle("Readout waveform: raw carrier at different trigger delays")
    figure.savefig(
        OUTPUT_DIR / "waveform_raw_by_trigger_delay.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_iq_windows(time_ns: np.ndarray, iq_envelope: np.ndarray) -> None:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 7),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for axis, delay_ns in zip(axes.flat, TRIGGER_DELAYS_NS):
        axis.plot(time_ns, iq_envelope * 1e3, color="tab:blue", linewidth=1.2)
        add_windows(axis, delay_ns)
        axis.set_title(f"integration window = {delay_ns:.0f}..{delay_ns + INTEGRATION_TIME_NS:.0f} ns")
        axis.grid(True, alpha=0.25)
    figure.supxlabel("Time after received marker (ns)")
    figure.supylabel("Demodulated |IQ| (mV)")
    figure.suptitle("Readout envelope sampled at different trigger delays")
    figure.savefig(
        OUTPUT_DIR / "waveform_iq_by_trigger_delay.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_iq_components(time_ns: np.ndarray, iq_average: np.ndarray) -> None:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 7),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for panel_index, (axis, delay_ns) in enumerate(
        zip(axes.flat, TRIGGER_DELAYS_NS)
    ):
        axis.plot(
            time_ns,
            iq_average.real * 1e3,
            color="tab:blue",
            linewidth=1.1,
            label="I" if panel_index == 0 else None,
        )
        axis.plot(
            time_ns,
            iq_average.imag * 1e3,
            color="tab:orange",
            linewidth=1.1,
            label="Q" if panel_index == 0 else None,
        )
        axis.plot(
            time_ns,
            np.abs(iq_average) * 1e3,
            color="tab:green",
            linewidth=1.0,
            alpha=0.8,
            label="|IQ|" if panel_index == 0 else None,
        )
        add_windows(axis, delay_ns)
        axis.set_title(
            f"integration window = {delay_ns:.0f}.."
            f"{delay_ns + INTEGRATION_TIME_NS:.0f} ns"
        )
        axis.grid(True, alpha=0.25)
    axes[0, 0].legend(loc="best")
    figure.supxlabel("Time after received marker (ns)")
    figure.supylabel("Demodulated voltage (mV)")
    figure.suptitle("Demodulated I, Q, and |IQ| at different trigger delays")
    figure.savefig(
        OUTPUT_DIR / "waveform_iq_components_by_trigger_delay.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_iq_constellations(iq_window_shots: np.ndarray) -> None:
    values_mV = iq_window_shots * 1e3
    coordinates = np.concatenate(
        (values_mV.real.reshape(-1), values_mV.imag.reshape(-1))
    )
    lower, upper = np.percentile(coordinates, [0.5, 99.5])
    margin = max(0.05, 0.08 * (upper - lower))
    limits = (lower - margin, upper + margin)

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(9, 8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for axis, delay_ns, shots in zip(
        axes.flat,
        TRIGGER_DELAYS_NS,
        values_mV,
    ):
        axis.scatter(
            shots.real,
            shots.imag,
            s=8,
            alpha=0.25,
            color="tab:blue",
            edgecolors="none",
        )
        mean_iq = np.mean(shots)
        axis.plot(mean_iq.real, mean_iq.imag, "x", color="tab:red", markersize=9)
        axis.axhline(0, color="0.5", linewidth=0.8, alpha=0.5)
        axis.axvline(0, color="0.5", linewidth=0.8, alpha=0.5)
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"trigger_delay = {delay_ns:.0f} ns")
        axis.grid(True, alpha=0.2)
    figure.supxlabel("Integrated I (mV)")
    figure.supylabel("Integrated Q (mV)")
    figure.suptitle("Per-shot IQ constellation by trigger-delay window")
    figure.savefig(
        OUTPUT_DIR / "waveform_iq_constellation_by_trigger_delay.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def add_windows(axis, delay_ns: float) -> None:
    axis.axvspan(0, MARKER_LENGTH_NS, color="tab:orange", alpha=0.18)
    axis.axvspan(
        delay_ns,
        delay_ns + INTEGRATION_TIME_NS,
        facecolor="none",
        edgecolor="tab:green",
        linewidth=1.5,
        hatch="//",
    )


def plot_integrated_iq(summaries) -> None:
    delays = np.asarray([row["trigger_delay_ns"] for row in summaries])
    magnitudes = np.asarray(
        [row["integrated_iq_magnitude_mV"] for row in summaries]
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(delays, magnitudes, "o-", color="tab:blue", linewidth=1.5)
    for delay, magnitude in zip(delays, magnitudes):
        axis.annotate(
            f"{magnitude:.3f} mV",
            (delay, magnitude),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
        )
    axis.set_xlabel("Integration trigger_delay (ns)")
    axis.set_ylabel("Integrated |IQ| (mV)")
    axis.set_title("Readout amplitude captured by each trigger-delay window")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "waveform_integrated_iq_vs_trigger_delay.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
