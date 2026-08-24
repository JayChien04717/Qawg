"""Acquire and plot zero-signal background versus QAWG trigger_delay."""

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

    traces: list[np.ndarray] = []
    shot_window_means: list[np.ndarray] = []
    summaries: list[dict[str, float | int | bool]] = []
    time_ns: np.ndarray | None = None
    shared_trace: np.ndarray | None = None
    shared_window_means: list[np.ndarray] = []
    shared_summaries: list[dict[str, float | int | bool]] = []
    try:
        for delay_ns in TRIGGER_DELAYS_NS:
            cfg = {
                **BASE_CFG,
                "marker_length": MARKER_LENGTH_NS * ns,
                "pulse_gain": 0.0,
                "trigger_delay": delay_ns * ns,
                "integrate_time": INTEGRATION_TIME_NS * ns,
            }
            program = MarkerValidationProgram(
                cfg,
                name=f"background_delay_{delay_ns:.0f}ns",
                final_delay_s=1 * us,
            )
            program.REMOVE_DC_OFFSET = False
            compiled = program.compile(hardware=experiment)
            result = compiled.acquire(n_average=N_AVERAGE)

            records = result.raw[:, 0, :]
            trace = np.mean(records, axis=0)
            start = int(round(delay_ns))
            stop = int(round(delay_ns + INTEGRATION_TIME_NS))
            window_means = np.mean(records[:, start:stop], axis=1)
            mean_volts = float(np.mean(window_means))
            std_volts = float(np.std(window_means, ddof=1))
            sem_volts = std_volts / np.sqrt(window_means.size)

            if time_ns is None:
                time_ns = result.raw_time_s * 1e9
            traces.append(trace)
            shot_window_means.append(window_means)
            summaries.append(
                {
                    "trigger_delay_ns": delay_ns,
                    "integration_time_ns": INTEGRATION_TIME_NS,
                    "marker_length_ns": MARKER_LENGTH_NS,
                    "n_average": N_AVERAGE,
                    "background_mean_mV": mean_volts * 1e3,
                    "background_shot_std_mV": std_volts * 1e3,
                    "background_sem_mV": sem_volts * 1e3,
                    "remove_dc_offset": False,
                    "external_trigger_lsb_masked": True,
                }
            )
            print(
                f"delay={delay_ns:5.0f} ns: "
                f"background={mean_volts * 1e3:+.6f} ± "
                f"{sem_volts * 1e3:.6f} mV (SEM)"
            )

        # Control: evaluate every delay window on the exact same raw records.
        # QAWG trigger_delay moves the integration slice, not the ATS record
        # start, so this separates within-record timing from run-to-run drift.
        shared_cfg = {
            **BASE_CFG,
            "marker_length": MARKER_LENGTH_NS * ns,
            "pulse_gain": 0.0,
            "trigger_delay": 0.0,
            "integrate_time": INTEGRATION_TIME_NS * ns,
        }
        shared_program = MarkerValidationProgram(
            shared_cfg,
            name="background_shared_acquisition",
            final_delay_s=1 * us,
        )
        shared_program.REMOVE_DC_OFFSET = False
        shared_result = shared_program.compile(hardware=experiment).acquire(
            n_average=N_AVERAGE
        )
        shared_records = shared_result.raw[:, 0, :]
        shared_trace = np.mean(shared_records, axis=0)
        for delay_ns in TRIGGER_DELAYS_NS:
            start = int(round(delay_ns))
            stop = int(round(delay_ns + INTEGRATION_TIME_NS))
            window_means = np.mean(shared_records[:, start:stop], axis=1)
            mean_volts = float(np.mean(window_means))
            std_volts = float(np.std(window_means, ddof=1))
            sem_volts = std_volts / np.sqrt(window_means.size)
            shared_window_means.append(window_means)
            shared_summaries.append(
                {
                    "trigger_delay_ns": delay_ns,
                    "integration_time_ns": INTEGRATION_TIME_NS,
                    "marker_length_ns": MARKER_LENGTH_NS,
                    "n_average": N_AVERAGE,
                    "background_mean_mV": mean_volts * 1e3,
                    "background_shot_std_mV": std_volts * 1e3,
                    "background_sem_mV": sem_volts * 1e3,
                    "shared_acquisition": True,
                    "remove_dc_offset": False,
                    "external_trigger_lsb_masked": True,
                }
            )
            print(
                f"shared delay={delay_ns:5.0f} ns: "
                f"background={mean_volts * 1e3:+.6f} ± "
                f"{sem_volts * 1e3:.6f} mV (SEM)"
            )
    finally:
        experiment.close()

    assert time_ns is not None
    assert shared_trace is not None
    trace_matrix = np.stack(traces)
    np.savez_compressed(
        OUTPUT_DIR / "trigger_delay_background_data.npz",
        time_ns=time_ns,
        trigger_delays_ns=np.asarray(TRIGGER_DELAYS_NS),
        integration_time_ns=INTEGRATION_TIME_NS,
        marker_length_ns=MARKER_LENGTH_NS,
        average_traces_volts=trace_matrix,
        shot_window_means_volts=np.stack(shot_window_means),
    )

    with (OUTPUT_DIR / "trigger_delay_background_summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    with (OUTPUT_DIR / "averaged_traces.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            ["time_ns"]
            + [f"trace_delay_{delay_ns:.0f}ns_mV" for delay_ns in TRIGGER_DELAYS_NS]
        )
        for sample_index, sample_time_ns in enumerate(time_ns):
            writer.writerow(
                [sample_time_ns]
                + [trace[sample_index] * 1e3 for trace in traces]
            )

    (OUTPUT_DIR / "trigger_delay_background_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n",
        encoding="utf-8",
    )

    plot_trace_comparison(time_ns, trace_matrix)
    plot_background_summary(summaries)

    np.savez_compressed(
        OUTPUT_DIR / "same_acquisition_trigger_delay_data.npz",
        time_ns=time_ns,
        trigger_delays_ns=np.asarray(TRIGGER_DELAYS_NS),
        integration_time_ns=INTEGRATION_TIME_NS,
        marker_length_ns=MARKER_LENGTH_NS,
        average_trace_volts=shared_trace,
        shot_window_means_volts=np.stack(shared_window_means),
    )
    write_summary_files(
        "same_acquisition_trigger_delay_summary",
        shared_summaries,
    )
    plot_shared_trace_windows(time_ns, shared_trace)
    plot_shared_background_summary(shared_summaries)


def write_summary_files(stem: str, summaries) -> None:
    with (OUTPUT_DIR / f"{stem}.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (OUTPUT_DIR / f"{stem}.json").write_text(
        json.dumps(summaries, indent=2) + "\n",
        encoding="utf-8",
    )


def plot_trace_comparison(time_ns: np.ndarray, traces: np.ndarray) -> None:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 7),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for axis, delay_ns, trace in zip(axes.flat, TRIGGER_DELAYS_NS, traces):
        axis.plot(time_ns, trace * 1e3, color="tab:blue", linewidth=1.0)
        axis.axvspan(
            0,
            MARKER_LENGTH_NS,
            color="tab:orange",
            alpha=0.18,
            label="Marker high",
        )
        axis.axvspan(
            delay_ns,
            delay_ns + INTEGRATION_TIME_NS,
            facecolor="none",
            edgecolor="tab:green",
            linewidth=1.5,
            hatch="//",
            label="Background window",
        )
        axis.set_title(f"trigger_delay = {delay_ns:.0f} ns")
        axis.grid(True, alpha=0.25)
    axes[0, 0].legend(loc="best")
    figure.supxlabel("Time after received marker (ns)")
    figure.supylabel("Averaged ADC background (mV)")
    figure.suptitle(
        "Independent ATS937x acquisitions at each integration trigger_delay\n"
        "Includes run-to-run ADC offset drift"
    )
    figure.savefig(
        OUTPUT_DIR / "background_traces_by_trigger_delay.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_background_summary(summaries) -> None:
    delays = np.asarray([row["trigger_delay_ns"] for row in summaries])
    means = np.asarray([row["background_mean_mV"] for row in summaries])
    sems = np.asarray([row["background_sem_mV"] for row in summaries])

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.errorbar(
        delays,
        means,
        yerr=sems,
        fmt="o-",
        color="tab:blue",
        capsize=4,
        linewidth=1.5,
    )
    axis.set_xlabel("Integration trigger_delay (ns)")
    axis.set_ylabel("Background voltage (mV)")
    axis.set_title(
        "Independent acquisitions: background voltage vs trigger delay\n"
        "Point differences include run-to-run ADC offset drift"
    )
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "background_voltage_vs_trigger_delay.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_shared_trace_windows(time_ns: np.ndarray, trace: np.ndarray) -> None:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 7),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for axis, delay_ns in zip(axes.flat, TRIGGER_DELAYS_NS):
        axis.plot(time_ns, trace * 1e3, color="tab:blue", linewidth=1.0)
        axis.axvspan(0, MARKER_LENGTH_NS, color="tab:orange", alpha=0.18)
        axis.axvspan(
            delay_ns,
            delay_ns + INTEGRATION_TIME_NS,
            facecolor="none",
            edgecolor="tab:green",
            linewidth=1.5,
            hatch="//",
        )
        axis.set_title(
            f"integration window = {delay_ns:.0f}.."
            f"{delay_ns + INTEGRATION_TIME_NS:.0f} ns"
        )
        axis.grid(True, alpha=0.25)
    figure.supxlabel("Time after received marker (ns)")
    figure.supylabel("Averaged ADC background (mV)")
    figure.suptitle(
        "One ATS937x acquisition with four trigger-delay analysis windows\n"
        "40 ns marker, external-trigger LSB masked, zero readout gain"
    )
    figure.savefig(
        OUTPUT_DIR / "same_acquisition_windows_by_trigger_delay.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_shared_background_summary(summaries) -> None:
    delays = np.asarray([row["trigger_delay_ns"] for row in summaries])
    means = np.asarray([row["background_mean_mV"] for row in summaries])
    sems = np.asarray([row["background_sem_mV"] for row in summaries])
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.errorbar(
        delays,
        means,
        yerr=sems,
        fmt="o-",
        color="tab:blue",
        capsize=4,
        linewidth=1.5,
    )
    axis.set_xlabel("Integration trigger_delay (ns)")
    axis.set_ylabel("Background voltage (mV)")
    axis.set_title("Same acquisition: background voltage vs trigger delay")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "same_acquisition_background_vs_trigger_delay.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
