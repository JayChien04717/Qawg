"""Compare legacy boxcar DSP with FIR traces and QM-style full demodulation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from QAWG import AWGAlazar, ExperimentProgram, LinearSweep, MHz, ns, us
from QAWG.alazar import (
    downconvert,
    fir_filter,
    integrate,
)


AWG_RESOURCE = "TCPIP0::192.168.10.171::inst0::INSTR"
OUTPUT_DIR = Path(__file__).resolve().parent / "qm_demod_comparison"
PULSE_LENGTHS_NS = np.arange(100.0, 601.0, 100.0)
SAMPLE_RATE_HZ = 1e9
IF_FREQUENCY_HZ = 50e6
BOXCAR_SAMPLES = 20
FIR_CUTOFF_HZ = 20e6
FIR_TAPS = 81
FIR_BETA = 8.0
N_AVERAGE = 1000


class LengthSweepProgram(ExperimentProgram):
    def _initialize(self, cfg):
        self.declare_gen(
            "loopback",
            ch=cfg["awg_ch"],
            amplitude_vpp=cfg["channel_amplitude_vpp"],
        )
        self.declare_readout(
            "ro",
            adc_channel=cfg["adc_channel"],
            length=cfg["acquire_length"],
            demod_freq=cfg["frequency"],
            waveform_ch=cfg["awg_ch"],
            marker_channel=cfg["marker_ch"],
            marker_length=40 * ns,
            marker_lead=cfg["marker_lead"],
            integrate_time=cfg["integrate_time"],
        )
        pulse_length = self.add_sweep(
            "pulse_length",
            LinearSweep(
                cfg["length_start"],
                cfg["length_stop"],
                cfg["length_steps"],
            ),
        )
        self.add_pulse(
            "length_sweep_pulse",
            gen="loopback",
            style="gaussian_square",
            length=pulse_length,
            edge_sigma=cfg["edge_sigma"],
            frequency=cfg["frequency"],
            gain=cfg["pulse_gain"],
            readout=True,
        )

    def _body(self, cfg):
        self.play("length_sweep_pulse", at=0)
        self.trigger("ro", trigger_delay=0)


def legacy_centered_boxcar(baseband: np.ndarray, samples: int) -> np.ndarray:
    """Exact pre-FIR AWGAlazar centered boxcar implementation."""
    left = (samples - 1) // 2
    right = samples // 2
    padded = np.pad(baseband, ((0, 0), (left, right)), mode="edge")
    cumulative = np.pad(np.cumsum(padded, axis=1), ((0, 0), (1, 0)))
    return (cumulative[:, samples:] - cumulative[:, :-samples]) / samples


def crossing_time_ns(
    trace: np.ndarray,
    fraction: float,
    *,
    rising: bool,
    search_start: int = 140,
) -> float:
    peak = float(np.max(trace))
    threshold = peak * fraction
    peak_index = int(np.argmax(trace))
    if rising:
        candidates = np.flatnonzero(
            trace[search_start : peak_index + 1] >= threshold
        )
        return (
            float(search_start + candidates[0])
            if candidates.size
            else float("nan")
        )
    candidates = np.flatnonzero(trace[peak_index:] <= threshold)
    return float(peak_index + candidates[0]) if candidates.size else float("nan")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = {
        "awg_ch": 1,
        "marker_ch": 1,
        "adc_channel": "CHB",
        "channel_amplitude_vpp": 0.5,
        "frequency": IF_FREQUENCY_HZ,
        "marker_lead": 100 * ns,
        "edge_sigma": 15 * ns,
        "pulse_gain": 1.0,
        "acquire_length": 1.5 * us,
        "integrate_time": 1.0 * us,
        "length_start": 100 * ns,
        "length_stop": 600 * ns,
        "length_steps": 6,
    }
    experiment = AWGAlazar.connect(
        AWG_RESOURCE,
        awg_sample_rate_hz=2.5e9,
        alazar_sample_rate_hz=SAMPLE_RATE_HZ,
        acquire_window_s=1.5 * us,
        trigger_slope="rising",
        trigger_level=140,
        mask_external_trigger_lsb=True,
        fir_cutoff_hz=FIR_CUTOFF_HZ,
        fir_number_of_taps=FIR_TAPS,
        fir_kaiser_beta=FIR_BETA,
    )
    try:
        print("AWG clock source:", experiment.awg.query("CLOCk:SOURce?").strip())
        program = LengthSweepProgram(cfg, final_delay_s=1 * us)
        program.REMOVE_DC_OFFSET = False
        result = program.compile(hardware=experiment).acquire(n_average=N_AVERAGE)
    finally:
        experiment.close()

    raw = result.raw
    shot_count, step_count, sample_count = raw.shape
    flat_raw = raw.reshape(shot_count * step_count, sample_count)
    baseband = downconvert(
        flat_raw,
        SAMPLE_RATE_HZ,
        IF_FREQUENCY_HZ,
    )
    legacy = legacy_centered_boxcar(baseband, BOXCAR_SAMPLES)
    fir = fir_filter(
        baseband,
        SAMPLE_RATE_HZ,
        FIR_CUTOFF_HZ,
        FIR_TAPS,
        FIR_BETA,
    )
    legacy = legacy.reshape(shot_count, step_count, sample_count)
    fir = fir.reshape(shot_count, step_count, sample_count)

    integration_start = 0
    integration_stop = 1000
    legacy_shots = np.mean(
        legacy[:, :, integration_start:integration_stop], axis=2
    )
    qm_shots = integrate(
        baseband.reshape(shot_count, step_count, sample_count),
        integration_start,
        integration_stop,
    )
    legacy_average = np.mean(legacy, axis=0)
    fir_average = np.mean(fir, axis=0)
    raw_average = np.mean(raw, axis=0)
    time_ns = result.raw_time_s / ns

    summaries = []
    for index, length_ns in enumerate(PULSE_LENGTHS_NS):
        legacy_envelope = np.abs(legacy_average[index])
        fir_envelope = np.abs(fir_average[index])
        legacy_rise = (
            crossing_time_ns(legacy_envelope, 0.9, rising=True)
            - crossing_time_ns(legacy_envelope, 0.1, rising=True)
        )
        fir_rise = (
            crossing_time_ns(fir_envelope, 0.9, rising=True)
            - crossing_time_ns(fir_envelope, 0.1, rising=True)
        )
        row = {
            "pulse_length_ns": float(length_ns),
            "legacy_peak_mV": float(np.max(legacy_envelope) * 1e3),
            "fir_peak_mV": float(np.max(fir_envelope) * 1e3),
            "legacy_rise_10_90_ns": legacy_rise,
            "fir_rise_10_90_ns": fir_rise,
            "legacy_final_abs_mean_mV": float(np.abs(np.mean(legacy_shots[:, index])) * 1e3),
            "qm_final_abs_mean_mV": float(np.abs(np.mean(qm_shots[:, index])) * 1e3),
            "final_difference_uV": float(
                np.abs(np.mean(qm_shots[:, index]) - np.mean(legacy_shots[:, index]))
                * 1e6
            ),
        }
        summaries.append(row)
        print(
            f"length={length_ns:3.0f} ns: "
            f"rise legacy/FIR={legacy_rise:.1f}/{fir_rise:.1f} ns, "
            f"final legacy/QM={row['legacy_final_abs_mean_mV']:.5f}/"
            f"{row['qm_final_abs_mean_mV']:.5f} mV"
        )

    np.savez_compressed(
        OUTPUT_DIR / "qm_demod_comparison_data.npz",
        raw_volts=raw,
        raw_time_s=result.raw_time_s,
        pulse_lengths_ns=PULSE_LENGTHS_NS,
        legacy_boxcar_iq_average_volts=legacy_average,
        fir_iq_average_volts=fir_average,
        legacy_final_iq_shots_volts=legacy_shots,
        qm_final_iq_shots_volts=qm_shots,
        fir_coefficients=coefficients,
    )
    with (OUTPUT_DIR / "qm_demod_comparison_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (OUTPUT_DIR / "qm_demod_comparison_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    plot_comparison(
        time_ns,
        raw_average,
        legacy_average,
        fir_average,
        legacy_shots,
        qm_shots,
    )
    plot_heatmaps(time_ns, legacy_average, fir_average)


def plot_comparison(
    time_ns: np.ndarray,
    raw_average: np.ndarray,
    legacy_average: np.ndarray,
    fir_average: np.ndarray,
    legacy_shots: np.ndarray,
    qm_shots: np.ndarray,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for index, length_ns in enumerate(PULSE_LENGTHS_NS):
        axes[0, 0].plot(
            time_ns,
            np.abs(legacy_average[index]) * 1e3,
            label=f"{length_ns:.0f} ns",
        )
        axes[0, 1].plot(
            time_ns,
            np.abs(fir_average[index]) * 1e3,
            label=f"{length_ns:.0f} ns",
        )
    axes[0, 0].set_title("Legacy: 20 ns centered boxcar")
    axes[0, 1].set_title("New acquire_decimate: 81-tap, 20 MHz FIR")
    for axis in axes[0]:
        axis.set_xlim(140, 850)
        axis.set_xlabel("Time after ATS trigger (ns)")
        axis.set_ylabel("|IQ| (mV)")
        axis.grid(True, alpha=0.25)
        axis.legend(title="AWG length", ncol=2)

    short_index = 0
    axes[1, 0].plot(
        time_ns,
        np.abs(legacy_average[short_index]) * 1e3,
        label="legacy boxcar",
        linewidth=2,
    )
    axes[1, 0].plot(
        time_ns,
        np.abs(fir_average[short_index]) * 1e3,
        label="FIR",
        linewidth=2,
    )
    axes[1, 0].set_xlim(160, 360)
    axes[1, 0].set_title("100 ns pulse: edge detail")
    axes[1, 0].set_xlabel("Time after ATS trigger (ns)")
    axes[1, 0].set_ylabel("|IQ| (mV)")
    axes[1, 0].grid(True, alpha=0.25)
    axes[1, 0].legend()

    legacy_mean = np.abs(np.mean(legacy_shots, axis=0)) * 1e3
    qm_mean = np.abs(np.mean(qm_shots, axis=0)) * 1e3
    legacy_sem = np.std(np.abs(legacy_shots), axis=0, ddof=1) / np.sqrt(legacy_shots.shape[0]) * 1e3
    qm_sem = np.std(np.abs(qm_shots), axis=0, ddof=1) / np.sqrt(qm_shots.shape[0]) * 1e3
    axes[1, 1].errorbar(PULSE_LENGTHS_NS, legacy_mean, yerr=legacy_sem, marker="o", label="legacy: mean(boxcar trace)")
    axes[1, 1].errorbar(PULSE_LENGTHS_NS, qm_mean, yerr=qm_sem, marker="s", label="QM full: weighted raw dot product")
    axes[1, 1].set_title("Final IQ shot, identical 0..1000 ns weights")
    axes[1, 1].set_xlabel("AWG pulse length (ns)")
    axes[1, 1].set_ylabel("|mean IQ| (mV)")
    axes[1, 1].grid(True, alpha=0.25)
    axes[1, 1].legend()
    figure.suptitle("Same ATS raw acquisition: legacy versus QM-style DSP")
    figure.savefig(OUTPUT_DIR / "qm_demod_comparison.png", dpi=180)
    plt.close(figure)


def plot_heatmaps(
    time_ns: np.ndarray,
    legacy_average: np.ndarray,
    fir_average: np.ndarray,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    extent = [time_ns[0], time_ns[-1], PULSE_LENGTHS_NS[0], PULSE_LENGTHS_NS[-1]]
    legacy_mv = np.abs(legacy_average) * 1e3
    fir_mv = np.abs(fir_average) * 1e3
    limit = max(float(np.max(legacy_mv)), float(np.max(fir_mv)))
    first = axes[0].imshow(legacy_mv, aspect="auto", origin="lower", extent=extent, vmin=0, vmax=limit)
    axes[0].set_title("Legacy 20 ns boxcar")
    axes[1].imshow(fir_mv, aspect="auto", origin="lower", extent=extent, vmin=0, vmax=limit)
    axes[1].set_title("FIR LPF")
    difference = (fir_mv - legacy_mv) * 1e3
    diff_limit = float(np.max(np.abs(difference)))
    third = axes[2].imshow(difference, aspect="auto", origin="lower", extent=extent, cmap="coolwarm", vmin=-diff_limit, vmax=diff_limit)
    axes[2].set_title("FIR - legacy (uV)")
    for axis in axes:
        axis.set_xlim(140, 850)
        axis.set_xlabel("Time after ATS trigger (ns)")
        axis.set_ylabel("AWG pulse length (ns)")
    figure.colorbar(first, ax=axes[:2], label="|IQ| (mV)")
    figure.colorbar(third, ax=axes[2], label="Difference (uV)")
    figure.suptitle("Time-resolved demodulation comparison")
    figure.savefig(OUTPUT_DIR / "qm_demod_heatmap_comparison.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
