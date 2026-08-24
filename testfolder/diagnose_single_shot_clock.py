"""Diagnose phase/timing jitter in the 0/180 degree loopback test."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from QAWG import AWGAlazar, ExperimentProgram, MHz, ValuesSweep, ns, us


AWG_RESOURCE = "TCPIP0::192.168.10.171::inst0::INSTR"
OUTPUT_DIR = Path(__file__).resolve().parent / "single_shot_clock"
N_AVERAGE = 3000
WINDOWS_NS = {
    "notebook_0_1000": (0, 1000),
    "signal_250_950": (250, 950),
    "steady_350_950": (350, 950),
    "steady_400_800": (400, 800),
}


class PhaseDiagnosticProgram(ExperimentProgram):
    def _initialize(self, cfg):
        self.declare_gen(
            "readout",
            ch=cfg["readout_ch"],
            amplitude_vpp=cfg["channel_amplitude_vpp"],
        )
        self.declare_readout(
            "ro",
            adc_channel=cfg["adc_channel"],
            length=cfg["readout_length"],
            demod_freq=cfg["frequency"],
            waveform_ch=cfg["readout_ch"],
            marker_channel=cfg["marker_ch"],
            marker_length=40 * ns,
            marker_lead=cfg["marker_lead"],
            integrate_time=cfg["readout_length"],
        )
        phase = self.add_sweep("phase", ValuesSweep((0.0, np.pi)))
        self.add_pulse(
            "readout_pulse",
            gen="readout",
            style="const",
            length=cfg["readout_length"],
            frequency=cfg["frequency"],
            phase=phase,
            gain=cfg["readout_gain"],
            readout=True,
        )

    def _body(self, cfg):
        self.play("readout_pulse", at=0)
        self.trigger("ro", trigger_delay=0)


def summarize_shots(name: str, phase_index: int, shots: np.ndarray) -> dict:
    mean = np.mean(shots)
    rotated = shots * np.exp(-1j * np.angle(mean))
    amplitude = np.abs(shots)
    phase_error = np.angle(shots * np.exp(-1j * np.angle(mean)))
    median_amplitude = np.median(amplitude)
    return {
        "window": name,
        "phase_degrees": phase_index * 180,
        "mean_i_mV": float(mean.real * 1e3),
        "mean_q_mV": float(mean.imag * 1e3),
        "mean_amplitude_mV": float(np.abs(mean) * 1e3),
        "radial_std_mV": float(np.std(rotated.real, ddof=1) * 1e3),
        "tangential_std_mV": float(np.std(rotated.imag, ddof=1) * 1e3),
        "phase_std_degrees": float(np.std(np.rad2deg(phase_error), ddof=1)),
        "low_amplitude_shots": int(np.count_nonzero(amplitude < 0.5 * median_amplitude)),
        "shots": int(shots.size),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = {
        "readout_ch": 1,
        "marker_ch": 1,
        "adc_channel": "CHB",
        "channel_amplitude_vpp": 0.5,
        "frequency": 50 * MHz,
        "readout_length": 1 * us,
        # Match the roughly 2--3 mV clusters in the notebook screenshot.  The
        # notebook cell currently says 0.01, but its displayed output is stale.
        "readout_gain": 1.0,
        "marker_lead": 100 * ns,
    }
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
        print("AWG clock source:", experiment.awg.query("CLOCk:SOURce?").strip())
        program = PhaseDiagnosticProgram(cfg, final_delay_s=1 * us)
        program.REMOVE_DC_OFFSET = False
        compiled = program.compile(hardware=experiment)
        result = compiled.acquire(n_average=N_AVERAGE)
    finally:
        experiment.close()

    raw = result.raw
    time_s = result.raw_time_s
    reference = np.exp(-1j * 2 * np.pi * cfg["frequency"] * time_s)
    downconverted = 2.0 * raw * reference[None, None, :]
    shot_sets: dict[str, np.ndarray] = {}
    summaries = []
    for name, (start_ns, stop_ns) in WINDOWS_NS.items():
        start = int(round(start_ns * 1e-9 * experiment.alazar_sample_rate_hz))
        stop = int(round(stop_ns * 1e-9 * experiment.alazar_sample_rate_hz))
        shots = np.mean(downconverted[:, :, start:stop], axis=2)
        shot_sets[name] = shots
        for phase_index in range(2):
            row = summarize_shots(name, phase_index, shots[:, phase_index])
            summaries.append(row)
            print(
                f"{name:18s} {row['phase_degrees']:3d} deg: "
                f"|IQ|={row['mean_amplitude_mV']:.4f} mV, "
                f"radial={row['radial_std_mV']:.4f} mV, "
                f"tangent={row['tangential_std_mV']:.4f} mV, "
                f"phase_std={row['phase_std_degrees']:.3f} deg, "
                f"low_amp={row['low_amplitude_shots']}"
            )

    np.savez_compressed(
        OUTPUT_DIR / "single_shot_clock_data.npz",
        raw_volts=raw,
        raw_time_s=time_s,
        phase_degrees=np.asarray((0.0, 180.0)),
        **{f"iq_{name}_volts": shots for name, shots in shot_sets.items()},
    )
    with (OUTPUT_DIR / "single_shot_clock_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (OUTPUT_DIR / "single_shot_clock_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    plot_constellations(shot_sets)
    plot_mean_iq_time(time_s, downconverted)


def plot_constellations(shot_sets: dict[str, np.ndarray]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for axis, (name, shots) in zip(axes.flat, shot_sets.items()):
        for phase_index, color in enumerate(("tab:blue", "tab:orange")):
            values = shots[:, phase_index] * 1e3
            axis.scatter(values.real, values.imag, s=5, alpha=0.18, color=color)
            mean = np.mean(values)
            axis.scatter(mean.real, mean.imag, s=50, color=color, edgecolor="black")
        start_ns, stop_ns = WINDOWS_NS[name]
        axis.set_title(f"{start_ns}..{stop_ns} ns")
        axis.set_xlabel("I (mV)")
        axis.set_ylabel("Q (mV)")
        axis.axis("equal")
        axis.grid(True, alpha=0.25)
    figure.suptitle("0/180 degree single-shot IQ: integration-window comparison")
    figure.savefig(OUTPUT_DIR / "single_shot_iq_by_window.png", dpi=180)
    plt.close(figure)


def plot_mean_iq_time(time_s: np.ndarray, downconverted: np.ndarray) -> None:
    mean_iq = np.mean(downconverted, axis=0) * 1e3
    time_ns = time_s * 1e9
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    for phase_index, color in enumerate(("tab:blue", "tab:orange")):
        axes[0].plot(time_ns, mean_iq[phase_index].real, color=color, label=f"{phase_index * 180} deg I")
        axes[0].plot(time_ns, mean_iq[phase_index].imag, color=color, linestyle="--", label=f"{phase_index * 180} deg Q")
        axes[1].plot(time_ns, np.abs(mean_iq[phase_index]), color=color, label=f"{phase_index * 180} deg")
    axes[0].set_ylabel("I, Q (mV)")
    axes[1].set_ylabel("|IQ| (mV)")
    axes[1].set_xlabel("Time after received marker (ns)")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend()
    figure.suptitle("Mean demodulated IQ versus time")
    figure.savefig(OUTPUT_DIR / "mean_iq_vs_time.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
