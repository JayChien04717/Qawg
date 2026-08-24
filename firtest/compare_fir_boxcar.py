"""Compare FIR-then-boxcar with direct boxcar on Rabi_WG.hdf5."""

from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from QAWG.alazar import downconvert, fir_filter, integrate


DATA_PATH = Path(__file__).with_name("Rabi_WG.hdf5")
OUTPUT_PATH = Path(__file__).with_name("Rabi_WG_fir_vs_boxcar.png")

FIR_CUTOFF_HZ = 50e6
FIR_TAPS = 61
FIR_KAISER_BETA = 8.0


def _complex_trace(dataset: h5py.Dataset) -> np.ndarray:
    """Decode Labber's (time, real/imag, sweep) trace representation."""
    values = np.asarray(dataset)
    return np.asarray(
        values[:, 0, :] + 1j * values[:, 1, :],
        dtype=np.complex128,
    ).T


def _normalized_rms_error(reference: np.ndarray, values: np.ndarray) -> float:
    """Return residual error after removing a complex offset and gain."""
    x = reference - np.mean(reference)
    y = values - np.mean(values)
    gain = np.vdot(x, y) / np.vdot(x, x)
    residual = y - gain * x
    return float(np.linalg.norm(residual) / np.linalg.norm(y))


def _median_runtime(operation, repeats: int = 7) -> float:
    durations = []
    for _ in range(repeats):
        start = perf_counter()
        operation()
        durations.append(perf_counter() - start)
    return float(np.median(durations))


def main() -> None:
    with h5py.File(DATA_PATH, "r") as handle:
        raw_dataset = handle[
            "Traces/Digitizer - Channel A - Average record"
        ]
        raw_records = np.asarray(raw_dataset[:, 0, :], dtype=float).T
        sample_interval_s = float(
            handle[
                "Traces/Digitizer - Channel A - Average record_t0dt"
            ][0, 1]
        )
        power_dbm = np.asarray(handle["Data/Data"][:, 0, 0], dtype=float)
        stored_demodulated = _complex_trace(
            handle[
                "Traces/Digitizer - Channel A - "
                "Average piecewise demodulated values"
            ]
        )
        demodulator = handle[
            "Instrument config/AlazarTech Signal Demodulator2.3.2 - 1, "
            "Digitizer at localhost"
        ]
        demodulation_frequency_hz = float(
            demodulator.attrs["Demodulation frequency"]
        )
        boxcar_time_s = float(demodulator.attrs["Demodulation length"])
        cursor_time_s = float(
            handle["Views/Current view"].attrs.get("cursor_1_x", 0.0)
        )

    sample_rate_hz = 1.0 / sample_interval_s
    boxcar_samples = int(round(boxcar_time_s * sample_rate_hz))
    start_sample = int(round(cursor_time_s * sample_rate_hz))
    stop_sample = start_sample + boxcar_samples
    if not 0 <= start_sample < stop_sample <= raw_records.shape[1]:
        raise ValueError("saved Labber cursor is outside the average record")

    baseband = downconvert(
        raw_records,
        sample_rate_hz,
        demodulation_frequency_hz,
    )
    filtered = fir_filter(
        baseband,
        sample_rate_hz,
        FIR_CUTOFF_HZ,
        FIR_TAPS,
        FIR_KAISER_BETA,
    )
    direct_boxcar = integrate(baseband, start_sample, stop_sample)
    fir_boxcar = integrate(filtered, start_sample, stop_sample)
    saved_labber = stored_demodulated[:, start_sample]

    magnitude_correlation = float(
        np.corrcoef(np.abs(direct_boxcar), np.abs(fir_boxcar))[0, 1]
    )
    relative_difference = np.abs(
        np.abs(fir_boxcar) - np.abs(direct_boxcar)
    ) / np.maximum(np.abs(direct_boxcar), np.finfo(float).eps)
    nrmse = _normalized_rms_error(direct_boxcar, fir_boxcar)
    labber_correlation = float(
        np.corrcoef(np.abs(direct_boxcar), np.abs(saved_labber))[0, 1]
    )
    direct_runtime_s = _median_runtime(
        lambda: integrate(baseband, start_sample, stop_sample)
    )
    fir_runtime_s = _median_runtime(
        lambda: integrate(
            fir_filter(
                baseband,
                sample_rate_hz,
                FIR_CUTOFF_HZ,
                FIR_TAPS,
                FIR_KAISER_BETA,
            ),
            start_sample,
            stop_sample,
        )
    )

    representative = int(np.argmax(np.abs(direct_boxcar)))
    time_ns = np.arange(raw_records.shape[1]) * sample_interval_s * 1e9
    window_start_ns = start_sample * sample_interval_s * 1e9
    window_stop_ns = stop_sample * sample_interval_s * 1e9

    figure, axes = plt.subplots(3, 1, figsize=(10, 10))
    axes[0].plot(
        power_dbm,
        np.abs(direct_boxcar) * 1e3,
        label="DDC + direct 20 ns boxcar",
        linewidth=2,
    )
    axes[0].plot(
        power_dbm,
        np.abs(fir_boxcar) * 1e3,
        "--",
        label="DDC + FIR + 20 ns boxcar",
    )
    axes[0].plot(
        power_dbm,
        np.abs(saved_labber) * 1e3,
        ":",
        label="Saved Labber demodulation",
    )
    axes[0].set(
        xlabel="Qubit drive power (dBm)",
        ylabel="Integrated |IQ| (mV)",
        title="Rabi result: FIR + boxcar versus direct boxcar",
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        power_dbm,
        (np.abs(fir_boxcar) - np.abs(direct_boxcar)) * 1e6,
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set(
        xlabel="Qubit drive power (dBm)",
        ylabel="FIR - direct (uV)",
        title=(
            f"Magnitude correlation={magnitude_correlation:.8f}, "
            f"median relative difference={np.median(relative_difference):.4%}"
        ),
    )
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(
        time_ns,
        np.abs(baseband[representative]) * 1e3,
        alpha=0.6,
        label="Unfiltered |DDC IQ|",
    )
    axes[2].plot(
        time_ns,
        np.abs(filtered[representative]) * 1e3,
        label="FIR-filtered |IQ|",
        linewidth=1.5,
    )
    axes[2].axvspan(
        window_start_ns,
        window_stop_ns,
        color="tab:orange",
        alpha=0.2,
        label="20 ns boxcar",
    )
    axes[2].set(
        xlim=(400, 800),
        xlabel="Time (ns)",
        ylabel="|IQ| (mV)",
        title=f"Representative waveform at {power_dbm[representative]:.3f} dBm",
    )
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    figure.suptitle(
        "Rabi_WG.hdf5\n"
        f"FIR={FIR_CUTOFF_HZ / 1e6:g} MHz, {FIR_TAPS} taps, "
        f"Kaiser beta={FIR_KAISER_BETA:g}; "
        f"window={window_start_ns:.0f}-{window_stop_ns:.0f} ns"
    )
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=180)
    plt.close(figure)

    print(f"Input: {DATA_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Sweep points: {power_dbm.size}")
    print(f"Average-record shape: {raw_records.shape}")
    print(f"Integration window: {window_start_ns:.0f}-{window_stop_ns:.0f} ns")
    print(f"Magnitude correlation: {magnitude_correlation:.10f}")
    print(
        "Median magnitude relative difference: "
        f"{np.median(relative_difference):.6%}"
    )
    print(f"Complex normalized RMS difference: {nrmse:.6%}")
    print(f"Direct boxcar vs saved Labber correlation: {labber_correlation:.12f}")
    print(f"Direct boxcar median runtime: {direct_runtime_s * 1e3:.3f} ms")
    print(f"FIR + boxcar median runtime: {fir_runtime_s * 1e3:.3f} ms")


if __name__ == "__main__":
    main()
