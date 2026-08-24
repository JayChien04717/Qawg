"""Compare the saved -12 dBm trace with and without FIR preprocessing."""

from __future__ import annotations

from pathlib import Path
import sys

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from QAWG.alazar import downconvert, fir_filter


DATA_PATH = Path(__file__).with_name("Rabi_WG.hdf5")
OUTPUT_PATH = Path(__file__).with_name(
    "Rabi_WG_minus12dBm_fir_vs_direct_boxcar.png"
)
ZOOM_OUTPUT_PATH = Path(__file__).with_name(
    "Rabi_WG_minus12dBm_three_region_zoom.png"
)
FFT_OUTPUT_PATH = Path(__file__).with_name(
    "Rabi_WG_minus12dBm_fft_0_50MHz.png"
)

SELECTED_POWER_DBM = -12.0
FIR_CUTOFF_HZ = 50e6
FIR_TAPS = 61
FIR_KAISER_BETA = 8.0
FLAT_START_S = 0.8e-6
FLAT_STOP_S = 1.65e-6
RISING_ZOOM_S = (0.15e-6, 0.65e-6)
PULSE_OFF_ZOOM_S = (1.68e-6, 1.92e-6)


def sliding_boxcar(values: np.ndarray, samples: int) -> np.ndarray:
    """Return every valid boxcar average along the final sample axis."""
    cumulative = np.pad(
        np.cumsum(values, axis=-1),
        ((0, 0), (1, 0)),
    )
    return (cumulative[:, samples:] - cumulative[:, :-samples]) / samples


def falling_time(
    time_s: np.ndarray,
    magnitude: np.ndarray,
) -> tuple[float, float, float]:
    """Return 90%, 10%, and 90-to-10 falling times."""
    baseline = float(np.mean(magnitude[(time_s >= 1.88e-6) & (time_s <= 1.96e-6)]))
    plateau = float(np.mean(magnitude[(time_s >= 1.2e-6) & (time_s <= 1.65e-6)]))
    normalized = (magnitude - baseline) / (plateau - baseline)
    search = int(np.searchsorted(time_s, PULSE_OFF_ZOOM_S[0]))
    at_90 = search + int(np.flatnonzero(normalized[search:] <= 0.9)[0])
    at_10 = search + int(np.flatnonzero(normalized[search:] <= 0.1)[0])
    return time_s[at_90], time_s[at_10], time_s[at_10] - time_s[at_90]


def main() -> None:
    with h5py.File(DATA_PATH, "r") as handle:
        raw_records = np.asarray(
            handle[
                "Traces/Digitizer - Channel A - Average record"
            ][:, 0, :],
            dtype=float,
        ).T
        saved = np.asarray(
            handle[
                "Traces/Digitizer - Channel A - "
                "Average piecewise demodulated values"
            ]
        )
        saved_demodulated = np.asarray(
            saved[:, 0, :] + 1j * saved[:, 1, :],
            dtype=np.complex128,
        ).T
        power_dbm = np.asarray(handle["Data/Data"][:, 0, 0], dtype=float)
        sample_interval_s = float(
            handle[
                "Traces/Digitizer - Channel A - Average record_t0dt"
            ][0, 1]
        )
        demodulator = handle[
            "Instrument config/AlazarTech Signal Demodulator2.3.2 - 1, "
            "Digitizer at localhost"
        ]
        demodulation_frequency_hz = float(
            demodulator.attrs["Demodulation frequency"]
        )
        boxcar_time_s = float(demodulator.attrs["Demodulation length"])

    selected = int(np.argmin(np.abs(power_dbm - SELECTED_POWER_DBM)))
    sample_rate_hz = 1.0 / sample_interval_s
    boxcar_samples = int(round(boxcar_time_s * sample_rate_hz))
    baseband = downconvert(
        raw_records[selected : selected + 1],
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
    direct_boxcar = sliding_boxcar(baseband, boxcar_samples)[0]
    fir_boxcar = sliding_boxcar(filtered, boxcar_samples)[0]
    saved_trace = saved_demodulated[selected]
    time_s = np.arange(direct_boxcar.size) * sample_interval_s
    time_us = time_s * 1e6

    direct_magnitude = np.abs(direct_boxcar)
    fir_magnitude = np.abs(fir_boxcar)
    saved_magnitude = np.abs(saved_trace)
    difference = fir_magnitude - direct_magnitude
    flat = (time_s >= FLAT_START_S) & (time_s <= FLAT_STOP_S)

    direct_flat_mean = float(np.mean(direct_magnitude[flat]))
    fir_flat_mean = float(np.mean(fir_magnitude[flat]))
    direct_flat_ripple = float(np.std(direct_magnitude[flat]))
    fir_flat_ripple = float(np.std(fir_magnitude[flat]))
    saved_correlation = float(
        np.corrcoef(direct_magnitude, saved_magnitude)[0, 1]
    )

    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].plot(
        time_us,
        saved_magnitude * 1e3,
        color="black",
        linewidth=2,
        label="Saved Labber trace",
    )
    axes[0, 0].plot(
        time_us,
        direct_magnitude * 1e3,
        "--",
        label="Recomputed: DDC + direct 20 ns boxcar",
    )
    axes[0, 0].set(
        xlabel="Time (us)",
        ylabel="|IQ| (mV)",
        title="Saved data reproduced without FIR",
    )
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(
        time_us,
        direct_magnitude * 1e3,
        label="No FIR: direct 20 ns boxcar",
    )
    axes[0, 1].plot(
        time_us,
        fir_magnitude * 1e3,
        "--",
        label="50 MHz FIR + 20 ns boxcar",
    )
    axes[0, 1].set(
        xlabel="Time (us)",
        ylabel="|IQ| (mV)",
        title=f"Complete {power_dbm[selected]:g} dBm time trace",
    )
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    direct_deviation = (direct_magnitude - direct_flat_mean) * 1e6
    fir_deviation = (fir_magnitude - fir_flat_mean) * 1e6
    axes[1, 0].plot(
        time_us[flat],
        direct_deviation[flat],
        label=f"No FIR: std={direct_flat_ripple * 1e6:.1f} uV",
    )
    axes[1, 0].plot(
        time_us[flat],
        fir_deviation[flat],
        label=f"With FIR: std={fir_flat_ripple * 1e6:.1f} uV",
    )
    axes[1, 0].set(
        xlabel="Time (us)",
        ylabel="Deviation from flat mean (uV)",
        title="Flat-top ripple (0.8-1.65 us)",
    )
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(time_us, difference * 1e3)
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set(
        xlabel="Time (us)",
        ylabel="FIR - no FIR (mV)",
        title="FIR changes mainly the pulse edges",
    )
    axes[1, 1].grid(True, alpha=0.3)

    figure.suptitle(
        "Rabi_WG.hdf5 time-trace comparison\n"
        f"{demodulation_frequency_hz / 1e6:g} MHz DDC, "
        f"{boxcar_time_s * 1e9:g} ns boxcar; FIR={FIR_CUTOFF_HZ / 1e6:g} "
        f"MHz/{FIR_TAPS} taps/beta {FIR_KAISER_BETA:g}"
    )
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=180)
    plt.close(figure)

    _, _, direct_fall = falling_time(
        time_s,
        direct_magnitude,
    )
    _, _, fir_fall = falling_time(time_s, fir_magnitude)
    regions = (
        ("Rising edge", RISING_ZOOM_S),
        ("Flat top", (FLAT_START_S, FLAT_STOP_S)),
        ("Pulse-off edge and decay", PULSE_OFF_ZOOM_S),
    )
    zoom_figure, zoom_axes = plt.subplots(3, 1, figsize=(11, 12))
    for axis, (title, limits) in zip(zoom_axes, regions):
        region = (time_s >= limits[0]) & (time_s <= limits[1])
        fir_label = "FIR"
        direct_label = "No FIR"
        if title == "Flat top":
            fir_label += f" (ripple std={fir_flat_ripple * 1e6:.1f} uV)"
            direct_label += f" (ripple std={direct_flat_ripple * 1e6:.1f} uV)"
        elif title == "Pulse-off edge and decay":
            fir_label += f" (90-10%={fir_fall * 1e9:.1f} ns)"
            direct_label += f" (90-10%={direct_fall * 1e9:.1f} ns)"
        axis.plot(
            time_us[region],
            fir_magnitude[region] * 1e3,
            color="tab:orange",
            linewidth=2,
            label=fir_label,
        )
        axis.plot(
            time_us[region],
            direct_magnitude[region] * 1e3,
            color="tab:blue",
            linestyle="--",
            linewidth=1.6,
            label=direct_label,
        )
        axis.set(
            xlabel="Time (us)",
            ylabel="|IQ| (mV)",
            title=title,
        )
        axis.grid(True, alpha=0.3)
        axis.legend()
    zoom_figure.suptitle(
        f"Rabi_WG.hdf5 at {power_dbm[selected]:g} dBm\n"
        f"{demodulation_frequency_hz / 1e6:g} MHz DDC, "
        f"{boxcar_time_s * 1e9:g} ns boxcar; FIR={FIR_CUTOFF_HZ / 1e6:g} "
        f"MHz/{FIR_TAPS} taps/beta {FIR_KAISER_BETA:g}"
    )
    zoom_figure.tight_layout()
    zoom_figure.savefig(ZOOM_OUTPUT_PATH, dpi=180)
    plt.close(zoom_figure)

    fft_window = np.hanning(np.count_nonzero(flat))
    fft_frequency_hz = np.fft.fftfreq(fft_window.size, sample_interval_s)

    def residual_fft_db(values: np.ndarray) -> np.ndarray:
        residual = values[flat] - np.mean(values[flat])
        amplitude = np.abs(np.fft.fft(residual * fft_window)) / np.sum(
            fft_window
        )
        return 20.0 * np.log10(np.maximum(amplitude, 1e-15))

    direct_fft_db = residual_fft_db(direct_boxcar)
    fir_fft_db = residual_fft_db(fir_boxcar)
    fft_zoom = (fft_frequency_hz >= 0.0) & (fft_frequency_hz <= 50e6)
    fft_figure, fft_axis = plt.subplots(figsize=(11, 6))
    fft_axis.plot(
        fft_frequency_hz[fft_zoom] / 1e6,
        fir_fft_db[fft_zoom],
        color="tab:orange",
        linewidth=2,
        label="FIR",
    )
    fft_axis.plot(
        fft_frequency_hz[fft_zoom] / 1e6,
        direct_fft_db[fft_zoom],
        color="tab:blue",
        linestyle="--",
        linewidth=1.6,
        label="No FIR",
    )
    fft_axis.set(
        xlim=(0.0, 50.0),
        xlabel="Frequency (MHz)",
        ylabel="FFT amplitude (dBV)",
        title="Flat-top residual FFT (0.8-1.65 us)",
    )
    fft_axis.grid(True, alpha=0.3)
    fft_axis.legend()
    fft_figure.suptitle(
        f"Rabi_WG.hdf5 at {power_dbm[selected]:g} dBm: 0-50 MHz\n"
        f"{demodulation_frequency_hz / 1e6:g} MHz DDC, "
        f"{boxcar_time_s * 1e9:g} ns boxcar; FIR={FIR_CUTOFF_HZ / 1e6:g} "
        f"MHz/{FIR_TAPS} taps/beta {FIR_KAISER_BETA:g}"
    )
    fft_figure.tight_layout()
    fft_figure.savefig(FFT_OUTPUT_PATH, dpi=180)
    plt.close(fft_figure)

    print(f"Input: {DATA_PATH}")
    print(f"Selected sweep point: {power_dbm[selected]:g} dBm (index {selected})")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Three-region zoom: {ZOOM_OUTPUT_PATH}")
    print(f"FFT zoom: {FFT_OUTPUT_PATH}")
    print(f"Saved vs recomputed no-FIR correlation: {saved_correlation:.12f}")
    print(
        "Saved vs recomputed no-FIR median magnitude difference: "
        f"{np.median(np.abs(saved_magnitude - direct_magnitude)) * 1e6:.3f} uV"
    )
    print(f"No-FIR flat mean: {direct_flat_mean * 1e3:.6f} mV")
    print(f"FIR flat mean: {fir_flat_mean * 1e3:.6f} mV")
    print(f"No-FIR flat ripple std: {direct_flat_ripple * 1e6:.3f} uV")
    print(f"FIR flat ripple std: {fir_flat_ripple * 1e6:.3f} uV")
    print(
        "Maximum magnitude change from FIR: "
        f"{np.max(np.abs(difference)) * 1e3:.6f} mV"
    )
    print(f"No-FIR pulse-off 90-10% time: {direct_fall * 1e9:.3f} ns")
    print(f"FIR pulse-off 90-10% time: {fir_fall * 1e9:.3f} ns")


if __name__ == "__main__":
    main()
