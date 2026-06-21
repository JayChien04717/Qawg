"""Analysis helpers for timing and integration-window calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .compiler import ExperimentResult


@dataclass(frozen=True)
class WindowAnalysis:
    step_index: int
    initial_trigger_delay_s: float
    measured_rise_s: float
    readout_duration_s: float
    suggested_trigger_delay_s: float
    integration_start_s: float
    integration_stop_s: float
    figure: Any
    axes: tuple[Any, Any]


def _interpolate_crossing(
    time_s: np.ndarray,
    values: np.ndarray,
    threshold: float,
    right: int,
) -> float:
    if right == 0:
        return float(time_s[0])
    left = right - 1
    y0, y1 = values[left], values[right]
    fraction = 0.0 if y1 == y0 else (threshold - y0) / (y1 - y0)
    return float(time_s[left] + fraction * (time_s[right] - time_s[left]))


def _centered_moving_average(
    values: np.ndarray,
    window_samples: int,
) -> np.ndarray:
    if window_samples <= 1:
        return values.copy()
    left = (window_samples - 1) // 2
    right = window_samples // 2
    padded = np.pad(values, (left, right), mode="edge")
    kernel = np.ones(window_samples, dtype=np.float64) / window_samples
    return np.convolve(padded, kernel, mode="valid")


def _lowpass_iq_envelope(
    iq_trace: np.ndarray,
    iq_time_s: np.ndarray,
    smoothing_time_s: float,
) -> np.ndarray:
    if iq_time_s.size < 2 or smoothing_time_s <= 0:
        return np.abs(iq_trace)
    dt_s = float(np.median(np.diff(iq_time_s)))
    if dt_s <= 0:
        return np.abs(iq_trace)
    window_samples = max(1, int(round(smoothing_time_s / dt_s)))
    if window_samples % 2 == 0:
        window_samples += 1
    window_samples = min(window_samples, iq_trace.size)
    return np.abs(_centered_moving_average(iq_trace, window_samples))


def _find_sustained_rise(
    time_s: np.ndarray,
    envelope: np.ndarray,
    threshold: float,
    *,
    minimum_duration_s: float,
) -> int | None:
    if time_s.size < 2:
        return None
    dt_s = float(np.median(np.diff(time_s)))
    if dt_s <= 0:
        return None
    minimum_samples = max(1, int(round(minimum_duration_s / dt_s)))
    above = envelope >= threshold
    transitions = np.diff(
        np.concatenate(([False], above, [False])).astype(np.int8)
    )
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    candidates: list[tuple[float, int]] = []
    for start, stop in zip(starts, stops):
        if stop - start < minimum_samples:
            continue
        mean_excess = float(np.mean(envelope[start:stop] - threshold))
        duration_s = float(time_s[stop - 1] - time_s[start] + dt_s)
        candidates.append((mean_excess * duration_s, int(start)))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def calculate_window(
    result: ExperimentResult,
    *,
    step: int = 0,
    trigger_lead_s: float = 20e-9,
    integration_guard_s: float = 20e-9,
    envelope_smoothing_s: float = 20e-9,
    minimum_high_time_s: float | None = None,
    plot: bool = True,
    report: bool = True,
) -> WindowAnalysis:
    """Recommend integration delay and IQ integration window.

    The measured rising edge is found near the compiled readout waveform.
    Integration duration comes from that waveform, so marker-edge transients
    cannot extend the suggested window.
    """
    if result.initial_trigger_delay_s is None:
        raise ValueError("Result does not contain initial trigger metadata")
    if result.readout_windows_s is None:
        raise ValueError("Result does not contain readout waveform metadata")
    if not 0 <= step < result.raw.shape[1]:
        raise IndexError("step is outside the sequence")
    if trigger_lead_s < 0 or integration_guard_s < 0:
        raise ValueError("timing guards cannot be negative")

    initial_trigger_s = float(result.initial_trigger_delay_s)
    readout_start_s, readout_stop_s = result.readout_windows_s[step]
    readout_duration_s = float(readout_stop_s - readout_start_s)
    if readout_duration_s <= 0:
        raise ValueError("Compiled readout waveform has no duration")

    raw_average = result.trace_average()[step]
    iq_average = result.iq_trace_average()[step]
    iq_envelope = _lowpass_iq_envelope(
        iq_average,
        result.iq_time_s,
        envelope_smoothing_s,
    )
    baseline_count = max(1, int(0.1 * iq_envelope.size))
    baseline = float(np.median(iq_envelope[:baseline_count]))
    peak = float(np.percentile(iq_envelope, 95))
    threshold = baseline + 0.5 * (peak - baseline)

    sustained_time_s = (
        max(50e-9, 0.10 * readout_duration_s)
        if minimum_high_time_s is None
        else float(minimum_high_time_s)
    )
    rise_index = _find_sustained_rise(
        result.iq_time_s,
        iq_envelope,
        threshold,
        minimum_duration_s=sustained_time_s,
    )
    if rise_index is None:
        raise ValueError(
            "No sustained readout envelope found in the acquired trace"
        )

    measured_rise_s = _interpolate_crossing(
        result.iq_time_s,
        iq_envelope,
        threshold,
        rise_index,
    )
    integration_start_s = max(0.0, measured_rise_s - trigger_lead_s)
    integration_stop_s = (
        integration_start_s + readout_duration_s + integration_guard_s
    )
    if result.acquire_window_s is not None:
        integration_stop_s = min(
            integration_stop_s,
            float(result.acquire_window_s),
        )
    suggested_trigger_s = integration_start_s

    raw_plot_time_s = result.raw_time_s
    iq_plot_time_s = result.iq_time_s

    figure = None
    axes: tuple[Any, Any] = (None, None)
    if plot:
        import matplotlib.pyplot as plt

        figure, plot_axes = plt.subplots(
            2,
            1,
            figsize=(12, 8),
            sharex=True,
        )
        axes = (plot_axes[0], plot_axes[1])
        axes[0].plot(raw_plot_time_s * 1e9, raw_average * 1e3)
        axes[1].plot(
            iq_plot_time_s * 1e9,
            iq_envelope * 1e3,
            label="|IQ|",
        )

        marker_label = None
        if result.marker_windows_s is not None:
            marker_start_s, marker_stop_s = result.marker_windows_s[step]
            marker_label = (
                f"Marker high "
                f"({(marker_stop_s - marker_start_s) * 1e9:.0f} ns)"
            )
            for axis in axes:
                axis.axvspan(
                    marker_start_s * 1e9,
                    marker_stop_s * 1e9,
                    facecolor="tab:blue",
                    alpha=0.05,
                    edgecolor="tab:blue",
                    linewidth=2,
                    label=marker_label,
                )

        readout_plot_start_s = measured_rise_s
        readout_plot_stop_s = readout_plot_start_s + readout_duration_s
        for axis in axes:
            axis.axvspan(
                readout_plot_start_s * 1e9,
                readout_plot_stop_s * 1e9,
                facecolor="tab:green",
                alpha=0.10,
                edgecolor="tab:green",
                linewidth=2,
                label=(
                    "Readout waveform "
                    f"({readout_duration_s * 1e9:.0f} ns)"
                ),
            )
            axis.axvspan(
                integration_start_s * 1e9,
                integration_stop_s * 1e9,
                facecolor="none",
                edgecolor="tab:orange",
                linewidth=2,
                hatch="//",
                label=(
                    "Suggested integration window "
                    f"({(integration_stop_s - integration_start_s) * 1e9:.0f} ns)"
                ),
            )

        axes[0].set_ylabel("ADC voltage (mV)")
        axes[0].set_title(
            "Raw average in acquired window "
            "(starts at received marker)"
        )
        axes[1].set_xlabel("Time in acquired window (ns)")
        axes[1].set_ylabel("|IQ| (mV)")
        axes[1].set_title("Demodulated readout envelope")
        for axis in axes:
            axis.grid(True, alpha=0.3)
            axis.legend()
        figure.tight_layout()

    if report:
        print(f"Configured integration delay: {initial_trigger_s * 1e9:.3f} ns")
        print(
            "Measured readout arrival in acquire window: "
            f"{measured_rise_s * 1e9:.3f} ns"
        )
        print(f"Compiled readout duration: {readout_duration_s * 1e9:.3f} ns")
        print(
            "Suggested integration delay in acquire window: "
            f"{integration_start_s * 1e9:.3f} ns"
        )
        print(
            "Suggested integration time: "
            f"{(integration_stop_s - integration_start_s) * 1e9:.3f} ns"
        )
        print(
            "Suggested integration delay from received marker: "
            f"{suggested_trigger_s * 1e9:.3f} ns"
        )
        print(f"DC offset removal: {result.remove_dc_offset}")

    return WindowAnalysis(
        step_index=step,
        initial_trigger_delay_s=initial_trigger_s,
        measured_rise_s=measured_rise_s,
        readout_duration_s=readout_duration_s,
        suggested_trigger_delay_s=suggested_trigger_s,
        integration_start_s=integration_start_s,
        integration_stop_s=integration_stop_s,
        figure=figure,
        axes=axes,
    )
