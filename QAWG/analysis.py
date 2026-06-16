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


def calculate_window(
    result: ExperimentResult,
    *,
    step: int = 0,
    trigger_lead_s: float = 20e-9,
    integration_guard_s: float = 20e-9,
    plot: bool = True,
    report: bool = True,
) -> WindowAnalysis:
    """Recommend ATS trigger delay and IQ integration window.

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
    iq_envelope = np.abs(result.iq_trace_average()[step])
    baseline_count = max(1, int(0.1 * iq_envelope.size))
    baseline = float(np.median(iq_envelope[:baseline_count]))
    peak = float(np.percentile(iq_envelope, 95))
    threshold = baseline + 0.5 * (peak - baseline)

    expected_rise_s = float(readout_start_s - initial_trigger_s)
    search_margin_s = max(50e-9, min(250e-9, 0.5 * readout_duration_s))
    search_start_s = max(0.0, expected_rise_s - search_margin_s)
    search_stop_s = min(
        float(result.iq_time_s[-1]),
        expected_rise_s + search_margin_s,
    )
    search_indices = np.flatnonzero(
        (result.iq_time_s >= search_start_s)
        & (result.iq_time_s <= search_stop_s)
        & (iq_envelope >= threshold)
    )
    if not search_indices.size:
        raise ValueError(
            "No readout rising edge found near the compiled waveform window"
        )

    measured_rise_s = _interpolate_crossing(
        result.iq_time_s,
        iq_envelope,
        threshold,
        int(search_indices[0]),
    )
    suggested_trigger_s = max(
        0.0,
        initial_trigger_s + measured_rise_s - trigger_lead_s,
    )
    integration_start_s = 0.0
    integration_stop_s = (
        trigger_lead_s + readout_duration_s + integration_guard_s
    )
    if result.acquire_window_s is not None:
        integration_stop_s = min(
            integration_stop_s,
            float(result.acquire_window_s),
        )

    trigger_shift_s = suggested_trigger_s - initial_trigger_s
    raw_plot_time_s = result.raw_time_s - trigger_shift_s
    iq_plot_time_s = result.iq_time_s - trigger_shift_s

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
            marker_start_s -= suggested_trigger_s
            marker_stop_s -= suggested_trigger_s
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

        readout_plot_start_s = measured_rise_s - trigger_shift_s
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
            "Raw average aligned to suggested post-trigger delay "
            f"({suggested_trigger_s * 1e9:.3f} ns)"
        )
        axes[1].set_xlabel("Time after suggested ATS trigger (ns)")
        axes[1].set_ylabel("|IQ| (mV)")
        axes[1].set_title("Demodulated readout envelope")
        for axis in axes:
            axis.grid(True, alpha=0.3)
            axis.legend()
        figure.tight_layout()

    if report:
        print(f"Initial post-trigger delay: {initial_trigger_s * 1e9:.3f} ns")
        print(f"Measured readout arrival: {measured_rise_s * 1e9:.3f} ns")
        print(f"Compiled readout duration: {readout_duration_s * 1e9:.3f} ns")
        print(
            "Suggested post-trigger delay: "
            f"{suggested_trigger_s * 1e9:.3f} ns"
        )
        print(
            "Suggested integration window: "
            f"{integration_start_s * 1e9:.3f} to "
            f"{integration_stop_s * 1e9:.3f} ns"
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
