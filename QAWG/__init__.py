"""Lazy public API for the QAWG hardware stack.

Keeping this module lazy allows ``QAWG.awg5200`` and ``QAWG.alazar`` to be
imported independently without importing the other hardware stack.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "CompiledExperiment": (".compiler", "CompiledExperiment"),
    "ExperimentProgram": (".compiler", "ExperimentProgram"),
    "ExperimentResult": (".compiler", "ExperimentResult"),
    "LinearSweep": (".compiler", "LinearSweep"),
    "SweepRef": (".compiler", "SweepRef"),
    "ValuesSweep": (".compiler", "ValuesSweep"),
    "MHz": (".compiler", "MHz"),
    "ns": (".compiler", "ns"),
    "us": (".compiler", "us"),
    "AWGAlazar": (".awg_alazar", "AWGAlazar"),
    "AWG5208": (".awg5200", "AWG5208"),
    "WindowAnalysis": (".analysis", "WindowAnalysis"),
    "calculate_window": (".analysis", "calculate_window"),
    "fit_resonator_linewidth": (".analysis", "fit_resonator_linewidth"),
    "Delay": (".timeline", "Delay"),
    "Parallel": (".timeline", "Parallel"),
    "Timeline": (".timeline", "Timeline"),
    "Waveform": (".timeline", "Waveform"),
    "align_channel_envelopes": (".timeline", "align_channel_envelopes"),
    "align_channels": (".timeline", "align_channels"),
    "channel_names": (".timeline", "channel_names"),
    "delay": (".timeline", "delay"),
    "delay_auto": (".timeline", "delay_auto"),
    "parallel": (".timeline", "parallel"),
    "waveform": (".timeline", "waveform"),
    "calibrate_iq_samples": (".tomography", "calibrate_iq_samples"),
    "coherent_density_matrix": (".tomography", "coherent_density_matrix"),
    "heterodyne_ml_density_matrix": (
        ".tomography",
        "heterodyne_ml_density_matrix",
    ),
    "normalize_heterodyne_reference": (
        ".tomography",
        "normalize_heterodyne_reference",
    ),
    "project_temporal_mode": (".tomography", "project_temporal_mode"),
    "temporal_mode_weights": (".tomography", "temporal_mode_weights"),
    "wigner_function": (".tomography", "wigner_function"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value

__all__ = [
    "AWG5208",
    "AWGAlazar",
    "CompiledExperiment",
    "Delay",
    "ExperimentProgram",
    "ExperimentResult",
    "LinearSweep",
    "MHz",
    "Parallel",
    "SweepRef",
    "Timeline",
    "ValuesSweep",
    "Waveform",
    "WindowAnalysis",
    "align_channel_envelopes",
    "align_channels",
    "calculate_window",
    "calibrate_iq_samples",
    "channel_names",
    "coherent_density_matrix",
    "delay",
    "delay_auto",
    "fit_resonator_linewidth",
    "heterodyne_ml_density_matrix",
    "normalize_heterodyne_reference",
    "ns",
    "parallel",
    "project_temporal_mode",
    "temporal_mode_weights",
    "us",
    "waveform",
    "wigner_function",
]
