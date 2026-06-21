"""Microwave temporal-mode tomography and cavity-decay characterization.

The ``full`` mode performs the complete hardware workflow:

1. scan the RF source and fit the complex cavity response,
2. calculate kappa and a truncated rising-exponential loading pulse,
3. load the cavity and acquire the free ring-down,
4. fit the measured ring-down and construct a normalized temporal mode,
5. save raw data, metadata, and diagnostic figures under ``tomodata``.

Math and data transformations are kept in small pure functions. Instrument
side effects are isolated in the acquisition functions near the end.

Single-HEMT tomography:

    python tomo.py hemt-stream --cavity-frequency 5.9e9 --if-frequency 50e6
    python tomo.py analyze-hemt --data-dir tomodata/<run> \
        --photon-scale-volts <calibrated_volts_per_sqrt_photon>

``hemt-stream`` alternates a zero-drive reference and driven signal in the
same AWG sequence. It saves one complex temporal-mode sample per shot, not
the full ADC trace. The reference must be vacuum at the device output; IQ
leakage, thermal photons, line loss, mode timing, and volts-to-photon scale
are experimental calibrations, not values this script can infer safely.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares

from QAWG import AWGAlazar, ExperimentProgram, ValuesSweep, ns, us
from QAWG.awg5200.waveforms import modulate_envelope
from QAWG.instrument import RohdeSchwarzSGS100A
from QAWG.tomography import (
    heterodyne_ml_density_matrix,
    normalize_heterodyne_reference,
    project_temporal_mode,
    wigner_function,
)


ComplexArray = npt.NDArray[np.complex128]
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class ResonatorFit:
    resonance_hz: float
    linewidth_hz: float
    kappa_rad_s: float
    amplitude_lifetime_s: float
    energy_lifetime_s: float
    fitted_response: ComplexArray
    residual_rms: float


@dataclass(frozen=True)
class RingdownFit:
    start_s: float
    stop_s: float
    amplitude_lifetime_s: float
    linewidth_hz: float
    detuning_hz: float
    complex_amplitude: complex
    fitted_envelope: ComplexArray
    residual_rms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "simulate",
            "connect",
            "scan",
            "load",
            "stream",
            "analyze-stream",
            "hemt-stream",
            "analyze-hemt",
            "full",
        ),
        nargs="?",
        default="simulate",
    )
    parser.add_argument(
        "--awg-resource",
        default="TCPIP0::192.168.10.171::inst0::INSTR",
    )
    parser.add_argument("--sgs-address", default="192.168.10.90")
    parser.add_argument(
        "--cavity-frequency",
        "--device-frequency",
        dest="cavity_frequency",
        type=float,
        default=5.898e9,
        help="RF device frequency; use --device-frequency for transmon experiments.",
    )
    parser.add_argument("--if-frequency", type=float, default=50e6)
    parser.add_argument(
        "--sideband",
        choices=("upper", "lower"),
        default="upper",
    )
    parser.add_argument("--sgs-power", type=float, default=-10.0)
    parser.add_argument("--awg-channel", type=int, default=1)
    parser.add_argument("--marker-channel", type=int, default=1)
    parser.add_argument("--adc-channel", default="CHB")
    parser.add_argument("--awg-amplitude-vpp", type=float, default=0.5)
    parser.add_argument("--scan-gain", type=float, default=0.15)
    parser.add_argument("--load-gain", type=float, default=0.35)
    parser.add_argument(
        "--tomo-drive-gain",
        type=float,
        default=0.15,
        help=(
            "AWG gain for the signal state. CALIBRATE: use the weakest drive "
            "that prepares the intended transmon emission."
        ),
    )
    parser.add_argument(
        "--tomo-drive-ns",
        type=float,
        default=1500.0,
        help="Signal-drive duration. Must cover the selected boxcar mode.",
    )
    parser.add_argument(
        "--tomo-mode",
        choices=("boxcar", "decay"),
        default="boxcar",
        help=(
            "boxcar for continuously driven fluorescence; decay for a "
            "released pulse after drive-off."
        ),
    )
    parser.add_argument(
        "--tomo-mode-start-ns",
        type=float,
        default=700.0,
        help=(
            "Mode start after received marker. CALIBRATE from averaged traces; "
            "this includes cable, mixer, filter, and trigger delay."
        ),
    )
    parser.add_argument(
        "--tomo-mode-ns",
        type=float,
        default=600.0,
        help="Temporal-mode window length.",
    )
    parser.add_argument(
        "--tomo-decay-ns",
        type=float,
        default=None,
        help=(
            "Amplitude decay time for --tomo-mode decay. CALIBRATE from the "
            "complex ring-down fit; do not substitute the energy lifetime."
        ),
    )
    parser.add_argument(
        "--photon-scale-volts",
        type=float,
        default=None,
        help=(
            "Calibrated temporal-mode volts per sqrt(photon). Required for "
            "single-HEMT density-matrix and Wigner reconstruction."
        ),
    )
    parser.add_argument("--scan-pulse-ns", type=float, default=600.0)
    parser.add_argument("--scan-span", type=float, default=30e6)
    parser.add_argument("--scan-points", type=int, default=61)
    parser.add_argument("--scan-shots", type=int, default=80)
    parser.add_argument(
        "--load-time-constants",
        type=float,
        default=5.0,
        help="Length of the truncated rising exponential in amplitude lifetimes.",
    )
    parser.add_argument("--load-duration-ns", type=float, default=None)
    parser.add_argument("--ringdown-guard-ns", type=float, default=8.0)
    parser.add_argument("--ringdown-fit-ns", type=float, default=250.0)
    parser.add_argument("--readout-ns", type=float, default=1500.0)
    parser.add_argument("--acquire-ns", type=float, default=2304.0)
    parser.add_argument("--marker-padding-ns", type=float, default=500.0)
    parser.add_argument("--trigger-delay-ns", type=float, default=580.0)
    parser.add_argument("--moving-average-ns", type=float, default=20.0)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument(
        "--total-shots",
        type=int,
        default=1_000_000,
        help="Shots per state for stream mode.",
    )
    parser.add_argument(
        "--chunk-shots",
        type=int,
        default=2_000,
        help="Shots per state acquired and processed before raw traces are discarded.",
    )
    parser.add_argument(
        "--pilot-shots",
        type=int,
        default=1_000,
        help="Initial shots per state used to determine the measured temporal mode.",
    )
    parser.add_argument("--moment-order", type=int, default=4)
    parser.add_argument("--cutoff", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument(
        "--analysis-shots",
        type=int,
        default=200_000,
        help="Maximum samples per state used by ideal-heterodyne ML.",
    )
    parser.add_argument("--checkpoint-chunks", type=int, default=5)
    parser.add_argument("--keep-qa-traces", type=int, default=32)
    parser.add_argument(
        "--resume-dir",
        type=Path,
        default=None,
        help="Resume stream mode in an existing run directory.",
    )
    parser.add_argument("--final-delay-us", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("tomodata"))
    parser.add_argument("--scan-file", type=Path, default=None)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Stream run directory for analyze-stream mode.",
    )
    parser.add_argument(
        "--mode-file",
        type=Path,
        default=None,
        help="Optional calibrated temporal mode (.npy or tomography .npz).",
    )
    parser.add_argument(
        "--leave-rf-on",
        action="store_true",
        help="Do not turn the RF source off on exit.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.adc_channel.upper() != "CHB":
        raise ValueError("This experiment is intentionally locked to Alazar CHB")
    if args.scan_points < 9:
        raise ValueError("scan-points must be at least 9")
    if args.scan_span <= 0:
        raise ValueError("scan-span must be positive")
    if args.acquire_ns < args.readout_ns:
        raise ValueError("acquire-ns must be at least readout-ns")
    if args.load_time_constants <= 0:
        raise ValueError("load-time-constants must be positive")
    if (
        not 0 < args.scan_gain <= 1
        or not 0 < args.load_gain <= 1
        or not 0 < args.tomo_drive_gain <= 1
    ):
        raise ValueError("scan-gain, load-gain, and tomo-drive-gain must be in (0, 1]")
    if args.total_shots < 1 or args.chunk_shots < 1 or args.pilot_shots < 1:
        raise ValueError("stream shot counts must be positive")
    if args.chunk_shots > args.total_shots:
        raise ValueError("chunk-shots cannot exceed total-shots")
    if not 1 <= args.moment_order <= 8:
        raise ValueError("moment-order must be between 1 and 8")
    if args.checkpoint_chunks < 1 or args.keep_qa_traces < 0:
        raise ValueError("checkpoint-chunks must be positive and keep-qa-traces nonnegative")
    if args.tomo_drive_ns <= 0 or args.tomo_mode_start_ns < 0 or args.tomo_mode_ns <= 0:
        raise ValueError("tomography drive/mode timing is invalid")
    if args.tomo_mode_start_ns + args.tomo_mode_ns > args.acquire_ns:
        raise ValueError("tomography mode extends beyond acquire-ns")
    if args.tomo_mode == "decay" and (
        args.tomo_decay_ns is None or args.tomo_decay_ns <= 0
    ):
        raise ValueError("--tomo-mode decay requires positive --tomo-decay-ns")
    if args.photon_scale_volts is not None and args.photon_scale_volts <= 0:
        raise ValueError("--photon-scale-volts must be positive")
    if args.resume_dir is not None and args.mode not in {"stream", "hemt-stream"}:
        raise ValueError("--resume-dir is only valid in streaming modes")
    if args.mode in {"analyze-stream", "analyze-hemt"} and args.data_dir is None:
        raise ValueError(f"{args.mode} requires --data-dir")
    if args.cutoff < 2 or args.iterations < 1 or args.analysis_shots < 1:
        raise ValueError("analysis cutoff, iterations, and shots must be positive")


def create_run_directory(root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root.resolve() / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(path: Path, values: dict[str, Any]) -> None:
    path.write_text(json.dumps(values, indent=2), encoding="utf-8")


def json_safe_arguments(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def sideband_sign(sideband: str) -> float:
    return -1.0 if sideband == "upper" else 1.0


def lo_for_cavity(
    cavity_frequency_hz: float,
    if_frequency_hz: float,
    sideband: str,
) -> float:
    return cavity_frequency_hz + sideband_sign(sideband) * if_frequency_hz


def frequency_axis(center_hz: float, span_hz: float, points: int) -> FloatArray:
    return np.linspace(center_hz - span_hz / 2, center_hz + span_hz / 2, points)


def mean_difference(
    reference_traces: npt.ArrayLike,
    signal_traces: npt.ArrayLike,
) -> ComplexArray:
    reference = np.asarray(reference_traces, dtype=np.complex128)
    signal = np.asarray(signal_traces, dtype=np.complex128)
    return np.asarray(np.mean(signal, axis=0) - np.mean(reference, axis=0))


def average_window(
    traces: npt.ArrayLike,
    start_sample: int,
    stop_sample: int,
) -> ComplexArray:
    values = np.asarray(traces, dtype=np.complex128)
    return np.asarray(np.mean(values[:, start_sample:stop_sample], axis=1))


def steady_state_window(
    marker_padding_s: float,
    pulse_length_s: float,
    sample_rate_hz: float,
    fraction: float = 0.25,
) -> tuple[int, int]:
    pulse_stop = marker_padding_s + pulse_length_s
    pulse_start = pulse_stop - fraction * pulse_length_s
    return (
        int(round(pulse_start * sample_rate_hz)),
        int(round(pulse_stop * sample_rate_hz)),
    )


def complex_resonator_response(
    frequencies_hz: npt.ArrayLike,
    resonance_hz: float,
    linewidth_hz: float,
    background: complex,
    slope_per_hz: complex,
    resonant_amplitude: complex,
) -> ComplexArray:
    frequencies = np.asarray(frequencies_hz, dtype=float)
    detuning = frequencies - resonance_hz
    pole = resonant_amplitude / (1.0 + 2.0j * detuning / linewidth_hz)
    return np.asarray(background + slope_per_hz * detuning + pole)


def half_height_linewidth_guess(
    frequencies_hz: npt.ArrayLike,
    response: npt.ArrayLike,
) -> float:
    frequencies = np.asarray(frequencies_hz, dtype=float)
    values = np.abs(np.asarray(response, dtype=np.complex128))
    baseline = 0.5 * (values[0] + values[-1])
    feature = np.abs(values - baseline)
    peak = int(np.argmax(feature))
    target = feature[peak] / 2.0
    selected = np.flatnonzero(feature >= target)
    if selected.size >= 2:
        return float(frequencies[selected[-1]] - frequencies[selected[0]])
    return float((frequencies[-1] - frequencies[0]) / 5.0)


def resonator_initial_parameters(
    frequencies_hz: npt.ArrayLike,
    response: npt.ArrayLike,
) -> FloatArray:
    frequencies = np.asarray(frequencies_hz, dtype=float)
    values = np.asarray(response, dtype=np.complex128)
    background = 0.5 * (values[0] + values[-1])
    feature = np.abs(values - background)
    center_index = int(np.argmax(feature))
    resonance = float(frequencies[center_index])
    linewidth = max(
        half_height_linewidth_guess(frequencies, values),
        float(np.median(np.diff(frequencies)) * 2.0),
    )
    amplitude = values[center_index] - background
    return np.asarray(
        [
            resonance,
            math.log(linewidth),
            background.real,
            background.imag,
            0.0,
            0.0,
            amplitude.real,
            amplitude.imag,
        ],
        dtype=float,
    )


def unpack_resonator_parameters(parameters: npt.ArrayLike) -> tuple[Any, ...]:
    p = np.asarray(parameters, dtype=float)
    return (
        float(p[0]),
        float(np.exp(p[1])),
        complex(p[2], p[3]),
        complex(p[4], p[5]) / 1e6,
        complex(p[6], p[7]),
    )


def complex_residual(
    prediction: npt.ArrayLike,
    observation: npt.ArrayLike,
) -> FloatArray:
    difference = np.asarray(prediction) - np.asarray(observation)
    return np.concatenate((difference.real, difference.imag))


def fit_complex_resonator(
    frequencies_hz: npt.ArrayLike,
    response: npt.ArrayLike,
) -> ResonatorFit:
    frequencies = np.asarray(frequencies_hz, dtype=float)
    values = np.asarray(response, dtype=np.complex128)
    initial = resonator_initial_parameters(frequencies, values)
    spacing = float(np.median(np.diff(frequencies)))
    span = float(frequencies[-1] - frequencies[0])

    def residual(parameters: FloatArray) -> FloatArray:
        model = complex_resonator_response(
            frequencies,
            *unpack_resonator_parameters(parameters),
        )
        return complex_residual(model, values)

    lower = np.asarray(
        [
            frequencies[0],
            math.log(max(spacing, 1.0)),
            -np.inf,
            -np.inf,
            -np.inf,
            -np.inf,
            -np.inf,
            -np.inf,
        ]
    )
    upper = np.asarray(
        [
            frequencies[-1],
            math.log(2.0 * span),
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
        ]
    )
    result = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        loss="soft_l1",
        x_scale="jac",
        max_nfev=20_000,
    )
    resonance, linewidth, background, slope, amplitude = (
        unpack_resonator_parameters(result.x)
    )
    fitted = complex_resonator_response(
        frequencies,
        resonance,
        linewidth,
        background,
        slope,
        amplitude,
    )
    kappa = 2.0 * np.pi * linewidth
    return ResonatorFit(
        resonance_hz=resonance,
        linewidth_hz=linewidth,
        kappa_rad_s=kappa,
        amplitude_lifetime_s=2.0 / kappa,
        energy_lifetime_s=1.0 / kappa,
        fitted_response=fitted,
        residual_rms=float(np.sqrt(np.mean(np.abs(fitted - values) ** 2))),
    )


def rising_exponential_envelope(
    number_of_samples: int,
    sample_rate_hz: float,
    amplitude_lifetime_s: float,
    peak: float = 1.0,
) -> FloatArray:
    if number_of_samples < 2:
        raise ValueError("rising exponential needs at least two samples")
    if sample_rate_hz <= 0 or amplitude_lifetime_s <= 0:
        raise ValueError("sample rate and lifetime must be positive")
    time_before_end = (
        np.arange(number_of_samples, dtype=float) - (number_of_samples - 1)
    ) / sample_rate_hz
    return np.asarray(peak * np.exp(time_before_end / amplitude_lifetime_s))


def normalized_mode(values: npt.ArrayLike) -> ComplexArray:
    mode = np.asarray(values, dtype=np.complex128).reshape(-1)
    norm = float(np.linalg.norm(mode))
    if norm <= np.finfo(float).eps:
        raise ValueError("mode norm is zero")
    return np.asarray(mode / norm)


def ideal_decay_mode(
    time_s: npt.ArrayLike,
    start_s: float,
    amplitude_lifetime_s: float,
    detuning_hz: float = 0.0,
) -> ComplexArray:
    time = np.asarray(time_s, dtype=float)
    relative = time - start_s
    values = np.zeros(time.size, dtype=np.complex128)
    active = relative >= 0
    values[active] = np.exp(
        -relative[active] / amplitude_lifetime_s
        + 2.0j * np.pi * detuning_hz * relative[active]
    )
    return normalized_mode(values)


def fit_complex_ringdown(
    time_s: npt.ArrayLike,
    envelope: npt.ArrayLike,
    start_s: float,
    stop_s: float,
    lifetime_guess_s: float,
) -> RingdownFit:
    time = np.asarray(time_s, dtype=float)
    values = np.asarray(envelope, dtype=np.complex128)
    mask = (time >= start_s) & (time <= stop_s)
    fit_time = time[mask]
    fit_values = values[mask]
    if fit_time.size < 10:
        raise ValueError("ring-down fit window has fewer than ten samples")
    relative = fit_time - fit_time[0]
    amplitude_guess = fit_values[0]
    phase = np.unwrap(np.angle(fit_values))
    strong = np.abs(fit_values) >= 0.2 * np.max(np.abs(fit_values))
    detuning_guess = 0.0
    if np.count_nonzero(strong) >= 3:
        detuning_guess = float(
            np.polyfit(relative[strong], phase[strong], 1)[0] / (2.0 * np.pi)
        )
    initial = np.asarray(
        [
            amplitude_guess.real,
            amplitude_guess.imag,
            math.log(lifetime_guess_s),
            detuning_guess,
            fit_values[-1].real,
            fit_values[-1].imag,
        ]
    )

    def model(parameters: FloatArray) -> ComplexArray:
        amplitude = complex(parameters[0], parameters[1])
        lifetime = float(np.exp(parameters[2]))
        detuning = float(parameters[3])
        offset = complex(parameters[4], parameters[5])
        return np.asarray(
            offset
            + amplitude
            * np.exp(
                -relative / lifetime
                + 2.0j * np.pi * detuning * relative
            )
        )

    def residual(parameters: FloatArray) -> FloatArray:
        return complex_residual(model(parameters), fit_values)

    result = least_squares(
        residual,
        initial,
        bounds=(
            np.asarray(
                [
                    -np.inf,
                    -np.inf,
                    math.log(lifetime_guess_s / 5.0),
                    -20e6,
                    -np.inf,
                    -np.inf,
                ]
            ),
            np.asarray(
                [
                    np.inf,
                    np.inf,
                    math.log(lifetime_guess_s * 5.0),
                    20e6,
                    np.inf,
                    np.inf,
                ]
            ),
        ),
        loss="soft_l1",
        x_scale="jac",
        max_nfev=20_000,
    )
    lifetime = float(np.exp(result.x[2]))
    fitted_window = model(result.x)
    fitted_full = np.full(values.size, np.nan + 1.0j * np.nan)
    fitted_full[mask] = fitted_window
    return RingdownFit(
        start_s=start_s,
        stop_s=stop_s,
        amplitude_lifetime_s=lifetime,
        linewidth_hz=1.0 / (np.pi * lifetime),
        detuning_hz=float(result.x[3]),
        complex_amplitude=complex(result.x[0], result.x[1]),
        fitted_envelope=fitted_full,
        residual_rms=float(
            np.sqrt(np.mean(np.abs(fitted_window - fit_values) ** 2))
        ),
    )


def moving_average(values: npt.ArrayLike, samples: int) -> FloatArray:
    data = np.asarray(values, dtype=float)
    if samples <= 1:
        return data.copy()
    kernel = np.ones(samples, dtype=float) / samples
    return np.convolve(data, kernel, mode="same")


def detect_ringdown_start(
    time_s: npt.ArrayLike,
    envelope: npt.ArrayLike,
    expected_drive_off_s: float,
    amplitude_lifetime_s: float,
    smoothing_s: float = 15e-9,
) -> float:
    time = np.asarray(time_s, dtype=float)
    values = np.asarray(envelope, dtype=np.complex128)
    sample_interval = float(np.median(np.diff(time)))
    smoothing_samples = max(1, int(round(smoothing_s / sample_interval)))
    smoothed_amplitude = moving_average(np.abs(values), smoothing_samples)
    search_stop = expected_drive_off_s + max(
        4.0 * amplitude_lifetime_s,
        250e-9,
    )
    mask = (time >= expected_drive_off_s) & (time <= search_stop)
    if np.count_nonzero(mask) < 3:
        raise ValueError("ring-down arrival search window is outside the trace")
    indices = np.flatnonzero(mask)
    return float(time[indices[np.argmax(smoothed_amplitude[indices])]])


def phase_with_amplitude_mask(
    values: npt.ArrayLike,
    threshold_fraction: float = 0.1,
) -> FloatArray:
    complex_values = np.asarray(values, dtype=np.complex128)
    amplitude = np.abs(complex_values)
    phase = np.unwrap(np.angle(complex_values))
    finite = np.isfinite(amplitude)
    if not np.any(finite):
        return np.full(amplitude.shape, np.nan)
    threshold = threshold_fraction * np.nanmax(amplitude)
    phase[(~finite) | (amplitude < threshold)] = np.nan
    return phase


def mode_overlap(first: npt.ArrayLike, second: npt.ArrayLike) -> float:
    left = normalized_mode(first)
    right = normalized_mode(second)
    return float(abs(np.vdot(left, right)) ** 2)


def complex_moment_sums(
    samples: npt.ArrayLike,
    maximum_order: int,
) -> ComplexArray:
    values = np.asarray(samples, dtype=np.complex128).reshape(-1)
    powers = np.vstack([values**order for order in range(maximum_order + 1)])
    return np.asarray(np.conjugate(powers) @ powers.T, dtype=np.complex128)


def complex_moments(
    moment_sums: npt.ArrayLike,
    sample_count: int,
) -> ComplexArray:
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    return np.asarray(moment_sums, dtype=np.complex128) / sample_count


def stream_diagnostics(
    reference_samples: npt.ArrayLike,
    signal_samples: npt.ArrayLike,
) -> dict[str, float]:
    reference = np.asarray(reference_samples, dtype=np.complex128)
    signal = np.asarray(signal_samples, dtype=np.complex128)
    reference_power = float(np.mean(np.abs(reference) ** 2))
    signal_power = float(np.mean(np.abs(signal) ** 2))
    return {
        "reference_mean_real": float(np.mean(reference).real),
        "reference_mean_imag": float(np.mean(reference).imag),
        "signal_mean_real": float(np.mean(signal).real),
        "signal_mean_imag": float(np.mean(signal).imag),
        "reference_second_moment": reference_power,
        "signal_second_moment": signal_power,
        "second_moment_difference": signal_power - reference_power,
    }


def stream_diagnostics_from_sums(
    reference_moment_sums: npt.ArrayLike,
    signal_moment_sums: npt.ArrayLike,
    sample_count: int,
) -> dict[str, float]:
    reference = complex_moments(reference_moment_sums, sample_count)
    signal = complex_moments(signal_moment_sums, sample_count)
    return {
        "reference_mean_real": float(reference[0, 1].real),
        "reference_mean_imag": float(reference[0, 1].imag),
        "signal_mean_real": float(signal[0, 1].real),
        "signal_mean_imag": float(signal[0, 1].imag),
        "reference_second_moment": float(reference[1, 1].real),
        "signal_second_moment": float(signal[1, 1].real),
        "second_moment_difference": float(
            (signal[1, 1] - reference[1, 1]).real
        ),
    }


def release_acquisition_chunk(experiment: AWGAlazar, *objects: Any) -> None:
    for name in (
        "last_downconverted_iq",
        "last_records_volts",
        "last_raw_codes",
        "last_shot_iq",
    ):
        if hasattr(experiment, name):
            setattr(experiment, name, None)
    del objects
    gc.collect()


def split_states(result: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states = list(result.axis("state"))
    reference_index = states.index("reference")
    signal_index = states.index("signal")
    return (
        result.iq_traces[:, reference_index, :],
        result.iq_traces[:, signal_index, :],
        result.raw[:, reference_index, :],
        result.raw[:, signal_index, :],
    )


class ScanProgram(ExperimentProgram):
    def _initialize(self, cfg: dict[str, Any]) -> None:
        self.state = self.add_sweep("state", ValuesSweep(("reference", "signal")))
        self.declare_gen(
            "cavity",
            ch=cfg["awg_channel"],
            amplitude_vpp=cfg["awg_amplitude_vpp"],
        )
        self.declare_readout(
            "ro",
            adc_channel="CHB",
            length=cfg["readout_length"],
            demod_freq=cfg["if_frequency"],
            waveform_ch=cfg["awg_channel"],
            marker_channel=cfg["marker_channel"],
            marker_padding=cfg["marker_padding"],
            integrate_time=cfg["readout_length"],
        )
        for name, gain in (("reference", 0.0), ("signal", cfg["scan_gain"])):
            self.add_pulse(
                name,
                gen="cavity",
                style="gaussian_square",
                length=cfg["scan_pulse_length"],
                edge_sigma=20 * ns,
                frequency=cfg["if_frequency"],
                gain=gain,
                readout=True,
            )

    def _body(self, cfg: dict[str, Any]) -> None:
        self.play("reference", at=0, when=("state", "reference"))
        self.play("signal", at=0, when=("state", "signal"))
        self.trigger("ro", trigger_delay=cfg["trigger_delay"])


class RisingExponentialProgram(ExperimentProgram):
    def _initialize(self, cfg: dict[str, Any]) -> None:
        self.state = self.add_sweep("state", ValuesSweep(("reference", "signal")))
        self.declare_gen(
            "cavity",
            ch=cfg["awg_channel"],
            amplitude_vpp=cfg["awg_amplitude_vpp"],
        )
        self.declare_readout(
            "ro",
            adc_channel="CHB",
            length=cfg["readout_length"],
            demod_freq=cfg["if_frequency"],
            waveform_ch=cfg["awg_channel"],
            marker_channel=cfg["marker_channel"],
            marker_padding=cfg["marker_padding"],
            integrate_time=cfg["readout_length"],
        )
        for name, gain in (("reference", 0.0), ("signal", cfg["load_gain"])):
            self.add_pulse(
                name,
                gen="cavity",
                style="exponential",
                length=cfg["load_duration"],
                decay=cfg["energy_lifetime"],
                frequency=cfg["if_frequency"],
                gain=gain,
                readout=True,
            )

    def _body(self, cfg: dict[str, Any]) -> None:
        self.play("reference", at=0, when=("state", "reference"))
        self.play("signal", at=0, when=("state", "signal"))
        self.trigger("ro", trigger_delay=cfg["trigger_delay"])

    @staticmethod
    def _render_pulse(
        pulse: Any,
        sample_rate_hz: float,
        amplitude_vpp: float,
    ) -> FloatArray:
        count = max(
            2,
            int(round((pulse.stop_s - pulse.start_s) * sample_rate_hz)),
        )
        if pulse.definition.style != "exponential":
            return ExperimentProgram._render_pulse(
                pulse,
                sample_rate_hz,
                amplitude_vpp,
            )
        if pulse.decay_s is None or pulse.decay_s <= 0:
            raise ValueError("Rising exponential requires decay > 0")
        amplitude_lifetime = 2.0 * pulse.decay_s
        envelope = rising_exponential_envelope(
            count,
            sample_rate_hz,
            amplitude_lifetime,
            peak=pulse.gain,
        )
        waveform = modulate_envelope(
            envelope,
            sample_rate_hz,
            pulse.frequency_hz,
            pulse.phase_radians,
        )
        return np.asarray(waveform * (amplitude_vpp / 2.0))


class SingleHemtTomographyProgram(ExperimentProgram):
    """Alternate vacuum-reference and driven-signal records."""

    def _initialize(self, cfg: dict[str, Any]) -> None:
        self.state = self.add_sweep("state", ValuesSweep(("reference", "signal")))
        self.declare_gen(
            "drive",
            ch=cfg["awg_channel"],
            amplitude_vpp=cfg["awg_amplitude_vpp"],
        )
        self.declare_readout(
            "ro",
            adc_channel="CHB",
            length=cfg["readout_length"],
            demod_freq=cfg["if_frequency"],
            waveform_ch=cfg["awg_channel"],
            marker_channel=cfg["marker_channel"],
            marker_padding=cfg["marker_padding"],
            integrate_time=cfg["readout_length"],
        )
        for name, gain in (("reference", 0.0), ("signal", cfg["tomo_drive_gain"])):
            self.add_pulse(
                name,
                gen="drive",
                style="const",
                length=cfg["tomo_drive_length"],
                frequency=cfg["if_frequency"],
                gain=gain,
                readout=True,
            )

    def _body(self, cfg: dict[str, Any]) -> None:
        self.play("reference", at=0, when=("state", "reference"))
        self.play("signal", at=0, when=("state", "signal"))
        self.trigger("ro", trigger_delay=cfg["trigger_delay"])


def common_program_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "awg_channel": args.awg_channel,
        "marker_channel": args.marker_channel,
        "awg_amplitude_vpp": args.awg_amplitude_vpp,
        "if_frequency": args.if_frequency,
        "readout_length": args.readout_ns * ns,
        "marker_padding": args.marker_padding_ns * ns,
        "trigger_delay": args.trigger_delay_ns * ns,
        "scan_gain": args.scan_gain,
        "scan_pulse_length": args.scan_pulse_ns * ns,
        "tomo_drive_gain": args.tomo_drive_gain,
        "tomo_drive_length": args.tomo_drive_ns * ns,
    }


def connect_hardware(
    args: argparse.Namespace,
) -> tuple[RohdeSchwarzSGS100A, AWGAlazar]:
    source = RohdeSchwarzSGS100A(args.sgs_address)
    try:
        source.off()
        source.power = args.sgs_power
        source.IQ_state = "on"
        source.pulsemod_state = "off"
        source.configure_lo_output(True, mode="LO")
        experiment = AWGAlazar.connect(
            args.awg_resource,
            awg_sample_rate_hz=2.5e9,
            alazar_sample_rate_hz=1e9,
            acquire_window_s=args.acquire_ns * ns,
            trigger_slope="rising",
            trigger_level=140,
            tone_frequency_hz=args.if_frequency,
            integrate_window_ns=(0.0, args.readout_ns),
            adc_channel="CHB",
            moving_average_time_s=args.moving_average_ns * ns,
            baseline_time_s=100 * ns,
            timeout_ms=30_000,
        )
    except BaseException:
        source.close()
        raise
    return source, experiment


def acquire_frequency_scan(
    args: argparse.Namespace,
    source: RohdeSchwarzSGS100A,
    experiment: AWGAlazar,
    run_dir: Path,
) -> tuple[FloatArray, ComplexArray, ResonatorFit]:
    frequencies = frequency_axis(
        args.cavity_frequency,
        args.scan_span,
        args.scan_points,
    )
    config = common_program_config(args)
    program = ScanProgram(config, final_delay_s=args.final_delay_us * us)
    compiled = program.compile(hardware=experiment)
    start, stop = steady_state_window(
        args.marker_padding_ns * ns,
        args.scan_pulse_ns * ns,
        experiment.alazar_sample_rate_hz,
    )
    responses = np.empty(frequencies.size, dtype=np.complex128)
    reference_means = []
    signal_means = []

    for index, frequency in enumerate(frequencies):
        source.off()
        source.frequency = lo_for_cavity(
            float(frequency),
            args.if_frequency,
            args.sideband,
        )
        source.on()
        result = compiled.acquire(n_average=args.scan_shots)
        reference, signal, _, _ = split_states(result)
        reference_means.append(np.mean(reference, axis=0))
        signal_means.append(np.mean(signal, axis=0))
        reference_samples = average_window(reference, start, stop)
        signal_samples = average_window(signal, start, stop)
        responses[index] = np.mean(signal_samples) - np.mean(reference_samples)
        print(
            f"scan {index + 1:02d}/{frequencies.size}: "
            f"{frequency / 1e9:.9f} GHz, "
            f"|response|={abs(responses[index]) * 1e3:.5f} mV"
        )

    fit = fit_complex_resonator(frequencies, responses)
    np.savez_compressed(
        run_dir / "01_frequency_scan.npz",
        cavity_frequency_hz=frequencies,
        complex_response=responses,
        fitted_response=fit.fitted_response,
        reference_mean_iq=np.asarray(reference_means),
        signal_mean_iq=np.asarray(signal_means),
        steady_state_start_sample=start,
        steady_state_stop_sample=stop,
        iq_time_s=result.iq_time_s,
    )
    save_json(
        run_dir / "01_frequency_scan.json",
        {
            "resonance_hz": fit.resonance_hz,
            "linewidth_hz": fit.linewidth_hz,
            "kappa_rad_s": fit.kappa_rad_s,
            "amplitude_lifetime_s": fit.amplitude_lifetime_s,
            "energy_lifetime_s": fit.energy_lifetime_s,
            "residual_rms": fit.residual_rms,
            "scan_center_hz": args.cavity_frequency,
            "scan_span_hz": args.scan_span,
            "scan_points": args.scan_points,
            "scan_shots": args.scan_shots,
            "scan_gain": args.scan_gain,
            "sgs_power_dbm": args.sgs_power,
        },
    )
    plot_frequency_scan(run_dir / "01_frequency_scan.png", frequencies, responses, fit)
    return frequencies, responses, fit


def plot_frequency_scan(
    path: Path,
    frequencies_hz: npt.ArrayLike,
    response: npt.ArrayLike,
    fit: ResonatorFit,
) -> None:
    frequencies = np.asarray(frequencies_hz) / 1e9
    values = np.asarray(response)
    fitted = fit.fitted_response
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(frequencies, np.abs(values) * 1e3, "o", label="data")
    axes[0, 0].plot(frequencies, np.abs(fitted) * 1e3, "-", label="complex fit")
    axes[0, 0].set(xlabel="Cavity frequency (GHz)", ylabel="|response| (mV)")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    axes[0, 1].plot(frequencies, np.unwrap(np.angle(values)), "o")
    axes[0, 1].plot(frequencies, np.unwrap(np.angle(fitted)), "-")
    axes[0, 1].set(xlabel="Cavity frequency (GHz)", ylabel="Unwrapped phase (rad)")
    axes[0, 1].grid(alpha=0.3)
    axes[1, 0].plot(values.real * 1e3, values.imag * 1e3, "o", label="data")
    axes[1, 0].plot(fitted.real * 1e3, fitted.imag * 1e3, "-", label="fit")
    axes[1, 0].set(xlabel="Re(response) (mV)", ylabel="Im(response) (mV)")
    axes[1, 0].axis("equal")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    residual = values - fitted
    axes[1, 1].plot(frequencies, np.abs(residual) * 1e3, "o-")
    axes[1, 1].set(xlabel="Cavity frequency (GHz)", ylabel="|residual| (mV)")
    axes[1, 1].grid(alpha=0.3)
    fig.suptitle(
        f"Complex cavity fit: f0={fit.resonance_hz / 1e9:.9f} GHz, "
        f"kappa/2pi={fit.linewidth_hz / 1e6:.3f} MHz"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def choose_load_duration_s(
    args: argparse.Namespace,
    amplitude_lifetime_s: float,
) -> float:
    if args.load_duration_ns is not None:
        return args.load_duration_ns * ns
    return args.load_time_constants * amplitude_lifetime_s


def plot_loading_pulse(
    path: Path,
    load_duration_s: float,
    amplitude_lifetime_s: float,
    sample_rate_hz: float,
) -> None:
    count = int(round(load_duration_s * sample_rate_hz))
    envelope = rising_exponential_envelope(
        count,
        sample_rate_hz,
        amplitude_lifetime_s,
    )
    time_ns = np.arange(count) / sample_rate_hz * 1e9
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(time_ns, envelope)
    axes[0].set(ylabel="Normalized input amplitude")
    axes[0].grid(alpha=0.3)
    axes[1].plot(time_ns, envelope**2)
    axes[1].set(xlabel="Time from loading start (ns)", ylabel="Normalized input power")
    axes[1].grid(alpha=0.3)
    fig.suptitle(
        f"Truncated time-reversed loading pulse; "
        f"tau_amp={amplitude_lifetime_s * 1e9:.2f} ns"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def acquire_rising_exponential(
    args: argparse.Namespace,
    source: RohdeSchwarzSGS100A,
    experiment: AWGAlazar,
    run_dir: Path,
    resonator_fit: ResonatorFit,
) -> tuple[Any, RingdownFit, ComplexArray]:
    load_duration_s = choose_load_duration_s(
        args,
        resonator_fit.amplitude_lifetime_s,
    )
    config = common_program_config(args) | {
        "load_gain": args.load_gain,
        "load_duration": load_duration_s,
        "energy_lifetime": resonator_fit.energy_lifetime_s,
    }
    plot_loading_pulse(
        run_dir / "02_loading_pulse.png",
        load_duration_s,
        resonator_fit.amplitude_lifetime_s,
        experiment.awg_sample_rate_hz,
    )
    save_json(
        run_dir / "02_loading_pulse.json",
        {
            "load_duration_s": load_duration_s,
            "load_time_constants": load_duration_s
            / resonator_fit.amplitude_lifetime_s,
            "amplitude_lifetime_s": resonator_fit.amplitude_lifetime_s,
            "energy_lifetime_s": resonator_fit.energy_lifetime_s,
            "load_gain": args.load_gain,
        },
    )
    program = RisingExponentialProgram(
        config,
        final_delay_s=args.final_delay_us * us,
    )
    compiled = program.compile(hardware=experiment)
    source.off()
    source.frequency = lo_for_cavity(
        resonator_fit.resonance_hz,
        args.if_frequency,
        args.sideband,
    )
    source.on()
    result = compiled.acquire(n_average=args.shots)
    reference, signal, raw_reference, raw_signal = split_states(result)
    measured_envelope = mean_difference(reference, signal)
    pulse_end_s = args.marker_padding_ns * ns + load_duration_s
    detected_arrival_s = detect_ringdown_start(
        result.iq_time_s,
        measured_envelope,
        pulse_end_s,
        resonator_fit.amplitude_lifetime_s,
    )
    fit_start_s = detected_arrival_s + args.ringdown_guard_ns * ns
    fit_stop_s = min(
        fit_start_s + args.ringdown_fit_ns * ns,
        float(result.iq_time_s[-1]),
    )
    ringdown_fit = fit_complex_ringdown(
        result.iq_time_s,
        measured_envelope,
        fit_start_s,
        fit_stop_s,
        resonator_fit.amplitude_lifetime_s,
    )
    measured_mode_values = measured_envelope.copy()
    measured_mode_values[
        (result.iq_time_s < fit_start_s) | (result.iq_time_s > fit_stop_s)
    ] = 0.0
    measured_mode = normalized_mode(measured_mode_values)
    ideal_mode = ideal_decay_mode(
        result.iq_time_s,
        fit_start_s,
        resonator_fit.amplitude_lifetime_s,
        ringdown_fit.detuning_hz,
    )
    overlap = mode_overlap(measured_mode, ideal_mode)
    reference_samples = project_temporal_mode(reference, measured_mode)
    signal_samples = project_temporal_mode(signal, measured_mode)
    np.savez_compressed(
        run_dir / "03_ringdown_acquisition.npz",
        raw_time_s=result.raw_time_s,
        iq_time_s=result.iq_time_s,
        reference_raw=raw_reference,
        signal_raw=raw_signal,
        reference_iq=reference,
        signal_iq=signal,
        measured_envelope=measured_envelope,
        ringdown_fit_envelope=ringdown_fit.fitted_envelope,
        measured_temporal_mode=measured_mode,
        ideal_temporal_mode=ideal_mode,
        reference_mode_samples=reference_samples,
        signal_mode_samples=signal_samples,
    )
    save_json(
        run_dir / "03_ringdown_acquisition.json",
        {
            "resonator_scan_fit": serializable_resonator_fit(resonator_fit),
            "ringdown_fit": serializable_ringdown_fit(ringdown_fit),
            "load_duration_s": load_duration_s,
            "pulse_end_s": pulse_end_s,
            "detected_ringdown_arrival_s": detected_arrival_s,
            "measured_chain_delay_s": detected_arrival_s - pulse_end_s,
            "fit_start_s": fit_start_s,
            "fit_stop_s": fit_stop_s,
            "mode_overlap_squared": overlap,
            "shots_per_state": args.shots,
            "sgs_lo_hz": source.frequency,
            "sgs_power_dbm": source.power,
        },
    )
    plot_ringdown(
        run_dir / "03_ringdown_fit.png",
        result.iq_time_s,
        measured_envelope,
        ringdown_fit,
        pulse_end_s,
        detected_arrival_s,
    )
    plot_temporal_modes(
        run_dir / "04_temporal_modes.png",
        result.iq_time_s,
        measured_mode,
        ideal_mode,
        overlap,
    )
    plot_mode_samples(
        run_dir / "05_mode_projected_iq.png",
        reference_samples,
        signal_samples,
    )
    return result, ringdown_fit, measured_mode


def rising_exponential_compiled_program(
    args: argparse.Namespace,
    experiment: AWGAlazar,
    resonator_fit: ResonatorFit,
) -> tuple[Any, float]:
    load_duration_s = choose_load_duration_s(
        args,
        resonator_fit.amplitude_lifetime_s,
    )
    config = common_program_config(args) | {
        "load_gain": args.load_gain,
        "load_duration": load_duration_s,
        "energy_lifetime": resonator_fit.energy_lifetime_s,
    }
    program = RisingExponentialProgram(
        config,
        final_delay_s=args.final_delay_us * us,
    )
    return program.compile(hardware=experiment), load_duration_s


def prepare_stream_mode(
    args: argparse.Namespace,
    compiled: Any,
    experiment: AWGAlazar,
    run_dir: Path,
    resonator_fit: ResonatorFit,
    load_duration_s: float,
) -> tuple[ComplexArray, Any, np.ndarray, np.ndarray]:
    pilot_count = min(args.pilot_shots, args.total_shots)
    result = compiled.acquire(n_average=pilot_count)
    reference, signal, raw_reference, raw_signal = split_states(result)
    measured_envelope = mean_difference(reference, signal)
    pulse_end_s = args.marker_padding_ns * ns + load_duration_s
    if args.mode_file is None:
        arrival_s = detect_ringdown_start(
            result.iq_time_s,
            measured_envelope,
            pulse_end_s,
            resonator_fit.amplitude_lifetime_s,
        )
        start_s = arrival_s + args.ringdown_guard_ns * ns
        stop_s = min(
            start_s + args.ringdown_fit_ns * ns,
            float(result.iq_time_s[-1]),
        )
        mode_values = measured_envelope.copy()
        mode_values[
            (result.iq_time_s < start_s) | (result.iq_time_s > stop_s)
        ] = 0.0
        mode = normalized_mode(mode_values)
        mode_source = "pilot coherent mean envelope"
    else:
        mode = load_temporal_mode(args.mode_file)
        if mode.size != result.iq_time_s.size:
            raise ValueError(
                "mode-file sample count does not match acquired IQ traces"
            )
        active = np.flatnonzero(np.abs(mode) > 0)
        if active.size == 0:
            raise ValueError("mode-file contains no active samples")
        start_s = float(result.iq_time_s[active[0]])
        stop_s = float(result.iq_time_s[active[-1]])
        arrival_s = start_s
        mode_source = str(args.mode_file.resolve())
    reference_samples = project_temporal_mode(reference, mode)
    signal_samples = project_temporal_mode(signal, mode)
    np.save(run_dir / "stream_temporal_mode.npy", mode)
    np.save(run_dir / "stream_iq_time_s.npy", result.iq_time_s)
    qa_count = min(args.keep_qa_traces, pilot_count)
    if qa_count:
        np.savez_compressed(
            run_dir / "stream_qa_traces.npz",
            iq_time_s=result.iq_time_s,
            reference_iq=reference[:qa_count],
            signal_iq=signal[:qa_count],
            reference_raw=raw_reference[:qa_count],
            signal_raw=raw_signal[:qa_count],
            mean_envelope=measured_envelope,
            temporal_mode=mode,
        )
    save_json(
        run_dir / "stream_mode_metadata.json",
        {
            "pilot_shots_per_state": pilot_count,
            "pulse_end_s": pulse_end_s,
            "detected_arrival_s": arrival_s,
            "chain_delay_s": arrival_s - pulse_end_s,
            "mode_start_s": start_s,
            "mode_stop_s": stop_s,
            "qa_traces_kept_per_state": qa_count,
            "mode_source": mode_source,
        },
    )
    return mode, result, reference_samples, signal_samples


def load_temporal_mode(path: Path) -> ComplexArray:
    if path.suffix.lower() == ".npy":
        return normalized_mode(np.load(path))
    data = np.load(path)
    for key in (
        "measured_temporal_mode",
        "stream_temporal_mode",
        "mean_temporal_mode",
    ):
        if key in data:
            return normalized_mode(data[key])
    raise KeyError(
        "mode npz must contain measured_temporal_mode, "
        "stream_temporal_mode, or mean_temporal_mode"
    )


def tomography_temporal_mode(
    args: argparse.Namespace,
    iq_time_s: npt.ArrayLike,
) -> ComplexArray:
    """Build the calibrated mode used for every signal/reference shot."""
    time = np.asarray(iq_time_s, dtype=float)
    if args.mode_file is not None:
        mode = load_temporal_mode(args.mode_file)
        if mode.size != time.size:
            raise ValueError("mode-file sample count does not match acquired IQ traces")
        return mode

    start_s = args.tomo_mode_start_ns * ns
    stop_s = start_s + args.tomo_mode_ns * ns
    sample_interval_s = float(np.median(np.diff(time)))
    start_sample = int(round((start_s - time[0]) / sample_interval_s))
    stop_sample = start_sample + int(
        round(args.tomo_mode_ns * ns / sample_interval_s)
    )
    if not 0 <= start_sample < stop_sample <= time.size:
        raise ValueError("tomography mode is outside the acquired IQ trace")
    if stop_sample - start_sample < 2:
        raise ValueError("tomography mode contains fewer than two ADC samples")
    values = np.zeros(time.size, dtype=np.complex128)
    if args.tomo_mode == "boxcar":
        # CALIBRATE: start after drive transients and stop before drive-off.
        drive_start_s = args.marker_padding_ns * ns
        drive_stop_s = drive_start_s + args.tomo_drive_ns * ns
        if start_s < drive_start_s or stop_s > drive_stop_s:
            raise ValueError("boxcar mode must lie inside the driven pulse")
        values[start_sample:stop_sample] = 1.0
    else:
        # CALIBRATE: amplitude lifetime, not energy lifetime (tau_amp = 2*tau_energy).
        values[start_sample:stop_sample] = np.exp(
            -(time[start_sample:stop_sample] - start_s)
            / (args.tomo_decay_ns * ns)
        )
    return normalized_mode(values)


def create_stream_sample_files(
    run_dir: Path,
    total_shots: int,
) -> tuple[np.memmap, np.memmap]:
    reference = np.lib.format.open_memmap(
        run_dir / "reference_mode_samples.npy",
        mode="w+",
        dtype=np.complex128,
        shape=(total_shots,),
    )
    signal = np.lib.format.open_memmap(
        run_dir / "signal_mode_samples.npy",
        mode="w+",
        dtype=np.complex128,
        shape=(total_shots,),
    )
    return reference, signal


def open_stream_sample_files(
    run_dir: Path,
    total_shots: int,
) -> tuple[np.memmap, np.memmap]:
    reference = np.load(
        run_dir / "reference_mode_samples.npy",
        mmap_mode="r+",
    )
    signal = np.load(
        run_dir / "signal_mode_samples.npy",
        mmap_mode="r+",
    )
    if reference.shape != (total_shots,) or signal.shape != (total_shots,):
        raise ValueError("resume files do not match --total-shots")
    return reference, signal


def save_stream_checkpoint(
    run_dir: Path,
    completed_shots: int,
    total_shots: int,
    reference_moment_sums: npt.ArrayLike,
    signal_moment_sums: npt.ArrayLike,
    convergence: list[dict[str, float]],
) -> None:
    np.savez_compressed(
        run_dir / "stream_moments.npz",
        completed_shots=completed_shots,
        reference_moment_sums=reference_moment_sums,
        signal_moment_sums=signal_moment_sums,
        reference_moments=complex_moments(
            reference_moment_sums,
            completed_shots,
        ),
        signal_moments=complex_moments(
            signal_moment_sums,
            completed_shots,
        ),
    )
    save_json(
        run_dir / "stream_checkpoint.json",
        {
            "completed_shots_per_state": completed_shots,
            "total_shots_per_state": total_shots,
            "completion_fraction": completed_shots / total_shots,
            "convergence": convergence,
        },
    )
    plot_stream_convergence(
        run_dir / "stream_convergence.png",
        convergence,
    )


def load_stream_checkpoint(
    run_dir: Path,
    maximum_order: int,
) -> tuple[int, ComplexArray, ComplexArray, list[dict[str, float]]]:
    checkpoint = json.loads((run_dir / "stream_checkpoint.json").read_text())
    moment_data = np.load(run_dir / "stream_moments.npz")
    expected_shape = (maximum_order + 1, maximum_order + 1)
    reference_sums = np.asarray(moment_data["reference_moment_sums"])
    signal_sums = np.asarray(moment_data["signal_moment_sums"])
    if reference_sums.shape != expected_shape or signal_sums.shape != expected_shape:
        raise ValueError("resume moment order does not match --moment-order")
    return (
        int(checkpoint["completed_shots_per_state"]),
        reference_sums,
        signal_sums,
        list(checkpoint.get("convergence", [])),
    )


def plot_stream_convergence(
    path: Path,
    convergence: list[dict[str, float]],
) -> None:
    if not convergence:
        return
    shots = np.asarray([item["shots"] for item in convergence])
    difference = np.asarray(
        [item["second_moment_difference"] for item in convergence]
    )
    reference_power = np.asarray(
        [item["reference_second_moment"] for item in convergence]
    )
    signal_power = np.asarray(
        [item["signal_second_moment"] for item in convergence]
    )
    signal_mean = np.asarray(
        [
            complex(item["signal_mean_real"], item["signal_mean_imag"])
            for item in convergence
        ]
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(shots, reference_power, "o-", label="reference")
    axes[0, 0].plot(shots, signal_power, "o-", label="signal")
    axes[0, 0].set(xlabel="Shots per state", ylabel=r"$\langle |S|^2\rangle$")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    axes[0, 1].plot(shots, difference, "o-")
    axes[0, 1].set(
        xlabel="Shots per state",
        ylabel="Signal-reference second moment",
    )
    axes[0, 1].grid(alpha=0.3)
    axes[1, 0].plot(shots, signal_mean.real, "o-", label="Re")
    axes[1, 0].plot(shots, signal_mean.imag, "o-", label="Im")
    axes[1, 0].set(xlabel="Shots per state", ylabel=r"$\langle S\rangle$")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    axes[1, 1].loglog(shots, np.abs(difference - difference[-1]) + 1e-18, "o-")
    axes[1, 1].set(
        xlabel="Shots per state",
        ylabel="Distance from latest second moment",
    )
    axes[1, 1].grid(alpha=0.3)
    fig.suptitle(f"Streaming acquisition convergence: {int(shots[-1]):,} shots/state")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def stream_large_acquisition(
    args: argparse.Namespace,
    source: RohdeSchwarzSGS100A,
    experiment: AWGAlazar,
    run_dir: Path,
    resonator_fit: ResonatorFit,
) -> None:
    compiled, load_duration_s = rising_exponential_compiled_program(
        args,
        experiment,
        resonator_fit,
    )
    source.off()
    source.frequency = lo_for_cavity(
        resonator_fit.resonance_hz,
        args.if_frequency,
        args.sideband,
    )
    source.on()

    if args.resume_dir is not None:
        mode = np.load(run_dir / "stream_temporal_mode.npy")
        reference_store, signal_store = open_stream_sample_files(
            run_dir,
            args.total_shots,
        )
        completed, reference_sums, signal_sums, convergence = (
            load_stream_checkpoint(run_dir, args.moment_order)
        )
        print(f"Resuming from {completed:,} shots/state")
    else:
        mode, pilot_result, pilot_reference, pilot_signal = prepare_stream_mode(
            args,
            compiled,
            experiment,
            run_dir,
            resonator_fit,
            load_duration_s,
        )
        reference_store, signal_store = create_stream_sample_files(
            run_dir,
            args.total_shots,
        )
        completed = pilot_reference.size
        reference_store[:completed] = pilot_reference
        signal_store[:completed] = pilot_signal
        reference_sums = complex_moment_sums(
            pilot_reference,
            args.moment_order,
        )
        signal_sums = complex_moment_sums(
            pilot_signal,
            args.moment_order,
        )
        first_diagnostic = stream_diagnostics(
            pilot_reference,
            pilot_signal,
        )
        convergence = [{"shots": completed, **first_diagnostic}]
        del pilot_reference, pilot_signal, pilot_result
        release_acquisition_chunk(experiment)
        reference_store.flush()
        signal_store.flush()
        save_stream_checkpoint(
            run_dir,
            completed,
            args.total_shots,
            reference_sums,
            signal_sums,
            convergence,
        )

    chunk_index = 0
    while completed < args.total_shots:
        count = min(args.chunk_shots, args.total_shots - completed)
        result = compiled.acquire(n_average=count)
        reference, signal, _, _ = split_states(result)
        reference_samples = project_temporal_mode(reference, mode)
        signal_samples = project_temporal_mode(signal, mode)
        stop = completed + count
        reference_store[completed:stop] = reference_samples
        signal_store[completed:stop] = signal_samples
        reference_sums += complex_moment_sums(
            reference_samples,
            args.moment_order,
        )
        signal_sums += complex_moment_sums(
            signal_samples,
            args.moment_order,
        )
        completed = stop
        chunk_index += 1
        diagnostics = stream_diagnostics_from_sums(
            reference_sums,
            signal_sums,
            completed,
        )
        convergence.append({"shots": completed, **diagnostics})
        reference_store.flush()
        signal_store.flush()
        del reference, signal, reference_samples, signal_samples, result
        release_acquisition_chunk(experiment)
        print(
            f"stream: {completed:,}/{args.total_shots:,} shots/state "
            f"({completed / args.total_shots:.1%}); raw chunk released"
        )
        if (
            chunk_index % args.checkpoint_chunks == 0
            or completed == args.total_shots
        ):
            save_stream_checkpoint(
                run_dir,
                completed,
                args.total_shots,
                reference_sums,
                signal_sums,
                convergence,
            )

    final_diagnostics = stream_diagnostics_from_sums(
        reference_sums,
        signal_sums,
        completed,
    )
    save_json(
        run_dir / "stream_summary.json",
        {
            "completed_shots_per_state": completed,
            "raw_traces_saved_per_state": args.keep_qa_traces,
            "raw_chunks_deleted_after_projection": True,
            "mode_sample_storage_bytes_per_state": completed * 16,
            "resonator_fit": serializable_resonator_fit(resonator_fit),
            **final_diagnostics,
        },
    )


def single_hemt_stream(
    args: argparse.Namespace,
    source: RohdeSchwarzSGS100A,
    experiment: AWGAlazar,
    run_dir: Path,
) -> None:
    """Acquire single-HEMT signal/reference samples without retaining raw traces."""
    print(
        "\nSingle-HEMT calibration checklist:\n"
        "  1. Verify RF/IF sideband and LO phase at the device frequency.\n"
        "  2. Null SGS IQ leakage; reference must not drive the transmon.\n"
        "  3. Use QA traces to calibrate --tomo-mode-start-ns/window.\n"
        "  4. For decay mode fit tau_amp; tau_amp = 2 * tau_energy.\n"
        "  5. Calibrate --photon-scale-volts before claiming Wigner negativity.\n"
        "  6. Measure input loss and thermal occupation separately.\n"
    )
    program = SingleHemtTomographyProgram(
        common_program_config(args),
        final_delay_s=args.final_delay_us * us,
    )
    compiled = program.compile(hardware=experiment)
    source.off()
    source.frequency = lo_for_cavity(
        args.cavity_frequency,
        args.if_frequency,
        args.sideband,
    )
    source.on()

    if args.resume_dir is None:
        reference_store, signal_store = create_stream_sample_files(
            run_dir,
            args.total_shots,
        )
        completed = 0
        reference_sums = np.zeros(
            (args.moment_order + 1, args.moment_order + 1),
            dtype=np.complex128,
        )
        signal_sums = reference_sums.copy()
        convergence: list[dict[str, float]] = []
        mode: ComplexArray | None = None
    else:
        mode = np.load(run_dir / "stream_temporal_mode.npy")
        reference_store, signal_store = open_stream_sample_files(
            run_dir,
            args.total_shots,
        )
        completed, reference_sums, signal_sums, convergence = (
            load_stream_checkpoint(run_dir, args.moment_order)
        )

    chunk_index = 0
    while completed < args.total_shots:
        count = min(args.chunk_shots, args.total_shots - completed)
        result = compiled.acquire(n_average=count)
        reference, signal, raw_reference, raw_signal = split_states(result)
        if mode is None:
            mode = tomography_temporal_mode(args, result.iq_time_s)
            np.save(run_dir / "stream_temporal_mode.npy", mode)
            np.save(run_dir / "stream_iq_time_s.npy", result.iq_time_s)
            qa_count = min(args.keep_qa_traces, count)
            if qa_count:
                np.savez_compressed(
                    run_dir / "stream_qa_traces.npz",
                    iq_time_s=result.iq_time_s,
                    reference_iq=reference[:qa_count],
                    signal_iq=signal[:qa_count],
                    reference_raw=raw_reference[:qa_count],
                    signal_raw=raw_signal[:qa_count],
                    temporal_mode=mode,
                )
            plot_hemt_mode_calibration(
                run_dir / "stream_mode_calibration.png",
                result.iq_time_s,
                reference,
                signal,
                mode,
            )
            save_json(
                run_dir / "stream_mode_metadata.json",
                {
                    "mode": args.tomo_mode,
                    "mode_start_s": args.tomo_mode_start_ns * ns,
                    "mode_stop_s": (
                        args.tomo_mode_start_ns + args.tomo_mode_ns
                    )
                    * ns,
                    "amplitude_decay_s": (
                        None
                        if args.tomo_decay_ns is None
                        else args.tomo_decay_ns * ns
                    ),
                    "mode_source": (
                        "command-line calibrated mode"
                        if args.mode_file is None
                        else str(args.mode_file.resolve())
                    ),
                    "calibration_attention": [
                        "Reference must be vacuum at the device output.",
                        "Null SGS IQ leakage so zero AWG gain does not drive the transmon.",
                        "Calibrate mode start from averaged signal-reference traces.",
                        "For decay mode use amplitude lifetime, not energy lifetime.",
                        "Calibrate volts per sqrt(photon) before interpreting Wigner negativity.",
                        "HEMT-input loss and thermal photons require a separate efficiency model.",
                    ],
                },
            )

        reference_samples = project_temporal_mode(reference, mode)
        signal_samples = project_temporal_mode(signal, mode)
        stop = completed + count
        reference_store[completed:stop] = reference_samples
        signal_store[completed:stop] = signal_samples
        reference_sums += complex_moment_sums(reference_samples, args.moment_order)
        signal_sums += complex_moment_sums(signal_samples, args.moment_order)
        completed = stop
        chunk_index += 1
        convergence.append(
            {
                "shots": completed,
                **stream_diagnostics_from_sums(
                    reference_sums,
                    signal_sums,
                    completed,
                ),
            }
        )
        reference_store.flush()
        signal_store.flush()
        del result, reference, signal, raw_reference, raw_signal
        del reference_samples, signal_samples
        release_acquisition_chunk(experiment)
        print(f"single-HEMT: {completed:,}/{args.total_shots:,} shots/state")
        if (
            chunk_index % args.checkpoint_chunks == 0
            or completed == args.total_shots
        ):
            save_stream_checkpoint(
                run_dir,
                completed,
                args.total_shots,
                reference_sums,
                signal_sums,
                convergence,
            )

    save_json(
        run_dir / "stream_summary.json",
        {
            "measurement": "single-HEMT signal/reference temporal-mode tomography",
            "completed_shots_per_state": completed,
            "rf_target_hz": args.cavity_frequency,
            "lo_hz": source.frequency,
            "if_hz": args.if_frequency,
            "raw_chunks_deleted_after_projection": True,
            "mode_sample_storage_bytes_per_state": completed * 16,
        },
    )
    shown = min(completed, 20_000)
    plot_mode_samples(
        run_dir / "stream_mode_projected_iq.png",
        np.asarray(reference_store[:shown]),
        np.asarray(signal_store[:shown]),
    )


def annihilation_operator(cutoff: int) -> ComplexArray:
    operator = np.zeros((cutoff, cutoff), dtype=np.complex128)
    for number in range(1, cutoff):
        operator[number - 1, number] = math.sqrt(number)
    return operator


def density_matrix_moments(
    density_matrix: npt.ArrayLike,
    maximum_order: int,
) -> ComplexArray:
    rho = np.asarray(density_matrix, dtype=np.complex128)
    cutoff = rho.shape[0]
    annihilation = annihilation_operator(cutoff)
    creation = np.conjugate(annihilation.T)
    a_powers = [np.eye(cutoff, dtype=np.complex128)]
    adag_powers = [np.eye(cutoff, dtype=np.complex128)]
    for _ in range(maximum_order):
        a_powers.append(a_powers[-1] @ annihilation)
        adag_powers.append(adag_powers[-1] @ creation)
    moments = np.empty(
        (maximum_order + 1, maximum_order + 1),
        dtype=np.complex128,
    )
    for creation_order in range(maximum_order + 1):
        for annihilation_order in range(maximum_order + 1):
            operator = (
                adag_powers[creation_order]
                @ a_powers[annihilation_order]
            )
            moments[creation_order, annihilation_order] = np.trace(
                rho @ operator
            )
    return moments


def named_operator_moments(density_matrix: npt.ArrayLike) -> dict[str, Any]:
    rho = np.asarray(density_matrix, dtype=np.complex128)
    annihilation = annihilation_operator(rho.shape[0])
    creation = np.conjugate(annihilation.T)

    def expectation(operator: ComplexArray) -> complex:
        return complex(np.trace(rho @ operator))

    operators = {
        "a": annihilation,
        "adag": creation,
        "adag_a": creation @ annihilation,
        "a_adag": annihilation @ creation,
        "a_a": annihilation @ annihilation,
        "adag_adag": creation @ creation,
        "adag2_a2": creation @ creation @ annihilation @ annihilation,
    }
    values: dict[str, Any] = {}
    for name, operator in operators.items():
        value = expectation(operator)
        values[name] = {
            "real": float(value.real),
            "imag": float(value.imag),
            "abs": float(abs(value)),
        }
    values["commutator_a_adag"] = {
        "real": float(expectation(annihilation @ creation - creation @ annihilation).real),
        "imag": 0.0,
        "abs": float(
            abs(expectation(annihilation @ creation - creation @ annihilation))
        ),
    }
    return values


def json_complex_matrix(values: npt.ArrayLike) -> list[list[dict[str, float]]]:
    matrix = np.asarray(values, dtype=np.complex128)
    return [
        [
            {"real": float(value.real), "imag": float(value.imag)}
            for value in row
        ]
        for row in matrix
    ]


def deconvolve_single_hemt_moments(
    measured_signal: npt.ArrayLike,
    measured_reference: npt.ArrayLike,
) -> ComplexArray:
    """Remove independent additive HEMT noise order by order.

    Matrices use ``[m, n] = <(S*)**m S**n>``. The reference is the same
    detector chain with vacuum at the signal input.
    """
    measured = np.asarray(measured_signal, dtype=np.complex128)
    noise = np.asarray(measured_reference, dtype=np.complex128)
    if measured.shape != noise.shape or measured.ndim != 2:
        raise ValueError("signal and reference moment matrices must match")
    maximum_order = measured.shape[0] - 1
    if measured.shape[1] != maximum_order + 1:
        raise ValueError("moment matrices must be square")

    signal = np.zeros_like(measured)
    signal[0, 0] = 1.0
    for m in range(maximum_order + 1):
        for n in range(maximum_order + 1):
            if m == 0 and n == 0:
                continue
            known = 0.0j
            for i in range(m + 1):
                for j in range(n + 1):
                    if i == m and j == n:
                        continue
                    known += (
                        math.comb(m, i)
                        * math.comb(n, j)
                        * signal[i, j]
                        * noise[m - i, n - j]
                    )
            signal[m, n] = measured[m, n] - known
    return signal


def fit_density_matrix_from_moments(
    target_moments: npt.ArrayLike,
    cutoff: int,
) -> ComplexArray:
    """Find a positive density matrix matching calibrated normal moments."""
    target = np.asarray(target_moments, dtype=np.complex128)
    maximum_order = target.shape[0] - 1
    initial_matrix = np.eye(cutoff, dtype=np.complex128) / math.sqrt(cutoff)
    initial = np.concatenate((initial_matrix.real.ravel(), initial_matrix.imag.ravel()))

    def density(parameters: FloatArray) -> ComplexArray:
        size = cutoff * cutoff
        factor = (
            parameters[:size].reshape(cutoff, cutoff)
            + 1.0j * parameters[size:].reshape(cutoff, cutoff)
        )
        rho = factor @ np.conjugate(factor.T)
        return np.asarray(rho / np.trace(rho), dtype=np.complex128)

    indices = [
        (m, n)
        for m in range(maximum_order + 1)
        for n in range(maximum_order + 1)
        if 0 < m + n <= maximum_order
    ]

    def residual(parameters: FloatArray) -> FloatArray:
        model = density_matrix_moments(density(parameters), maximum_order)
        values = []
        for m, n in indices:
            scale = max(1e-3, abs(target[m, n]))
            delta = (model[m, n] - target[m, n]) / scale
            values.extend((delta.real, delta.imag))
        return np.asarray(values, dtype=float)

    result = least_squares(
        residual,
        initial,
        loss="soft_l1",
        max_nfev=10_000,
    )
    return density(result.x)


def analyze_single_hemt_run(
    data_dir: Path,
    *,
    photon_scale_volts: float | None,
    cutoff: int,
    maximum_order: int,
    analysis_shots: int,
) -> dict[str, Any]:
    """Noise-deconvolve one HEMT chain and reconstruct a physical state."""
    if photon_scale_volts is None:
        raise ValueError(
            "analyze-hemt requires --photon-scale-volts from a coherent-state "
            "or independently calibrated photon source"
        )
    run_dir = data_dir.resolve()
    reference_all = np.load(run_dir / "reference_mode_samples.npy", mmap_mode="r")
    signal_all = np.load(run_dir / "signal_mode_samples.npy", mmap_mode="r")
    checkpoint = json.loads(
        (run_dir / "stream_checkpoint.json").read_text(encoding="utf-8")
    )
    used = min(int(checkpoint["completed_shots_per_state"]), analysis_shots)
    reference = np.array(reference_all[:used])
    signal = np.array(signal_all[:used])
    del reference_all, signal_all

    # CALIBRATE: remove electronics/IQ leakage offset before moment inversion.
    offset = complex(np.mean(reference))
    reference_alpha = (reference - offset) / photon_scale_volts
    signal_alpha = (signal - offset) / photon_scale_volts
    reference_moments = complex_moments(
        complex_moment_sums(reference_alpha, maximum_order),
        used,
    )
    measured_moments = complex_moments(
        complex_moment_sums(signal_alpha, maximum_order),
        used,
    )
    normal_moments = deconvolve_single_hemt_moments(
        measured_moments,
        reference_moments,
    )
    rho = fit_density_matrix_from_moments(normal_moments, cutoff)
    fitted_moments = density_matrix_moments(rho, maximum_order)
    axis = np.linspace(-2.5, 2.5, 51)
    padded_rho = np.zeros(
        (max(32, cutoff + 20), max(32, cutoff + 20)),
        dtype=np.complex128,
    )
    padded_rho[:cutoff, :cutoff] = rho
    wigner = wigner_function(padded_rho, axis, axis)
    population = np.maximum(np.real(np.diag(rho)), 0.0)
    named = named_operator_moments(rho)
    result = {
        "analysis": "single-HEMT reference-noise moment inversion",
        "analysis_shots_per_state": used,
        "photon_scale_volts_per_sqrt_photon": photon_scale_volts,
        "reference_offset_real": float(offset.real),
        "reference_offset_imag": float(offset.imag),
        "reference_detector_moments": json_complex_matrix(reference_moments),
        "measured_signal_plus_noise_moments": json_complex_matrix(measured_moments),
        "noise_deconvolved_normal_moments": json_complex_matrix(normal_moments),
        "physical_fit_moments": json_complex_matrix(fitted_moments),
        "density_matrix": json_complex_matrix(rho),
        "photon_population": population.tolist(),
        "mean_photon_number": float(named["adag_a"]["real"]),
        "purity": float(np.real(np.trace(rho @ rho))),
        "wigner_minimum": float(np.min(wigner)),
        "wigner_origin": float(
            2.0 / np.pi * np.sum(((-1.0) ** np.arange(cutoff)) * population)
        ),
        "calibration_warning": (
            "Wigner values are physical only if photon-scale, vacuum reference, "
            "input loss, thermal occupation, IQ leakage, and temporal mode are calibrated."
        ),
    }
    np.savez_compressed(
        run_dir / "hemt_wigner_analysis.npz",
        reference_alpha=reference_alpha,
        signal_alpha=signal_alpha,
        detector_moments=reference_moments,
        deconvolved_moments=normal_moments,
        rho=rho,
        wigner_axis=axis,
        wigner=wigner,
    )
    save_json(run_dir / "hemt_wigner_analysis.json", result)
    plot_stream_wigner_analysis(
        run_dir / "hemt_wigner_analysis.png",
        reference_alpha,
        signal_alpha,
        rho,
        fitted_moments,
        axis,
        wigner,
    )
    return result


def analyze_stream_run(
    data_dir: Path,
    *,
    cutoff: int,
    iterations: int,
    maximum_order: int,
    analysis_shots: int,
) -> dict[str, Any]:
    run_dir = data_dir.resolve()
    reference_all = np.load(
        run_dir / "reference_mode_samples.npy",
        mmap_mode="r",
    )
    signal_all = np.load(
        run_dir / "signal_mode_samples.npy",
        mmap_mode="r",
    )
    checkpoint = json.loads(
        (run_dir / "stream_checkpoint.json").read_text(encoding="utf-8")
    )
    completed = int(checkpoint["completed_shots_per_state"])
    used = min(completed, analysis_shots)
    reference = np.array(reference_all[:used])
    signal = np.array(signal_all[:used])
    del reference_all, signal_all
    reference_alpha, (signal_alpha,), offset, scale = (
        normalize_heterodyne_reference(reference, signal)
    )
    rho = heterodyne_ml_density_matrix(
        signal_alpha,
        cutoff=cutoff,
        iterations=iterations,
        dilution=0.35,
    )
    moments = density_matrix_moments(rho, maximum_order)
    named = named_operator_moments(rho)
    wigner_cutoff = max(32, cutoff + 20)
    padded_rho = np.zeros(
        (wigner_cutoff, wigner_cutoff),
        dtype=np.complex128,
    )
    padded_rho[:cutoff, :cutoff] = rho
    axis = np.linspace(-2.5, 2.5, 51)
    wigner = wigner_function(padded_rho, axis, axis)
    population = np.maximum(np.real(np.diag(rho)), 0.0)
    normalized_displacement = complex(
        np.mean(signal_alpha) - np.mean(reference_alpha)
    )
    coherent_photon_proxy = float(abs(normalized_displacement) ** 2)
    mean_photon_number = float(named["adag_a"]["real"])
    factorial_second_moment = float(named["adag2_a2"]["real"])
    g2_zero = (
        factorial_second_moment / mean_photon_number**2
        if mean_photon_number > 0
        else float("nan")
    )
    result = {
        "completed_shots_per_state": completed,
        "analysis_shots_per_state": used,
        "cutoff": cutoff,
        "iterations": iterations,
        "vacuum_reference_offset_real": float(offset.real),
        "vacuum_reference_offset_imag": float(offset.imag),
        "vacuum_reference_rms_scale": float(scale),
        "raw_voltage_moments_reference": json_complex_matrix(
            complex_moments(
                complex_moment_sums(reference, maximum_order),
                used,
            )
        ),
        "raw_voltage_moments_signal": json_complex_matrix(
            complex_moments(
                complex_moment_sums(signal, maximum_order),
                used,
            )
        ),
        "ideal_heterodyne_density_matrix": json_complex_matrix(rho),
        "ideal_heterodyne_normal_moments": json_complex_matrix(moments),
        "operator_moments": named,
        "normalized_displacement_real": float(normalized_displacement.real),
        "normalized_displacement_imag": float(normalized_displacement.imag),
        "coherent_photon_number_proxy": coherent_photon_proxy,
        "photon_population": population.tolist(),
        "mean_photon_number": mean_photon_number,
        "g2_zero": g2_zero,
        "purity": float(np.real(np.trace(rho @ rho))),
        "wigner_minimum": float(np.min(wigner)),
        "wigner_origin": float(
            2.0
            / np.pi
            * np.sum(((-1.0) ** np.arange(cutoff)) * population)
        ),
        "warning": (
            "Vacuum-normalized ideal-heterodyne diagnostic only. The Wigner "
            "calculation pads the reconstructed density matrix to suppress "
            "finite-cutoff displacement artifacts. "
            "Photon units and Wigner negativity require calibrated detection "
            "efficiency or amplifier-noise moment inversion."
        ),
    }
    np.savez_compressed(
        run_dir / "stream_wigner_analysis.npz",
        reference_alpha=reference_alpha,
        signal_alpha=signal_alpha,
        rho=rho,
        normal_ordered_moments=moments,
        photon_population=population,
        wigner_axis=axis,
        wigner=wigner,
    )
    save_json(run_dir / "stream_wigner_analysis.json", result)
    plot_stream_wigner_analysis(
        run_dir / "stream_wigner_analysis.png",
        reference_alpha,
        signal_alpha,
        rho,
        moments,
        axis,
        wigner,
    )
    return result


def plot_stream_wigner_analysis(
    path: Path,
    reference_alpha: npt.ArrayLike,
    signal_alpha: npt.ArrayLike,
    density_matrix: npt.ArrayLike,
    moments: npt.ArrayLike,
    axis: npt.ArrayLike,
    wigner: npt.ArrayLike,
) -> None:
    reference = np.asarray(reference_alpha)
    signal = np.asarray(signal_alpha)
    rho = np.asarray(density_matrix)
    moment_values = np.asarray(moments)
    wigner_values = np.asarray(wigner)
    display_count = min(10_000, reference.size)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].scatter(
        reference[:display_count].real,
        reference[:display_count].imag,
        s=3,
        alpha=0.12,
        label="reference",
    )
    axes[0, 0].scatter(
        signal[:display_count].real,
        signal[:display_count].imag,
        s=3,
        alpha=0.12,
        label="signal",
    )
    axes[0, 0].set(xlabel="Re(alpha)", ylabel="Im(alpha)", title="Normalized IQ")
    axes[0, 0].axis("equal")
    axes[0, 0].legend()
    image = axes[0, 1].contourf(
        axis,
        axis,
        wigner_values,
        levels=41,
        cmap="RdBu_r",
    )
    axes[0, 1].contour(
        axis,
        axis,
        wigner_values,
        levels=[0.0],
        colors="black",
        linewidths=1,
    )
    axes[0, 1].set(xlabel="Re(alpha)", ylabel="Im(alpha)", title="Wigner diagnostic")
    axes[0, 1].axis("equal")
    fig.colorbar(image, ax=axes[0, 1])
    axes[1, 0].bar(np.arange(rho.shape[0]), np.real(np.diag(rho)))
    axes[1, 0].set(xlabel="Fock number", ylabel="Population", title="Density matrix diagonal")
    moment_image = axes[1, 1].imshow(
        np.abs(moment_values),
        origin="lower",
        cmap="viridis",
    )
    axes[1, 1].set(
        xlabel="annihilation order m",
        ylabel="creation order n",
        title=r"$|\langle(a^\dagger)^n a^m\rangle|$",
    )
    fig.colorbar(moment_image, ax=axes[1, 1])
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def serializable_resonator_fit(fit: ResonatorFit) -> dict[str, float]:
    values = asdict(fit)
    values.pop("fitted_response")
    return {key: float(value) for key, value in values.items()}


def serializable_ringdown_fit(fit: RingdownFit) -> dict[str, float]:
    return {
        "start_s": fit.start_s,
        "stop_s": fit.stop_s,
        "amplitude_lifetime_s": fit.amplitude_lifetime_s,
        "linewidth_hz": fit.linewidth_hz,
        "detuning_hz": fit.detuning_hz,
        "complex_amplitude_real": fit.complex_amplitude.real,
        "complex_amplitude_imag": fit.complex_amplitude.imag,
        "residual_rms": fit.residual_rms,
    }


def plot_ringdown(
    path: Path,
    time_s: npt.ArrayLike,
    envelope: npt.ArrayLike,
    fit: RingdownFit,
    pulse_end_s: float,
    detected_arrival_s: float,
) -> None:
    time_ns = np.asarray(time_s) * 1e9
    values = np.asarray(envelope)
    fitted = fit.fitted_envelope
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(time_ns, np.abs(values) * 1e3, label="measured")
    axes[0].plot(time_ns, np.abs(fitted) * 1e3, label="complex exponential fit")
    axes[0].axvline(pulse_end_s * 1e9, color="black", linestyle="--", label="drive off")
    axes[0].axvline(
        detected_arrival_s * 1e9,
        color="tab:red",
        linestyle=":",
        label="detected ring-down arrival",
    )
    axes[0].set(ylabel="|signal-reference| (mV)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(time_ns, values.real * 1e3, label="I")
    axes[1].plot(time_ns, values.imag * 1e3, label="Q")
    axes[1].plot(time_ns, fitted.real * 1e3, "--")
    axes[1].plot(time_ns, fitted.imag * 1e3, "--")
    axes[1].set(ylabel="Complex envelope (mV)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[2].plot(time_ns, phase_with_amplitude_mask(values), label="measured")
    axes[2].plot(time_ns, phase_with_amplitude_mask(fitted), label="fit")
    axes[2].set(xlabel="Time after marker edge (ns)", ylabel="Phase (rad)")
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    fig.suptitle(
        f"Free ring-down: tau_amp={fit.amplitude_lifetime_s * 1e9:.2f} ns, "
        f"kappa/2pi={fit.linewidth_hz / 1e6:.3f} MHz, "
        f"detuning={fit.detuning_hz / 1e6:.3f} MHz"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_temporal_modes(
    path: Path,
    time_s: npt.ArrayLike,
    measured_mode: npt.ArrayLike,
    ideal_mode: npt.ArrayLike,
    overlap: float,
) -> None:
    time_ns = np.asarray(time_s) * 1e9
    measured = np.asarray(measured_mode)
    ideal = np.asarray(ideal_mode)
    phase_alignment = np.vdot(ideal, measured)
    if abs(phase_alignment) > 0:
        ideal = ideal * phase_alignment / abs(phase_alignment)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(time_ns, np.abs(measured), label="measured matched mode")
    axes[0].plot(time_ns, np.abs(ideal), "--", label="ideal exponential mode")
    axes[0].set(ylabel="|f(t)|")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(time_ns, phase_with_amplitude_mask(measured), label="measured")
    axes[1].plot(time_ns, phase_with_amplitude_mask(ideal), "--", label="ideal")
    axes[1].set(xlabel="Time after marker edge (ns)", ylabel="Phase (rad)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.suptitle(f"Temporal-mode comparison; squared overlap={overlap:.4f}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_mode_samples(
    path: Path,
    reference_samples: npt.ArrayLike,
    signal_samples: npt.ArrayLike,
) -> None:
    reference = np.asarray(reference_samples)
    signal = np.asarray(signal_samples)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(reference.real * 1e3, reference.imag * 1e3, s=8, alpha=0.25)
    axes[0].scatter(signal.real * 1e3, signal.imag * 1e3, s=8, alpha=0.25)
    axes[0].set(xlabel="I-mode (mV)", ylabel="Q-mode (mV)", title="Projected IQ samples")
    axes[0].axis("equal")
    axes[0].grid(alpha=0.3)
    bins = 50
    axes[1].hist(np.abs(reference) * 1e3, bins=bins, alpha=0.6, label="reference")
    axes[1].hist(np.abs(signal) * 1e3, bins=bins, alpha=0.6, label="signal")
    axes[1].set(xlabel="Projected magnitude (mV)", ylabel="Count")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_hemt_mode_calibration(
    path: Path,
    time_s: npt.ArrayLike,
    reference_traces: npt.ArrayLike,
    signal_traces: npt.ArrayLike,
    mode: npt.ArrayLike,
) -> None:
    """Plot the measured timing and the samples selected by the mode."""
    time_ns = np.asarray(time_s) * 1e9
    reference = np.mean(np.asarray(reference_traces), axis=0)
    signal = np.mean(np.asarray(signal_traces), axis=0)
    difference = signal - reference
    weights = np.asarray(mode)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(time_ns, np.abs(reference) * 1e3, label="reference")
    axes[0].plot(time_ns, np.abs(signal) * 1e3, label="signal")
    axes[0].plot(time_ns, np.abs(difference) * 1e3, label="|signal-reference|")
    axes[0].set(ylabel="Mean envelope (mV)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(time_ns, np.abs(weights), label="|temporal mode|")
    axes[1].plot(time_ns, weights.real, "--", label="Re(mode)")
    axes[1].set(
        xlabel="Time after received marker (ns)",
        ylabel="Normalized weight",
    )
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.suptitle("CALIBRATE: signal timing and temporal-mode window")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def load_scan_fit(path: Path) -> ResonatorFit:
    data = np.load(path)
    return fit_complex_resonator(
        data["cavity_frequency_hz"],
        data["complex_response"],
    )


def simulate_workflow(args: argparse.Namespace, run_dir: Path) -> None:
    rng = np.random.default_rng(args.seed)
    frequencies = frequency_axis(
        args.cavity_frequency,
        args.scan_span,
        args.scan_points,
    )
    true_linewidth = 5.7e6
    response = complex_resonator_response(
        frequencies,
        args.cavity_frequency + 0.7e6,
        true_linewidth,
        0.15e-3 - 0.05e-3j,
        (0.001e-3 + 0.0005e-3j) / 1e6,
        0.35e-3 + 0.22e-3j,
    )
    response += 0.006e-3 * (
        rng.normal(size=frequencies.size)
        + 1.0j * rng.normal(size=frequencies.size)
    )
    fit = fit_complex_resonator(frequencies, response)
    np.savez_compressed(
        run_dir / "01_frequency_scan.npz",
        cavity_frequency_hz=frequencies,
        complex_response=response,
        fitted_response=fit.fitted_response,
    )
    plot_frequency_scan(run_dir / "01_frequency_scan.png", frequencies, response, fit)
    load_duration = choose_load_duration_s(args, fit.amplitude_lifetime_s)
    plot_loading_pulse(
        run_dir / "02_loading_pulse.png",
        load_duration,
        fit.amplitude_lifetime_s,
        2.5e9,
    )
    time_s = np.arange(int(args.acquire_ns), dtype=float) * ns
    pulse_end = args.marker_padding_ns * ns + load_duration
    envelope = 0.4e-3 * np.exp(
        -(time_s - pulse_end) / fit.amplitude_lifetime_s
        + 2.0j * np.pi * 0.25e6 * (time_s - pulse_end)
    )
    envelope[time_s < pulse_end] = 0
    envelope += 0.004e-3 * (
        rng.normal(size=time_s.size) + 1.0j * rng.normal(size=time_s.size)
    )
    ringdown = fit_complex_ringdown(
        time_s,
        envelope,
        pulse_end + args.ringdown_guard_ns * ns,
        pulse_end + args.ringdown_fit_ns * ns,
        fit.amplitude_lifetime_s,
    )
    plot_ringdown(
        run_dir / "03_ringdown_fit.png",
        time_s,
        envelope,
        ringdown,
        pulse_end,
        pulse_end,
    )
    measured = envelope.copy()
    measured[time_s < ringdown.start_s] = 0
    measured = normalized_mode(measured)
    ideal = ideal_decay_mode(
        time_s,
        ringdown.start_s,
        fit.amplitude_lifetime_s,
        ringdown.detuning_hz,
    )
    plot_temporal_modes(
        run_dir / "04_temporal_modes.png",
        time_s,
        measured,
        ideal,
        mode_overlap(measured, ideal),
    )
    save_json(
        run_dir / "summary.json",
        {
            "mode": "simulation",
            "resonator_fit": serializable_resonator_fit(fit),
            "ringdown_fit": serializable_ringdown_fit(ringdown),
        },
    )


def run_hardware(args: argparse.Namespace, run_dir: Path) -> None:
    source: RohdeSchwarzSGS100A | None = None
    experiment: AWGAlazar | None = None
    try:
        source, experiment = connect_hardware(args)
        print("SGS:", source.idn())
        print("AWG/Alazar connected; ADC channel CHB")
        save_json(
            run_dir / "00_configuration.json",
            {
                **json_safe_arguments(args),
                "sgs_idn": source.idn(),
            },
        )
        if args.mode == "connect":
            return
        if args.mode == "hemt-stream":
            single_hemt_stream(args, source, experiment, run_dir)
            return

        fit: ResonatorFit
        if args.mode in {"scan", "full"}:
            _, _, fit = acquire_frequency_scan(
                args,
                source,
                experiment,
                run_dir,
            )
            print(
                f"cavity fit: f0={fit.resonance_hz / 1e9:.9f} GHz, "
                f"kappa/2pi={fit.linewidth_hz / 1e6:.4f} MHz, "
                f"tau_amp={fit.amplitude_lifetime_s * 1e9:.3f} ns"
            )
            if args.mode == "scan":
                return
        elif args.scan_file is not None:
            fit = load_scan_fit(args.scan_file)
        else:
            raise ValueError(
                "load/stream mode requires --scan-file; use full to scan first"
            )

        if args.mode == "stream":
            stream_large_acquisition(
                args,
                source,
                experiment,
                run_dir,
                fit,
            )
            return

        _, ringdown, _ = acquire_rising_exponential(
            args,
            source,
            experiment,
            run_dir,
            fit,
        )
        comparison = {
            "scan_linewidth_hz": fit.linewidth_hz,
            "ringdown_linewidth_hz": ringdown.linewidth_hz,
            "fractional_difference": (
                ringdown.linewidth_hz - fit.linewidth_hz
            )
            / fit.linewidth_hz,
        }
        save_json(run_dir / "summary.json", comparison)
        print(
            f"ring-down fit: kappa/2pi={ringdown.linewidth_hz / 1e6:.4f} MHz, "
            f"tau_amp={ringdown.amplitude_lifetime_s * 1e9:.3f} ns, "
            f"detuning={ringdown.detuning_hz / 1e6:.4f} MHz"
        )
    finally:
        if source is not None and not args.leave_rf_on:
            try:
                source.off()
            except Exception as exc:
                print(f"Warning: could not turn SGS RF off: {exc}")
        if experiment is not None:
            try:
                experiment.close()
            except Exception as exc:
                print(f"Warning: could not close AWG/Alazar: {exc}")
        if source is not None:
            try:
                source.close()
            except Exception as exc:
                print(f"Warning: could not close SGS: {exc}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.mode == "analyze-stream":
        result = analyze_stream_run(
            args.data_dir,
            cutoff=args.cutoff,
            iterations=args.iterations,
            maximum_order=args.moment_order,
            analysis_shots=args.analysis_shots,
        )
        print("Analyzed:", args.data_dir.resolve())
        print(
            f"<adag a>={result['mean_photon_number']:.6g}, "
            f"Wmin={result['wigner_minimum']:.6g}, "
            f"W(0)={result['wigner_origin']:.6g}"
        )
        return
    if args.mode == "analyze-hemt":
        result = analyze_single_hemt_run(
            args.data_dir,
            photon_scale_volts=args.photon_scale_volts,
            cutoff=args.cutoff,
            maximum_order=args.moment_order,
            analysis_shots=args.analysis_shots,
        )
        print("Analyzed:", args.data_dir.resolve())
        print(
            f"<adag a>={result['mean_photon_number']:.6g}, "
            f"Wmin={result['wigner_minimum']:.6g}, "
            f"W(0)={result['wigner_origin']:.6g}"
        )
        return
    run_dir = (
        args.resume_dir.resolve()
        if args.resume_dir is not None
        else create_run_directory(args.output_dir)
    )
    if args.resume_dir is not None and not run_dir.is_dir():
        raise FileNotFoundError(f"resume directory does not exist: {run_dir}")
    print("Data directory:", run_dir)
    if args.mode == "simulate":
        simulate_workflow(args, run_dir)
    else:
        run_hardware(args, run_dir)
    print("Completed:", run_dir)


if __name__ == "__main__":
    main()
