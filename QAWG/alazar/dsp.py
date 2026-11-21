"""Pure NumPy downconversion, FIR filtering, and IQ integration."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .constants import CHANNEL_A

if TYPE_CHECKING:
    from .device import CaptureData


def _records(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    records = np.asarray(values, dtype=np.float64)
    if records.ndim != 2 or records.shape[1] == 0:
        raise ValueError("records must have shape (number_of_records, samples)")
    return records


def downconvert(
    records_volts: npt.ArrayLike,
    sample_rate_hz: float,
    frequency_hz: float,
    reference_phase_radians: float = 0.0,
) -> npt.NDArray[np.complex128]:
    """Mix real voltage records to complex baseband."""
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    records = _records(records_volts)
    time = np.arange(records.shape[1]) / sample_rate_hz
    reference = np.exp(
        -1j * (2.0 * np.pi * frequency_hz * time + reference_phase_radians)
    )
    return 2.0 * records * reference


def integrate(
    baseband_iq: npt.ArrayLike,
    start_sample: int = 0,
    stop_sample: int | None = None,
    integration_weights: npt.ArrayLike | None = None,
) -> npt.NDArray[np.complex128]:
    """Reduce each IQ record to one optionally weighted complex value."""
    values = np.asarray(baseband_iq, dtype=np.complex128)
    if values.ndim < 2 or values.shape[-1] == 0:
        raise ValueError("baseband_iq must end with a non-empty sample axis")
    stop = values.shape[-1] if stop_sample is None else int(stop_sample)
    start = int(start_sample)
    if not 0 <= start < stop <= values.shape[-1]:
        raise ValueError("integration window is outside the acquired record")
    count = stop - start
    if integration_weights is None:
        return np.mean(values[..., start:stop], axis=-1)
    weights = np.asarray(integration_weights, dtype=np.complex128)
    if weights.ndim != 1 or weights.size != count:
        raise ValueError("integration_weights must match the integration window")
    if not np.all(np.isfinite(weights)):
        raise ValueError("integration_weights must be finite")
    normalization = float(np.sum(np.abs(weights)))
    if normalization == 0:
        raise ValueError("integration_weights cannot all be zero")
    return np.sum(values[..., start:stop] * weights, axis=-1) / normalization


@functools.lru_cache(maxsize=32)
def _cached_lowpass_fir(
    sample_rate_hz: float,
    cutoff_hz: float,
    number_of_taps: int,
    kaiser_beta: float,
) -> npt.NDArray[np.float64]:
    index = np.arange(number_of_taps) - (number_of_taps - 1) / 2
    normalized_cutoff = cutoff_hz / sample_rate_hz
    coefficients = (
        2
        * normalized_cutoff
        * np.sinc(2 * normalized_cutoff * index)
        * np.kaiser(number_of_taps, kaiser_beta)
    )
    return np.asarray(coefficients / np.sum(coefficients), dtype=np.float64)


def design_lowpass_fir(
    sample_rate_hz: float,
    cutoff_hz: float,
    number_of_taps: int = 41,
    kaiser_beta: float = 6.0,
) -> npt.NDArray[np.float64]:
    """Design an odd-length, unity-gain Kaiser low-pass FIR."""
    if sample_rate_hz <= 0 or not 0 < cutoff_hz < sample_rate_hz / 2:
        raise ValueError("cutoff_hz must be between DC and Nyquist")
    taps = int(number_of_taps)
    if taps < 3 or taps % 2 == 0:
        raise ValueError("number_of_taps must be odd and at least 3")
    if kaiser_beta < 0:
        raise ValueError("kaiser_beta cannot be negative")
    return _cached_lowpass_fir(
        float(sample_rate_hz),
        float(cutoff_hz),
        taps,
        float(kaiser_beta),
    ).copy()


def fir_lowpass_iq(
    baseband_iq: npt.ArrayLike,
    coefficients: npt.ArrayLike,
) -> npt.NDArray[np.complex128]:
    """Apply a centered linear-phase FIR along the final sample axis."""
    values = np.asarray(baseband_iq, dtype=np.complex128)
    taps = np.asarray(coefficients, dtype=np.float64)
    if values.ndim < 2 or values.shape[-1] == 0:
        raise ValueError("baseband_iq must end with a non-empty sample axis")
    if taps.ndim != 1 or taps.size < 1 or taps.size % 2 == 0:
        raise ValueError("coefficients must be a non-empty odd-length vector")
    if not np.all(np.isfinite(taps)):
        raise ValueError("coefficients must be finite")
    full_count = values.shape[-1] + taps.size - 1
    fft_count = 1 << (full_count - 1).bit_length()
    full = np.fft.ifft(
        np.fft.fft(values, n=fft_count, axis=-1)
        * np.fft.fft(taps, n=fft_count),
        axis=-1,
    )[..., :full_count]
    delay = (taps.size - 1) // 2
    return full[..., delay : delay + values.shape[-1]]


def fir_filter(
    baseband_iq: npt.ArrayLike,
    sample_rate_hz: float,
    cutoff_hz: float,
    number_of_taps: int = 41,
    kaiser_beta: float = 6.0,
) -> npt.NDArray[np.complex128]:
    """Design and apply the standard low-pass FIR."""
    return fir_lowpass_iq(
        baseband_iq,
        design_lowpass_fir(
            sample_rate_hz,
            cutoff_hz,
            number_of_taps,
            kaiser_beta,
        ),
    )


def correct_interleaving_offsets(
    records_volts: npt.ArrayLike,
    stop_sample: int,
    period: int = 2,
    start_sample: int = 0,
) -> npt.NDArray[np.float64]:
    """Remove periodic ADC-core offsets using a baseline window."""
    records = _records(records_volts)
    if period < 1 or not 0 <= start_sample < stop_sample <= records.shape[1]:
        raise ValueError("invalid interleaving baseline window")
    if stop_sample - start_sample < period:
        raise ValueError("baseline window must include every interleaving phase")
    corrected = records.copy()
    samples = np.arange(records.shape[1])
    baseline = np.arange(start_sample, stop_sample)
    for phase in range(period):
        phase_samples = baseline[baseline % period == phase]
        offset = np.mean(records[:, phase_samples], axis=1, keepdims=True)
        corrected[:, samples % period == phase] -= offset
    return corrected


def moving_average_iq(
    baseband_iq: npt.ArrayLike,
    window_samples: int,
) -> npt.NDArray[np.complex128]:
    """Boxcar-average complex traces; only fully covered windows are kept.

    Output length is ``samples - window_samples + 1``; use
    :func:`moving_average_time_axis` for the matching window-center times.
    """
    values = np.asarray(baseband_iq, dtype=np.complex128)
    if values.ndim != 2:
        raise ValueError("baseband_iq must have shape (records, samples)")
    if not 1 <= window_samples <= values.shape[1]:
        raise ValueError("window_samples must fit inside each record")
    cumulative = np.cumsum(
        np.pad(values, ((0, 0), (1, 0))), axis=1
    )
    return (
        cumulative[:, window_samples:] - cumulative[:, :-window_samples]
    ) / window_samples


def moving_average_time_axis(
    number_of_samples: int,
    sample_rate_hz: float,
    window_samples: int,
) -> npt.NDArray[np.float64]:
    """Return center times of the :func:`moving_average_iq` outputs."""
    if not 1 <= window_samples <= number_of_samples:
        raise ValueError("window_samples must fit inside the acquired record")
    count = number_of_samples - window_samples + 1
    first_center = (window_samples - 1) / (2.0 * sample_rate_hz)
    return first_center + np.arange(count) / sample_rate_hz


def phase_align_iq(
    baseband_iq: npt.ArrayLike,
    start_sample: int,
    stop_sample: int,
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.float64]]:
    """Rotate each record so its reference-window phase is zero.

    Only use this when the source has no phase reference to the digitizer;
    with a shared clock it biases noisy shots (see
    :func:`clock_referenced_envelope`).
    """
    values = np.asarray(baseband_iq, dtype=np.complex128)
    if values.ndim != 2:
        raise ValueError("baseband_iq must have shape (records, samples)")
    if not 0 <= start_sample < stop_sample <= values.shape[1]:
        raise ValueError("phase reference window is outside the record")
    reference = np.mean(values[:, start_sample:stop_sample], axis=1)
    if np.any(np.abs(reference) == 0):
        raise ValueError("phase reference window contains a zero-amplitude shot")
    phases = np.angle(reference)
    return values * np.exp(-1j * phases[:, None]), phases


def clock_referenced_envelope(
    records_volts: npt.ArrayLike,
    sample_rate_hz: float,
    frequency_hz: float,
    window_samples: int,
    baseline_stop_sample: int | None = None,
) -> tuple[
    npt.NDArray[np.complex128],
    npt.NDArray[np.complex128],
    npt.NDArray[np.float64],
]:
    """Envelope for AWG and digitizer sharing a reference clock.

    Returns ``(per_record, coherent_average, rms_envelope)``. Phase is never
    estimated from the signal, so there is no rectification bias.
    """
    records = _records(records_volts)
    if baseline_stop_sample is not None:
        records = correct_interleaving_offsets(records, baseline_stop_sample)
    filtered = moving_average_iq(
        downconvert(records, sample_rate_hz, frequency_hz), window_samples
    )
    return (
        filtered,
        np.mean(filtered, axis=0),
        np.sqrt(np.mean(np.abs(filtered) ** 2, axis=0)),
    )


def remove_record_dc_offset(
    records_volts: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Subtract the per-record mean voltage from each trace."""
    records = _records(records_volts)
    return records - np.mean(records, axis=1, keepdims=True)


def reference_phase(
    reference_records_volts: npt.ArrayLike,
    sample_rate_hz: float,
    frequency_hz: float,
    start_sample: int = 0,
    stop_sample: int | None = None,
    *,
    min_amplitude: float = 0.0,
) -> npt.NDArray[np.float64]:
    """Return each record's phase (rad) of a reference tone.

    Slices ``start_sample:stop_sample`` before downconversion while preserving
    the sample-0 phase origin, avoiding unnecessary full-record mixing.
    If ``min_amplitude > 0``, raises ``ValueError`` when any record's
    reference tone drops below that threshold (e.g. unplugged or muted ref).
    """
    records = _records(reference_records_volts)
    start = int(start_sample)
    stop = records.shape[1] if stop_sample is None else int(stop_sample)
    if not 0 <= start < stop <= records.shape[1]:
        raise ValueError("reference window is outside the acquired record")
    if min_amplitude < 0:
        raise ValueError("min_amplitude cannot be negative")
    windowed = records[:, start:stop]
    phase_origin = 2.0 * np.pi * frequency_hz * (start / sample_rate_hz)
    iq = np.mean(
        downconvert(windowed, sample_rate_hz, frequency_hz, phase_origin),
        axis=1,
    )
    if min_amplitude > 0.0:
        amplitudes = np.abs(iq)
        min_seen = float(np.min(amplitudes))
        if min_seen < min_amplitude:
            raise ValueError(
                f"reference tone amplitude dropped to {min_seen:.3e} V "
                f"(below min_amplitude={min_amplitude:.3e} V)"
            )
    return np.angle(iq)


def circular_mean(angles_rad: npt.ArrayLike) -> float:
    """Mean direction of angles; unaffected by the +/-pi wrap."""
    return float(np.angle(np.mean(np.exp(1j * np.asarray(angles_rad)))))


def apply_reference_correction(
    baseband_iq: npt.ArrayLike,
    reference_phase_rad: npt.ArrayLike,
    *,
    center_rad: float | None = None,
    scale: float = 1.0,
    unwrap: bool = False,
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.float64]]:
    """Remove per-record trigger/clock phase using a reference channel.

    A timing error dt shifts a tone at frequency f by ``2*pi*f*dt``.  The
    reference deviation (its phase minus ``center_rad``) is multiplied by
    ``scale`` (= signal frequency / reference frequency for a pure timing
    error, 1 for a common phase) and removed from every record.
    ``center_rad`` defaults to the circular mean over the records.

    When ``unwrap=True`` (recommended for slow phase drift when ``scale`` is
    non-integer), the reference phase is unwrapped across records before
    scaling so crossing ``+/-pi`` does not introduce a ``2*pi*scale`` jump.

    Returns ``(corrected_baseband, deviation_rad)``.
    """
    values = np.asarray(baseband_iq, dtype=np.complex128)
    phase = np.asarray(reference_phase_rad, dtype=np.float64).reshape(-1)
    if values.ndim != 2 or values.shape[0] != phase.size:
        raise ValueError("need one reference phase per baseband record")
    center = circular_mean(phase) if center_rad is None else float(center_rad)
    if unwrap and phase.size > 0:
        first_delta = float(np.angle(np.exp(1j * (phase[0] - center))))
        deviation = first_delta + (np.unwrap(phase) - phase[0])
    else:
        deviation = np.angle(np.exp(1j * (phase - center)))
    corrected = values * np.exp(-1j * scale * deviation)[:, None]
    return corrected, deviation


@dataclass(frozen=True)
class DemodulationConfig:
    """Configuration for standalone ATS9371 receive-side DSP."""

    sample_rate_hz: float
    tone_frequency_hz: float
    integrate_start_sample: int = 0
    integrate_stop_sample: int | None = None
    integration_weights: npt.ArrayLike | None = None
    reference_phase_radians: float = 0.0
    baseline_stop_sample: int | None = None
    remove_dc_offset: bool = True
    fir_cutoff_hz: float = 20e6
    fir_number_of_taps: int = 41
    fir_kaiser_beta: float = 6.0
    reference_frequency_hz: float | None = None
    reference_start_sample: int | None = None
    reference_stop_sample: int | None = None
    reference_center_rad: float | None = None
    reference_unwrap: bool = False
    reference_min_amplitude: float = 0.0


@dataclass(frozen=True)
class DemodulationResult:
    """Outputs from one receive-side demodulation pass."""

    signal_volts: npt.NDArray[np.float64]
    reference_volts: npt.NDArray[np.float64] | None
    baseband_iq: npt.NDArray[np.complex128]
    filtered_iq: npt.NDArray[np.complex128] | None
    shots_iq: npt.NDArray[np.complex128]
    reference_deviation_rad: npt.NDArray[np.float64] | None


class DemodulationPipeline:
    """Standalone receive-side DSP chain for Alazar9371 captures.

    Performs optional interleaving-offset removal, DC removal, downconversion,
    optional second-channel phase-reference correction, optional FIR lowpass
    filtering, and windowed IQ integration without requiring an AWG wrapper.
    """

    def __init__(self, config: DemodulationConfig) -> None:
        self.config = config

    def clean(self, records_volts: npt.ArrayLike) -> npt.NDArray[np.float64]:
        records = _records(records_volts)
        if self.config.baseline_stop_sample is not None:
            records = correct_interleaving_offsets(
                records,
                stop_sample=self.config.baseline_stop_sample,
                period=2,
            )
        if self.config.remove_dc_offset:
            records = remove_record_dc_offset(records)
        return records

    def process(
        self,
        signal_volts: npt.ArrayLike,
        reference_volts: npt.ArrayLike | None = None,
        *,
        compute_fir: bool = True,
        already_cleaned: bool = False,
    ) -> DemodulationResult:
        cfg = self.config
        signal = (
            _records(signal_volts)
            if already_cleaned
            else self.clean(signal_volts)
        )
        reference = (
            None
            if reference_volts is None
            else (
                _records(reference_volts)
                if already_cleaned
                else self.clean(reference_volts)
            )
        )
        baseband = downconvert(
            signal,
            cfg.sample_rate_hz,
            cfg.tone_frequency_hz,
            cfg.reference_phase_radians,
        )
        deviation: npt.NDArray[np.float64] | None = None
        if reference is not None:
            f_ref = (
                cfg.tone_frequency_hz
                if cfg.reference_frequency_hz is None
                else cfg.reference_frequency_hz
            )
            ref_start = (
                cfg.integrate_start_sample
                if cfg.reference_start_sample is None
                else cfg.reference_start_sample
            )
            ref_stop = (
                cfg.integrate_stop_sample
                if cfg.reference_stop_sample is None
                else cfg.reference_stop_sample
            )
            phase = reference_phase(
                reference,
                cfg.sample_rate_hz,
                f_ref,
                ref_start,
                ref_stop,
                min_amplitude=cfg.reference_min_amplitude,
            )
            scale = (
                1.0 if f_ref == 0.0 else cfg.tone_frequency_hz / f_ref
            )
            baseband, deviation = apply_reference_correction(
                baseband,
                phase,
                center_rad=cfg.reference_center_rad,
                scale=scale,
                unwrap=cfg.reference_unwrap,
            )

        filtered = (
            fir_filter(
                baseband,
                cfg.sample_rate_hz,
                cfg.fir_cutoff_hz,
                cfg.fir_number_of_taps,
                cfg.fir_kaiser_beta,
            )
            if compute_fir
            else None
        )
        shots = integrate(
            baseband,
            cfg.integrate_start_sample,
            cfg.integrate_stop_sample,
            cfg.integration_weights,
        )
        return DemodulationResult(
            signal_volts=signal,
            reference_volts=reference,
            baseband_iq=baseband,
            filtered_iq=filtered,
            shots_iq=shots,
            reference_deviation_rad=deviation,
        )

    def process_capture(
        self,
        capture: "CaptureData",
        *,
        signal_channel: int = CHANNEL_A,
        reference_channel: int | None = None,
        compute_fir: bool = True,
    ) -> DemodulationResult:
        signal = capture.channel(signal_channel)
        reference = (
            None
            if reference_channel is None
            else capture.channel(reference_channel)
        )
        return self.process(signal, reference, compute_fir=compute_fir)


__all__ = [
    "DemodulationConfig",
    "DemodulationPipeline",
    "DemodulationResult",
    "apply_reference_correction",
    "circular_mean",
    "clock_referenced_envelope",
    "correct_interleaving_offsets",
    "design_lowpass_fir",
    "downconvert",
    "fir_filter",
    "fir_lowpass_iq",
    "integrate",
    "moving_average_iq",
    "moving_average_time_axis",
    "phase_align_iq",
    "reference_phase",
    "remove_record_dc_offset",
]
