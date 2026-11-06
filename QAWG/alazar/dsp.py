"""Pure NumPy downconversion, FIR filtering, and IQ integration."""

from __future__ import annotations

import functools

import numpy as np
import numpy.typing as npt


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


__all__ = [
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
]
