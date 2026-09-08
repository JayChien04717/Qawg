"""Pure NumPy downconversion, FIR filtering, and IQ integration."""

from __future__ import annotations

import functools

import numpy as np
import numpy.typing as npt


@functools.lru_cache(maxsize=16)
def _mixing_reference(samples, sample_rate_hz, frequency_hz, phase):
    reference = 2.0 * np.exp(-1j * (
        2.0 * np.pi * frequency_hz * np.arange(samples) / sample_rate_hz + phase
    ))
    reference.flags.writeable = False
    return reference


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
    if not np.all(np.isfinite([sample_rate_hz, frequency_hz, reference_phase_radians])) or sample_rate_hz <= 0:
        raise ValueError("DDC parameters must be finite and sample_rate_hz positive")
    records = _records(records_volts)
    # Bound retained cache memory for long acquisitions; no shot-sized real temporary.
    make_reference = (_mixing_reference if records.shape[1] <= 65536
                      else _mixing_reference.__wrapped__)
    reference = make_reference(records.shape[1], float(sample_rate_hz),
                               float(frequency_hz), float(reference_phase_radians))
    return records * reference


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
    # einsum avoids an intermediate array the size of the entire integration window.
    return np.einsum('...n,n->...', values[..., start:stop], weights,
                     optimize=False) / normalization


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
    *,
    batch_records: int = 128,
) -> npt.NDArray[np.complex128]:
    """Centered FIR with bounded FFT batches; output owns only its exact shape.

    batch_records bounds simultaneous records in FFT scratch arrays. The full
    input/output still reside in RAM; this is not acquisition streaming.
    """
    values = np.asarray(baseband_iq, dtype=np.complex128)
    taps = np.asarray(coefficients, dtype=np.float64)
    if values.ndim < 2 or values.shape[-1] == 0:
        raise ValueError("baseband_iq must end with a non-empty sample axis")
    if taps.ndim != 1 or taps.size < 1 or taps.size % 2 == 0:
        raise ValueError("coefficients must be a non-empty odd-length vector")
    if not np.all(np.isfinite(taps)):
        raise ValueError("coefficients must be finite")
    if isinstance(batch_records, bool) or int(batch_records) != batch_records or batch_records < 1:
        raise ValueError("batch_records must be a positive integer")
    full_count = values.shape[-1] + taps.size - 1
    fft_count = 1 << (full_count - 1).bit_length()
    kernel = np.fft.fft(taps, n=fft_count)
    output = np.empty(values.shape, dtype=np.complex128)
    rows = int(np.prod(values.shape[:-1]))
    dest = output.reshape(rows, values.shape[-1])
    # Avoid copying an entire strided input just to flatten its leading axes.
    source = values.reshape(rows, values.shape[-1]) if values.flags.c_contiguous else None
    delay = (taps.size - 1) // 2
    for start in range(0, rows, int(batch_records)):
        stop = min(start + int(batch_records), rows)
        if source is not None:
            block = source[start:stop]
        else:
            indices = np.unravel_index(np.arange(start, stop), values.shape[:-1])
            block = values[indices]
        spectrum = np.fft.fft(block, n=fft_count, axis=-1)
        spectrum *= kernel
        filtered = np.fft.ifft(spectrum, axis=-1)
        dest[start:stop] = filtered[:, delay:delay + values.shape[-1]]
    return output


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


def weighted_full_demodulate(
    records_volts: npt.ArrayLike,
    sample_rate_hz: float,
    intermediate_frequency_hz: float,
    start_sample: int = 0,
    stop_sample: int | None = None,
    integration_weights: npt.ArrayLike | None = None,
    reference_phase_radians: float = 0.0,
) -> npt.NDArray[np.complex128]:
    """Compatibility helper combining downconversion and integration."""
    return integrate(
        downconvert(
            records_volts,
            sample_rate_hz,
            intermediate_frequency_hz,
            reference_phase_radians,
        ),
        start_sample,
        stop_sample,
        integration_weights,
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


# Compatibility names for existing callers.
digital_downconvert = downconvert
weighted_integrate_iq = integrate


__all__ = [
    "correct_interleaving_offsets",
    "design_lowpass_fir",
    "digital_downconvert",
    "downconvert",
    "fir_filter",
    "fir_lowpass_iq",
    "integrate",
    "weighted_full_demodulate",
    "weighted_integrate_iq",
]
