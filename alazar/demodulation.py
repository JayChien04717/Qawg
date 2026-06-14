"""Pure functions for dispersive and waveguide-QED demodulation."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def require_records(
    records: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    values = np.asarray(records, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("records must have shape (number_of_records, samples)")
    if values.shape[1] == 0:
        raise ValueError("records must contain at least one sample")
    return values


def digital_downconvert(
    records_volts: npt.NDArray[np.float64],
    sample_rate_hz: float,
    intermediate_frequency_hz: float,
    reference_phase_radians: float = 0.0,
) -> npt.NDArray[np.complex128]:
    """Mix real IF records to complex baseband without low-pass filtering."""
    records = require_records(records_volts)
    time = np.arange(records.shape[1], dtype=np.float64) / sample_rate_hz
    phase = 2.0 * np.pi * intermediate_frequency_hz * time
    reference = np.exp(-1j * (phase + reference_phase_radians))
    return 2.0 * records * reference


def integrate_iq_window(
    baseband: npt.NDArray[np.complex128],
    start_sample: int = 0,
    stop_sample: int | None = None,
) -> npt.NDArray[np.complex128]:
    """Average a baseband time window into one complex IQ point per record."""
    values = np.asarray(baseband, dtype=np.complex128)
    if values.ndim != 2:
        raise ValueError("baseband must have shape (number_of_records, samples)")
    stop = values.shape[1] if stop_sample is None else stop_sample
    if not 0 <= start_sample < stop <= values.shape[1]:
        raise ValueError("integration window is outside the acquired record")
    return np.mean(values[:, start_sample:stop], axis=1)


def dispersive_demodulate(
    records_volts: npt.NDArray[np.float64],
    sample_rate_hz: float,
    intermediate_frequency_hz: float,
    start_sample: int = 0,
    stop_sample: int | None = None,
    reference_phase_radians: float = 0.0,
) -> npt.NDArray[np.complex128]:
    """Return one integrated IQ point per triggered readout record."""
    baseband = digital_downconvert(
        records_volts,
        sample_rate_hz,
        intermediate_frequency_hz,
        reference_phase_radians,
    )
    return integrate_iq_window(baseband, start_sample, stop_sample)


def seconds_to_samples(duration_s: float, sample_rate_hz: float) -> int:
    if duration_s < 0:
        raise ValueError("duration_s cannot be negative")
    return int(round(duration_s * sample_rate_hz))


def dispersive_demodulate_seconds(
    records_volts: npt.NDArray[np.float64],
    sample_rate_hz: float,
    intermediate_frequency_hz: float,
    integration_delay_s: float,
    integration_time_s: float,
    reference_phase_radians: float = 0.0,
) -> npt.NDArray[np.complex128]:
    """Return one IQ point per record using a window specified in seconds."""
    if integration_time_s <= 0:
        raise ValueError("integration_time_s must be positive")
    start_sample = seconds_to_samples(integration_delay_s, sample_rate_hz)
    window_samples = seconds_to_samples(integration_time_s, sample_rate_hz)
    if window_samples < 1:
        raise ValueError("integration_time_s is shorter than one sample")
    return dispersive_demodulate(
        records_volts,
        sample_rate_hz,
        intermediate_frequency_hz,
        start_sample,
        start_sample + window_samples,
        reference_phase_radians,
    )


from scipy.ndimage import gaussian_filter1d

# def moving_average_iq(
#     baseband: npt.NDArray[np.complex128],
#     window_samples: int,
# ) -> npt.NDArray[np.complex128]:
#     """Low-pass a complex IQ trace with a valid boxcar moving average."""
#     values = np.asarray(baseband, dtype=np.complex128)
#     if values.ndim != 2:
#         raise ValueError("baseband must have shape (number_of_records, samples)")
#     if not 1 <= window_samples <= values.shape[1]:
#         raise ValueError("window_samples must fit inside each record")

#     padded = np.pad(values, ((0, 0), (1, 0)), mode="constant")
#     cumulative = np.cumsum(padded, axis=1)
#     window_sums = cumulative[:, window_samples:] - cumulative[:, :-window_samples]
#     return window_sums / window_samples


def moving_average_iq(
    baseband: npt.NDArray[np.complex128],
    sigma_samples: float,
) -> npt.NDArray[np.complex128]:
    """使用高斯濾波器替代傳統矩形移動平均，獲得極平滑的基頻包絡."""
    values = np.asarray(baseband, dtype=np.complex128)
    if values.ndim != 2:
        raise ValueError("baseband must have shape (number_of_records, samples)")

    # 對時間軸 (axis=1) 進行高斯平滑
    # sigma_samples 代表高斯標準差的點數，通常設為 (預期視窗點數 / 4)
    smooth_real = gaussian_filter1d(values.real, sigma=sigma_samples, axis=1)
    smooth_imag = gaussian_filter1d(values.imag, sigma=sigma_samples, axis=1)

    return smooth_real + 1j * smooth_imag


def moving_average_time_axis(
    number_of_samples: int,
    sample_rate_hz: float,
    window_samples: int,
) -> npt.NDArray[np.float64]:
    """Return center times for the valid moving-average output samples."""
    if not 1 <= window_samples <= number_of_samples:
        raise ValueError("window_samples must fit inside the acquired record")
    output_length = number_of_samples - window_samples + 1
    first_center = (window_samples - 1) / (2.0 * sample_rate_hz)
    return first_center + np.arange(output_length) / sample_rate_hz


def emission_demodulate(
    records_volts: npt.NDArray[np.float64],
    sample_rate_hz: float,
    intermediate_frequency_hz: float,
    window_samples: int,
    reference_phase_radians: float = 0.0,
) -> npt.NDArray[np.complex128]:
    """Return a time-resolved, moving-averaged complex emission envelope."""
    baseband = digital_downconvert(
        records_volts,
        sample_rate_hz,
        intermediate_frequency_hz,
        reference_phase_radians,
    )
    return moving_average_iq(baseband, window_samples)


def average_shots(
    shot_values: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128] | np.complex128:
    """Average the first axis, which represents repeated AWG triggers."""
    values = np.asarray(shot_values, dtype=np.complex128)
    if values.ndim not in (1, 2):
        raise ValueError("shot_values must be IQ points or IQ time traces")
    if values.shape[0] == 0:
        raise ValueError("shot_values must contain at least one shot")
    return np.mean(values, axis=0)


def subtract_baseline(
    records_volts: npt.NDArray[np.float64],
    stop_sample: int,
    start_sample: int = 0,
) -> npt.NDArray[np.float64]:
    """Subtract one pre-pulse DC baseline from every acquired record."""
    records = require_records(records_volts)
    if not 0 <= start_sample < stop_sample <= records.shape[1]:
        raise ValueError("baseline window is outside the acquired record")
    baseline = np.mean(records[:, start_sample:stop_sample], axis=1, keepdims=True)
    return records - baseline


def correct_interleaving_offsets(
    records_volts: npt.NDArray[np.float64],
    stop_sample: int,
    period: int = 2,
    start_sample: int = 0,
) -> npt.NDArray[np.float64]:
    """Remove periodic ADC-core offsets using a pre-pulse baseline window."""
    records = require_records(records_volts)
    if period < 1:
        raise ValueError("period must be positive")
    if not 0 <= start_sample < stop_sample <= records.shape[1]:
        raise ValueError("baseline window is outside the acquired record")
    if stop_sample - start_sample < period:
        raise ValueError("baseline window must include every interleaving phase")

    corrected = records.copy()
    sample_indices = np.arange(records.shape[1])
    baseline_indices = np.arange(start_sample, stop_sample)
    for phase in range(period):
        phase_baseline = baseline_indices[baseline_indices % period == phase]
        offset = np.mean(records[:, phase_baseline], axis=1, keepdims=True)
        corrected[:, sample_indices % period == phase] -= offset
    return corrected


def phase_align_iq(
    baseband: npt.NDArray[np.complex128],
    start_sample: int,
    stop_sample: int,
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.float64]]:
    """Rotate each shot so its reference-window phase is zero."""
    values = np.asarray(baseband, dtype=np.complex128)
    if values.ndim != 2:
        raise ValueError("baseband must have shape (number_of_records, samples)")
    if not 0 <= start_sample < stop_sample <= values.shape[1]:
        raise ValueError("phase reference window is outside the record")
    reference = np.mean(values[:, start_sample:stop_sample], axis=1)
    if np.any(np.abs(reference) == 0):
        raise ValueError("phase reference window contains a zero-amplitude shot")
    phases = np.angle(reference)
    aligned = values * np.exp(-1j * phases[:, None])
    return aligned, phases


def recover_coherent_envelope(
    records_volts: npt.NDArray[np.float64],
    sample_rate_hz: float,
    intermediate_frequency_hz: float,
    baseline_stop_sample: int,
    phase_start_sample: int,
    phase_stop_sample: int,
    window_samples: int,
) -> tuple[
    npt.NDArray[np.complex128],
    npt.NDArray[np.complex128],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Recover phase-aligned and phase-insensitive averaged pulse envelopes."""
    corrected = correct_interleaving_offsets(
        records_volts,
        baseline_stop_sample,
        period=2,
    )
    baseband = digital_downconvert(
        corrected,
        sample_rate_hz,
        intermediate_frequency_hz,
    )
    aligned, shot_phases = phase_align_iq(
        baseband,
        phase_start_sample,
        phase_stop_sample,
    )
    filtered = moving_average_iq(aligned, window_samples)
    coherent_average = np.mean(filtered, axis=0)
    magnitude_average = np.mean(np.abs(filtered), axis=0)
    return filtered, coherent_average, magnitude_average, shot_phases


def recover_clock_referenced_envelope(
    records_volts: npt.NDArray[np.float64],
    sample_rate_hz: float,
    intermediate_frequency_hz: float,
    baseline_stop_sample: int,
    window_samples: int,
) -> tuple[
    npt.NDArray[np.complex128],
    npt.NDArray[np.complex128],
    npt.NDArray[np.float64],
]:
    """Recover an envelope without estimating phase from the measured pulse.

    Use this path when the AWG and digitizer share a reference clock. It avoids
    the positive bias created by rotating every noisy shot with its own phase.
    """
    corrected = correct_interleaving_offsets(
        records_volts,
        baseline_stop_sample,
        period=2,
    )
    baseband = digital_downconvert(
        corrected,
        sample_rate_hz,
        intermediate_frequency_hz,
    )
    filtered = moving_average_iq(baseband, window_samples)
    coherent_average = np.mean(filtered, axis=0)
    rms_envelope = np.sqrt(np.mean(np.abs(filtered) ** 2, axis=0))
    return filtered, coherent_average, rms_envelope
