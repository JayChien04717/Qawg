"""Offline, shot-preserving fluorescence DSP. No hardware imports or quantum calibration."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

from .dsp import design_lowpass_fir, downconvert, fir_lowpass_iq


@dataclass(frozen=True)
class IFProcessor:
    """Verified pass/stop bands for real IF; centered FIR retains full sample rate."""
    sample_rate_hz: float = 1e9
    if_hz: float = 50e6
    passband_hz: float = 30e6
    stopband_hz: float = 70e6
    taps: int = 181
    beta: float = 8.0
    max_passband_error_db: float = 0.1
    min_stopband_attenuation_db: float = 60.0

    def __post_init__(self):
        values = [self.sample_rate_hz, self.if_hz, self.passband_hz,
                  self.stopband_hz, self.beta, self.max_passband_error_db,
                  self.min_stopband_attenuation_db]
        if not np.all(np.isfinite(values)):
            raise ValueError("filter settings must be finite")
        if not 0 < self.passband_hz < abs(self.if_hz) < self.sample_rate_hz / 2:
            raise ValueError("real IF requires 0 < passband < abs(IF) < Nyquist")
        if not self.passband_hz < self.stopband_hz < self.sample_rate_hz / 2:
            raise ValueError("require passband < stopband < Nyquist")
        if abs(self.if_hz) + self.passband_hz >= self.sample_rate_hz / 2:
            raise ValueError("desired IF band reaches ADC Nyquist")
        image_center = min(2*abs(self.if_hz), self.sample_rate_hz-2*abs(self.if_hz))
        if self.stopband_hz > image_center - self.passband_hz:
            raise ValueError("stopband does not reject the nearest aliased IF image")
        if int(self.taps) != self.taps or self.taps < 3 or self.taps % 2 != 1:
            raise ValueError("taps must be an odd integer >= 3")
        if self.beta < 0 or self.max_passband_error_db <= 0 or self.min_stopband_attenuation_db <= 0:
            raise ValueError("invalid filter tolerance or beta")
        response = self.response_summary()
        if response['passband_error_db'] > self.max_passband_error_db:
            raise ValueError("FIR fails requested passband tolerance; increase taps or transition width")
        if response['stopband_attenuation_db'] < self.min_stopband_attenuation_db:
            raise ValueError("FIR fails requested stopband attenuation")

    @property
    def coefficients(self):
        return design_lowpass_fir(self.sample_rate_hz,
                                 (self.passband_hz+self.stopband_hz)/2,
                                 int(self.taps), self.beta)

    @property
    def edge_samples(self):
        return (int(self.taps)-1)//2

    def response_summary(self):
        count = max(65536, 1 << (int(self.taps)*32-1).bit_length())
        f = np.fft.rfftfreq(count, 1/self.sample_rate_hz)
        db = 20*np.log10(np.maximum(abs(np.fft.rfft(self.coefficients,count)), 1e-300))
        return dict(passband_error_db=float(np.max(abs(db[f<=self.passband_hz]))),
                    stopband_attenuation_db=float(-np.max(db[f>=self.stopband_hz])))

    def process(self, records):
        records = np.asarray(records)
        if np.iscomplexobj(records) or records.ndim != 2 or records.shape[0] == 0:
            raise ValueError("expected nonempty real IF records (shots, samples)")
        if records.shape[1] <= 2*self.edge_samples or not np.all(np.isfinite(records)):
            raise ValueError("record must be finite and longer than FIR guards")
        return fir_lowpass_iq(downconvert(records,self.sample_rate_hz,self.if_hz),self.coefficients)


def reference_phase_correct(signal_iq, reference_iq, window, *, min_reference_amplitude,
                            expected_reference_phase=0.0):
    """Correct A using independent B. Reject weak pilots; never estimate phase from A.

    threshold is in reference IQ volts. Raises instead of silently dropping shots.
    expected_reference_phase preserves deliberately programmed pilot phases.
    """
    signal = np.asarray(signal_iq,dtype=complex)
    reference = np.asarray(reference_iq,dtype=complex)
    if signal.ndim != 2 or signal.shape != reference.shape or signal.shape[0] == 0:
        raise ValueError("signal/reference must have matching (shots, samples) shapes")
    if not np.all(np.isfinite(signal)) or not np.all(np.isfinite(reference)):
        raise ValueError("nonfinite signal or reference")
    start,stop = window
    if not 0 <= start < stop <= signal.shape[1]:
        raise ValueError("pilot window outside record")
    if not np.isfinite(min_reference_amplitude) or min_reference_amplitude <= 0:
        raise ValueError("provide a positive calibrated pilot amplitude threshold")
    pilot = reference[:,start:stop].mean(axis=1)
    if np.any(abs(pilot) < min_reference_amplitude):
        raise ValueError("weak reference shot; inspect pilot before phase correction")
    expected = np.asarray(expected_reference_phase,dtype=float)
    if expected.ndim > 1 or (expected.ndim == 1 and expected.shape != pilot.shape) or not np.all(np.isfinite(expected)):
        raise ValueError("expected phase must be finite scalar or one value per shot")
    phase = np.angle(pilot)-expected
    return signal*np.exp(-1j*phase[:,None]), phase


def save_mode_shots(path, *, iq_shots, time_s, weights, processor,
                    metadata, reference_phase=None, raw_codes=None):
    """Save all mode samples and full IQ with units/provenance; never overwrite.

    This stores electrical mode coordinates, NOT calibrated photon amplitudes.
    Use separate files/metadata conditions for signal, vacuum and coherent reference.
    """
    import h5py
    from ..tomography import project_temporal_mode
    iq = np.asarray(iq_shots,dtype=complex)
    time = np.asarray(time_s,dtype=float)
    mode = np.asarray(weights,dtype=complex)
    if iq.ndim != 2 or iq.shape[0] == 0 or time.shape != (iq.shape[1],) or mode.shape != time.shape:
        raise ValueError("expected IQ (shots, samples) and matching time/weights")
    if not all(np.all(np.isfinite(a)) for a in (iq,time,mode)):
        raise ValueError("nonfinite data")
    if time.size < 2 or not np.allclose(np.diff(time),1/processor.sample_rate_hz,rtol=1e-6,atol=1e-15):
        raise ValueError("time spacing does not match processor")
    if not np.isclose(np.linalg.norm(mode),1,rtol=1e-6,atol=1e-10):
        raise ValueError("weights must have unit discrete L2 norm")
    edge = processor.edge_samples
    if np.any(mode[:edge] != 0) or np.any(mode[-edge:] != 0):
        raise ValueError("mode overlaps zero-padded FIR guards")
    phase = None if reference_phase is None else np.asarray(reference_phase,dtype=float)
    if phase is not None and (phase.shape != (iq.shape[0],) or not np.all(np.isfinite(phase))):
        raise ValueError("reference_phase must contain one finite phase per shot")
    raw = None if raw_codes is None else np.asarray(raw_codes)
    if raw is not None and (raw.dtype != np.uint16 or raw.ndim not in (2,3)
                           or raw.shape[0] != iq.shape[0] or raw.shape[-1] != iq.shape[1]):
        raise ValueError("raw_codes must be uint16 with matching shot/sample axes")
    meta = json.dumps(metadata,ensure_ascii=False,allow_nan=False)
    settings = json.dumps(processor.__dict__,allow_nan=False)
    samples = project_temporal_mode(iq,mode)
    with h5py.File(Path(path),'x') as f:
        f.attrs['schema'] = 'QAWG.mode_shots.v1'
        f.attrs['metadata_json'] = meta
        f.attrs['processor_json'] = settings
        f.attrs['mode_units'] = 'electrical voltage projection; not photon calibrated'
        f.create_dataset('iq_shots',data=iq,compression='gzip')
        f.create_dataset('time_s',data=time)
        f.create_dataset('mode_weights',data=mode)
        f.create_dataset('mode_iq',data=samples)
        f.create_dataset('fir_coefficients',data=processor.coefficients)
        if phase is not None: f.create_dataset('reference_phase_rad',data=phase)
        if raw is not None: f.create_dataset('raw_codes',data=raw,compression='gzip')
    return samples
