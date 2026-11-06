import numpy as np
import pytest

from QAWG.alazar import (
    CaptureConfig,
    apply_reference_correction,
    downconvert,
    integrate,
    reference_phase,
)
from QAWG.alazar.constants import CHANNEL_A, CHANNEL_AB, CHANNEL_B
from QAWG.alazar.ats9371 import (
    BoardInfo,
    CaptureSession,
    DmaBuffer,
    copy_dma_buffer,
)
from QAWG.alazar.device import CaptureData

SAMPLE_RATE = 1e9
SAMPLES = 1024


def _records(shifts_s, frequency, phase):
    time = np.arange(SAMPLES) / SAMPLE_RATE
    return np.cos(
        2 * np.pi * frequency * (time[None, :] + shifts_s[:, None]) + phase
    )


def test_reference_removes_sub_sample_timing_jitter():
    rng = np.random.default_rng(1)
    shifts = rng.uniform(-0.5e-9, 0.5e-9, 400)  # +/- half an ADC sample
    signal = _records(shifts, 50e6, 0.4)
    reference = _records(shifts, 50e6, -1.1)

    raw = integrate(downconvert(signal, SAMPLE_RATE, 50e6), 100, 900)
    assert np.std(np.angle(raw * np.exp(-1j * np.angle(raw.mean())))) > 0.08

    phase = reference_phase(reference, SAMPLE_RATE, 50e6, 100, 900)
    corrected, _ = apply_reference_correction(
        downconvert(signal, SAMPLE_RATE, 50e6), phase
    )
    shots = integrate(corrected, 100, 900)
    spread = np.angle(shots * np.exp(-1j * np.angle(shots.mean())))
    assert np.std(spread) < 1e-3
    np.testing.assert_allclose(np.abs(shots), 1.0, atol=1e-3)


def test_reference_at_other_frequency_uses_timing_scale():
    rng = np.random.default_rng(2)
    shifts = rng.uniform(-0.5e-9, 0.5e-9, 200)
    signal = _records(shifts, 80e6, 0.0)
    reference = _records(shifts, 20e6, 0.7)
    phase = reference_phase(reference, SAMPLE_RATE, 20e6, 100, 900)
    corrected, _ = apply_reference_correction(
        downconvert(signal, SAMPLE_RATE, 80e6), phase, scale=80e6 / 20e6
    )
    shots = integrate(corrected, 100, 900)
    spread = np.angle(shots * np.exp(-1j * np.angle(shots.mean())))
    assert np.std(spread) < 2e-3


def test_reference_correction_keeps_a_real_common_offset():
    shifts = np.zeros(10)
    signal = _records(shifts, 50e6, 0.4)
    phase = np.full(10, 0.9)
    corrected, deviation = apply_reference_correction(
        downconvert(signal, SAMPLE_RATE, 50e6), phase, center_rad=0.9
    )
    np.testing.assert_allclose(deviation, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        corrected, downconvert(signal, SAMPLE_RATE, 50e6)
    )


def test_apply_reference_correction_validates_shapes():
    with pytest.raises(ValueError):
        apply_reference_correction(np.ones((3, 8), complex), np.zeros(2))


def test_capture_config_accepts_two_channels():
    config = CaptureConfig(samples_per_record=256, channel=CHANNEL_AB)
    assert config.channel_count == 2
    assert CaptureConfig(samples_per_record=256).channel_count == 1
    with pytest.raises(ValueError):
        CaptureConfig(samples_per_record=256, channel=4)


def test_dual_channel_dma_buffer_is_split_record_then_channel(monkeypatch):
    import ctypes

    records, channels, samples = 3, 2, 8
    data = np.arange(records * channels * samples, dtype=np.uint16)
    backing = (ctypes.c_uint16 * data.size)(*data)
    session = CaptureSession(
        handle=1,
        buffers=[],
        samples_per_record=samples,
        records_per_buffer=records,
        records_per_acquisition=records,
        buffers_per_acquisition=1,
        bits_per_sample=16,
        channel_count=channels,
    )
    buffer = DmaBuffer(ctypes.addressof(backing), data.size, data.size * 2)
    result = copy_dma_buffer(session, buffer)
    assert result.shape == (records, channels, samples)
    np.testing.assert_array_equal(result[1, 0], data[16:24])  # record 1, CHA
    np.testing.assert_array_equal(result[1, 1], data[24:32])  # record 1, CHB


def test_capture_data_channel_selects_a_or_b():
    volts = np.zeros((4, 2, 16))
    volts[:, 0] = 1.0
    volts[:, 1] = 2.0
    capture = CaptureData(volts, volts, SAMPLE_RATE)
    assert np.all(capture.channel(CHANNEL_A) == 1.0)
    assert np.all(capture.channel(CHANNEL_B) == 2.0)


def test_awg_alazar_acquire_uses_reference_channel():
    from QAWG.awg_alazar import AWGAlazar

    rng = np.random.default_rng(3)
    shots = 64
    shifts = rng.uniform(-0.5e-9, 0.5e-9, shots)

    class Source:
        def stop(self):
            pass

        def run(self, wait_until_ready=False):
            pass

    class Digitizer:
        def capture(self, config, *, start_callback=None):
            assert config.channel == CHANNEL_AB
            start_callback()
            cha = 0.01 * _records(shifts, 50e6, 0.2)  # reference tone on CHA
            chb = 0.02 * _records(shifts, 50e6, 0.9)  # signal on CHB
            volts = np.stack([cha, chb], axis=1)
            return CaptureData(volts, volts, SAMPLE_RATE)

    def make(reference_channel):
        controller = AWGAlazar(
            Source(),
            object(),
            BoardInfo(1, 0, 0, 12),
            awg_sample_rate_hz=1e9,
            alazar_sample_rate_hz=1e9,
            acquire_window_ns=1024,
            tone_frequency_hz=50e6,
            integrate_window_ns=(100, 900),
            adc_channel="CHB",
            reference_channel=reference_channel,
            remove_dc_offset=False,
        )
        controller.alazar = Digitizer()
        return controller

    controller = make("CHA")
    corrected, _ = controller.acquire(n_average=shots)
    assert controller.last_reference_deviation_rad.shape == (shots,)
    phase = np.angle(corrected * np.exp(-1j * np.angle(corrected.mean())))
    assert np.std(phase) < 2e-3
    np.testing.assert_allclose(np.abs(corrected), 0.02, rtol=1e-2)

    with pytest.raises(ValueError):
        make("CHB")  # reference must differ from the signal channel
