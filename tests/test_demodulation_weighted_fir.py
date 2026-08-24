import numpy as np

from QAWG.alazar import (
    design_lowpass_fir,
    digital_downconvert,
    fir_lowpass_iq,
    weighted_full_demodulate,
    weighted_integrate_iq,
)


def test_weighted_full_demodulation_recovers_tone_amplitude_and_phase():
    sample_rate = 1e9
    frequency = 50e6
    sample_count = 1000
    amplitude = 0.012
    phase = 0.37
    time = np.arange(sample_count) / sample_rate
    records = amplitude * np.cos(2 * np.pi * frequency * time + phase)

    result = weighted_full_demodulate(
        records[None, :],
        sample_rate,
        frequency,
    )

    np.testing.assert_allclose(result[0], amplitude * np.exp(1j * phase), atol=1e-14)


def test_weighted_integration_selects_requested_samples():
    baseband = np.asarray([[1 + 2j, 3 + 4j, 8 + 1j, 9 + 2j]])
    result = weighted_integrate_iq(
        baseband,
        1,
        4,
        integration_weights=[1.0, 0.0, 1.0],
    )
    np.testing.assert_allclose(result, [(3 + 4j + 9 + 2j) / 2])


def test_fir_has_unity_dc_gain_and_suppresses_100mhz_image():
    sample_rate = 1e9
    taps = design_lowpass_fir(sample_rate, 30e6, 81, 8.0)
    np.testing.assert_allclose(np.sum(taps), 1.0, atol=1e-14)

    sample_count = 4096
    time = np.arange(sample_count) / sample_rate
    baseband = 2.0 + np.exp(1j * 2 * np.pi * 100e6 * time)
    filtered = fir_lowpass_iq(baseband[None, :], taps)[0]
    interior = filtered[200:-200]

    assert filtered.shape == (sample_count,)
    assert np.std(interior) < 2e-4
    np.testing.assert_allclose(np.mean(interior), 2.0, atol=2e-4)


def test_fir_downconversion_keeps_record_shape():
    records = np.ones((3, 257))
    baseband = digital_downconvert(records, 1e9, 50e6)
    taps = design_lowpass_fir(1e9, 30e6)
    filtered = fir_lowpass_iq(baseband, taps)
    assert filtered.shape == records.shape
