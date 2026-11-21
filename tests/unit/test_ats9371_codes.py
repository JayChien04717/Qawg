import numpy as np

from QAWG.alazar.ats9371 import adc_codes_to_volts


def _stored_12_bit_codes(*codes: int) -> np.ndarray:
    return np.left_shift(np.asarray(codes, dtype=np.uint16), 4)


def test_external_trigger_lsb_mask_maps_both_states_to_same_voltage():
    raw = _stored_12_bit_codes(2048, 2049)

    unmasked = adc_codes_to_volts(raw, 12, 0.4)
    masked = adc_codes_to_volts(
        raw,
        12,
        0.4,
        mask_external_trigger_lsb=True,
    )

    assert unmasked[1] - unmasked[0] == 0.4 / 2048
    assert masked[0] == masked[1]


def test_external_trigger_lsb_mask_uses_quantization_bin_midpoint():
    raw = _stored_12_bit_codes(2048, 2049, 2050, 2051)
    masked = adc_codes_to_volts(
        raw,
        12,
        0.4,
        mask_external_trigger_lsb=True,
    )

    half_lsb = 0.5 * 0.4 / 2048
    np.testing.assert_allclose(
        masked,
        [half_lsb, half_lsb, 5 * half_lsb, 5 * half_lsb],
    )
