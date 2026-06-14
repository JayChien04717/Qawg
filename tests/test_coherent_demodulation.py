from __future__ import annotations

import unittest

import numpy as np

from alazar.ats9371 import configure_clock
from alazar.constants import (
    CLOCK_EDGE_RISING,
    EXTERNAL_CLOCK_10MHZ_REF,
    EXTERNAL_SAMPLE_RATE_1000MSPS,
    INTERNAL_CLOCK,
    SAMPLE_RATE_1000MSPS,
)
from alazar.demodulation import (
    correct_interleaving_offsets,
    recover_clock_referenced_envelope,
    recover_coherent_envelope,
)


class FakeApi:
    def __init__(self) -> None:
        self.clock_call: tuple[int, int, int, int, int] | None = None

    def set_capture_clock(
        self,
        handle: int,
        source: int,
        sample_rate: int,
        edge: int,
        decimation: int,
    ) -> None:
        self.clock_call = (handle, source, sample_rate, edge, decimation)


class ClockTests(unittest.TestCase):
    def test_external_10mhz_reference_clock(self) -> None:
        api = FakeApi()
        configure_clock(api, handle=123, use_external_10mhz_reference=True)
        self.assertEqual(
            api.clock_call,
            (
                123,
                EXTERNAL_CLOCK_10MHZ_REF,
                EXTERNAL_SAMPLE_RATE_1000MSPS,
                CLOCK_EDGE_RISING,
                0,
            ),
        )

    def test_internal_clock_uses_sample_rate_enum(self) -> None:
        api = FakeApi()
        configure_clock(api, handle=123)
        self.assertEqual(
            api.clock_call,
            (
                123,
                INTERNAL_CLOCK,
                SAMPLE_RATE_1000MSPS,
                CLOCK_EDGE_RISING,
                0,
            ),
        )


class CoherentDemodulationTests(unittest.TestCase):
    def test_interleaving_offset_correction_removes_odd_even_spur(self) -> None:
        records = np.zeros((3, 100), dtype=np.float64)
        records[:, 0::2] = -0.001
        records[:, 1::2] = 0.002
        records[:, 40:60] += 0.1

        corrected = correct_interleaving_offsets(
            records,
            stop_sample=20,
            period=2,
        )

        np.testing.assert_allclose(corrected[:, :20], 0.0, atol=1e-15)
        np.testing.assert_allclose(corrected[:, 40:60], 0.1, atol=1e-15)

    def test_clock_referenced_recovery_does_not_create_a_noise_pulse(self) -> None:
        rng = np.random.default_rng(4321)
        records = 0.002 * rng.standard_normal((1000, 1024))

        _, coherent, rms = recover_clock_referenced_envelope(
            records,
            sample_rate_hz=1e9,
            intermediate_frequency_hz=50e6,
            baseline_stop_sample=100,
            window_samples=20,
        )

        reference_window = np.mean(np.abs(coherent[300:500]))
        outside_window = np.mean(np.abs(coherent[600:800]))
        self.assertLess(reference_window, 0.0002)
        self.assertLess(abs(reference_window - outside_window), 0.0001)
        self.assertGreater(np.mean(rms), np.mean(np.abs(coherent)))

    def test_random_shot_phase_is_aligned_before_average(self) -> None:
        sample_rate_hz = 1e9
        frequency_hz = 50e6
        number_of_samples = 1024
        number_of_shots = 200
        time = np.arange(number_of_samples) / sample_rate_hz
        envelope = np.zeros(number_of_samples)
        envelope[200:700] = 0.1
        rng = np.random.default_rng(1234)
        phases = rng.uniform(-np.pi, np.pi, number_of_shots)
        records = envelope[None, :] * np.cos(
            2 * np.pi * frequency_hz * time[None, :] + phases[:, None]
        )
        records += 0.002 * rng.standard_normal(records.shape)
        records += rng.normal(0.01, 0.002, (number_of_shots, 1))

        _, coherent, magnitude, measured_phases = recover_coherent_envelope(
            records,
            sample_rate_hz,
            frequency_hz,
            baseline_stop_sample=150,
            phase_start_sample=300,
            phase_stop_sample=600,
            window_samples=20,
        )

        self.assertEqual(measured_phases.shape, (number_of_shots,))
        self.assertGreater(np.mean(np.abs(coherent[300:550])), 0.09)
        self.assertGreater(np.mean(magnitude[300:550]), 0.09)
        self.assertLess(np.mean(np.abs(coherent[:100])), 0.002)


if __name__ == "__main__":
    unittest.main()
