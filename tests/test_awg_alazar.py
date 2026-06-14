from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from QAWG.alazar import BoardInfo
from QAWG.alazar import AcquisitionConfig
from QAWG.alazar.constants import CHANNEL_A, CHANNEL_B
from QAWG.alazar.constants import (
    TRIGGER_SLOPE_NEGATIVE,
    TRIGGER_SLOPE_POSITIVE,
)
from QAWG.awg_alazar import (
    AWGAlazar,
    normalize_adc_channel,
    normalize_trigger_slope,
    records_per_buffer_for,
)


class AWGAlazarTests(unittest.TestCase):
    def make_experiment(self, **overrides: object) -> AWGAlazar:
        settings: dict[str, object] = {
            "awg_sample_rate_hz": 2.5e9,
            "alazar_sample_rate_hz": 1e9,
            "tone_frequency_hz": 50e6,
            "trigger_delay_s": 100e-9,
            "acquire_window_ns": 256,
            "integrate_window_ns": (20, 220),
            "moving_average_time_s": 20e-9,
            "use_external_10mhz_reference": True,
        }
        settings.update(overrides)
        return AWGAlazar(
            Mock(),
            Mock(),
            BoardInfo(handle=1, kind=1, memory_samples=1_000_000, bits_per_sample=12),
            **settings,
        )

    def test_time_conversions_use_the_correct_instrument_clock(self) -> None:
        experiment = self.make_experiment()

        self.assertEqual(experiment.ns2cycles(100), 250)
        self.assertEqual(experiment.ns2cycles(100, inst="dac"), 250)
        self.assertEqual(experiment.ns2cycles(100, inst="adc"), 100)
        self.assertEqual(experiment.cycles2ns(250), 100.0)
        self.assertEqual(experiment.cycles2ns(100, inst="adc"), 100.0)
        self.assertEqual(experiment.trigger_delay_samples, 100)
        self.assertEqual(experiment.acquire_window_cycles, 256)
        self.assertEqual(experiment.integrate_window_cycles, (20, 220))
        self.assertEqual(experiment.integrate_samples, 200)

    def test_time_conversion_rejects_unknown_instrument(self) -> None:
        experiment = self.make_experiment()

        with self.assertRaisesRegex(ValueError, "dac.*adc"):
            experiment.ns2cycles(100, inst="unknown")

    def test_adc_channel_accepts_names_and_zero_based_indices(self) -> None:
        self.assertEqual(normalize_adc_channel("CHA"), CHANNEL_A)
        self.assertEqual(normalize_adc_channel("cha"), CHANNEL_A)
        self.assertEqual(normalize_adc_channel(0), CHANNEL_A)
        self.assertEqual(normalize_adc_channel("CHB"), CHANNEL_B)
        self.assertEqual(normalize_adc_channel("chb"), CHANNEL_B)
        self.assertEqual(normalize_adc_channel(1), CHANNEL_B)

    def test_adc_channel_rejects_unknown_values(self) -> None:
        for value in ("A", "CHC", 2, True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "CHA.*CHB.*0.*1"):
                    normalize_adc_channel(value)

    def test_trigger_slope_accepts_edge_names(self) -> None:
        self.assertEqual(
            normalize_trigger_slope("rising"),
            TRIGGER_SLOPE_POSITIVE,
        )
        self.assertEqual(
            normalize_trigger_slope("falling"),
            TRIGGER_SLOPE_NEGATIVE,
        )

    def test_records_per_buffer_is_a_divisor(self) -> None:
        self.assertEqual(records_per_buffer_for(1000), 100)
        self.assertEqual(records_per_buffer_for(32), 32)
        self.assertEqual(records_per_buffer_for(37), 37)

    def test_acquire_window_is_aligned_up_without_shortening(self) -> None:
        experiment = self.make_experiment(
            acquire_window_ns=1200,
            integrate_window_ns=(100, 1200),
        )

        self.assertEqual(experiment.acquire_window_cycles, 1280)

    def test_acquisition_config_alignment_never_shortens_record(self) -> None:
        config = AcquisitionConfig(
            tone_frequency_hz=0,
            samples_per_record=257,
        )

        self.assertEqual(config.samples_per_record, 384)
        from QAWG.alazar.ats9371 import validate_acquisition_config

        validate_acquisition_config(config)

    def test_integrate_time_uses_trace_start(self) -> None:
        experiment = self.make_experiment(
            acquire_window_ns=1500,
            integrate_window_ns=None,
            integrate_time_s=1e-6,
        )

        self.assertEqual(experiment.integrate_window_cycles, (0, 1000))
        self.assertEqual(experiment.integrate_samples, 1000)

    def test_integration_defaults_to_requested_acquire_window(self) -> None:
        experiment = self.make_experiment(
            acquire_window_ns=1500,
            integrate_window_ns=None,
        )

        self.assertEqual(experiment.integrate_window_cycles, (0, 1500))

    def test_rejects_two_integration_forms(self) -> None:
        with self.assertRaisesRegex(ValueError, "not both"):
            self.make_experiment(
                integrate_window_ns=(20, 220),
                integrate_time_s=100e-9,
            )

    def test_acquire_decimate_returns_time_resolved_average(self) -> None:
        experiment = self.make_experiment()
        time_s = np.arange(256) / experiment.alazar_sample_rate_hz
        records = np.array(
            [
                0.1 * np.cos(2 * np.pi * 50e6 * time_s),
                0.1 * np.cos(2 * np.pi * 50e6 * time_s),
                0.1 * np.cos(2 * np.pi * 50e6 * time_s),
                0.1 * np.cos(2 * np.pi * 50e6 * time_s),
            ]
        )

        with patch.object(
            experiment,
            "_capture_records",
            return_value=records,
        ):
            output_time_s, average_iq = experiment.acquire_decimate(4)

        self.assertEqual(output_time_s[0], 0.0)
        self.assertEqual(average_iq.size, 237)
        np.testing.assert_allclose(average_iq.real, 0.1, atol=1e-12)
        np.testing.assert_allclose(average_iq.imag, 0.0, atol=1e-12)
        self.assertEqual(experiment.last_shot_iq.shape, (4, 237))

    def test_acquire_decimate_supports_various_filters(self) -> None:
        experiment = self.make_experiment()
        time_s = np.arange(256) / experiment.alazar_sample_rate_hz
        records = np.array(
            [
                0.1 * np.cos(2 * np.pi * 50e6 * time_s),
                0.1 * np.cos(2 * np.pi * 50e6 * time_s),
            ]
        )

        with patch.object(
            experiment,
            "_capture_records",
            return_value=records,
        ):
            t_s, avg_iq = experiment.acquire_decimate(2, filter_type="butterworth")
            self.assertEqual(avg_iq.size, 256)
            self.assertEqual(experiment.last_shot_iq.shape, (2, 256))

            t_s, avg_iq = experiment.acquire_decimate(2, filter_type="elliptic")
            self.assertEqual(avg_iq.size, 256)
            self.assertEqual(experiment.last_shot_iq.shape, (2, 256))

            t_s, avg_iq = experiment.acquire_decimate(2, filter_type="notch")
            self.assertEqual(avg_iq.size, 256)
            self.assertEqual(experiment.last_shot_iq.shape, (2, 256))

            with self.assertRaises(ValueError):
                experiment.acquire_decimate(2, filter_type="invalid")

    def test_acquire_returns_one_integrated_iq_point(self) -> None:
        experiment = self.make_experiment()
        time_s = np.arange(256) / experiment.alazar_sample_rate_hz
        record = np.zeros(256)
        record[20:220] = (
            0.1 * np.cos(2 * np.pi * 50e6 * time_s[20:220])
        )
        records = np.repeat(record[None, :], 4, axis=0)

        with patch.object(
            experiment,
            "_capture_records",
            return_value=records,
        ):
            iq, downconverted_iq = experiment.acquire(4)

        self.assertAlmostEqual(iq.real, 0.1, places=12)
        self.assertAlmostEqual(iq.imag, 0.0, places=12)
        self.assertEqual(downconverted_iq.shape, (4, 256))
        np.testing.assert_array_equal(
            downconverted_iq,
            experiment.last_downconverted_iq,
        )
        self.assertEqual(experiment.last_shot_iq.shape, (4,))
        self.assertEqual(experiment.last_time_s[0], 0.0)
        self.assertAlmostEqual(experiment.last_time_s[1], 1e-9)

    def test_acquire_sequence_traces_averages_matching_steps(self) -> None:
        experiment = self.make_experiment()
        time_s = np.arange(256) / experiment.alazar_sample_rate_hz
        step_zero = 0.1 * np.cos(2 * np.pi * 50e6 * time_s)
        step_one = 0.2 * np.cos(2 * np.pi * 50e6 * time_s)
        records = np.array(
            [step_zero, step_one, step_zero, step_one, step_zero, step_one]
        )

        with patch.object(
            experiment,
            "_capture_records",
            return_value=records,
        ) as mock_capture:
            raw_time, average_records, iq_time, average_iq = (
                experiment.acquire_sequence_traces(
                    number_of_steps=2,
                    number_of_averages=3,
                )
            )

        mock_capture.assert_called_once_with(n_average=6)
        self.assertEqual(average_records.shape, (2, 256))
        self.assertEqual(average_iq.shape, (2, 237))
        np.testing.assert_allclose(average_records[0], step_zero)
        np.testing.assert_allclose(average_records[1], step_one)
        np.testing.assert_allclose(average_iq[0].real, 0.1, atol=1e-12)
        np.testing.assert_allclose(average_iq[1].real, 0.2, atol=1e-12)
        self.assertEqual(raw_time.size, 256)
        self.assertEqual(iq_time.size, 237)
        self.assertEqual(
            experiment.last_sequence_records_volts.shape,
            (3, 2, 256),
        )

    def test_acquisition_config_respects_n_average(self) -> None:
        experiment = self.make_experiment()
        self.assertEqual(experiment._acquisition_config(4).num_averages, 4)

        config = experiment._acquisition_config(n_average=8)
        self.assertEqual(config.num_averages, 8)
        self.assertEqual(config.records_per_buffer, 8)

    def test_acquire_and_decimate_pass_n_average_to_capture_records(self) -> None:
        experiment = self.make_experiment()

        with patch.object(
            experiment,
            "_capture_records",
            return_value=np.zeros((8, 256)),
        ) as mock_capture:
            experiment.acquire_decimate(n_average=8)
            mock_capture.assert_called_once_with(n_average=8)

            mock_capture.reset_mock()
            experiment.acquire(n_average=8)
            mock_capture.assert_called_once_with(n_average=8)

    def test_acquire_records_returns_time_axis_and_raw_records(self) -> None:
        experiment = self.make_experiment()
        records = np.ones((5, 256))

        with patch.object(
            experiment,
            "_capture_records",
            return_value=records,
        ) as mock_capture:
            time_s, acquired = experiment.acquire_records(n_average=5)

        mock_capture.assert_called_once_with(n_average=5)
        self.assertEqual(time_s.shape, (256,))
        self.assertAlmostEqual(time_s[1], 1e-9)
        np.testing.assert_array_equal(acquired, records)
        self.assertIsNot(acquired, records)

    def test_capture_diagnostics_reports_adc_resolution_and_offset(self) -> None:
        experiment = self.make_experiment(adc_channel="CHB")
        experiment.last_raw_codes = np.array([[32768, 32784]], dtype=np.uint16)
        experiment.last_records_volts = np.array([[0.0, 0.0001953125]])

        diagnostics = experiment.capture_diagnostics()

        self.assertEqual(diagnostics["adc_channel"], "CHB")
        self.assertAlmostEqual(diagnostics["adc_lsb_mv"], 0.1953125)
        self.assertEqual(diagnostics["raw_code_min"], 32768)
        self.assertEqual(diagnostics["raw_code_max"], 32784)

    def test_rejects_software_range_that_hardware_did_not_configure(self) -> None:
        with self.assertRaisesRegex(ValueError, "fixed.*400 mV"):
            self.make_experiment(input_range_volts=0.2)

    def test_integrate_window_must_fit_inside_acquire_window(self) -> None:
        with self.assertRaisesRegex(ValueError, "fit inside"):
            self.make_experiment(integrate_window_ns=(20, 300))

    def test_configure_applies_both_sample_clocks_and_trigger_delay(self) -> None:
        experiment = self.make_experiment()

        with patch("QAWG.awg_alazar.configure_ats9371") as configure:
            experiment.configure()

        experiment.awg.set_awg_mode.assert_called_once_with()
        experiment.awg.use_external_10mhz_reference.assert_called_once_with()
        experiment.awg.set_sample_rate.assert_called_once_with(2.5e9)
        trigger = configure.call_args.args[2]
        self.assertEqual(trigger.delay_samples, 100)
        self.assertEqual(trigger.slope, TRIGGER_SLOPE_POSITIVE)
        self.assertEqual(trigger.level, 140)
        self.assertEqual(configure.call_args.kwargs["channel"], CHANNEL_A)

    def test_channel_b_is_used_for_configuration_and_acquisition(self) -> None:
        experiment = self.make_experiment(adc_channel=1)

        with patch("QAWG.awg_alazar.configure_ats9371") as configure:
            experiment.configure()

        self.assertEqual(experiment.adc_channel_name, "CHB")
        self.assertEqual(configure.call_args.kwargs["channel"], CHANNEL_B)
        self.assertEqual(experiment._acquisition_config(4).channel, CHANNEL_B)

    def test_configure_experiment_applies_readout_owned_settings(self) -> None:
        experiment = self.make_experiment()

        with patch("QAWG.awg_alazar.configure_ats9371") as configure:
            experiment.configure_experiment(
                tone_frequency_hz=75e6,
                trigger_delay_s=30e-9,
                integrate_time_s=120e-9,
                adc_channel="CHB",
            )

        self.assertEqual(experiment.tone_frequency_hz, 75e6)
        self.assertEqual(experiment.trigger_delay_samples, 30)
        self.assertEqual(experiment.integrate_window_cycles, (0, 120))
        self.assertEqual(experiment.adc_channel_name, "CHB")
        self.assertEqual(configure.call_args.kwargs["channel"], CHANNEL_B)

    def test_compiled_acquisition_is_owned_by_hardware_coordinator(self) -> None:
        experiment = self.make_experiment(acquire_window_ns=256)
        readout = SimpleNamespace(
            name="ro",
            length_s=100e-9,
            integrate_time_s=80e-9,
            demod_frequency_hz=50e6,
            adc_channel="CHA",
        )
        compiled = SimpleNamespace(
            readout=readout,
            trigger_delay_s=30e-9,
            number_of_sequence_steps=2,
            axes={"gain": np.array([0.1, 0.2])},
            point_coordinates=({"gain": 0.1}, {"gain": 0.2}),
        )
        experiment._uploaded_compiled = compiled
        experiment.last_sequence_records_volts = np.ones((3, 2, 256))
        experiment.last_sequence_shot_iq = np.ones(
            (3, 2, 237),
            dtype=complex,
        )

        with (
            patch.object(experiment, "configure_experiment") as configure,
            patch.object(
                experiment,
                "acquire_sequence_traces",
                return_value=(
                    np.arange(256) / 1e9,
                    np.ones((2, 256)),
                    np.arange(237) / 1e9,
                    np.ones((2, 237), dtype=complex),
                ),
            ) as acquire,
        ):
            result = experiment.acquire_compiled_experiment(
                compiled,
                n_average=3,
            )

        configure.assert_called_once_with(
            tone_frequency_hz=50e6,
            trigger_delay_s=30e-9,
            integrate_time_s=80e-9,
            adc_channel="CHA",
        )
        acquire.assert_called_once_with(
            number_of_steps=2,
            number_of_averages=3,
            filter_type="boxcar",
        )
        self.assertEqual(result.raw.shape, (3, 2, 256))
        self.assertEqual(result.shots().shape, (3, 2))

    def test_process_multiplex_integrate(self) -> None:
        from QAWG.alazar import AlazarProcessor
        processor = AlazarProcessor(sample_rate_hz=1e9)
        time_s = np.arange(1000) / 1e9
        # Construct raw signal with 50 MHz and 150 MHz components
        sig_50 = 0.1 * np.cos(2 * np.pi * 50e6 * time_s)
        sig_150 = 0.05 * np.cos(2 * np.pi * 150e6 * time_s)
        records = np.array([sig_50 + sig_150])

        results = processor.process_multiplex_integrate(
            records_volts=records,
            tone_frequencies_hz={"q0": 50e6, "q1": 150e6},
            integrate_start=0,
            integrate_stop=1000,
        )

        self.assertIn("q0", results)
        self.assertIn("q1", results)

        _, _, avg_q0 = results["q0"]
        _, _, avg_q1 = results["q1"]

        self.assertAlmostEqual(avg_q0.real, 0.1, places=3)
        self.assertAlmostEqual(avg_q0.imag, 0.0, places=3)
        self.assertAlmostEqual(avg_q1.real, 0.05, places=3)
        self.assertAlmostEqual(avg_q1.imag, 0.0, places=3)

    def test_apply_butterworth_lpf(self) -> None:
        from QAWG.alazar import AlazarProcessor
        processor = AlazarProcessor(sample_rate_hz=1e9)
        time_s = np.arange(1000) / 1e9
        baseband = np.exp(1j * 2 * np.pi * 10e6 * time_s) + np.exp(1j * 2 * np.pi * 200e6 * time_s)
        baseband_records = np.array([baseband])

        filtered = processor.apply_butterworth_lpf(baseband_records, cutoff_hz=30e6, order=4)

        fft_orig = np.abs(np.fft.fft(baseband))
        fft_filt = np.abs(np.fft.fft(filtered[0]))

        self.assertGreater(fft_filt[10], 0.8 * fft_orig[10])
        self.assertLess(fft_filt[200], 0.05 * fft_orig[200])


if __name__ == "__main__":
    unittest.main()
