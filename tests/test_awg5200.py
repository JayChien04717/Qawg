from __future__ import annotations

import re
import unittest

import numpy as np

from QAWG.awg5200 import (
    AWG5208,
    TriggerConfig,
    align_channel_envelopes,
    align_channels,
    cosine_square,
    cosine_square_ns,
    delay,
    delay_auto,
    gaussian_cosine_ns,
    gaussian_square_ns,
    marker_window,
    marker_window_ns,
    ns_to_samples,
    parallel,
    sine,
    trigger_channel_for,
    waveform,
)
from QAWG.awg5200.driver import ieee_block
from QAWG.awg5200.waveforms import make_wfmx, pack_markers


class FakeTransport:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.raw_messages: list[bytes] = []
        self.responses = {
            "*IDN?": "TEKTRONIX,AWG5208,B012345,7.1",
            "*OPC?": "1",
            "SYSTem:ERRor:CODE?": "0",
            "AWGControl:RSTATe?": "1",
        }
        self.closed = False

    def write(self, command: str) -> None:
        self.commands.append(command)

    def query(self, command: str) -> str:
        self.commands.append(command)
        return self.responses.get(command, "0")

    def write_raw(self, message: bytes) -> None:
        self.raw_messages.append(message)

    def close(self) -> None:
        self.closed = True


class WaveformTests(unittest.TestCase):
    def test_sine_shape(self) -> None:
        values = sine(2400, 2.5e9, 100e6, 0.2)
        self.assertEqual(values.shape, (2400,))
        self.assertLessEqual(np.max(np.abs(values)), 0.2)

    def test_nanosecond_helpers(self) -> None:
        self.assertEqual(ns_to_samples(240, 10e9), 2400)
        waveform = gaussian_cosine_ns(
            duration_ns=240,
            sample_rate_hz=10e9,
            frequency_hz=50e6,
            sigma_ns=20,
            amplitude_volts=0.2,
            center_ns=120,
        )
        marker = marker_window_ns(240, 10e9, 100, 140)
        self.assertEqual(waveform.shape, (2400,))
        self.assertEqual(marker.shape, (2400,))
        self.assertEqual(np.count_nonzero(marker), 400)

    def test_gaussian_square_ns(self) -> None:
        envelope = gaussian_square_ns(
            duration_ns=100,
            sample_rate_hz=2.5e9,
            edge_sigma_ns=10,
            amplitude_volts=0.2,
        )
        self.assertEqual(envelope.size, 250)
        self.assertLess(envelope[0], 0.001)
        self.assertEqual(envelope[125], 0.2)

    def test_cosine_square_has_symmetric_edges_and_flat_top(self) -> None:
        envelope = cosine_square(
            number_of_samples=100,
            sample_rate_hz=1e9,
            edge_length_s=20e-9,
            amplitude_volts=0.2,
        )

        self.assertEqual(envelope[0], 0.0)
        self.assertEqual(envelope[-1], 0.0)
        self.assertEqual(envelope[19], 0.2)
        self.assertTrue(np.all(envelope[20:80] == 0.2))
        np.testing.assert_allclose(envelope[:20], envelope[-20:][::-1])

    def test_cosine_square_ns_converts_edge_length(self) -> None:
        envelope = cosine_square_ns(
            duration_ns=100,
            sample_rate_hz=2.5e9,
            edge_length_ns=20,
            amplitude_volts=0.2,
        )

        self.assertEqual(envelope.size, 250)
        self.assertEqual(envelope[0], 0.0)
        self.assertEqual(envelope[50], 0.2)

    def test_marker_bits(self) -> None:
        first = marker_window(2400, 10, 20)
        second = marker_window(2400, 15, 25)
        packed = pack_markers((first, second), 2400)
        self.assertEqual(packed[9], 0)
        self.assertEqual(packed[12], 1)
        self.assertEqual(packed[17], 3)
        self.assertEqual(packed[22], 2)

    def test_trigger_channel_matches_reference_waveform(self) -> None:
        reference = np.zeros(10_000)
        reference[4750:4800] = np.sin(np.linspace(0, 4 * np.pi, 50))
        trigger_waveform, trigger_marker = trigger_channel_for(
            reference_waveform=reference,
        )
        self.assertEqual(trigger_waveform.shape, reference.shape)
        self.assertEqual(trigger_marker.shape, reference.shape)
        self.assertTrue(np.all(trigger_waveform == 0))
        self.assertTrue(np.all(trigger_marker[4751:4799]))
        self.assertFalse(trigger_marker[4750])
        self.assertFalse(trigger_marker[4799])

    def test_delay_is_relative_to_previous_start(self) -> None:
        first = waveform(np.ones(100), fc=0, ch=2)
        second = waveform(np.ones(50), fc=0, ch=3)
        rendered = align_channels(
            first / delay(50e-9) / second,
            sample_rate_hz=1e9,
        )
        self.assertEqual(rendered[2].size, 5000)
        self.assertTrue(np.all(rendered[2][:100] == 1))
        self.assertTrue(np.all(rendered[3][50:100] == 1))

    def test_leading_delay_offsets_first_waveform(self) -> None:
        readout = waveform(np.ones(500), fc=0, ch=3)
        rendered = align_channels(
            delay(1e-6) / readout,
            sample_rate_hz=1e9,
        )
        self.assertTrue(np.all(rendered[3][:1000] == 0))
        self.assertTrue(np.all(rendered[3][1000:1500] == 1))

    def test_aligned_envelope_is_independent_of_carrier_phase(self) -> None:
        envelope = np.hanning(200)
        phase_zero = delay(1e-6) / waveform(
            envelope, fc=50e6, ch=3, phase_radians=0.0
        )
        phase_quadrature = delay(1e-6) / waveform(
            envelope, fc=50e6, ch=3, phase_radians=np.pi / 2
        )

        zero = align_channel_envelopes(phase_zero, 1e9)[3]
        quadrature = align_channel_envelopes(phase_quadrature, 1e9)[3]

        np.testing.assert_array_equal(zero, quadrature)
        np.testing.assert_array_equal(zero[1000:1200], envelope)

    def test_delay_auto_is_relative_to_previous_end(self) -> None:
        first = waveform(np.ones(100), fc=0, ch=2)
        second = waveform(np.ones(50), fc=0, ch=3)
        rendered = align_channels(
            first / delay_auto(20e-9) / second,
            sample_rate_hz=1e9,
        )
        self.assertTrue(np.all(rendered[3][120:170] == 1))
        self.assertTrue(np.all(rendered[3][:120] == 0))
        self.assertEqual(rendered[3].size, 5000)

    def test_parallel_group_starts_together_and_waits_for_longest(self) -> None:
        qubit1 = waveform(np.ones(100), fc=0, ch=2)
        qubit2 = waveform(np.ones(200), fc=0, ch=4)
        readout = waveform(np.ones(50), fc=0, ch=3)

        rendered = align_channels(
            parallel(qubit1, qubit2) / delay_auto(20e-9) / readout,
            sample_rate_hz=1e9,
        )

        self.assertTrue(np.all(rendered[2][:100] == 1))
        self.assertTrue(np.all(rendered[4][:200] == 1))
        self.assertTrue(np.all(rendered[3][:220] == 0))
        self.assertTrue(np.all(rendered[3][220:270] == 1))

    def test_parallel_waveforms_can_share_one_channel(self) -> None:
        first = waveform(np.ones(100), fc=0, ch=2)
        second = waveform(np.full(50, 0.5), fc=0, ch=2)
        rendered = align_channels(
            parallel(first, second) / delay_auto(0) / waveform(
                np.ones(10), fc=0, ch=3
            ),
            sample_rate_hz=1e9,
        )
        self.assertTrue(np.all(rendered[2][:50] == 1.5))
        self.assertTrue(np.all(rendered[2][50:100] == 1.0))

    def test_timeline_extends_past_default_duration(self) -> None:
        long_pulse = waveform(np.ones(6000), fc=0, ch=2)
        rendered = align_channels(long_pulse / delay_auto(0) / waveform(
            np.ones(100), fc=0, ch=3
        ), sample_rate_hz=1e9)
        self.assertEqual(rendered[2].size, 6100)
        self.assertEqual(rendered[3].size, 6100)

    def test_wfmx_contains_float_and_marker_data(self) -> None:
        waveform = np.zeros(2400)
        marker = marker_window(2400, 100, 200)
        result = make_wfmx(waveform, 0.5, (marker,))
        self.assertIn(b"<NumberSamples>2400</NumberSamples>", result)
        match = re.search(br'<DataFile offset="(\d{9})"', result)
        self.assertIsNotNone(match)
        offset = int(match.group(1))
        self.assertEqual(result[offset : offset + 4], b"\x00\x00\x00\x00")
        self.assertEqual(len(result) - offset, 2400 * 5)

    def test_waveform_range_error_reports_required_vpp(self) -> None:
        with self.assertRaisesRegex(ValueError, r"use amplitude_vpp >= 0.6 V"):
            make_wfmx(np.full(2400, 0.3), amplitude_vpp=0.5)

    def test_ieee_block(self) -> None:
        self.assertEqual(ieee_block(b"abcd"), b"#14abcd")


class DriverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.transport = FakeTransport()
        self.awg = AWG5208(self.transport)

    def test_identity(self) -> None:
        self.awg.verify_identity()

    def test_configure_and_trigger(self) -> None:
        self.awg.set_sample_rate(2.5e9)
        self.awg.configure_trigger(TriggerConfig(level_volts=0.4))
        self.awg.force_trigger()
        self.assertIn("CLOCk:SRATe 2500000000", self.transport.commands)
        self.assertIn("TRIGger:A:LEVel 0.4", self.transport.commands)
        self.assertIn("TRIGger:IMMediate ATRigger", self.transport.commands)

    def test_rejects_sample_rate_above_hardware_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "2.5 GSa/s"):
            self.awg.set_sample_rate(2.5e9 + 1)
        self.assertFalse(
            any(command.startswith("CLOCk:SRATe") for command in self.transport.commands)
        )

    def test_run_can_wait_until_ready(self) -> None:
        self.awg.run(wait_until_ready=True)
        self.assertEqual(
            self.transport.commands[-2:],
            ["AWGControl:RUN", "*OPC?"],
        )

    def test_clear_all_stops_outputs_and_clears_assets(self) -> None:
        self.awg._waveforms["old"] = np.zeros(2400)
        self.awg._assigned_waveforms[2] = "old"

        self.awg.clear_all()

        self.assertEqual(self.transport.commands[0], "AWGControl:STOP")
        for channel in range(1, 9):
            self.assertIn(f"OUTPut{channel}:STATe 0", self.transport.commands)
            self.assertIn(
                f"SOURce{channel}:CASSet:CLEAR",
                self.transport.commands,
            )
        self.assertIn("SLISt:SEQuence:DELete ALL", self.transport.commands)
        self.assertIn("WLISt:WAVeform:DELete ALL", self.transport.commands)
        self.assertEqual(self.transport.commands[-1], "*OPC?")
        self.assertEqual(self.awg._waveforms, {})
        self.assertEqual(self.awg._activity_waveforms, {})
        self.assertEqual(self.awg._assigned_waveforms, {})

    def test_upload_and_assign(self) -> None:
        envelope = np.zeros(2400)
        self.awg.set_sample_rate(2.5e9)
        name = self.awg.upload_waveform(
            envelope,
            fc=50e6,
            ch=8,
            amplitude_vpp=0.5,
            name="readout",
        )
        self.assertEqual(name, "readout")
        self.assertTrue(
            self.transport.raw_messages[0].startswith(
                b'MMEMory:DATA "readout.wfmx",#'
            )
        )
        self.assertIn(
            'SOURce8:CASSet:WAVeform "readout"', self.transport.commands
        )
        self.assertIn("OUTPut8:STATe 1", self.transport.commands)
        self.assertLess(
            self.transport.commands.index("WLISt:WAVeform:DELete ALL"),
            self.transport.commands.index('MMEMory:DELete "readout.wfmx"'),
        )

    def test_upload_requires_sample_rate(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "set_sample_rate"):
            self.awg.upload_waveform(np.zeros(2400), fc=50e6, ch=3)

    def test_create_sequence_assigns_tracks_and_loops(self) -> None:
        self.awg._waveforms.update(
            {
                "marker_100ns": np.zeros(3750),
                "marker_200ns": np.zeros(3750),
                "readout_100ns": np.zeros(3750),
                "readout_200ns": np.zeros(3750),
            }
        )

        name = self.awg.create_sequence(
            "length_sweep",
            {
                1: ["marker_100ns", "marker_200ns"],
                3: ["readout_100ns", "readout_200ns"],
            },
        )

        self.assertEqual(name, "length_sweep")
        self.assertIn(
            'SLISt:SEQuence:NEW "length_sweep",2,2',
            self.transport.commands,
        )
        self.assertIn(
            'SLISt:SEQuence:STEP1:TASSet1:WAVeform '
            '"length_sweep","marker_100ns"',
            self.transport.commands,
        )
        self.assertIn(
            'SLISt:SEQuence:STEP2:TASSet2:WAVeform '
            '"length_sweep","readout_200ns"',
            self.transport.commands,
        )
        self.assertIn(
            'SLISt:SEQuence:STEP2:GOTO "length_sweep",1',
            self.transport.commands,
        )
        self.assertIn(
            'SOURce1:CASSet:SEQuence "length_sweep",1',
            self.transport.commands,
        )
        self.assertIn(
            'SOURce3:CASSet:SEQuence "length_sweep",2',
            self.transport.commands,
        )

    def test_upload_modulates_envelope_using_sample_rate(self) -> None:
        self.awg.set_sample_rate(2.5e9)
        envelope = np.full(2400, 0.2)
        name = self.awg.upload_waveform(
            envelope,
            fc=50e6,
            ch=3,
            name="modulated",
        )
        expected = envelope * np.sin(
            2 * np.pi * 50e6 * np.arange(2400) / 2.5e9
        )
        np.testing.assert_allclose(self.awg._waveforms[name], expected, atol=1e-13)
        np.testing.assert_array_equal(
            self.awg._activity_waveforms[name],
            envelope,
        )
        self.assertIn("SOURce3:DAC:RESolution 16", self.transport.commands)

    def test_upload_fc_zero_preserves_envelope(self) -> None:
        self.awg.set_sample_rate(2.5e9)
        envelope = np.full(2400, 0.2)
        name = self.awg.upload_waveform(envelope, fc=0, ch=3, name="dc")
        np.testing.assert_array_equal(self.awg._waveforms[name], envelope)

    def test_marker_uses_assigned_waveform_length_and_active_region(self) -> None:
        waveform = np.zeros(3000)
        waveform[1000:1200] = 0.2 * np.hanning(200)
        name = self.awg._upload_waveform_data("pulse", waveform, 0.5)
        self.awg.prepare_channel(3, name, 0.5)

        marker_name = self.awg.marker(waveform_ch=3, marker_ch=1)

        self.assertEqual(marker_name, "marker_ch1_for_ch3")
        self.assertEqual(self.awg._waveforms[marker_name].size, waveform.size)
        self.assertIn("SOURce1:DAC:RESolution 15", self.transport.commands)
        self.assertIn(
            'SOURce1:CASSet:WAVeform "marker_ch1_for_ch3"',
            self.transport.commands,
        )

    def test_marker_can_share_analog_waveform_channel(self) -> None:
        self.awg.set_sample_rate(2.5e9)
        envelope = np.zeros(3000)
        envelope[1000:2000] = 0.2
        name = self.awg.upload_waveform(envelope, fc=0, ch=3, name="readout")

        marker_name = self.awg.marker(waveform_ch=3, marker_ch=3)

        np.testing.assert_array_equal(
            self.awg._waveforms[marker_name],
            self.awg._waveforms[name],
        )
        self.assertIn("SOURce3:DAC:RESolution 15", self.transport.commands)

    def test_upload_timeline_assigns_aligned_channels(self) -> None:
        self.awg.set_sample_rate(1e9)
        qubit = waveform(
            np.full(2400, 0.1),
            fc=50e6,
            ch=2,
        )
        readout = waveform(
            np.full(2400, 0.1),
            fc=20e6,
            ch=3,
        )
        names = self.awg.upload_timeline(
            qubit / delay(50e-9) / readout,
            name_prefix="experiment",
        )
        self.assertEqual(names, {2: "experiment_ch2", 3: "experiment_ch3"})
        self.assertEqual(
            self.awg._waveforms["experiment_ch2"].size,
            self.awg._waveforms["experiment_ch3"].size,
        )
        np.testing.assert_allclose(
            self.awg._activity_waveforms["experiment_ch3"][50:2450],
            0.1,
        )
        self.assertEqual(
            self.transport.commands.count("WLISt:WAVeform:DELete ALL"),
            1,
        )

    def test_upload_timeline_uses_explicit_channel_names(self) -> None:
        self.awg.set_sample_rate(1e9)
        qubit = waveform(np.full(2400, 0.1), fc=0, ch=2, name="qubit")
        readout = waveform(np.full(2400, 0.1), fc=0, ch=3, name="readout")

        names = self.awg.upload_timeline(
            qubit / delay_auto(0) / readout,
            name_prefix="fallback",
        )

        self.assertEqual(names, {2: "qubit", 3: "readout"})
        self.assertEqual(self.awg._assigned_waveforms[2], "qubit")
        self.assertEqual(self.awg._assigned_waveforms[3], "readout")

    def test_upload_timeline_range_error_reports_channel(self) -> None:
        self.awg.set_sample_rate(1e9)
        too_large = waveform(np.full(2400, 0.3), fc=0, ch=4)
        with self.assertRaisesRegex(ValueError, r"Channel 4: waveform peak 0.3 V"):
            self.awg.upload_timeline(too_large, amplitude_vpp={4: 0.5})

    def test_waveform_gain(self) -> None:
        raw_envelope = np.ones(100)
        wf = waveform(raw_envelope, fc=10e6, ch=3, gain=0.5)
        np.testing.assert_allclose(wf.envelope, 0.5)
        self.assertEqual(wf.gain, 0.5)

    def test_rejects_invalid_channel(self) -> None:
        with self.assertRaises(ValueError):
            self.awg.set_output(9, True)


if __name__ == "__main__":
    unittest.main()
