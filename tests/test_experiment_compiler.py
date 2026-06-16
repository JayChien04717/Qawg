from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from QAWG import (
    CavityRingdownProgram,
    ExperimentProgram,
    ExperimentResult,
    LinearSweep,
    SingleShotProgram,
    ns,
    us,
)
from QAWG import PowerRabiProgram, T1Program
from QAWG.awg5200 import trigger_channel_for


class DelayProgram(ExperimentProgram):
    def _initialize(self, cfg):
        self.declare_gen("drive", ch=3)
        self.declare_readout(
            "ro",
            adc_channel="CHA",
            length=1 * us,
            demod_freq=50e6,
            waveform_ch=3,
            integrate_time=800 * ns,
        )
        self.delay = self.add_sweep(
            "delay",
            LinearSweep(0, 200 * ns, 6),
        )
        self.add_pulse(
            "pulse",
            gen="drive",
            style="gaussian",
            length=100 * ns,
            sigma=15 * ns,
            frequency=50e6,
            gain=0.02,
        )

    def _body(self, cfg):
        self.delay_auto(self.delay)
        self.play("pulse")
        self.trigger("ro", trigger_delay=30 * ns)


class ExperimentCompilerTests(unittest.TestCase):
    def test_cavity_ringdown_starts_acquisition_after_drive(self) -> None:
        cfg = {
            "frequency": 50e6,
            "drive_length": 2 * us,
            "ringdown_guard": 40 * ns,
            "acquire_length": 1 * us,
            "drive_gain": 0.02,
            "edge_sigma": 20 * ns,
        }
        compiled = CavityRingdownProgram(cfg).compile(
            sample_rate_hz=2.5e9
        )

        self.assertEqual(
            compiled.trigger_delay_s,
            cfg["drive_length"] + cfg["ringdown_guard"],
        )
        self.assertEqual(compiled.readout.length_s, cfg["acquire_length"])
        self.assertTrue(np.any(compiled.marker_waveforms[0]))
        self.assertEqual(compiled.preview(3).shape[0], 1)

    def test_exponential_pulse_uses_energy_lifetime(self) -> None:
        class ExponentialProgram(ExperimentProgram):
            def _initialize(self, cfg):
                self.declare_gen("drive", ch=3)
                self.declare_readout(
                    "ro",
                    adc_channel="CHA",
                    length=200 * ns,
                    demod_freq=0,
                    waveform_ch=3,
                )
                self.add_pulse(
                    "decay",
                    gen="drive",
                    style="exponential",
                    length=200 * ns,
                    decay=100 * ns,
                    frequency=0,
                    gain=0.1,
                )

            def _body(self, cfg):
                self.play("decay", at=0)
                self.trigger("ro")

        compiled = ExponentialProgram({}).compile(sample_rate_hz=1e9)
        waveform = compiled.preview(3)[0]
        self.assertAlmostEqual(waveform[0], 0.1)
        self.assertAlmostEqual(
            waveform[100] / waveform[0],
            np.exp(-0.5),
        )

    def test_delay_sweep_renders_trace_by_trace_steps(self) -> None:
        compiled = DelayProgram({}).compile(sample_rate_hz=2.5e9)

        self.assertEqual(compiled.number_of_sequence_steps, 6)
        self.assertEqual(compiled.trigger_delay_s, 30 * ns)
        np.testing.assert_allclose(
            compiled.axis("delay") / ns,
            [0, 40, 80, 120, 160, 200],
        )
        self.assertEqual(compiled.preview(3).shape[0], 6)

        starts = []
        for step_index, trace in enumerate(compiled.preview(3)):
            starts.append(np.flatnonzero(np.abs(trace) > 1e-5)[0])
            _, expected_marker = trigger_channel_for(trace)
            np.testing.assert_array_equal(
                compiled.marker_waveforms[step_index],
                expected_marker,
            )
        np.testing.assert_array_equal(np.diff(starts), 100)

    def test_tagged_readout_adds_marker_padding_and_shifts_shot(self) -> None:
        class TaggedReadoutProgram(ExperimentProgram):
            def _initialize(self, cfg):
                self.declare_gen("readout", ch=1)
                self.declare_gen("control", ch=2)
                self.declare_readout(
                    "ro",
                    adc_channel="CHA",
                    length=200 * ns,
                    demod_freq=50e6,
                    waveform_ch=1,
                    marker_channel=1,
                    marker_padding=500 * ns,
                )
                self.add_pulse(
                    "control",
                    gen="control",
                    style="const",
                    length=50 * ns,
                    frequency=0,
                    gain=0.01,
                )
                self.add_pulse(
                    "readout",
                    gen="readout",
                    style="const",
                    length=200 * ns,
                    frequency=0,
                    gain=0.02,
                    readout=True,
                )

            def _body(self, cfg):
                self.play("control", at=0)
                self.play("readout", at=100 * ns)
                self.trigger("ro")

        compiled = TaggedReadoutProgram({}).compile(
            sample_rate_hz=1e9
        )

        self.assertEqual(compiled.number_of_sequence_steps, 1)
        self.assertEqual(compiled.trigger_delay_s, 500 * ns)
        np.testing.assert_allclose(
            compiled.readout_windows_s[0],
            [500 * ns, 700 * ns],
        )
        readout = compiled.preview(1)[0]
        control = compiled.preview(2)[0]
        self.assertEqual(np.flatnonzero(np.abs(readout) > 0)[0], 500)
        self.assertEqual(np.flatnonzero(np.abs(control) > 0)[0], 400)
        marker = np.flatnonzero(compiled.marker_waveforms[0])
        self.assertEqual(marker[0], 0)
        self.assertEqual(marker[-1] + 1, 1200)

    def test_readout_length_sweep_changes_marker_stop(self) -> None:
        class TaggedLengthSweepProgram(ExperimentProgram):
            def _initialize(self, cfg):
                self.declare_gen("readout", ch=1)
                self.declare_readout(
                    "ro",
                    adc_channel="CHA",
                    length=200 * ns,
                    demod_freq=50e6,
                    waveform_ch=1,
                    marker_channel=1,
                    marker_padding=500 * ns,
                )
                length = self.add_sweep(
                    "length",
                    LinearSweep(100 * ns, 200 * ns, 2),
                )
                self.add_pulse(
                    "readout",
                    gen="readout",
                    style="const",
                    length=length,
                    frequency=0,
                    gain=0.02,
                    readout=True,
                )

            def _body(self, cfg):
                self.play("readout", at=0)
                self.trigger("ro")

        compiled = TaggedLengthSweepProgram({}).compile(
            sample_rate_hz=1e9
        )

        self.assertEqual(compiled.number_of_sequence_steps, 2)
        marker_stops = [
            np.flatnonzero(marker)[-1] + 1
            for marker in compiled.marker_waveforms
        ]
        self.assertEqual(marker_stops, [1100, 1200])

    def test_program_dc_offset_option_is_compiled(self) -> None:
        program = DelayProgram({})
        program.REMOVE_DC_OFFSET = True
        compiled = program.compile(sample_rate_hz=2.5e9)
        self.assertTrue(compiled.remove_dc_offset)

    def test_tagged_readout_delay_sweep_keeps_relative_timing(self) -> None:
        class TaggedDelaySweepProgram(ExperimentProgram):
            def _initialize(self, cfg):
                self.declare_gen("readout", ch=1)
                self.declare_readout(
                    "ro",
                    adc_channel="CHA",
                    length=100 * ns,
                    demod_freq=0,
                    waveform_ch=1,
                    marker_channel=1,
                    marker_padding=500 * ns,
                )
                delay = self.add_sweep(
                    "delay",
                    LinearSweep(0, 200 * ns, 3),
                )
                self.add_pulse(
                    "readout",
                    gen="readout",
                    style="const",
                    length=100 * ns,
                    frequency=0,
                    gain=0.02,
                    readout=True,
                )
                self.delay = delay

            def _body(self, cfg):
                self.delay_auto(self.delay)
                self.play("readout")
                self.trigger("ro")

        compiled = TaggedDelaySweepProgram({}).compile(
            sample_rate_hz=1e9
        )
        starts = [
            np.flatnonzero(np.abs(trace) > 0)[0]
            for trace in compiled.preview(1)
        ]
        self.assertEqual(starts, [500, 600, 700])

    def test_trigger_delay_cannot_change_between_sequence_steps(self) -> None:
        class SweptTriggerDelayProgram(ExperimentProgram):
            def _initialize(self, cfg):
                self.declare_gen("drive", ch=3)
                self.declare_readout(
                    "ro",
                    adc_channel="CHA",
                    length=1 * us,
                    demod_freq=50e6,
                    waveform_ch=3,
                )
                self.delay = self.add_sweep(
                    "trigger_delay",
                    LinearSweep(0, 40 * ns, 2),
                )
                self.add_pulse(
                    "pulse",
                    gen="drive",
                    style="gaussian",
                    length=100 * ns,
                    sigma=15 * ns,
                    frequency=50e6,
                    gain=0.02,
                )

            def _body(self, cfg):
                self.play("pulse")
                self.trigger("ro", trigger_delay=self.delay)

        with self.assertRaisesRegex(ValueError, "same.*sequence step"):
            SweptTriggerDelayProgram({}).compile(sample_rate_hz=2.5e9)

    def test_power_rabi_changes_waveform_gain_per_step(self) -> None:
        cfg = {
            "f_res": 50e6,
            "f_ge": 100e6,
            "gain_start": 0.01,
            "gain_stop": 0.05,
            "steps": 5,
            "qubit_len": 100 * ns,
            "qubit_sigma": 15 * ns,
            "res_len": 500 * ns,
            "res_gain": 0.02,
            "ro_len": 500 * ns,
        }
        compiled = PowerRabiProgram(cfg).compile(sample_rate_hz=2.5e9)
        peaks = np.max(np.abs(compiled.preview(4)), axis=1)

        self.assertEqual(compiled.number_of_sequence_steps, 5)
        self.assertTrue(np.all(np.diff(peaks) > 0))

    def test_cosine_square_applies_gain_carrier_and_phase(self) -> None:
        class CosineSquareProgram(ExperimentProgram):
            def _initialize(self, cfg):
                self.declare_gen("drive", ch=3)
                self.declare_readout(
                    "ro",
                    adc_channel="CHA",
                    length=100 * ns,
                    demod_freq=0,
                    waveform_ch=3,
                )
                self.add_pulse(
                    "pulse",
                    gen="drive",
                    style="cosine_square",
                    length=100 * ns,
                    edge_length=20 * ns,
                    frequency=100e6,
                    phase=np.pi / 2,
                    gain=0.02,
                )

            def _body(self, cfg):
                self.play("pulse")
                self.trigger("ro")

        compiled = CosineSquareProgram({}).compile(sample_rate_hz=1e9)
        waveform_values = compiled.preview(3)[0, :100]

        self.assertEqual(waveform_values[0], 0.0)
        self.assertAlmostEqual(waveform_values[20], 0.02, places=12)
        self.assertLessEqual(np.max(np.abs(waveform_values)), 0.02)

    def test_cosine_square_requires_edge_length(self) -> None:
        class MissingEdgeProgram(ExperimentProgram):
            def _initialize(self, cfg):
                self.declare_gen("drive", ch=3)
                self.declare_readout(
                    "ro",
                    adc_channel="CHA",
                    length=100 * ns,
                    demod_freq=0,
                    waveform_ch=3,
                )
                self.add_pulse(
                    "pulse",
                    gen="drive",
                    style="cosine_square",
                    length=100 * ns,
                    frequency=0,
                )

            def _body(self, cfg):
                self.play("pulse")
                self.trigger("ro")

        with self.assertRaisesRegex(ValueError, "edge_length"):
            MissingEdgeProgram({}).compile(sample_rate_hz=1e9)

    def test_t1_uses_fixed_step_length_for_all_delays(self) -> None:
        cfg = {
            "f_res": 50e6,
            "f_ge": 100e6,
            "delay_start": 0,
            "delay_stop": 2 * us,
            "steps": 5,
            "pi_len": 100 * ns,
            "pi_sigma": 15 * ns,
            "pi_gain": 0.02,
            "res_len": 500 * ns,
            "res_gain": 0.02,
            "ro_len": 500 * ns,
        }
        compiled = T1Program(cfg).compile(sample_rate_hz=2.5e9)

        shapes = {values.shape[1] for values in compiled.channel_waveforms.values()}
        self.assertEqual(shapes, {compiled.preview(3).shape[1]})
        self.assertEqual(compiled.number_of_sequence_steps, 5)

    def test_result_keeps_shots_and_averages_explicitly(self) -> None:
        raw = np.arange(3 * 2 * 4, dtype=float).reshape(3, 2, 4)
        iq_traces = raw.astype(complex)
        iq_shots = np.mean(iq_traces, axis=2)
        result = ExperimentResult(
            axes={"gain": np.array([0.1, 0.2])},
            point_coordinates=({"gain": 0.1}, {"gain": 0.2}),
            raw=raw,
            iq_traces=iq_traces,
            iq_shots=iq_shots,
            raw_time_s=np.arange(4),
            iq_time_s=np.arange(4),
        )

        self.assertEqual(result.shots("ro").shape, (3, 2))
        np.testing.assert_allclose(result.trace_average("ro"), raw.mean(axis=0))
        np.testing.assert_allclose(result.iq_average("ro"), iq_shots.mean(axis=0))

    def test_single_shot_ground_and_excited_steps_differ(self) -> None:
        cfg = {
            "f_res": 50e6,
            "f_ge": 100e6,
            "pi_len": 100 * ns,
            "pi_sigma": 15 * ns,
            "pi_gain": 0.02,
            "res_len": 500 * ns,
            "res_gain": 0.02,
            "ro_len": 500 * ns,
        }
        compiled = SingleShotProgram(cfg).compile(sample_rate_hz=2.5e9)

        self.assertEqual(compiled.axis("state").tolist(), ["g", "e"])
        ground = compiled.preview(4)[0]
        excited = compiled.preview(4)[1]
        self.assertTrue(np.all(ground == 0))
        self.assertGreater(np.max(np.abs(excited)), 0)

    def test_acquire_uses_one_n_average_axis(self) -> None:
        compiled = DelayProgram({}).compile(sample_rate_hz=2.5e9)

        class FakeHardware:
            def acquire_compiled_experiment(
                self,
                plan,
                n_average,
                *,
                filter_type,
            ):
                self.called_with = (
                    plan,
                    n_average,
                    filter_type,
                )
                return ExperimentResult(
                    axes=plan.axes,
                    point_coordinates=plan.point_coordinates,
                    raw=np.ones((n_average, plan.number_of_sequence_steps, 128)),
                    iq_traces=np.ones(
                        (n_average, plan.number_of_sequence_steps, 100),
                        dtype=complex,
                    ),
                    iq_shots=np.ones(
                        (n_average, plan.number_of_sequence_steps),
                        dtype=complex,
                    ),
                    raw_time_s=np.arange(128) / 1e9,
                    iq_time_s=np.arange(100) / 1e9,
                )

        hardware = FakeHardware()
        compiled.bind(hardware)

        result = compiled.acquire(n_average=7)

        self.assertIs(hardware.called_with[0], compiled)
        self.assertEqual(hardware.called_with[1:], (7, "boxcar"))
        self.assertEqual(result.raw.shape, (7, 6, 128))
        self.assertEqual(result.shots("ro").shape, (7, 6))
        self.assertEqual(result.trace_average("ro").shape, (6, 128))

    def test_only_one_ro_readout_is_supported(self) -> None:
        class InvalidReadoutProgram(ExperimentProgram):
            def _initialize(self, cfg):
                self.declare_readout(
                    "other",
                    adc_channel="CHA",
                    length=1 * us,
                    demod_freq=50e6,
                    marker_length=40 * ns,
                )

            def _body(self, cfg):
                pass

        with self.assertRaisesRegex(ValueError, "only readout 'ro'"):
            InvalidReadoutProgram({})

    def test_compile_checks_hardware_acquisition_window(self) -> None:
        hardware = SimpleNamespace(
            awg_sample_rate_hz=2.5e9,
            acquire_window_ns=500,
        )

        with self.assertRaisesRegex(
            ValueError,
            "integration window exceeds",
        ):
            DelayProgram({}).compile(hardware=hardware)


if __name__ == "__main__":
    unittest.main()
