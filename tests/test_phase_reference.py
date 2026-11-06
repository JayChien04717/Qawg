import numpy as np
import pytest

from QAWG import ValuesSweep, ns
from QAWG.compiler import ExperimentProgram
from QAWG.timeline import align_channels, delay_auto, waveform

SAMPLE_RATE = 2.5e9
FREQUENCY = 125e6  # 20 samples/cycle: 100-sample windows hold whole cycles


class TwoPulseProgram(ExperimentProgram):
    """Two identical pulses; the second sits at a swept time."""

    def _initialize(self, cfg):
        self.declare_gen("q", ch=2, amplitude_vpp=2.0)
        self.declare_gen("r", ch=3, amplitude_vpp=2.0)
        self.declare_readout(
            "ro", adc_channel="CHA", length=200 * ns,
            demod_freq=FREQUENCY, waveform_ch=3, marker_channel=1,
        )
        self.gap = self.add_sweep("gap", ValuesSweep([100 * ns, 136 * ns]))
        self.add_pulse("p", gen="q", style="const", length=40 * ns,
                       frequency=FREQUENCY, phase=0.3)
        self.add_pulse("ro_p", gen="r", style="const", length=200 * ns,
                       frequency=FREQUENCY, phase=0.3, readout=True)

    def _body(self, cfg):
        self.play("p", at=0)
        self.play("p", at=self.gap)
        self.play("ro_p", at=0)
        self.trigger("ro")


def _compile(**kwargs):
    return TwoPulseProgram({}, **kwargs).compile(sample_rate_hz=SAMPLE_RATE)


def test_global_phase_equals_one_continuous_carrier():
    compiled = _compile()
    shift = round((compiled.trigger_delay_s) * SAMPLE_RATE)
    for step, gap in enumerate((100 * ns, 136 * ns)):
        wave = compiled.channel_waveforms[2][step]
        start = round(gap * SAMPLE_RATE) + shift
        n = np.arange(start, start + 100)
        expected = np.sin(2 * np.pi * FREQUENCY * n / SAMPLE_RATE + 0.3)
        np.testing.assert_allclose(wave[start:start + 100], expected, atol=1e-12)


def test_pulse_reference_restarts_carrier_at_each_pulse():
    compiled = _compile(phase_reference="pulse")
    wave = compiled.channel_waveforms[2][1]
    start = round(136 * ns * SAMPLE_RATE) + round(compiled.trigger_delay_s * SAMPLE_RATE)
    local = np.arange(100)
    expected = np.sin(2 * np.pi * FREQUENCY * local / SAMPLE_RATE + 0.3)
    np.testing.assert_allclose(wave[start:start + 100], expected, atol=1e-12)


def _window_phases(compiled):
    n_total = compiled.channel_waveforms[2].shape[1]
    ref = np.exp(-2j * np.pi * FREQUENCY * np.arange(n_total) / SAMPLE_RATE)
    shift = round(compiled.trigger_delay_s * SAMPLE_RATE)
    phases = []
    for step, gap in enumerate((100 * ns, 136 * ns)):
        start = round(gap * SAMPLE_RATE) + shift
        window = slice(start, start + 100)
        wave = compiled.channel_waveforms[2][step]
        phases.append(np.angle(np.sum(wave[window] * ref[window])))
    return phases


def test_second_pulse_phase_is_independent_of_its_position():
    # Ramsey requirement: a pulse of fixed phase has the same phase in the
    # shared frame wherever it is placed; fringes then come only from the
    # qubit's detuning during the gap.
    first, second = _window_phases(_compile())
    np.testing.assert_allclose(first, second, atol=1e-9)


def test_pulse_reference_phase_depends_on_position():
    # Legacy mode: a 36 ns shift at 125 MHz is 4.5 turns -> pi of error.
    first, second = _window_phases(_compile(phase_reference="pulse"))
    assert abs(abs(np.angle(np.exp(1j * (first - second)))) - np.pi) < 1e-9


def test_invalid_phase_reference_is_rejected():
    with pytest.raises(ValueError):
        _compile(phase_reference="bogus")


def test_timeline_global_phase_matches_continuous_carrier():
    first = waveform(np.ones(50), FREQUENCY, 1)
    second = waveform(np.ones(50), FREQUENCY, 1)
    timeline = first / delay_auto(37 / SAMPLE_RATE) / second
    wave = align_channels(timeline, SAMPLE_RATE)[1]
    n = np.arange(wave.size)
    carrier = np.sin(2 * np.pi * FREQUENCY * n / SAMPLE_RATE)
    np.testing.assert_allclose(wave[:50], carrier[:50], atol=1e-12)
    np.testing.assert_allclose(wave[87:137], carrier[87:137], atol=1e-12)
    local = align_channels(timeline, SAMPLE_RATE, phase_reference="pulse")[1]
    assert not np.allclose(local[87:137], carrier[87:137], atol=1e-3)
