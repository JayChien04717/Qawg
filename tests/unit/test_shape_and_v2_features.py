"""Tests for waveform primitives, Shape builder, IQArray, and plotting ported from QuantumCompilerV2."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from QAWG import (
    ExperimentProgram,
    IQArray,
    LinearSweep,
    MHz,
    SHAPE_REGISTRY,
    Shape,
    delay_auto,
    ns,
    parallel,
    register_shape,
    us,
    waveform,
)
from QAWG.awg5200.waveforms import (
    cosine,
    exp_falling_pulse,
    exp_rising_pulse,
    exp_square,
    pwm,
    pwm_width,
    square,
)


def test_new_waveform_primitives():
    fs = 1e9
    n = 100

    c = cosine(n, fs, frequency_hz=50e6, amplitude_volts=0.4, phase_radians=0.0)
    assert c.shape == (n,)
    np.testing.assert_allclose(c[0], 0.4, atol=1e-12)

    sq = square(n, fs, start_s=20e-9, flat_s=30e-9, amplitude_volts=0.5, offset_volts=0.1)
    np.testing.assert_allclose(sq[:20], 0.1)
    np.testing.assert_allclose(sq[20:51], 0.6)
    np.testing.assert_allclose(sq[51:], 0.1)

    er = exp_rising_pulse(n, fs, peak_s=50e-9, time_constant_s=10e-9, amplitude_volts=2.0)
    np.testing.assert_allclose(er[50], 2.0, atol=1e-12)
    np.testing.assert_allclose(er[40], 2.0 * np.exp(-1.0), atol=1e-12)
    np.testing.assert_allclose(er[51:], 0.0)

    ef = exp_falling_pulse(n, fs, peak_s=20e-9, time_constant_s=10e-9, amplitude_volts=2.0)
    np.testing.assert_allclose(ef[:20], 0.0)
    np.testing.assert_allclose(ef[20], 2.0, atol=1e-12)
    np.testing.assert_allclose(ef[30], 2.0 * np.exp(-1.0), atol=1e-12)

    es = exp_square(n, fs, first_peak_s=20e-9, flat_s=30e-9, time_constant_s=10e-9, amplitude_volts=1.5)
    np.testing.assert_allclose(es[20:51], 1.5, atol=1e-12)
    np.testing.assert_allclose(es[10], 1.5 * np.exp(-1.0), atol=1e-12)
    np.testing.assert_allclose(es[60], 1.5 * np.exp(-1.0), atol=1e-12)

    p1 = pwm(100, fs, period_s=20e-9, duty_cycle=0.25, on_value=1.0, off_value=-1.0)
    p2 = pwm_width(100, fs, period_s=20e-9, on_duration_s=5e-9, on_value=1.0, off_value=-1.0)
    np.testing.assert_allclose(p1, p2)
    assert p1[0] == 1.0
    assert p1[10] == -1.0


def test_shape_algebra_and_rsub_fix():
    fs = 1e9
    g = Shape("gaussian", {"peak_x": 50e-9, "sigma": 10e-9}, span=100e-9, sampling_rate=fs, name="gauss")
    c = Shape("const", {"lv": 1.0}, span=100e-9, sample_rate=fs)

    # Scalar & binary ops
    np.testing.assert_allclose((2.0 * g).y, 2.0 * g.y)
    np.testing.assert_allclose((-g).y, -g.y)
    np.testing.assert_allclose(abs(-g).y, g.y)
    np.testing.assert_allclose((c - g).y, 1.0 - g.y)
    # V2 bug fix: 1.0 - g must equal 1 - g.y (NOT g.y - 1)
    np.testing.assert_allclose((1.0 - g).y, 1.0 - g.y)
    np.testing.assert_allclose((g - 1.0).y, g.y - 1.0)
    np.testing.assert_allclose((g / 2.0).y, g.y / 2.0)
    np.testing.assert_allclose((g ** 2).y, g.y ** 2)

    # Reverse & roll
    np.testing.assert_allclose(g.reverse().y, g.y[::-1])
    np.testing.assert_allclose(g.roll(10).y, np.roll(g.y, 10))
    np.testing.assert_allclose(g.roll(10e-9).y, np.roll(g.y, 10))

    # Derivative (DRAG check: analytical d/dt exp(-0.5*((t-mu)/sigma)^2))
    dg = g.diff()
    analytical = -((g.x - 50e-9) / (10e-9 ** 2)) * g.y
    np.testing.assert_allclose(dg.y[5:-5], analytical[5:-5], rtol=2e-2, atol=1e5)

    # FFT peak check on pure sine
    s = Shape("sine2", {"frequency": 100e6}, span=1e-6, sample_rate=fs, name="tone")
    peak_freq = abs(s.f[np.argmax(np.abs(s.yf))])
    np.testing.assert_allclose(peak_freq, 100e6, rtol=1e-6)


def test_shape_registry_custom_function():
    @register_shape("test_ramp_custom")
    def _ramp(x, slope=1.0):
        return slope * x

    try:
        assert "test_ramp_custom" in SHAPE_REGISTRY
        s = Shape("test_ramp_custom", {"slope": 1e9}, span=10e-9, sample_rate=1e9)
        np.testing.assert_allclose(s.y, np.arange(10, dtype=np.float64))
    finally:
        SHAPE_REGISTRY.pop("test_ramp_custom", None)


def test_shape_temporal_ops_are_immutable():
    fs = 1e9
    a = Shape("const", {"lv": 1.0}, span=10e-9, sample_rate=fs, name="a")
    b = Shape("const", {"lv": 2.0}, span=20e-9, sample_rate=fs, name="b")

    # Concat with | and repeat
    ab = a | b
    assert ab.n_samples == 30
    np.testing.assert_allclose(ab.y[:10], 1.0)
    np.testing.assert_allclose(ab.y[10:], 2.0)

    rep = a.repeat(3)
    assert rep.n_samples == 30
    np.testing.assert_allclose(rep.y, 1.0)

    # Shift (zero pad)
    delayed = a >> 5e-9
    assert delayed.n_samples == 15
    np.testing.assert_allclose(delayed.y[:5], 0.0)
    np.testing.assert_allclose(delayed.y[5:], 1.0)

    extended = a << 5
    assert extended.n_samples == 15
    np.testing.assert_allclose(extended.y[:10], 1.0)
    np.testing.assert_allclose(extended.y[10:], 0.0)

    # Immutable alignment
    a_hh, b_hh = a.align_to(b, mode="head_to_head")
    assert a.n_samples == 10 and b.n_samples == 20  # originals untouched
    assert a_hh.n_samples == 20 and b_hh.n_samples == 20
    np.testing.assert_allclose(a_hh.y[:10], 1.0)
    np.testing.assert_allclose(a_hh.y[10:], 0.0)

    a_tt, b_tt = a.align_to(b, mode="tail_to_tail")
    np.testing.assert_allclose(a_tt.y[:10], 0.0)
    np.testing.assert_allclose(a_tt.y[10:], 1.0)

    a_th, b_th = a.align_to(b, mode="tail_to_head")
    assert a_th.n_samples == 30 and b_th.n_samples == 30
    np.testing.assert_allclose(a_th.y[:10], 1.0)
    np.testing.assert_allclose(b_th.y[10:], 2.0)


def test_iqarray_rf_helpers_and_slicing():
    t = np.linspace(0, 2 * np.pi, 100)
    iq = IQArray(0.5 * np.exp(1j * t), Z0=50.0, name="lo")
    assert iq.name == "lo"
    assert iq.Z0 == 50.0

    # Slicing preserves metadata via __array_finalize__
    sub = iq[:10]
    assert isinstance(sub, IQArray)
    assert sub.Z0 == 50.0
    assert sub.name == "lo"

    # Scale to 0 dBm into 50 Ohm: Vp = sqrt(2 * 50 * 1e-3) = sqrt(0.1)
    scaled = iq.scale_to_dbm(0.0)
    np.testing.assert_allclose(scaled.dBm, 0.0, atol=1e-10)
    np.testing.assert_allclose(scaled.amplitude, np.sqrt(0.1), atol=1e-12)

    # +6 dB gain ~ 2x amplitude
    boosted = iq.gain(20.0)
    np.testing.assert_allclose(boosted.amplitude, 10.0 * iq.amplitude, atol=1e-12)

    # Phase shift
    shifted = iq.phase_shift(np.pi / 2)
    np.testing.assert_allclose(shifted.vI, -iq.vQ, atol=1e-12)
    np.testing.assert_allclose(shifted.vQ, iq.vI, atol=1e-12)

    fig, ax = iq.iq_plot(show=False)
    plt.close(fig)


def test_shape_timeline_and_compiler_integration_and_plots():
    fs = 1e9
    g = Shape("gaussian", {"peak_x": 40e-9, "sigma": 10e-9}, span=80e-9, sample_rate=fs, name="drive")
    drag_q = (0.1 * g.diff()).normalize(0.05)
    drag_q.name = "drag_q"

    # Direct pass to timeline.waveform & to_waveform
    w1 = waveform(0.2 * g, fc=50e6, ch=1)
    assert w1.name == "drive"
    w2 = drag_q.to_waveform(fc=50e6, ch=2, phase_radians=np.pi / 2)
    ro = Shape("const", {"lv": 0.15}, span=200e-9, sample_rate=fs, name="ro").to_waveform(fc=100e6, ch=3)

    tl = parallel(w1, w2) / delay_auto(20e-9) / ro

    # Matplotlib & Plotly plotting (smoke tests, show=False)
    fig, _ = g.plot(backend="matplotlib", show=False)
    plt.close(fig)
    fig_fft, _ = g.plot_fft(backend="matplotlib", show=False)
    plt.close(fig_fft)
    pfig = g.plot(backend="plotly", show=False)
    assert len(pfig.data) == 1

    fig_tl, _ = tl.plot(fs, total_duration_s=1e-6, backend="matplotlib", show=False)
    plt.close(fig_tl)
    fig_sch, _ = tl.plot_schedule(fs, backend="matplotlib", show=False)
    plt.close(fig_sch)
    pfig_sch = tl.plot_schedule(fs, backend="plotly", show=False)
    assert len(pfig_sch.data) == 3

    # ExperimentProgram with custom Shape & new pulse styles
    class CustomShapeProgram(ExperimentProgram):
        def _initialize(self, cfg):
            self.declare_gen("drive", ch=1, amplitude_vpp=0.5)
            self.declare_gen("ro_gen", ch=2, amplitude_vpp=0.5)
            self.declare_readout(
                "ro",
                adc_channel="A",
                length=400 * ns,
                demod_freq=50 * MHz,
                waveform_ch=2,
            )
            gain_sweep = self.add_sweep("gain", LinearSweep(0.2, 0.8, 3))
            self.add_pulse(
                "drag_i",
                gen="drive",
                style="custom",
                envelope=g,
                frequency=50 * MHz,
                gain=gain_sweep,
            )
            self.add_pulse(
                "absorber",
                gen="drive",
                style="exp_rising_pulse",
                length=100 * ns,
                decay=25 * ns,
                frequency=0.0,
                gain=0.5,
            )
            self.add_pulse(
                "ro_pulse",
                gen="ro_gen",
                style="square",
                length=400 * ns,
                frequency=50 * MHz,
                gain=0.5,
                readout=True,
            )

        def _body(self, cfg):
            self.play("drag_i")
            self.delay_auto(20 * ns)
            self.play("absorber")
            self.delay_auto(20 * ns)
            self.play("ro_pulse")
            self.trigger("ro")

    compiled = CustomShapeProgram({}).compile(sample_rate_hz=fs)
    assert compiled.number_of_sequence_steps == 3
    assert np.max(np.abs(compiled.preview(1)[2])) > np.max(np.abs(compiled.preview(1)[0]))

    fig_c, _ = compiled.plot(step=1, backend="matplotlib", show=False)
    plt.close(fig_c)
    pfig_c = compiled.plot(step=1, backend="plotly", show=False)
    assert len(pfig_c.data) >= 2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS: {name}")
    print("ALL PASSED")
