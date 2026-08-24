import numpy as np

from QAWG.compiler import ExperimentProgram, ExperimentResult, ns


class TaggedReadoutProgram(ExperimentProgram):
    def _initialize(self, cfg):
        self.declare_gen("readout", ch=1, amplitude_vpp=0.5)
        self.declare_readout(
            "ro",
            adc_channel="CHB",
            length=600 * ns,
            demod_freq=50e6,
            waveform_ch=1,
            marker_channel=1,
            marker_length=cfg["marker_length"],
            marker_lead=cfg["marker_lead"],
            integrate_time=100 * ns,
        )
        self.add_pulse(
            "readout",
            gen="readout",
            style="const",
            length=600 * ns,
            frequency=50e6,
            gain=0.5,
            readout=True,
        )

    def _body(self, cfg):
        self.play("readout", at=0)
        if cfg.get("explicit_trigger_delay") is None:
            self.trigger("ro")
        else:
            self.trigger("ro", trigger_delay=cfg["explicit_trigger_delay"])


def test_marker_length_is_independent_of_tagged_readout_window():
    compiled = TaggedReadoutProgram(
        {
            "marker_length": 40 * ns,
            "marker_lead": 100 * ns,
        }
    ).compile(sample_rate_hz=1e9)

    marker_indices = np.flatnonzero(compiled.marker_waveforms[0])
    assert marker_indices[0] == 0
    assert marker_indices[-1] == 39
    np.testing.assert_allclose(
        compiled.readout_windows_s[0],
        [100 * ns, 700 * ns],
    )
    assert compiled.trigger_delay_s == 100 * ns


def test_explicit_trigger_delay_is_independent_of_marker_lead():
    compiled = TaggedReadoutProgram(
        {
            "marker_length": 40 * ns,
            "marker_lead": 100 * ns,
            "explicit_trigger_delay": 225 * ns,
        }
    ).compile(sample_rate_hz=1e9)

    assert compiled.trigger_delay_s == 225 * ns
    np.testing.assert_allclose(
        compiled.readout_windows_s[0],
        [100 * ns, 700 * ns],
    )


def test_dc_offset_removal_defaults_true_and_can_be_disabled():
    cfg = {
        "marker_length": 40 * ns,
        "marker_lead": 100 * ns,
    }
    assert TaggedReadoutProgram(cfg).compile(
        sample_rate_hz=1e9
    ).remove_dc_offset
    assert not TaggedReadoutProgram(
        cfg,
        remove_dc_offset=False,
    ).compile(sample_rate_hz=1e9).remove_dc_offset

    program = TaggedReadoutProgram(cfg)
    program.remove_dc_offset = False
    assert not program.compile(sample_rate_hz=1e9).remove_dc_offset


def test_marker_parameters_are_validated_separately():
    for key in ("marker_length", "marker_lead"):
        cfg = {
            "marker_length": 40 * ns,
            "marker_lead": 100 * ns,
        }
        cfg[key] = -1 * ns if key == "marker_lead" else 0.0
        try:
            TaggedReadoutProgram(cfg)
        except ValueError as exc:
            assert key in str(exc)
        else:
            raise AssertionError(f"invalid {key} was accepted")


def test_result_reduction_names_match_the_saved_data_kind():
    raw = np.arange(12, dtype=float).reshape(2, 2, 3)
    iq_traces = raw.astype(complex) * (1 + 0.5j)
    iq_shots = np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]])
    result = ExperimentResult(
        axes={"frequency": np.array([5.8e9, 5.9e9])},
        point_coordinates=({}, {}),
        raw=raw,
        iq_traces=iq_traces,
        iq_shots=iq_shots,
        raw_time_s=np.arange(3, dtype=float),
        iq_time_s=np.arange(3, dtype=float),
    )

    np.testing.assert_allclose(result.raw_average(), np.mean(raw, axis=0))
    np.testing.assert_allclose(
        result.iq_average(),
        np.mean(iq_traces, axis=0),
    )
    np.testing.assert_allclose(result.s21(), np.mean(iq_shots, axis=0))
