import numpy as np

from QAWG.alazar import (
    Alazar9371,
    CaptureConfig,
    CaptureData,
    downconvert,
    fir_filter,
    integrate,
)
from QAWG.alazar.ats9371 import BoardInfo
from QAWG.awg5200 import AWG5208, AWGProgram, AWGStep
from QAWG.awg_alazar import AWGAlazar


class NullTransport:
    def write(self, command):
        pass

    def query(self, command):
        return "0"

    def write_raw(self, data):
        pass

    def close(self):
        pass


class RecordingAWG(AWG5208):
    def __init__(self):
        super().__init__(NullTransport())
        self.assigned = []
        self.created_tracks = None

    def clear_all(self):
        self._waveforms.clear()

    def upload_waveform_asset(
        self,
        name,
        waveform_volts,
        amplitude_vpp=0.5,
        markers=(),
        overwrite=True,
    ):
        self._waveforms[name] = np.asarray(waveform_volts)
        return name

    def set_channel_amplitude(self, channel, amplitude_vpp):
        pass

    def set_channel_resolution(self, channel, bits):
        pass

    def set_marker_levels(self, channel, marker, low_volts, high_volts):
        pass

    def assign_waveform(self, channel, waveform_name):
        self.assigned.append((channel, waveform_name))

    def set_output(self, channel, enabled):
        pass

    def create_sequence(
        self,
        name,
        tracks,
        *,
        repetitions=1,
        goto_step=1,
        enabled=True,
    ):
        self.created_tracks = tracks
        return name


def _step(value):
    return AWGStep(waveforms={1: np.full(8, value, dtype=float)})


def test_awg_upload_program_selects_single_waveform():
    awg = RecordingAWG()
    result = awg.upload_program(
        AWGProgram("single", (_step(1.0),), {1: 0.5})
    )

    assert not result.is_sequence
    assert awg.assigned == [(1, "single_s0000_ch1")]
    assert awg.created_tracks is None


def test_awg_upload_program_owns_sequence_playback_order():
    awg = RecordingAWG()
    result = awg.upload_program(
        AWGProgram("sweep", (_step(0.0), _step(1.0), _step(2.0)), {1: 0.5})
    )

    assert result.is_sequence
    assert awg.created_tracks == {
        1: [
            "sweep_s0001_ch1",
            "sweep_s0002_ch1",
            "sweep_s0000_ch1",
        ]
    }


def test_alazar_capture_starts_source_after_dma(monkeypatch):
    import QAWG.alazar.device as device

    events = []

    class Session:
        active = True
        started = False

    session = Session()
    raw_codes = np.asarray([[32768, 32784]], dtype=np.uint16)

    monkeypatch.setattr(
        device,
        "arm_capture",
        lambda api, board, config: events.append("arm") or session,
    )

    def start(api, current):
        events.append("start")
        current.started = True

    monkeypatch.setattr(device, "start_capture", start)
    monkeypatch.setattr(
        device,
        "wait_for_capture",
        lambda api, current, timeout: events.append("wait") or raw_codes,
    )

    def abort(api, current):
        events.append("abort")
        current.active = False

    monkeypatch.setattr(device, "abort_capture", abort)
    monkeypatch.setattr(
        device,
        "free_capture",
        lambda current: events.append("free"),
    )

    alazar = Alazar9371(object(), BoardInfo(1, 0, 0, 12))
    capture = alazar.capture(
        CaptureConfig(samples_per_record=256),
        start_callback=lambda: events.append("source"),
    )

    assert events == ["arm", "start", "source", "wait", "abort", "free"]
    np.testing.assert_array_equal(capture.raw_codes, raw_codes)
    assert capture.records_volts.shape == raw_codes.shape


def test_core_dsp_supports_trace_and_integration():
    sample_rate = 1e9
    frequency = 50e6
    sample_count = 1000
    amplitude = 0.012
    phase = 0.37
    time = np.arange(sample_count) / sample_rate
    records = amplitude * np.cos(2 * np.pi * frequency * time + phase)
    capture = CaptureData(
        raw_codes=np.zeros((1, sample_count), dtype=np.uint16),
        records_volts=records[None, :],
        sample_rate_hz=sample_rate,
    )
    baseband = downconvert(
        capture.records_volts,
        capture.sample_rate_hz,
        frequency,
    )
    filtered = fir_filter(baseband, sample_rate, 20e6)
    shots = integrate(baseband, 100, 900)

    assert filtered.shape == capture.records_volts.shape
    np.testing.assert_allclose(
        shots[0],
        amplitude * np.exp(1j * phase),
        atol=1e-14,
    )


def test_qawg_coordinator_only_orders_source_and_capture():
    events = []

    class Source:
        def stop(self):
            events.append("stop")

        def run(self, wait_until_ready=False):
            events.append("run")

        def close(self):
            events.append("close")

    class Digitizer:
        def capture(self, config, *, start_callback=None):
            events.append("arm")
            start_callback()
            events.append("collect")
            time = np.arange(config.samples_per_record) / config.sample_rate_hz
            records = 0.01 * np.cos(2 * np.pi * 50e6 * time)
            return CaptureData(
                raw_codes=np.zeros(
                    (config.records_per_acquisition, config.samples_per_record),
                    dtype=np.uint16,
                ),
                records_volts=np.tile(
                    records,
                    (config.records_per_acquisition, 1),
                ),
                sample_rate_hz=config.sample_rate_hz,
            )

    controller = AWGAlazar(
        Source(),
        object(),
        BoardInfo(1, 0, 0, 12),
        awg_sample_rate_hz=1e9,
        alazar_sample_rate_hz=1e9,
        acquire_window_ns=1000,
        tone_frequency_hz=50e6,
        integrate_window_ns=(100, 900),
    )
    controller.alazar = Digitizer()

    shots, average = controller.acquire(n_average=2)

    assert events == ["stop", "arm", "run", "collect", "stop"]
    assert shots.shape == (2,)
    np.testing.assert_allclose(average, 0.01, atol=1e-14)


def test_qawg_coordinator_compiles_and_binds_program():
    controller = object.__new__(AWGAlazar)

    class Program:
        def compile(self, *, hardware):
            assert hardware is controller
            return "compiled"

    assert controller.compile(Program()) == "compiled"
