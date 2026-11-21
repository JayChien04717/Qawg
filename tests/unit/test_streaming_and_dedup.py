"""Tests for streaming Alazar capture, standalone DSP, AWG dedup, and waveform fixes."""

import subprocess
import sys

import numpy as np

from QAWG.alazar import (
    Alazar9371,
    BoardInfo,
    CaptureConfig,
    DemodulationConfig,
    DemodulationPipeline,
)
from QAWG.alazar.constants import CHANNEL_A, CHANNEL_AB, CHANNEL_B
from QAWG.awg5200 import AWG5208, AWGProgram, AWGStep
from QAWG.awg5200.waveforms import triangle


def test_alazar_stream_capture_and_standalone_demod_pipeline(monkeypatch):
    """Alazar9371.stream_capture + DemodulationPipeline work without AWG."""
    import QAWG.alazar.device as device

    sample_rate = 1e9
    samples = 512
    total_records = 12
    chunk_records = 4
    rng = np.random.default_rng(5)
    shifts = rng.uniform(-0.4e-9, 0.4e-9, total_records)
    time = np.arange(samples) / sample_rate

    sig_v = 0.05 * np.cos(
        2 * np.pi * 50e6 * (time[None, :] + shifts[:, None]) + 0.6
    )
    ref_v = 0.08 * np.cos(
        2 * np.pi * 50e6 * (time[None, :] + shifts[:, None]) - 0.3
    )

    # Encode volts to 12-bit shifted uint16 ADC codes (+/-0.4 V)
    def volts_to_codes(v):
        codes = np.round(v / 0.4 * 2048.0 + 2048.0).astype(np.uint16)
        return np.left_shift(codes, 4)

    all_codes = np.stack([volts_to_codes(sig_v), volts_to_codes(ref_v)], axis=1)

    class Session:
        active = True
        started = False

    session = Session()
    monkeypatch.setattr(device, "arm_capture", lambda api, board, cfg: session)

    def start(api, current):
        current.started = True

    monkeypatch.setattr(device, "start_capture", start)
    monkeypatch.setattr(
        device,
        "iter_capture_buffers",
        lambda api, current, timeout: (
            all_codes[i : i + chunk_records]
            for i in range(0, total_records, chunk_records)
        ),
    )
    monkeypatch.setattr(device, "abort_capture", lambda api, current: None)
    monkeypatch.setattr(device, "free_capture", lambda current: None)

    alazar = Alazar9371(object(), BoardInfo(1, 0, 0, 12))
    pipeline = DemodulationPipeline(
        DemodulationConfig(
            sample_rate_hz=sample_rate,
            tone_frequency_hz=50e6,
            integrate_start_sample=64,
            integrate_stop_sample=448,
            remove_dc_offset=False,
            reference_center_rad=-0.3,
        )
    )
    collected_shots = []
    n_delivered = alazar.stream_capture(
        CaptureConfig(
            samples_per_record=samples,
            records_per_acquisition=total_records,
            channel=CHANNEL_AB,
        ),
        lambda chunk: collected_shots.append(
            pipeline.process_capture(
                chunk,
                signal_channel=CHANNEL_A,
                reference_channel=CHANNEL_B,
                compute_fir=False,
            ).shots_iq
        ),
    )
    assert n_delivered == total_records
    shots = np.concatenate(collected_shots)
    assert shots.shape == (total_records,)
    phase_std = np.std(np.angle(shots * np.exp(-1j * np.angle(shots.mean()))))
    assert phase_std < 5e-3
    np.testing.assert_allclose(np.abs(shots), 0.05, rtol=2e-2)


def test_awg_upload_program_deduplicates_identical_step_waveforms():
    """Constant ref channel across 5 steps is uploaded only once."""

    class NullTransport:
        def write(self, command):
            pass

        def query(self, command):
            return "0"

        def write_raw(self, data):
            pass

        def close(self):
            pass

    class CountingAWG(AWG5208):
        def __init__(self):
            super().__init__(NullTransport())
            self.uploaded_names = []
            self.created_tracks = None

        def clear_all(self):
            self._waveforms.clear()

        def upload_waveform_asset(
            self, name, waveform_volts, amplitude_vpp=0.5, markers=(), overwrite=True
        ):
            self.uploaded_names.append(name)
            self._waveforms[name] = np.asarray(waveform_volts)
            return name

        def set_channel_amplitude(self, channel, amplitude_vpp):
            pass

        def set_channel_resolution(self, channel, bits):
            pass

        def set_marker_levels(self, channel, marker, low_volts, high_volts):
            pass

        def set_output(self, channel, enabled):
            pass

        def create_sequence(
            self, name, tracks, *, repetitions=1, goto_step=1, enabled=True
        ):
            self.created_tracks = tracks
            return name

    ref_wave = np.ones(16, dtype=float) * 0.1
    steps = tuple(
        AWGStep(
            waveforms={
                1: np.full(16, 0.02 * i, dtype=float),
                2: ref_wave,
            }
        )
        for i in range(5)
    )
    awg = CountingAWG()
    result = awg.upload_program(
        AWGProgram("dedup", steps, {1: 0.5, 2: 0.5})
    )
    assert result.is_sequence
    # 5 distinct waveforms on ch1 + 1 deduplicated waveform on ch2 = 6 total (not 10!)
    assert len(awg.uploaded_names) == 6
    assert len(set(result.waveform_names[2])) == 1
    assert len(set(result.waveform_names[1])) == 5


def test_triangle_supports_negative_amplitude():
    pos = triangle(101, 1e9, 20e-9, amplitude_volts=0.4)
    neg = triangle(101, 1e9, 20e-9, amplitude_volts=-0.4)
    np.testing.assert_allclose(neg, -pos, atol=1e-15)
    assert np.min(neg) < -0.39


def test_awg5200_has_no_circular_import_with_parent_timeline():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import QAWG.awg5200; "
                "assert 'QAWG.timeline' not in sys.modules; "
                "assert 'QAWG.alazar' not in sys.modules"
            ),
        ],
        check=True,
    )
