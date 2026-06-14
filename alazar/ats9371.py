"""Functional building blocks for triggered ATS9371 NPT acquisition."""

from __future__ import annotations

import ctypes
import sys
from contextlib import suppress
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .ats_api import ATSApi
from .constants import (
    ADMA_EXTERNAL_STARTCAPTURE,
    ADMA_NPT,
    ATS9371_BOARD_KIND,
    BW_LIMIT_DISABLE,
    CHANNEL_A,
    CHANNEL_B,
    CLOCK_EDGE_RISING,
    DC_COUPLING,
    ETR_TTL,
    EXTERNAL_CLOCK_10MHZ_REF,
    EXTERNAL_SAMPLE_RATE_1000MSPS,
    IMPEDANCE_50_OHM,
    INPUT_RANGE_PM_400_MV,
    INTERNAL_CLOCK,
    MIN_SAMPLES_PER_RECORD,
    SAMPLE_RATE_1000MSPS,
    SAMPLES_PER_RECORD_ALIGNMENT,
    TRIG_DISABLE,
    TRIG_ENGINE_J,
    TRIG_ENGINE_K,
    TRIG_ENGINE_OP_J,
    TRIG_EXTERNAL,
    TRIGGER_SLOPE_POSITIVE,
)
from .demodulation import dispersive_demodulate

MEM_COMMIT = 0x1000
MEM_RELEASE = 0x8000
PAGE_READWRITE = 0x04


@dataclass(frozen=True)
class BoardInfo:
    handle: int
    kind: int
    memory_samples: int
    bits_per_sample: int


@dataclass(frozen=True)
class TriggerConfig:
    coupling: int = DC_COUPLING
    trigger_range: int = ETR_TTL
    level: int = 140
    delay_samples: int = 0
    timeout_ticks: int = 0


@dataclass(frozen=True)
class AcquisitionConfig:
    sample_rate_hz: float = 1e9
    tone_frequency_hz: float = 100e6
    samples_per_record: int = 2560
    num_averages: int = 1
    records_per_buffer: int = 1
    dma_buffer_count: int = 4
    input_range_volts: float = 0.4
    timeout_ms: int = 5000
    channel: int = CHANNEL_A

    @property
    def records_per_acquisition(self) -> int:
        """One triggered record is acquired for each average."""
        return self.num_averages


@dataclass
class DmaBuffer:
    address: int
    sample_count: int
    size_bytes: int


@dataclass
class CaptureSession:
    handle: int
    buffers: list[DmaBuffer]
    samples_per_record: int
    records_per_buffer: int
    records_per_acquisition: int
    buffers_per_acquisition: int
    bits_per_sample: int
    active: bool = True
    started: bool = False
    released: bool = False


@dataclass(frozen=True)
class ToneAnalysis:
    complex_amplitude: npt.NDArray[np.complex128]
    amplitude_volts: npt.NDArray[np.float64]
    phase_radians: npt.NDArray[np.float64]
    peak_frequency_hz: npt.NDArray[np.float64]


def validate_acquisition_config(config: AcquisitionConfig) -> None:
    if config.sample_rate_hz != 1e9:
        raise ValueError("This first driver path is fixed to ATS9371 at 1 GS/s")
    if config.channel not in (CHANNEL_A, CHANNEL_B):
        raise ValueError("channel must be CHANNEL_A or CHANNEL_B")
    if config.samples_per_record < MIN_SAMPLES_PER_RECORD:
        raise ValueError(f"samples_per_record must be >= {MIN_SAMPLES_PER_RECORD}")
    if config.samples_per_record % SAMPLES_PER_RECORD_ALIGNMENT:
        raise ValueError(
            "samples_per_record must be a multiple of "
            f"{SAMPLES_PER_RECORD_ALIGNMENT}"
        )
    if config.num_averages < 1:
        raise ValueError("num_averages must be positive")
    if config.records_per_buffer < 1:
        raise ValueError("records_per_buffer must be positive")
    if config.records_per_buffer > config.records_per_acquisition:
        raise ValueError(
            "records_per_buffer cannot exceed num_averages"
        )
    if config.records_per_acquisition % config.records_per_buffer:
        raise ValueError(
            "num_averages must be divisible by records_per_buffer"
        )
    if config.dma_buffer_count < 2:
        raise ValueError("dma_buffer_count must be at least 2")
    if not 0 < config.tone_frequency_hz < config.sample_rate_hz / 2:
        raise ValueError("tone_frequency_hz must be between DC and Nyquist")


def open_ats9371(
    api: ATSApi, system_id: int = 1, board_id: int = 1
) -> BoardInfo:
    handle = api.get_board(system_id, board_id)
    return get_board_info(api, handle)


def get_board_info(api: ATSApi, handle: int) -> BoardInfo:
    kind = api.get_board_kind(handle)
    memory_samples, bits_per_sample = api.get_channel_info(handle)
    if kind != ATS9371_BOARD_KIND:
        raise RuntimeError(
            f"Expected ATS9371 board kind {ATS9371_BOARD_KIND}, found {kind}"
        )
    return BoardInfo(handle, kind, memory_samples, bits_per_sample)


def configure_clock(
    api: ATSApi,
    handle: int,
    use_external_10mhz_reference: bool = False,
) -> None:
    source = (
        EXTERNAL_CLOCK_10MHZ_REF
        if use_external_10mhz_reference
        else INTERNAL_CLOCK
    )
    sample_rate = (
        EXTERNAL_SAMPLE_RATE_1000MSPS
        if use_external_10mhz_reference
        else SAMPLE_RATE_1000MSPS
    )
    api.set_capture_clock(
        handle,
        source,
        sample_rate,
        CLOCK_EDGE_RISING,
        0,
    )


def configure_channel(
    api: ATSApi, handle: int, channel: int = CHANNEL_A
) -> None:
    if channel not in (CHANNEL_A, CHANNEL_B):
        raise ValueError("channel must be CHANNEL_A or CHANNEL_B")
    api.input_control(
        handle,
        channel,
        DC_COUPLING,
        INPUT_RANGE_PM_400_MV,
        IMPEDANCE_50_OHM,
    )
    api.set_bw_limit(handle, channel, BW_LIMIT_DISABLE)


def configure_channel_a(api: ATSApi, handle: int) -> None:
    configure_channel(api, handle, CHANNEL_A)


def configure_external_trigger(
    api: ATSApi, handle: int, trigger: TriggerConfig
) -> None:
    api.set_external_trigger(handle, trigger.coupling, trigger.trigger_range)
    api.set_trigger_operation(
        handle,
        TRIG_ENGINE_OP_J,
        TRIG_ENGINE_J,
        TRIG_EXTERNAL,
        TRIGGER_SLOPE_POSITIVE,
        trigger.level,
        TRIG_ENGINE_K,
        TRIG_DISABLE,
        TRIGGER_SLOPE_POSITIVE,
        128,
    )
    api.set_trigger_delay(handle, trigger.delay_samples)
    api.set_trigger_timeout(handle, trigger.timeout_ticks)


def configure_ats9371(
    api: ATSApi,
    board: BoardInfo,
    trigger: TriggerConfig = TriggerConfig(),
    use_external_10mhz_reference: bool = False,
) -> None:
    configure_clock(api, board.handle, use_external_10mhz_reference)
    configure_channel(api, board.handle, CHANNEL_A)
    configure_channel(api, board.handle, CHANNEL_B)
    configure_external_trigger(api, board.handle, trigger)


def bytes_per_sample(bits_per_sample: int) -> int:
    return (bits_per_sample + 7) // 8


def _kernel32() -> ctypes.WinDLL:
    if sys.platform != "win32":
        raise RuntimeError("ATS9371 DMA acquisition requires Windows")
    kernel32 = ctypes.windll.kernel32
    kernel32.VirtualAlloc.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    kernel32.VirtualAlloc.restype = ctypes.c_void_p
    kernel32.VirtualFree.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_ulong,
    ]
    kernel32.VirtualFree.restype = ctypes.c_int
    return kernel32


def allocate_dma_memory(size_bytes: int) -> int:
    address = _kernel32().VirtualAlloc(
        None, size_bytes, MEM_COMMIT, PAGE_READWRITE
    )
    if not address:
        raise MemoryError(f"VirtualAlloc failed for {size_bytes} bytes")
    return int(address)


def release_dma_memory(address: int) -> None:
    if not _kernel32().VirtualFree(ctypes.c_void_p(address), 0, MEM_RELEASE):
        raise RuntimeError("VirtualFree failed")


def arm_capture(
    api: ATSApi, board: BoardInfo, config: AcquisitionConfig
) -> CaptureSession:
    validate_acquisition_config(config)
    buffers_per_acquisition = (
        config.records_per_acquisition // config.records_per_buffer
    )
    allocated_buffer_count = min(
        config.dma_buffer_count, buffers_per_acquisition
    )
    sample_count = config.samples_per_record * config.records_per_buffer
    size_bytes = sample_count * bytes_per_sample(board.bits_per_sample)
    buffers: list[DmaBuffer] = []

    try:
        for _ in range(allocated_buffer_count):
            address = allocate_dma_memory(size_bytes)
            buffers.append(DmaBuffer(address, sample_count, size_bytes))

        api.set_record_size(board.handle, 0, config.samples_per_record)
        api.before_async_read(
            board.handle,
            config.channel,
            0,
            config.samples_per_record,
            config.records_per_buffer,
            config.records_per_acquisition,
            ADMA_NPT | ADMA_EXTERNAL_STARTCAPTURE,
        )
        for buffer in buffers:
            api.post_async_buffer(
                board.handle, buffer.address, buffer.size_bytes
            )
    except BaseException:
        with suppress(Exception):
            api.abort_async_read(board.handle)
        for buffer in buffers:
            release_dma_memory(buffer.address)
        raise

    return CaptureSession(
        handle=board.handle,
        buffers=buffers,
        samples_per_record=config.samples_per_record,
        records_per_buffer=config.records_per_buffer,
        records_per_acquisition=config.records_per_acquisition,
        buffers_per_acquisition=buffers_per_acquisition,
        bits_per_sample=board.bits_per_sample,
    )


def start_capture(api: ATSApi, session: CaptureSession) -> None:
    api.start_capture(session.handle)
    session.started = True


def copy_dma_buffer(
    session: CaptureSession, buffer: DmaBuffer
) -> npt.NDArray[np.uint16]:
    if bytes_per_sample(session.bits_per_sample) != 2:
        raise RuntimeError(
            f"This first driver expects 16-bit transfers, got "
            f"{session.bits_per_sample} ADC bits"
        )
    array_type = ctypes.c_uint16 * buffer.sample_count
    view = np.ctypeslib.as_array(array_type.from_address(buffer.address))
    return view.copy().reshape(
        session.records_per_buffer, session.samples_per_record
    )


def wait_for_capture(
    api: ATSApi, session: CaptureSession, timeout_ms: int
) -> npt.NDArray[np.uint16]:
    if not session.started:
        raise RuntimeError("Call start_capture before wait_for_capture")

    records = np.empty(
        (session.records_per_acquisition, session.samples_per_record),
        dtype=np.uint16,
    )
    buffer_count = len(session.buffers)

    for completed in range(session.buffers_per_acquisition):
        buffer = session.buffers[completed % buffer_count]
        api.wait_async_buffer(session.handle, buffer.address, timeout_ms)

        record_start = completed * session.records_per_buffer
        record_stop = record_start + session.records_per_buffer
        records[record_start:record_stop] = copy_dma_buffer(session, buffer)

        if completed + buffer_count < session.buffers_per_acquisition:
            api.post_async_buffer(
                session.handle, buffer.address, buffer.size_bytes
            )

    return records


def abort_capture(api: ATSApi, session: CaptureSession) -> None:
    if session.active:
        api.abort_async_read(session.handle)
        session.active = False
        session.started = False


def free_capture(session: CaptureSession) -> None:
    if session.active:
        raise RuntimeError("Call abort_capture before free_capture")
    if not session.released:
        for buffer in session.buffers:
            release_dma_memory(buffer.address)
        session.released = True


def unpack_adc_codes(
    raw_samples: npt.NDArray[np.uint16], bits_per_sample: int
) -> npt.NDArray[np.uint16]:
    storage_bits = raw_samples.dtype.itemsize * 8
    shift = storage_bits - bits_per_sample
    if shift < 0:
        raise ValueError("ADC resolution is larger than the transfer data type")
    return np.right_shift(raw_samples, shift)


def adc_codes_to_volts(
    raw_samples: npt.NDArray[np.uint16],
    bits_per_sample: int,
    input_range_volts: float,
) -> npt.NDArray[np.float64]:
    codes = unpack_adc_codes(raw_samples, bits_per_sample).astype(np.float64)
    code_midpoint = float(1 << (bits_per_sample - 1))
    return (codes - code_midpoint) * (input_range_volts / code_midpoint)


def demodulate_tone(
    records_volts: npt.NDArray[np.float64],
    sample_rate_hz: float,
    tone_frequency_hz: float,
) -> npt.NDArray[np.complex128]:
    return dispersive_demodulate(
        records_volts, sample_rate_hz, tone_frequency_hz
    )


def estimate_peak_frequency(
    records_volts: npt.NDArray[np.float64], sample_rate_hz: float
) -> npt.NDArray[np.float64]:
    centered = records_volts - records_volts.mean(axis=1, keepdims=True)
    spectrum = np.abs(np.fft.rfft(centered, axis=1))
    frequencies = np.fft.rfftfreq(records_volts.shape[1], 1.0 / sample_rate_hz)
    peak_indices = np.argmax(spectrum[:, 1:], axis=1) + 1
    return frequencies[peak_indices]


def analyze_tone(
    records_volts: npt.NDArray[np.float64],
    sample_rate_hz: float,
    tone_frequency_hz: float,
) -> ToneAnalysis:
    complex_amplitude = demodulate_tone(
        records_volts, sample_rate_hz, tone_frequency_hz
    )
    return ToneAnalysis(
        complex_amplitude=complex_amplitude,
        amplitude_volts=np.abs(complex_amplitude),
        phase_radians=np.angle(complex_amplitude),
        peak_frequency_hz=estimate_peak_frequency(records_volts, sample_rate_hz),
    )
