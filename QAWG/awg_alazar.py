"""Unified AWG5208 and ATS9371 experiment control."""

from __future__ import annotations

import hashlib
from contextlib import suppress
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from .alazar import (
    ATSApi,
    Alazar9371,
    BoardInfo,
    CaptureConfig,
    CaptureData,
    TriggerConfig,
    correct_interleaving_offsets,
    downconvert,
    fir_filter,
    integrate,
    open_ats9371,
)
from .alazar.constants import (
    CHANNEL_A,
    CHANNEL_B,
    MIN_SAMPLES_PER_RECORD,
    SAMPLES_PER_RECORD_ALIGNMENT,
    TRIGGER_SLOPE_NEGATIVE,
    TRIGGER_SLOPE_POSITIVE,
)
from .awg5200 import AWG5208, AWGProgram, AWGStep

if TYPE_CHECKING:
    from .compiler import (
        CompiledExperiment,
        ExperimentProgram,
        ExperimentResult,
    )


def validate_average_count(n_average: int) -> int:
    averages = int(n_average)
    if averages < 1:
        raise ValueError("n_average must be positive")
    return averages


def remove_record_dc_offset(
    records: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return records - np.mean(records, axis=1, keepdims=True)


def normalize_adc_channel(channel: str | int) -> int:
    """Map CHA/CHB or zero-based 0/1 to an ATS channel constant."""
    if isinstance(channel, str):
        name = channel.strip().upper()
        if name == "CHA":
            return CHANNEL_A
        if name == "CHB":
            return CHANNEL_B
    elif isinstance(channel, int) and not isinstance(channel, bool):
        if channel == 0:
            return CHANNEL_A
        if channel == 1:
            return CHANNEL_B
    raise ValueError("adc_channel must be 'CHA', 'CHB', 0, or 1")


def normalize_trigger_slope(slope: str | int) -> int:
    """Map rising/falling names to ATS trigger-slope constants."""
    if isinstance(slope, str):
        name = slope.strip().lower()
        if name in {"rising", "positive"}:
            return TRIGGER_SLOPE_POSITIVE
        if name in {"falling", "negative"}:
            return TRIGGER_SLOPE_NEGATIVE
    elif slope in (TRIGGER_SLOPE_POSITIVE, TRIGGER_SLOPE_NEGATIVE):
        return int(slope)
    raise ValueError(
        "trigger_slope must be 'rising', 'falling', "
        "TRIGGER_SLOPE_POSITIVE, or TRIGGER_SLOPE_NEGATIVE"
    )


class AWGAlazar:
    """Coordinate AWG playback, triggered acquisition, and IQ averaging."""

    def __init__(
        self,
        awg: AWG5208,
        ats_api: ATSApi,
        ats_board: BoardInfo,
        *,
        awg_sample_rate_hz: float,
        alazar_sample_rate_hz: float,
        acquire_window_ns: float,
        tone_frequency_hz: float = 0.0,
        trigger_delay_s: float = 0.0,
        integrate_time_s: float | None = None,
        integrate_window_ns: tuple[float, float] | None = None,
        adc_channel: str | int = "CHA",
        fir_cutoff_hz: float = 20e6,
        fir_number_of_taps: int = 41,
        fir_kaiser_beta: float = 6.0,
        integration_weights: npt.ArrayLike | None = None,
        reference_phase_radians: float = 0.0,
        trigger_slope: str | int = "rising",
        trigger_level: int = 140,
        input_range_volts: float = 0.4,
        timeout_ms: int = 5000,
        dma_buffer_count: int = 4,
        maximum_records_per_buffer: int = 100,
        use_external_10mhz_reference: bool = True,
        baseline_time_s: float | None = None,
        mask_external_trigger_lsb: bool = True,
        remove_dc_offset: bool = True,
    ) -> None:
        self.awg = awg
        self.ats_api = ats_api
        self.ats_board = ats_board
        self.alazar = Alazar9371(ats_api, ats_board)
        self.awg_sample_rate_hz = float(awg_sample_rate_hz)
        self.alazar_sample_rate_hz = float(alazar_sample_rate_hz)
        self.tone_frequency_hz = float(tone_frequency_hz)
        self.trigger_delay_s = float(trigger_delay_s)
        self.acquire_window_ns = float(acquire_window_ns)
        if integrate_time_s is not None and integrate_window_ns is not None:
            raise ValueError(
                "Use integrate_time_s or integrate_window_ns, not both"
            )
        if integrate_time_s is not None:
            self.integrate_time_s = float(integrate_time_s)
            self.integrate_window_ns = (
                self.trigger_delay_s * 1e9,
                (self.trigger_delay_s + self.integrate_time_s) * 1e9,
            )
        elif integrate_window_ns is not None:
            self.integrate_window_ns = (
                float(integrate_window_ns[0]),
                float(integrate_window_ns[1]),
            )
            self.integrate_time_s = (
                self.integrate_window_ns[1]
                - self.integrate_window_ns[0]
            ) * 1e-9
        else:
            self.integrate_time_s = self.acquire_window_ns * 1e-9
            self.integrate_window_ns = (0.0, self.acquire_window_ns)
        self.adc_channel = normalize_adc_channel(adc_channel)
        self.fir_cutoff_hz = float(fir_cutoff_hz)
        self.fir_number_of_taps = int(fir_number_of_taps)
        self.fir_kaiser_beta = float(fir_kaiser_beta)
        self.integration_weights = (
            None
            if integration_weights is None
            else np.asarray(integration_weights, dtype=np.complex128).copy()
        )
        self.reference_phase_radians = float(reference_phase_radians)
        self.trigger_slope = normalize_trigger_slope(trigger_slope)
        self.trigger_level = int(trigger_level)
        self.input_range_volts = float(input_range_volts)
        self.timeout_ms = int(timeout_ms)
        self.dma_buffer_count = int(dma_buffer_count)
        self.maximum_records_per_buffer = int(maximum_records_per_buffer)
        self.use_external_10mhz_reference = bool(use_external_10mhz_reference)
        self.baseline_time_s = baseline_time_s
        self.mask_external_trigger_lsb = bool(mask_external_trigger_lsb)
        self.remove_dc_offset = bool(remove_dc_offset)

        self.last_raw_codes: npt.NDArray[np.uint16] | None = None
        self.last_records_volts: npt.NDArray[np.float64] | None = None
        self.last_downconverted_iq: (
            npt.NDArray[np.complex128] | None
        ) = None
        self.last_shot_iq: npt.NDArray[np.complex128] | None = None
        self.last_time_s: npt.NDArray[np.float64] | None = None
        self.last_capture: CaptureData | None = None
        self._uploaded_compiled_key: str | None = None
        self._uploaded_sequence_name: str | None = None

        self._validate_settings()

    @classmethod
    def connect(
        cls,
        awg_resource: str,
        *,
        awg_sample_rate_hz: float,
        alazar_sample_rate_hz: float = 1e9,
        acquire_window_s: float | None = None,
        acquire_window_ns: float | None = None,
        trigger_slope: str | int = "rising",
        trigger_level: int = 140,
        awg_timeout_ms: int = 60_000,
        ats_system_id: int = 1,
        ats_board_id: int = 1,
        **settings: Any,
    ) -> "AWGAlazar":
        """Connect both instruments and apply fixed hardware settings."""
        if acquire_window_s is None and acquire_window_ns is None:
            raise ValueError(
                "Provide acquire_window_s (preferred) or acquire_window_ns"
            )
        if acquire_window_s is not None and acquire_window_ns is not None:
            raise ValueError(
                "Use acquire_window_s or acquire_window_ns, not both"
            )
        requested_window_ns = (
            float(acquire_window_s) * 1e9
            if acquire_window_s is not None
            else float(acquire_window_ns)
        )
        awg = AWG5208.connect(awg_resource, timeout_ms=awg_timeout_ms)
        try:
            ats_api = ATSApi()
            ats_board = open_ats9371(ats_api, ats_system_id, ats_board_id)
            experiment = cls(
                awg,
                ats_api,
                ats_board,
                awg_sample_rate_hz=awg_sample_rate_hz,
                alazar_sample_rate_hz=alazar_sample_rate_hz,
                acquire_window_ns=requested_window_ns,
                trigger_slope=trigger_slope,
                trigger_level=trigger_level,
                **settings,
            )
            experiment.configure()
            return experiment
        except BaseException:
            awg.close()
            raise

    def _validate_settings(self) -> None:
        if not 1.49e3 <= self.awg_sample_rate_hz <= 2.5e9:
            raise ValueError(
                "awg_sample_rate_hz must be between 1.49 kSa/s and 2.5 GSa/s"
            )
        if self.alazar_sample_rate_hz != 1e9:
            raise ValueError("ATS9371 acquisition is currently fixed to 1 GS/s")
        if self.input_range_volts != 0.4:
            raise ValueError(
                "The current ATS9371 driver configures a fixed +/-400 mV range"
            )
        if not 0 <= self.tone_frequency_hz < self.alazar_sample_rate_hz / 2:
            raise ValueError("tone_frequency_hz must be between DC and Nyquist")
        if self.trigger_delay_s < 0:
            raise ValueError("trigger_delay_s cannot be negative")
        if not 0 <= self.trigger_level <= 255:
            raise ValueError("trigger_level must be an ATS code from 0 to 255")
        if self.acquire_window_ns <= 0:
            raise ValueError("acquire_window_ns must be positive")
        integrate_start_ns, integrate_stop_ns = self.integrate_window_ns
        if not 0 <= integrate_start_ns < integrate_stop_ns:
            raise ValueError(
                "integrate_window_ns must satisfy 0 <= start < stop"
            )
        if integrate_stop_ns > self.acquire_window_ns:
            raise ValueError(
                "integrate_window_ns must fit inside acquire_window_ns"
            )
        if not 0 < self.fir_cutoff_hz < self.alazar_sample_rate_hz / 2:
            raise ValueError("fir_cutoff_hz must be between DC and Nyquist")
        if self.fir_number_of_taps < 3 or self.fir_number_of_taps % 2 == 0:
            raise ValueError("fir_number_of_taps must be odd and at least 3")
        if self.fir_kaiser_beta < 0:
            raise ValueError("fir_kaiser_beta cannot be negative")
        if self.baseline_time_s is not None:
            if not 0 < self.baseline_time_s * 1e9 <= self.acquire_window_ns:
                raise ValueError(
                    "baseline_time_s must fit inside acquire_window_ns"
                )

    @property
    def trigger_delay_samples(self) -> int:
        """Integration-window delay from the start of the acquired record."""
        return self.ns2cycles(self.trigger_delay_s * 1e9, inst="adc")

    @property
    def adc_channel_name(self) -> str:
        return "CHA" if self.adc_channel == CHANNEL_A else "CHB"

    @property
    def adc_lsb_volts(self) -> float:
        return self.input_range_volts / (
            1 << (self.ats_board.bits_per_sample - 1)
        )

    @property
    def integrate_window_cycles(self) -> tuple[int, int]:
        start_ns, stop_ns = self.integrate_window_ns
        return (
            self.ns2cycles(start_ns, inst="adc"),
            self.ns2cycles(stop_ns, inst="adc"),
        )

    @property
    def integrate_samples(self) -> int:
        start, stop = self.integrate_window_cycles
        return stop - start

    @property
    def acquire_window_cycles(self) -> int:
        requested = self.ns2cycles(self.acquire_window_ns, inst="adc")
        alignment = SAMPLES_PER_RECORD_ALIGNMENT
        aligned = ((requested + alignment - 1) // alignment) * alignment
        return max(aligned, MIN_SAMPLES_PER_RECORD)

    def _sample_rate_for(self, inst: str) -> float:
        instrument = inst.lower()
        if instrument == "dac":
            return self.awg_sample_rate_hz
        if instrument == "adc":
            return self.alazar_sample_rate_hz
        raise ValueError("inst must be 'dac' for AWG or 'adc' for Alazar")

    def ns2cycles(self, duration_ns: float, inst: str = "dac") -> int:
        """Convert nanoseconds to AWG DAC or Alazar ADC sample cycles."""
        if duration_ns < 0:
            raise ValueError("duration_ns cannot be negative")
        return int(round(duration_ns * 1e-9 * self._sample_rate_for(inst)))

    def cycles2ns(self, cycles: int, inst: str = "dac") -> float:
        """Convert AWG DAC or Alazar ADC sample cycles to nanoseconds."""
        if cycles < 0:
            raise ValueError("cycles cannot be negative")
        return cycles / self._sample_rate_for(inst) * 1e9

    def configure(self) -> None:
        """Configure clocks, AWG mode/sample rate, and current ATS settings."""
        self.awg.set_awg_mode()
        if self.use_external_10mhz_reference:
            self.awg.use_external_10mhz_reference()
        self.awg.set_sample_rate(self.awg_sample_rate_hz)
        self.alazar.configure(
            trigger=TriggerConfig(
                slope=self.trigger_slope,
                level=self.trigger_level,
                delay_samples=0,
                timeout_ticks=0,
            ),
            use_external_10mhz_reference=self.use_external_10mhz_reference,
            channel=self.adc_channel,
        )

    def compile(self, program: "ExperimentProgram") -> "CompiledExperiment":
        """Compile and bind a program using this connected hardware setup."""
        return program.compile(hardware=self)

    def configure_experiment(
        self,
        *,
        tone_frequency_hz: float,
        trigger_delay_s: float,
        integrate_time_s: float,
        adc_channel: str | int,
    ) -> None:
        """Acquire from the marker and integrate after trigger_delay_s."""
        self.tone_frequency_hz = float(tone_frequency_hz)
        self.trigger_delay_s = float(trigger_delay_s)
        self.integrate_time_s = float(integrate_time_s)
        self.integrate_window_ns = (
            self.trigger_delay_s * 1e9,
            (self.trigger_delay_s + self.integrate_time_s) * 1e9,
        )
        self.adc_channel = normalize_adc_channel(adc_channel)
        self._validate_settings()
        self.alazar.configure(
            trigger=TriggerConfig(
                slope=self.trigger_slope,
                level=self.trigger_level,
                delay_samples=0,
                timeout_ticks=0,
            ),
            use_external_10mhz_reference=self.use_external_10mhz_reference,
            channel=self.adc_channel,
        )

    @staticmethod
    def _compiled_marker_tuple(
        compiled: "CompiledExperiment",
        step_index: int,
    ) -> tuple[npt.NDArray[np.bool_], ...]:
        active = compiled.marker_waveforms[step_index]
        return tuple(
            active
            if marker == compiled.readout.marker_number
            else np.zeros_like(active)
            for marker in range(1, compiled.readout.marker_number + 1)
        )

    @staticmethod
    def _compiled_key(compiled: "CompiledExperiment") -> str:
        """Fingerprint mutable ndarray content so stale uploads are detected."""
        digest = hashlib.blake2b(digest_size=16)
        digest.update(compiled.program_name.encode("utf-8"))
        digest.update(repr(compiled.readout).encode("utf-8"))
        digest.update(repr(sorted(compiled.channel_amplitudes_vpp.items())).encode())
        for channel, values in sorted(compiled.channel_waveforms.items()):
            array = np.ascontiguousarray(values)
            digest.update(str(channel).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.dtype.str.encode())
            digest.update(array.tobytes())
        markers = np.ascontiguousarray(compiled.marker_waveforms)
        digest.update(str(markers.shape).encode())
        digest.update(markers.tobytes())
        return digest.hexdigest()

    def upload_compiled_experiment(
        self,
        compiled: "CompiledExperiment",
    ) -> str:
        """Translate a compiled plan and let the AWG choose waveform/sequence."""
        marker_channel = compiled.readout.marker_channel
        channels = set(compiled.channel_waveforms)
        channels.add(marker_channel)
        amplitudes_vpp = dict(compiled.channel_amplitudes_vpp)
        amplitudes_vpp.setdefault(marker_channel, 0.5)
        steps: list[AWGStep] = []
        for step_index in range(compiled.number_of_sequence_steps):
            steps.append(
                AWGStep(
                    waveforms={
                        channel: (
                            compiled.channel_waveforms[channel][step_index]
                            if channel in compiled.channel_waveforms
                            else np.zeros(compiled.marker_waveforms.shape[1])
                        )
                        for channel in channels
                    },
                    markers={
                        marker_channel: self._compiled_marker_tuple(
                            compiled, step_index
                        )
                    },
                )
            )

        uploaded = self.awg.upload_program(
            AWGProgram(
                name=compiled.program_name,
                steps=tuple(steps),
                amplitudes_vpp=amplitudes_vpp,
                marker_levels={
                    (marker_channel, compiled.readout.marker_number): (
                        compiled.readout.marker_low_volts,
                        compiled.readout.marker_high_volts,
                    )
                },
            )
        )
        self._uploaded_compiled_key = self._compiled_key(compiled)
        self._uploaded_sequence_name = uploaded.name if uploaded.is_sequence else None
        return uploaded.name

    def acquire_compiled_experiment(
        self,
        compiled: "CompiledExperiment",
        n_average: int,
    ) -> "ExperimentResult":
        """Configure, execute, and collect one compiled experiment plan."""
        from .compiler import ExperimentResult

        integrate_time_s = (
            compiled.readout.length_s
            if compiled.readout.integrate_time_s is None
            else compiled.readout.integrate_time_s
        )
        self.configure_experiment(
            tone_frequency_hz=compiled.readout.demod_frequency_hz,
            trigger_delay_s=compiled.trigger_delay_s,
            integrate_time_s=integrate_time_s,
            adc_channel=compiled.readout.adc_channel,
        )
        if self._uploaded_compiled_key != self._compiled_key(compiled):
            self.upload_compiled_experiment(compiled)

        steps = compiled.number_of_sequence_steps
        averages = validate_average_count(n_average)
        capture = self._capture_data(
            n_average=steps * averages,
            remove_dc_offset=compiled.remove_dc_offset,
        )
        records = capture.records_volts
        downconverted, weighted_shots, _ = self._demodulate(records)
        raw_time_s = (
            np.arange(records.shape[1], dtype=np.float64)
            / self.alazar_sample_rate_hz
        )
        iq_time_s = raw_time_s[: downconverted.shape[1]]
        raw_by_point = records.reshape(
            averages,
            steps,
            records.shape[1],
        )
        iq_by_point = downconverted.reshape(
            averages,
            steps,
            downconverted.shape[1],
        )
        self.last_time_s = iq_time_s

        marker_windows_s = None
        if compiled.marker_waveforms is not None:
            marker_windows_s = np.zeros(
                (compiled.number_of_sequence_steps, 2),
                dtype=np.float64,
            )
            for step_index, marker in enumerate(compiled.marker_waveforms):
                active = np.flatnonzero(marker)
                if active.size:
                    marker_windows_s[step_index] = (
                        active[0] / compiled.sample_rate_hz,
                        (active[-1] + 1) / compiled.sample_rate_hz,
                    )
        return ExperimentResult(
            axes={
                name: values.copy()
                for name, values in compiled.axes.items()
            },
            point_coordinates=compiled.point_coordinates,
            raw=raw_by_point.copy(),
            iq_traces=iq_by_point.copy(),
            iq_shots=weighted_shots.reshape(averages, steps),
            raw_time_s=raw_time_s,
            iq_time_s=iq_time_s,
            readout_name=compiled.readout.name,
            initial_trigger_delay_s=compiled.trigger_delay_s,
            readout_windows_s=(
                compiled.readout_windows_s.copy()
            ),
            marker_windows_s=marker_windows_s,
            acquire_window_s=self.acquire_window_ns * 1e-9,
            remove_dc_offset=compiled.remove_dc_offset,
            external_trigger_lsb_masked=self.mask_external_trigger_lsb,
        )

    def _acquisition_config(self, n_average: int) -> CaptureConfig:
        num_averages = validate_average_count(n_average)
        return CaptureConfig(
            samples_per_record=self.acquire_window_cycles,
            records_per_acquisition=num_averages,
            sample_rate_hz=self.alazar_sample_rate_hz,
            maximum_records_per_buffer=self.maximum_records_per_buffer,
            dma_buffer_count=self.dma_buffer_count,
            input_range_volts=self.input_range_volts,
            timeout_ms=self.timeout_ms,
            channel=self.adc_channel,
            mask_external_trigger_lsb=self.mask_external_trigger_lsb,
        )

    def _capture_data(
        self,
        n_average: int,
        *,
        remove_dc_offset: bool = False,
    ) -> CaptureData:
        config = self._acquisition_config(n_average=n_average)
        if self._uploaded_sequence_name is None:
            self.awg.stop()
        else:
            self.awg.rewind_sequence(self._uploaded_sequence_name)
        try:
            capture = self.alazar.capture(
                config,
                start_callback=lambda: self.awg.run(wait_until_ready=False),
            )
        finally:
            with suppress(Exception):
                self.awg.stop()

        records = capture.records_volts
        if self.baseline_time_s is not None:
            records = correct_interleaving_offsets(
                records,
                stop_sample=self.ns2cycles(
                    self.baseline_time_s * 1e9,
                    inst="adc",
                ),
                period=2,
            )
        if remove_dc_offset:
            records = remove_record_dc_offset(records)
        if records is not capture.records_volts:
            capture = CaptureData(
                raw_codes=capture.raw_codes,
                records_volts=records,
                sample_rate_hz=capture.sample_rate_hz,
            )

        self.last_capture = capture
        self.last_raw_codes = capture.raw_codes
        self.last_records_volts = records
        return capture

    def _demodulate(
        self,
        records: npt.NDArray[np.float64],
    ) -> tuple[
        npt.NDArray[np.complex128],
        npt.NDArray[np.complex128],
        slice,
    ]:
        """Downconvert once; return FIR traces, integrated shots, window.

        Integration uses the unfiltered baseband so the FIR does not smear
        the window edges. Also records ``last_downconverted_iq`` and
        ``last_shot_iq``.
        """
        baseband = downconvert(
            records,
            self.alazar_sample_rate_hz,
            self.tone_frequency_hz,
            self.reference_phase_radians,
        )
        filtered = self._lowpass_downconverted_iq(baseband)
        window = self._integration_slice(filtered.shape[1])
        shots = integrate(
            baseband,
            window.start,
            window.stop,
            self.integration_weights,
        )
        self.last_downconverted_iq = filtered
        self.last_shot_iq = shots
        return filtered, shots, window

    def _integration_slice(self, number_of_samples: int) -> slice:
        start, stop = self.integrate_window_cycles
        if not 0 <= start < stop <= number_of_samples:
            raise ValueError("integration window is outside the acquired record")
        return slice(start, stop)

    def _lowpass_downconverted_iq(
        self,
        baseband: npt.NDArray[np.complex128],
    ) -> npt.NDArray[np.complex128]:
        return fir_filter(
            baseband,
            self.alazar_sample_rate_hz,
            self.fir_cutoff_hz,
            self.fir_number_of_taps,
            self.fir_kaiser_beta,
        )

    def acquire_decimate(
        self,
        n_average: int,
        *,
        remove_dc_offset: bool | None = None,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.complex128],
    ]:
        """Return full-rate raw and FIR-filtered time-resolved waveforms."""
        capture = self._capture_data(
            n_average,
            remove_dc_offset=(
                self.remove_dc_offset
                if remove_dc_offset is None
                else bool(remove_dc_offset)
            ),
        )
        filtered, _, window = self._demodulate(capture.records_volts)
        self.last_time_s = capture.time_s
        return (
            capture.time_s[window],
            capture.records_volts[:, window].copy(),
            filtered[:, window].copy(),
        )

    def acquire(
        self,
        n_average: int,
        *,
        remove_dc_offset: bool | None = None,
    ) -> tuple[
        npt.NDArray[np.complex128],
        np.complex128,
    ]:
        """Return per-shot IQ points and their mean from the integration window."""
        capture = self._capture_data(
            n_average,
            remove_dc_offset=(
                self.remove_dc_offset
                if remove_dc_offset is None
                else bool(remove_dc_offset)
            ),
        )
        window = self._integration_slice(capture.records_volts.shape[1])
        baseband = downconvert(
            capture.records_volts,
            capture.sample_rate_hz,
            self.tone_frequency_hz,
            self.reference_phase_radians,
        )
        shots = integrate(
            baseband,
            window.start,
            window.stop,
            self.integration_weights,
        )
        self.last_downconverted_iq = baseband
        self.last_shot_iq = shots
        self.last_time_s = capture.time_s
        return shots.copy(), np.complex128(np.mean(shots))

    def close(self) -> None:
        self.awg.close()

    def __enter__(self) -> "AWGAlazar":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
