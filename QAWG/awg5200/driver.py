"""OOP hardware boundary for the Tektronix AWG5208."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt

from .transport import ScpiTransport, open_visa_transport
from ..timeline import (
    Parallel,
    Timeline,
    Waveform,
    align_channel_envelopes,
    align_channels,
    channel_names,
)
from .waveforms import make_wfmx, modulate_envelope, trigger_channel_for

CHANNEL_COUNT = 8
MAX_SAMPLE_RATE_HZ = 2.5e9
DEFAULT_WAVEFORM_DIRECTORY = r"\Users\OEM\Documents"


class TriggerInput(str, Enum):
    A = "A"
    B = "B"


class TriggerSlope(str, Enum):
    RISING = "RISing"
    FALLING = "FALLing"


@dataclass(frozen=True)
class TriggerConfig:
    input: TriggerInput = TriggerInput.A
    level_volts: float = 0.5
    slope: TriggerSlope = TriggerSlope.RISING
    impedance_ohms: int = 50


def validate_channel(channel: int) -> int:
    if not 1 <= channel <= CHANNEL_COUNT:
        raise ValueError(f"channel must be between 1 and {CHANNEL_COUNT}")
    return channel


def quote_scpi(value: str) -> str:
    if '"' in value or "\n" in value or "\r" in value:
        raise ValueError("SCPI strings cannot contain quotes or line endings")
    return f'"{value}"'


def ieee_block(data: bytes) -> bytes:
    length = str(len(data)).encode("ascii")
    if len(length) > 9:
        raise ValueError("binary block is too large")
    return b"#" + str(len(length)).encode("ascii") + length + data


def normalize_filename(name: str) -> str:
    filename = name if name.lower().endswith(".wfmx") else f"{name}.wfmx"
    quote_scpi(filename)
    return filename


def waveform_name_from_filename(filename: str) -> str:
    return filename[:-5] if filename.lower().endswith(".wfmx") else filename


class AWG5208:
    def __init__(
        self,
        transport: ScpiTransport,
        waveform_directory: str = DEFAULT_WAVEFORM_DIRECTORY,
    ) -> None:
        self._transport = transport
        self.waveform_directory = waveform_directory.rstrip("\\")
        self._waveforms: dict[str, npt.NDArray[np.float64]] = {}
        self._activity_waveforms: dict[str, npt.NDArray[np.float64]] = {}
        self._assigned_waveforms: dict[int, str] = {}
        self._sequences: dict[str, dict[int, tuple[str, ...]]] = {}
        self._sample_rate_hz: float | None = None

    @classmethod
    def connect(
        cls,
        resource_name: str,
        timeout_ms: int = 60_000,
        backend: str | None = None,
    ) -> "AWG5208":
        transport = open_visa_transport(resource_name, timeout_ms, backend)
        instrument = cls(transport)
        instrument.verify_identity()
        return instrument

    def write(self, command: str) -> None:
        self._transport.write(command)

    def query(self, command: str) -> str:
        return self._transport.query(command).strip()

    def identify(self) -> str:
        return self.query("*IDN?")

    def verify_identity(self) -> None:
        identity = self.identify().upper()
        if "TEKTRONIX" not in identity or "AWG5208" not in identity:
            raise RuntimeError(f"Expected Tektronix AWG5208, received {identity!r}")

    def close(self) -> None:
        self._transport.close()

    def reset(self) -> None:
        self.write("*RST")

    def wait_until_complete(self) -> None:
        self.query("*OPC?")

    def error(self) -> str:
        return self.query("SYSTem:ERRor?")

    def set_awg_mode(self) -> None:
        self.write("INSTrument:MODE AWG")

    def use_external_10mhz_reference(self) -> None:
        """Lock the internal sample clock to the fixed external reference."""
        self.write("CLOCk:SOURce EFIX")

    def set_sample_rate(self, sample_rate_hz: float) -> None:
        if not 1.49e3 <= sample_rate_hz <= MAX_SAMPLE_RATE_HZ:
            raise ValueError(
                "This AWG5208 supports sample_rate_hz between "
                "1.49 kSa/s and 2.5 GSa/s"
            )
        self.write(f"CLOCk:SRATe {sample_rate_hz:.12g}")
        self._sample_rate_hz = float(sample_rate_hz)

    def set_channel_amplitude(self, channel: int, amplitude_vpp: float) -> None:
        validate_channel(channel)
        if not 0.25 <= amplitude_vpp <= 1.5:
            raise ValueError("AWG5208 amplitude must be between 0.25 and 1.5 Vpp")
        self.write(f"SOURce{channel}:VOLTage {amplitude_vpp:.12g}")

    def set_channel_offset(self, channel: int, offset_volts: float) -> None:
        validate_channel(channel)
        if not -2.0 <= offset_volts <= 2.0:
            raise ValueError("offset_volts must be between -2 and 2 V")
        self.write(
            f"SOURce{channel}:VOLTage:LEVel:IMMediate:OFFSet "
            f"{offset_volts:.12g}"
        )

    def set_channel_resolution(self, channel: int, bits: int) -> None:
        validate_channel(channel)
        if bits not in range(12, 17):
            raise ValueError("AWG5208 resolution must be 12, 13, 14, 15, or 16 bits")
        self.write(f"SOURce{channel}:DAC:RESolution {bits}")

    def set_output(self, channel: int, enabled: bool) -> None:
        validate_channel(channel)
        self.write(f"OUTPut{channel}:STATe {int(enabled)}")

    def set_marker_levels(
        self,
        channel: int,
        marker: int,
        low_volts: float,
        high_volts: float,
    ) -> None:
        validate_channel(channel)
        if not 1 <= marker <= 4:
            raise ValueError("marker must be between 1 and 4")
        if low_volts >= high_volts:
            raise ValueError("marker low level must be below its high level")
        self.write(
            f"SOURce{channel}:MARKer{marker}:VOLTage:LOW {low_volts:.12g}"
        )
        self.write(
            f"SOURce{channel}:MARKer{marker}:VOLTage:HIGH {high_volts:.12g}"
        )

    def configure_trigger(self, config: TriggerConfig) -> None:
        prefix = f"TRIGger:{config.input.value}"
        self.write(f"{prefix}:LEVel {config.level_volts:.12g}")
        self.write(f"{prefix}:SLOPe {config.slope.value}")
        self.write(f"{prefix}:IMPedance {config.impedance_ohms}")

    def force_trigger(self, trigger: TriggerInput = TriggerInput.A) -> None:
        self.write(f"TRIGger:IMMediate {trigger.value}TRigger")

    def run(self, wait_until_ready: bool = False) -> None:
        self.write("AWGControl:RUN")
        if wait_until_ready:
            self.wait_until_complete()

    def stop(self) -> None:
        self.write("AWGControl:STOP")

    def run_state(self) -> int:
        return int(self.query("AWGControl:RSTATe?"))

    def clear_all(self) -> None:
        """Stop playback and clear all channel, waveform, and sequence assets."""
        self.stop()
        for channel in range(1, CHANNEL_COUNT + 1):
            self.set_output(channel, False)
            self.write(f"SOURce{channel}:CASSet:CLEAR")
        self.write("SLISt:SEQuence:DELete ALL")
        self.write("WLISt:WAVeform:DELete ALL")
        self.wait_until_complete()
        self._waveforms.clear()
        self._activity_waveforms.clear()
        self._assigned_waveforms.clear()
        self._sequences.clear()

    def set_current_directory(self, path: str) -> None:
        self.write(f"MMEMory:CDIRectory {quote_scpi(path)}")

    def delete_file(self, filename: str) -> None:
        self.write(f"MMEMory:DELete {quote_scpi(filename)}")

    def upload_file(self, filename: str, contents: bytes) -> None:
        self.set_current_directory(self.waveform_directory)
        command = f"MMEMory:DATA {quote_scpi(filename)},".encode("ascii")
        self._transport.write_raw(command + ieee_block(contents))

    def load_waveform_file(self, filename: str) -> None:
        path = f"C:{self.waveform_directory}\\{filename}"
        self.write(f"MMEMory:OPEN {quote_scpi(path)}")
        self.wait_until_complete()

    def _upload_waveform_data(
        self,
        name: str,
        waveform_volts: npt.ArrayLike,
        amplitude_vpp: float,
        markers: tuple[npt.ArrayLike, ...] = (),
        overwrite: bool = True,
    ) -> str:
        filename = normalize_filename(name)
        waveform = np.asarray(waveform_volts, dtype=np.float64).reshape(-1)
        contents = make_wfmx(waveform, amplitude_vpp, markers)
        if overwrite:
            self.delete_file(filename)
            self.query("SYSTem:ERRor:CODE?")
        self.upload_file(filename, contents)
        self.load_waveform_file(filename)
        waveform_name = waveform_name_from_filename(filename)
        self._waveforms[waveform_name] = waveform.copy()
        return waveform_name

    def upload_waveform_asset(
        self,
        name: str,
        waveform_volts: npt.ArrayLike,
        amplitude_vpp: float = 0.5,
        markers: tuple[npt.ArrayLike, ...] = (),
        overwrite: bool = True,
    ) -> str:
        """Upload one WFMX asset without assigning it to a channel."""
        return self._upload_waveform_data(
            name=name,
            waveform_volts=waveform_volts,
            amplitude_vpp=amplitude_vpp,
            markers=markers,
            overwrite=overwrite,
        )

    def create_sequence(
        self,
        name: str,
        tracks: dict[int, Sequence[str]],
        *,
        repetitions: int | Sequence[int] = 1,
        goto_step: int | None = 1,
        enabled: bool = True,
    ) -> str:
        """Create and assign a sequence list from uploaded waveform assets."""
        if not tracks:
            raise ValueError("tracks cannot be empty")
        for channel in tracks:
            validate_channel(channel)

        track_waveforms = {
            channel: tuple(waveform_names)
            for channel, waveform_names in tracks.items()
        }
        step_counts = {len(waveform_names) for waveform_names in track_waveforms.values()}
        if len(step_counts) != 1 or not step_counts or next(iter(step_counts)) < 1:
            raise ValueError("all tracks must contain the same positive number of steps")
        step_count = next(iter(step_counts))

        if isinstance(repetitions, int):
            repeat_counts = (repetitions,) * step_count
        else:
            repeat_counts = tuple(int(value) for value in repetitions)
        if len(repeat_counts) != step_count or any(value < 1 for value in repeat_counts):
            raise ValueError("repetitions must provide one positive count per step")
        if goto_step is not None and not 1 <= goto_step <= step_count:
            raise ValueError("goto_step must refer to an existing sequence step")
        for waveform_names in track_waveforms.values():
            for waveform_name in waveform_names:
                if waveform_name not in self._waveforms:
                    raise ValueError(
                        f"waveform {waveform_name!r} was not uploaded by this driver session"
                    )

        sequence_name = name.removesuffix(".seqx")
        quote_scpi(sequence_name)
        ordered_channels = tuple(sorted(track_waveforms))
        self.write(
            f"SLISt:SEQuence:NEW {quote_scpi(sequence_name)},"
            f"{step_count},{len(ordered_channels)}"
        )
        for step_index in range(1, step_count + 1):
            for track_index, channel in enumerate(ordered_channels, start=1):
                waveform_name = track_waveforms[channel][step_index - 1]
                self.write(
                    f"SLISt:SEQuence:STEP{step_index}:TASSet{track_index}:"
                    f"WAVeform {quote_scpi(sequence_name)},"
                    f"{quote_scpi(waveform_name)}"
                )
            self.write(
                f"SLISt:SEQuence:STEP{step_index}:RCOunt "
                f"{quote_scpi(sequence_name)},{repeat_counts[step_index - 1]}"
            )
        if goto_step is not None:
            self.write(
                f"SLISt:SEQuence:STEP{step_count}:GOTO "
                f"{quote_scpi(sequence_name)},{goto_step}"
            )

        for track_index, channel in enumerate(ordered_channels, start=1):
            self.write(
                f"SOURce{channel}:CASSet:SEQuence "
                f"{quote_scpi(sequence_name)},{track_index}"
            )
            self.set_output(channel, enabled)
        self.wait_until_complete()
        self._sequences[sequence_name] = track_waveforms
        return sequence_name

    def rewind_sequence(self, sequence_name: str) -> None:
        """Stop and reassign a sequence so playback restarts from step one."""
        if sequence_name not in self._sequences:
            raise ValueError(
                f"sequence {sequence_name!r} was not created by this driver session"
            )
        self.stop()
        tracks = self._sequences[sequence_name]
        for track_index, channel in enumerate(sorted(tracks), start=1):
            self.write(
                f"SOURce{channel}:CASSet:SEQuence "
                f"{quote_scpi(sequence_name)},{track_index}"
            )
        self.wait_until_complete()

    def upload_waveform(
        self,
        waveform_array: npt.ArrayLike,
        fc: float,
        ch: int,
        amplitude_vpp: float = 0.5,
        name: str | None = None,
        phase_radians: float = 0.0,
        enabled: bool = True,
        clear_before_upload: bool = True,
    ) -> str:
        """Modulate an envelope, upload it, and assign it to one channel."""
        validate_channel(ch)
        if self._sample_rate_hz is None:
            raise RuntimeError("Call set_sample_rate() before upload_waveform()")
        if not 0 <= fc <= self._sample_rate_hz / 2:
            raise ValueError("fc must be between DC and the Nyquist frequency")

        envelope = np.asarray(waveform_array, dtype=np.float64).reshape(-1)
        if envelope.size < 1:
            raise ValueError("waveform_array cannot be empty")
        if clear_before_upload:
            self.clear_all()
        waveform = modulate_envelope(
            envelope,
            self._sample_rate_hz,
            fc,
            phase_radians,
        )
        waveform_name = self._upload_waveform_data(
            name=name or f"ch{ch}_{fc / 1e6:g}MHz",
            waveform_volts=waveform,
            amplitude_vpp=amplitude_vpp,
        )
        self._activity_waveforms[waveform_name] = envelope.copy()
        self.set_channel_resolution(ch, 16)
        self.prepare_channel(
            channel=ch,
            waveform_name=waveform_name,
            amplitude_vpp=amplitude_vpp,
            enabled=enabled,
        )
        return waveform_name

    def upload_timeline(
        self,
        timeline: Timeline | Waveform | Parallel,
        amplitude_vpp: float | dict[int, float] = 0.5,
        name_prefix: str = "timeline",
        total_duration_s: float = 5e-6,
        enabled: bool = True,
        clear_before_upload: bool = True,
    ) -> dict[int, str]:
        """Align, upload, and assign a multi-channel timeline."""
        if self._sample_rate_hz is None:
            raise RuntimeError("Call set_sample_rate() before upload_timeline()")
        channel_data = align_channels(
            timeline,
            self._sample_rate_hz,
            total_duration_s=total_duration_s,
        )
        channel_envelopes = align_channel_envelopes(
            timeline,
            self._sample_rate_hz,
            total_duration_s=total_duration_s,
        )
        explicit_names = channel_names(timeline)
        if clear_before_upload:
            self.clear_all()
        uploaded: dict[int, str] = {}
        for channel, values in channel_data.items():
            channel_amplitude = (
                amplitude_vpp[channel]
                if isinstance(amplitude_vpp, dict)
                else amplitude_vpp
            )
            try:
                waveform_name = self._upload_waveform_data(
                    name=explicit_names.get(channel, f"{name_prefix}_ch{channel}"),
                    waveform_volts=values,
                    amplitude_vpp=channel_amplitude,
                )
            except ValueError as exc:
                raise ValueError(f"Channel {channel}: {exc}") from exc
            self._activity_waveforms[waveform_name] = channel_envelopes[
                channel
            ].copy()
            self.set_channel_resolution(channel, 16)
            self.prepare_channel(
                channel,
                waveform_name,
                channel_amplitude,
                enabled=enabled,
            )
            uploaded[channel] = waveform_name
        return uploaded

    def assign_waveform(self, channel: int, waveform_name: str) -> None:
        validate_channel(channel)
        self.write(
            f"SOURce{channel}:CASSet:WAVeform {quote_scpi(waveform_name)}"
        )
        self._assigned_waveforms[channel] = waveform_name

    def marker(
        self,
        waveform_ch: int,
        marker_ch: int,
        marker_number: int = 1,
        low_volts: float = 0.0,
        high_volts: float = 1.2,
        threshold_ratio: float = 1e-3,
        padding_samples: int = 0,
        amplitude_vpp: float = 0.5,
        envelope_waveform: npt.ArrayLike | None = None,
    ) -> str:
        """Create a marker channel aligned to the waveform on another channel."""
        validate_channel(waveform_ch)
        validate_channel(marker_ch)
        if not 1 <= marker_number <= 4:
            raise ValueError("marker_number must be between 1 and 4")
        if waveform_ch not in self._assigned_waveforms:
            raise ValueError(f"channel {waveform_ch} has no assigned waveform")

        waveform_name = self._assigned_waveforms[waveform_ch]
        if waveform_name not in self._waveforms:
            raise ValueError(
                "The assigned waveform was not uploaded by this driver session"
            )
        ref_wave = (
            np.asarray(envelope_waveform, dtype=np.float64).reshape(-1)
            if envelope_waveform is not None
            else self._activity_waveforms.get(
                waveform_name,
                self._waveforms[waveform_name],
            )
        )
        if ref_wave.size != self._waveforms[waveform_name].size:
            raise ValueError(
                "envelope_waveform must match the assigned waveform length"
            )
        zero_waveform, active_marker = trigger_channel_for(
            ref_wave,
            threshold_ratio=threshold_ratio,
            padding_samples=padding_samples,
        )
        marker_waveform = (
            self._waveforms[waveform_name].copy()
            if waveform_ch == marker_ch
            else zero_waveform
        )
        markers = tuple(
            active_marker if index == marker_number else np.zeros_like(active_marker)
            for index in range(1, marker_number + 1)
        )
        trigger_name = self._upload_waveform_data(
            name=f"marker_ch{marker_ch}_for_ch{waveform_ch}",
            waveform_volts=marker_waveform,
            amplitude_vpp=amplitude_vpp,
            markers=markers,
        )
        self._activity_waveforms[trigger_name] = ref_wave.copy()
        self.set_channel_resolution(marker_ch, 16 - marker_number)
        self.set_marker_levels(
            marker_ch,
            marker_number,
            low_volts,
            high_volts,
        )
        self.prepare_channel(
            marker_ch,
            trigger_name,
            amplitude_vpp,
            enabled=True,
        )
        return trigger_name

    def prepare_channel(
        self,
        channel: int,
        waveform_name: str,
        amplitude_vpp: float,
        enabled: bool = True,
    ) -> None:
        self.set_channel_amplitude(channel, amplitude_vpp)
        self.assign_waveform(channel, waveform_name)
        self.set_output(channel, enabled)

    def __enter__(self) -> "AWG5208":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
