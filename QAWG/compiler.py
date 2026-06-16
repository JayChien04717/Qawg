"""Host-side experiment compiler for AWG5208 and ATS9371 workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Union

import numpy as np
import numpy.typing as npt

from .awg5200.waveforms import (
    MIN_WAVEFORM_SAMPLES,
    constant,
    cosine_square,
    gaussian,
    gaussian_square,
    modulate_envelope,
    trigger_channel_for,
)

ns = 1e-9
us = 1e-6
MHz = 1e6


class ValueExpression:
    """A compile-time value resolved from one sweep point."""

    def resolve(self, point: dict[str, Any]) -> float:
        raise NotImplementedError

    def __add__(self, other: ValueLike) -> "BinaryExpression":
        return BinaryExpression(self, "+", as_expression(other))

    def __radd__(self, other: ValueLike) -> "BinaryExpression":
        return BinaryExpression(as_expression(other), "+", self)

    def __sub__(self, other: ValueLike) -> "BinaryExpression":
        return BinaryExpression(self, "-", as_expression(other))

    def __rsub__(self, other: ValueLike) -> "BinaryExpression":
        return BinaryExpression(as_expression(other), "-", self)


ValueLike = Union[float, int, ValueExpression]


@dataclass(frozen=True)
class ConstantExpression(ValueExpression):
    value: float

    def resolve(self, point: dict[str, Any]) -> float:
        return self.value


@dataclass(frozen=True)
class SweepRef(ValueExpression):
    name: str

    def resolve(self, point: dict[str, Any]) -> float:
        try:
            return float(point[self.name])
        except KeyError as exc:
            raise KeyError(f"Unknown sweep {self.name!r}") from exc


@dataclass(frozen=True)
class BinaryExpression(ValueExpression):
    left: ValueExpression
    operator: str
    right: ValueExpression

    def resolve(self, point: dict[str, Any]) -> float:
        left = self.left.resolve(point)
        right = self.right.resolve(point)
        if self.operator == "+":
            return left + right
        if self.operator == "-":
            return left - right
        raise ValueError(f"Unsupported operator {self.operator!r}")


def as_expression(value: ValueLike) -> ValueExpression:
    if isinstance(value, ValueExpression):
        return value
    return ConstantExpression(float(value))


@dataclass(frozen=True)
class LinearSweep:
    start: float
    stop: float
    count: int

    def values(self) -> npt.NDArray[np.float64]:
        if self.count < 1:
            raise ValueError("sweep count must be positive")
        return np.linspace(self.start, self.stop, self.count)


@dataclass(frozen=True)
class ValuesSweep:
    data: tuple[Any, ...]

    def __init__(self, values: Any) -> None:
        object.__setattr__(self, "data", tuple(values))
        if not self.data:
            raise ValueError("sweep values cannot be empty")

    def values(self) -> npt.NDArray[Any]:
        return np.asarray(self.data)


@dataclass(frozen=True)
class GeneratorDeclaration:
    name: str
    channel: int
    amplitude_vpp: float


@dataclass(frozen=True)
class ReadoutDeclaration:
    name: str
    adc_channel: str | int
    length_s: float
    demod_frequency_hz: float
    waveform_channel: int | None
    marker_channel: int
    marker_number: int = 1
    marker_length_s: float | None = None
    marker_padding_s: float = 500 * ns
    marker_low_volts: float = 0.0
    marker_high_volts: float = 1.2
    integrate_time_s: float | None = None


@dataclass(frozen=True)
class PulseDefinition:
    name: str
    generator: str
    style: str
    length_s: ValueExpression
    frequency_hz: ValueExpression
    phase_radians: ValueExpression
    gain: ValueExpression
    sigma_s: ValueExpression | None = None
    edge_sigma_s: ValueExpression | None = None
    edge_length_s: ValueExpression | None = None
    decay_s: ValueExpression | None = None
    is_readout: bool = False


@dataclass(frozen=True)
class PlayEvent:
    pulse_name: str
    at_s: ValueExpression | None
    when: tuple[str, Any] | None = None


@dataclass(frozen=True)
class DelayEvent:
    duration_s: ValueExpression


@dataclass(frozen=True)
class TriggerEvent:
    readout_name: str
    trigger_delay_s: ValueExpression | None


ProgramEvent = Union[PlayEvent, DelayEvent, TriggerEvent]


@dataclass(frozen=True)
class ScheduledPulse:
    definition: PulseDefinition
    start_s: float
    stop_s: float
    frequency_hz: float
    phase_radians: float
    gain: float
    sigma_s: float | None
    edge_sigma_s: float | None
    edge_length_s: float | None
    decay_s: float | None


@dataclass(frozen=True)
class ScheduledPoint:
    coordinates: dict[str, Any]
    pulses: tuple[ScheduledPulse, ...]
    trigger_delays_s: dict[str, float]
    duration_s: float


@dataclass
class ExperimentResult:
    """Unaveraged records plus explicit reduction helpers."""

    axes: dict[str, npt.NDArray[Any]]
    point_coordinates: tuple[dict[str, Any], ...]
    raw: npt.NDArray[np.float64]
    iq_traces: npt.NDArray[np.complex128]
    iq_shots: npt.NDArray[np.complex128]
    raw_time_s: npt.NDArray[np.float64]
    iq_time_s: npt.NDArray[np.float64]
    readout_name: str = "ro"
    initial_trigger_delay_s: float | None = None
    readout_windows_s: npt.NDArray[np.float64] | None = None
    marker_windows_s: npt.NDArray[np.float64] | None = None
    acquire_window_s: float | None = None
    remove_dc_offset: bool = False

    def _check_readout(self, readout: str) -> None:
        if readout != self.readout_name:
            raise KeyError(f"Unknown readout {readout!r}")

    def axis(self, name: str) -> npt.NDArray[Any]:
        return self.axes[name].copy()

    def trace_average(self, readout: str = "ro") -> npt.NDArray[np.float64]:
        self._check_readout(readout)
        return np.mean(self.raw, axis=0)

    def iq_trace_average(
        self, readout: str = "ro"
    ) -> npt.NDArray[np.complex128]:
        self._check_readout(readout)
        return np.mean(self.iq_traces, axis=0)

    def shots(self, readout: str = "ro") -> npt.NDArray[np.complex128]:
        self._check_readout(readout)
        return self.iq_shots.copy()

    def iq_average(
        self, readout: str = "ro"
    ) -> npt.NDArray[np.complex128]:
        self._check_readout(readout)
        return np.mean(self.iq_shots, axis=0)


@dataclass
class CompiledExperiment:
    """Rendered sequence plan and record-layout contract."""

    program_name: str
    sample_rate_hz: float
    step_duration_s: float
    axes: dict[str, npt.NDArray[Any]]
    point_coordinates: tuple[dict[str, Any], ...]
    channel_waveforms: dict[int, npt.NDArray[np.float64]]
    marker_waveforms: npt.NDArray[np.bool_]
    readout_windows_s: npt.NDArray[np.float64]
    readout: ReadoutDeclaration
    trigger_delay_s: float
    channel_amplitudes_vpp: dict[int, float]
    remove_dc_offset: bool = False
    _hardware: Any | None = None

    @property
    def number_of_sequence_steps(self) -> int:
        return len(self.point_coordinates)

    def axis(self, name: str) -> npt.NDArray[Any]:
        return self.axes[name].copy()

    def preview(self, channel: int) -> npt.NDArray[np.float64]:
        return self.channel_waveforms[channel].copy()

    def bind(self, hardware: Any) -> "CompiledExperiment":
        self._hardware = hardware
        return self

    def upload(self, hardware: Any | None = None) -> str:
        """Compatibility wrapper delegated to the hardware coordinator."""
        target = hardware or self._hardware
        if target is None:
            raise RuntimeError("Bind an AWGAlazar instance before upload")
        self._hardware = target
        return target.upload_compiled_experiment(self)

    def acquire(
        self,
        n_average: int,
        *,
        hardware: Any | None = None,
        filter_type: str = "boxcar",
    ) -> ExperimentResult:
        """Compatibility wrapper delegated to the hardware coordinator."""
        target = hardware or self._hardware
        if target is None:
            raise RuntimeError("Bind an AWGAlazar instance before acquire")
        self._hardware = target
        return target.acquire_compiled_experiment(
            self,
            n_average=n_average,
            filter_type=filter_type,
        )


class ExperimentProgram:
    """Declarative program compiled into an AWG sequence and ATS record plan."""

    REMOVE_DC_OFFSET = False

    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        name: str | None = None,
        final_delay_s: float = 1 * us,
    ) -> None:
        self.cfg = cfg
        self.name = name or self.__class__.__name__
        self.final_delay_s = float(final_delay_s)
        self.generators: dict[str, GeneratorDeclaration] = {}
        self.readouts: dict[str, ReadoutDeclaration] = {}
        self.sweeps: dict[str, LinearSweep | ValuesSweep] = {}
        self.pulses: dict[str, PulseDefinition] = {}
        self.events: list[ProgramEvent] = []
        self._initialize(cfg)
        self._body(cfg)
        if "ro" not in self.readouts:
            raise ValueError("Programs must declare the 'ro' readout")

    def _initialize(self, cfg: dict[str, Any]) -> None:
        raise NotImplementedError

    def _body(self, cfg: dict[str, Any]) -> None:
        raise NotImplementedError

    def declare_gen(
        self,
        name: str,
        *,
        ch: int,
        amplitude_vpp: float = 0.5,
    ) -> None:
        if name in self.generators:
            raise ValueError(f"Generator {name!r} is already declared")
        self.generators[name] = GeneratorDeclaration(name, ch, amplitude_vpp)

    def declare_readout(
        self,
        name: str = "ro",
        *,
        adc_channel: str | int,
        length: float,
        demod_freq: float,
        waveform_ch: int | None = None,
        marker_channel: int = 1,
        marker_number: int = 1,
        marker_length: float | None = None,
        marker_padding: float = 500 * ns,
        integrate_time: float | None = None,
    ) -> None:
        if name != "ro":
            raise ValueError("The current compiler supports only readout 'ro'")
        if self.readouts:
            raise ValueError("The current compiler supports only one readout")
        if length <= 0:
            raise ValueError("Readout length must be positive")
        if integrate_time is not None and integrate_time <= 0:
            raise ValueError("Readout integrate_time must be positive")
        if integrate_time is not None and integrate_time > length:
            raise ValueError(
                "Readout integrate_time cannot exceed readout length"
            )
        if waveform_ch is None and marker_length is None:
            raise ValueError("Provide waveform_ch or marker_length")
        if waveform_ch is not None and marker_length is not None:
            raise ValueError("Use waveform_ch or marker_length, not both")
        if marker_length is not None and marker_length <= 0:
            raise ValueError("Readout marker_length must be positive")
        if marker_padding < 0:
            raise ValueError("Readout marker_padding cannot be negative")
        self.readouts[name] = ReadoutDeclaration(
            name=name,
            adc_channel=adc_channel,
            length_s=float(length),
            demod_frequency_hz=float(demod_freq),
            waveform_channel=(
                None if waveform_ch is None else int(waveform_ch)
            ),
            marker_channel=marker_channel,
            marker_number=marker_number,
            marker_length_s=(
                None if marker_length is None else float(marker_length)
            ),
            marker_padding_s=float(marker_padding),
            integrate_time_s=(
                None if integrate_time is None else float(integrate_time)
            ),
        )

    def add_sweep(
        self,
        name: str,
        sweep: LinearSweep | ValuesSweep,
    ) -> SweepRef:
        if name in self.sweeps:
            raise ValueError(f"Sweep {name!r} is already declared")
        self.sweeps[name] = sweep
        return SweepRef(name)

    def add_pulse(
        self,
        name: str,
        *,
        gen: str,
        style: str,
        length: ValueLike,
        frequency: ValueLike,
        phase: ValueLike = 0.0,
        gain: ValueLike = 1.0,
        sigma: ValueLike | None = None,
        edge_sigma: ValueLike | None = None,
        edge_length: ValueLike | None = None,
        decay: ValueLike | None = None,
        readout: bool = False,
    ) -> None:
        if gen not in self.generators:
            raise KeyError(f"Generator {gen!r} is not declared")
        pulse_style = style.lower()
        if pulse_style not in {
            "const",
            "gaussian",
            "gaussian_square",
            "cosine_square",
            "exponential",
        }:
            raise ValueError(
                "style must be 'const', 'gaussian', 'gaussian_square', "
                "'cosine_square', or 'exponential'"
            )
        self.pulses[name] = PulseDefinition(
            name=name,
            generator=gen,
            style=pulse_style,
            length_s=as_expression(length),
            frequency_hz=as_expression(frequency),
            phase_radians=as_expression(phase),
            gain=as_expression(gain),
            sigma_s=None if sigma is None else as_expression(sigma),
            edge_sigma_s=(
                None if edge_sigma is None else as_expression(edge_sigma)
            ),
            edge_length_s=(
                None if edge_length is None else as_expression(edge_length)
            ),
            decay_s=None if decay is None else as_expression(decay),
            is_readout=bool(readout),
        )

    def play(
        self,
        pulse_name: str,
        *,
        at: ValueLike | None = None,
        when: tuple[str, Any] | None = None,
    ) -> None:
        if pulse_name not in self.pulses:
            raise KeyError(f"Pulse {pulse_name!r} is not declared")
        if when is not None and when[0] not in self.sweeps:
            raise KeyError(f"Sweep {when[0]!r} is not declared")
        self.events.append(
            PlayEvent(
                pulse_name=pulse_name,
                at_s=None if at is None else as_expression(at),
                when=when,
            )
        )

    def delay_auto(self, duration: ValueLike) -> None:
        self.events.append(DelayEvent(as_expression(duration)))

    def trigger(
        self,
        readout: str = "ro",
        *,
        trigger_delay: ValueLike | None = None,
    ) -> None:
        if readout not in self.readouts:
            raise KeyError(f"Readout {readout!r} is not declared")
        self.events.append(
            TriggerEvent(
                readout_name=readout,
                trigger_delay_s=(
                    None
                    if trigger_delay is None
                    else as_expression(trigger_delay)
                ),
            )
        )

    def compile(
        self,
        *,
        sample_rate_hz: float | None = None,
        hardware: Any | None = None,
    ) -> CompiledExperiment:
        if sample_rate_hz is None:
            if hardware is None:
                raise ValueError(
                    "Provide sample_rate_hz or an AWGAlazar hardware instance"
                )
            sample_rate_hz = float(hardware.awg_sample_rate_hz)
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        points, axes = self._sweep_points()
        scheduled = tuple(self._schedule_point(point) for point in points)
        tagged_starts = [
            pulse.start_s
            for scheduled_point in scheduled
            for pulse in scheduled_point.pulses
            if pulse.definition.is_readout
        ]
        if tagged_starts:
            padding_s = self.readouts["ro"].marker_padding_s
            sequence_shift_s = max(0.0, padding_s - min(tagged_starts))
            if sequence_shift_s:
                scheduled = tuple(
                    self._shift_scheduled_point(point, sequence_shift_s)
                    for point in scheduled
                )
        trigger_delays = {
            point.trigger_delays_s["ro"] for point in scheduled
        }
        if len(trigger_delays) != 1:
            raise ValueError(
                "ATS trigger_delay must be the same for every sequence step"
            )
        trigger_delay_s = trigger_delays.pop()
        step_duration_s = max(point.duration_s for point in scheduled)
        step_samples = max(
            MIN_WAVEFORM_SAMPLES,
            int(np.ceil(step_duration_s * sample_rate_hz)),
        )

        channel_amplitudes = {
            declaration.channel: declaration.amplitude_vpp
            for declaration in self.generators.values()
        }
        channel_waveforms = {
            channel: np.zeros((len(points), step_samples), dtype=np.float64)
            for channel in channel_amplitudes
        }
        readout = self.readouts["ro"]
        integrate_time_s = (
            readout.length_s
            if readout.integrate_time_s is None
            else readout.integrate_time_s
        )
        if (
            hardware is not None
            and hasattr(hardware, "acquire_window_ns")
            and integrate_time_s * 1e9 > hardware.acquire_window_ns
        ):
            raise ValueError(
                "Readout integration window exceeds the hardware "
                "acquisition window"
            )
        if (
            readout.waveform_channel is not None
            and readout.waveform_channel not in channel_waveforms
        ):
            raise ValueError(
                f"Readout waveform_ch {readout.waveform_channel} "
                "is not a declared generator channel"
            )
        markers = np.zeros((len(points), step_samples), dtype=np.bool_)
        readout_windows_s = np.zeros((len(points), 2), dtype=np.float64)

        for point_index, scheduled_point in enumerate(scheduled):
            for pulse in scheduled_point.pulses:
                generator = self.generators[pulse.definition.generator]
                values = self._render_pulse(pulse, sample_rate_hz)
                start = int(round(pulse.start_s * sample_rate_hz))
                stop = start + values.size
                channel_waveforms[generator.channel][
                    point_index, start:stop
                ] += values

            if readout.waveform_channel is None:
                active_marker = np.zeros(step_samples, dtype=np.bool_)
                marker_samples = max(
                    1,
                    int(round(readout.marker_length_s * sample_rate_hz)),
                )
                active_marker[:marker_samples] = True
            else:
                tagged_readout_pulses = [
                    pulse
                    for pulse in scheduled_point.pulses
                    if pulse.definition.is_readout
                ]
                if tagged_readout_pulses:
                    readout_windows_s[point_index] = (
                        min(
                            pulse.start_s
                            for pulse in tagged_readout_pulses
                        ),
                        max(
                            pulse.stop_s
                            for pulse in tagged_readout_pulses
                        ),
                    )
                    marker_start_s = max(
                        0.0,
                        min(
                            pulse.start_s
                            for pulse in tagged_readout_pulses
                        )
                        - readout.marker_padding_s,
                    )
                    marker_stop_s = (
                        max(
                            pulse.stop_s
                            for pulse in tagged_readout_pulses
                        )
                        + readout.marker_padding_s
                    )
                    marker_start = int(
                        round(marker_start_s * sample_rate_hz)
                    )
                    marker_stop = int(
                        round(marker_stop_s * sample_rate_hz)
                    )
                    active_marker = np.zeros(
                        step_samples,
                        dtype=np.bool_,
                    )
                    active_marker[marker_start:marker_stop] = True
                else:
                    reference = channel_waveforms[
                        readout.waveform_channel
                    ][point_index]
                    active = np.flatnonzero(np.abs(reference) > 0)
                    readout_windows_s[point_index] = (
                        active[0] / sample_rate_hz,
                        (active[-1] + 1) / sample_rate_hz,
                    )
                    _, active_marker = trigger_channel_for(
                        reference,
                    )
            if readout.waveform_channel is None:
                readout_windows_s[point_index] = (
                    0.0,
                    readout.length_s,
                )
            markers[point_index] = active_marker

        return CompiledExperiment(
            program_name=self.name,
            sample_rate_hz=sample_rate_hz,
            step_duration_s=step_samples / sample_rate_hz,
            axes=axes,
            point_coordinates=tuple(points),
            channel_waveforms=channel_waveforms,
            marker_waveforms=markers,
            readout_windows_s=readout_windows_s,
            readout=readout,
            trigger_delay_s=trigger_delay_s,
            channel_amplitudes_vpp=channel_amplitudes,
            remove_dc_offset=bool(self.REMOVE_DC_OFFSET),
            _hardware=hardware,
        )

    def _sweep_points(
        self,
    ) -> tuple[list[dict[str, Any]], dict[str, npt.NDArray[Any]]]:
        axes = {name: sweep.values() for name, sweep in self.sweeps.items()}
        if not axes:
            return [{}], {}
        names = tuple(axes)
        points = [
            dict(zip(names, values))
            for values in product(*(axes[name] for name in names))
        ]
        return points, axes

    def _schedule_point(self, point: dict[str, Any]) -> ScheduledPoint:
        cursor_s = 0.0
        pulses: list[ScheduledPulse] = []
        trigger_delays: dict[str, float] = {}
        for event in self.events:
            if isinstance(event, DelayEvent):
                cursor_s += event.duration_s.resolve(point)
                continue
            if isinstance(event, PlayEvent):
                if event.when is not None:
                    sweep_name, expected_value = event.when
                    if point[sweep_name] != expected_value:
                        continue
                definition = self.pulses[event.pulse_name]
                start_s = (
                    cursor_s
                    if event.at_s is None
                    else event.at_s.resolve(point)
                )
                length_s = definition.length_s.resolve(point)
                if start_s < 0 or length_s <= 0:
                    raise ValueError("Pulse start and length must be positive")
                stop_s = start_s + length_s
                pulses.append(
                    ScheduledPulse(
                        definition=definition,
                        start_s=start_s,
                        stop_s=stop_s,
                        frequency_hz=definition.frequency_hz.resolve(point),
                        phase_radians=definition.phase_radians.resolve(point),
                        gain=definition.gain.resolve(point),
                        sigma_s=(
                            None
                            if definition.sigma_s is None
                            else definition.sigma_s.resolve(point)
                        ),
                        edge_sigma_s=(
                            None
                            if definition.edge_sigma_s is None
                            else definition.edge_sigma_s.resolve(point)
                        ),
                        edge_length_s=(
                            None
                            if definition.edge_length_s is None
                            else definition.edge_length_s.resolve(point)
                        ),
                        decay_s=(
                            None
                            if definition.decay_s is None
                            else definition.decay_s.resolve(point)
                        ),
                    )
                )
                if event.at_s is None:
                    cursor_s = stop_s
                continue
            readout = self.readouts[event.readout_name]
            has_tagged_readout = any(
                definition.is_readout
                for definition in self.pulses.values()
            )
            trigger_delay_s = (
                readout.marker_padding_s
                if event.trigger_delay_s is None and has_tagged_readout
                else (
                    0.0
                    if event.trigger_delay_s is None
                    else event.trigger_delay_s.resolve(point)
                )
            )
            if trigger_delay_s < 0:
                raise ValueError("ATS trigger_delay cannot be negative")
            trigger_delays[event.readout_name] = trigger_delay_s

        if "ro" not in trigger_delays:
            raise ValueError("Program body must trigger the 'ro' readout")
        readout = self.readouts["ro"]
        tagged_readout_pulses = [
            pulse for pulse in pulses if pulse.definition.is_readout
        ]
        if tagged_readout_pulses:
            if readout.waveform_channel is None:
                raise ValueError(
                    "Tagged readout pulses require readout waveform_ch"
                )
            tagged_channels = {
                self.generators[pulse.definition.generator].channel
                for pulse in tagged_readout_pulses
            }
            if tagged_channels != {readout.waveform_channel}:
                raise ValueError(
                    "Tagged readout pulses must use the readout waveform_ch"
                )
            readout_start = min(
                pulse.start_s for pulse in tagged_readout_pulses
            )
            readout_stop = readout_start + readout.length_s
            marker_stop = (
                max(pulse.stop_s for pulse in tagged_readout_pulses)
                + readout.marker_padding_s
            )
        elif readout.waveform_channel is None:
            readout_stop = readout.length_s
            marker_stop = float(readout.marker_length_s)
        else:
            reference_generators = {
                name
                for name, declaration in self.generators.items()
                if declaration.channel == readout.waveform_channel
            }
            reference_pulses = [
                pulse
                for pulse in pulses
                if pulse.definition.generator in reference_generators
            ]
            if not reference_pulses:
                raise ValueError(
                    f"Readout waveform_ch {readout.waveform_channel} "
                    "has no active pulse"
                )
            readout_stop = (
                min(pulse.start_s for pulse in reference_pulses)
                + readout.length_s
            )
            marker_stop = max(pulse.stop_s for pulse in reference_pulses)
        duration_s = max(
            [
                cursor_s,
                *[pulse.stop_s for pulse in pulses],
                readout_stop,
                marker_stop,
            ]
        ) + self.final_delay_s
        return ScheduledPoint(
            coordinates=point.copy(),
            pulses=tuple(pulses),
            trigger_delays_s=trigger_delays,
            duration_s=duration_s,
        )

    @staticmethod
    def _shift_scheduled_point(
        point: ScheduledPoint,
        shift_s: float,
    ) -> ScheduledPoint:
        shifted_pulses = tuple(
            ScheduledPulse(
                definition=pulse.definition,
                start_s=pulse.start_s + shift_s,
                stop_s=pulse.stop_s + shift_s,
                frequency_hz=pulse.frequency_hz,
                phase_radians=pulse.phase_radians,
                gain=pulse.gain,
                sigma_s=pulse.sigma_s,
                edge_sigma_s=pulse.edge_sigma_s,
                edge_length_s=pulse.edge_length_s,
                decay_s=pulse.decay_s,
            )
            for pulse in point.pulses
        )
        return ScheduledPoint(
            coordinates=point.coordinates.copy(),
            pulses=shifted_pulses,
            trigger_delays_s=point.trigger_delays_s.copy(),
            duration_s=point.duration_s + shift_s,
        )

    @staticmethod
    def _render_pulse(
        pulse: ScheduledPulse,
        sample_rate_hz: float,
    ) -> npt.NDArray[np.float64]:
        count = max(
            1,
            int(round((pulse.stop_s - pulse.start_s) * sample_rate_hz)),
        )
        style = pulse.definition.style
        if style == "const":
            envelope = constant(count, pulse.gain)
        elif style == "gaussian":
            sigma_s = pulse.sigma_s
            if sigma_s is None or sigma_s <= 0:
                raise ValueError("Gaussian pulses require sigma > 0")
            envelope = gaussian(
                count,
                sample_rate_hz,
                sigma_s,
                pulse.gain,
            )
        elif style == "gaussian_square":
            sigma_s = pulse.edge_sigma_s
            if sigma_s is None or sigma_s <= 0:
                raise ValueError(
                    "Gaussian-square pulses require edge_sigma > 0"
                )
            try:
                envelope = gaussian_square(
                    count,
                    sample_rate_hz,
                    sigma_s,
                    pulse.gain,
                )
            except ValueError as exc:
                raise ValueError(
                    "Gaussian-square pulse is too short for edge_sigma"
                ) from exc
        elif style == "cosine_square":
            edge_length_s = pulse.edge_length_s
            if edge_length_s is None or edge_length_s <= 0:
                raise ValueError(
                    "Cosine-square pulses require edge_length > 0"
                )
            try:
                envelope = cosine_square(
                    count,
                    sample_rate_hz,
                    edge_length_s,
                    pulse.gain,
                )
            except ValueError as exc:
                raise ValueError(
                    "Cosine-square pulse is too short for edge_length"
                ) from exc
        else:
            decay_s = pulse.decay_s
            if decay_s is None or decay_s <= 0:
                raise ValueError(
                    "Exponential pulses require decay > 0"
                )
            time_s = np.arange(count, dtype=np.float64) / sample_rate_hz
            envelope = pulse.gain * np.exp(-time_s / (2.0 * decay_s))

        return modulate_envelope(
            envelope,
            sample_rate_hz,
            pulse.frequency_hz,
            pulse.phase_radians,
        )
