"""Functional cross-channel waveform timing for QAWG programs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from .awg5200.waveforms import (
    MIN_WAVEFORM_SAMPLES,
    FloatArray,
    modulate_envelope,
)


@dataclass(frozen=True)
class Waveform:
    envelope: FloatArray
    fc: float
    ch: int
    phase_radians: float = 0.0
    name: str | None = None
    gain: float = 1.0

    def __truediv__(self, other: object) -> "Timeline":
        return Timeline((self,)) / other


@dataclass(frozen=True)
class Parallel:
    waveforms: tuple[Waveform, ...]

    def __truediv__(self, other: object) -> "Timeline":
        return Timeline((self,)) / other


@dataclass(frozen=True)
class Delay:
    duration_s: float
    reference: Literal["start", "end"]

    def __truediv__(self, other: object) -> "Timeline":
        return Timeline((self,)) / other


@dataclass(frozen=True)
class Timeline:
    items: tuple[Waveform | Parallel | Delay, ...]

    def __truediv__(self, other: object) -> "Timeline":
        if not isinstance(other, (Waveform, Parallel, Delay)):
            return NotImplemented
        return Timeline(self.items + (other,))


def waveform(
    waveform_array: npt.ArrayLike,
    fc: float,
    ch: int,
    phase_radians: float = 0.0,
    name: str | None = None,
    gain: float = 1.0,
) -> Waveform:
    envelope = np.asarray(waveform_array, dtype=np.float64).reshape(-1) * gain
    if envelope.size < 1:
        raise ValueError("waveform_array cannot be empty")
    if not np.all(np.isfinite(envelope)):
        raise ValueError("waveform_array must contain only finite values")
    if not 1 <= ch <= 8:
        raise ValueError("ch must be between 1 and 8")
    if fc < 0:
        raise ValueError("fc cannot be negative")
    return Waveform(envelope.copy(), fc, ch, phase_radians, name, gain)


def parallel(*waveforms: Waveform) -> Parallel:
    if not waveforms:
        raise ValueError("parallel() requires at least one waveform")
    return Parallel(tuple(waveforms))


def delay(duration_s: float) -> Delay:
    """Start the next waveform this long after the previous waveform starts."""
    if duration_s < 0:
        raise ValueError("delay duration cannot be negative")
    return Delay(duration_s, "start")


def delay_auto(duration_s: float) -> Delay:
    """Start the next waveform this long after the previous waveform ends."""
    if duration_s < 0:
        raise ValueError("delay duration cannot be negative")
    return Delay(duration_s, "end")


def _scheduled_waveforms(
    timeline: Timeline,
    sample_rate_hz: float,
) -> list[tuple[Waveform, int]]:
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if not timeline.items:
        raise ValueError("timeline cannot be empty")

    scheduled: list[tuple[Waveform, int]] = []
    previous_start = 0
    previous_stop = 0
    pending_delay: Delay | None = None
    leading_offset = 0

    for item in timeline.items:
        if isinstance(item, Delay):
            if pending_delay is not None:
                raise ValueError("each delay must be between two waveforms")
            if not scheduled:
                leading_offset = int(round(item.duration_s * sample_rate_hz))
                pending_delay = item
                continue
            pending_delay = item
            continue

        group = item.waveforms if isinstance(item, Parallel) else (item,)
        if not scheduled:
            start = leading_offset
            pending_delay = None
        elif pending_delay is None:
            raise ValueError("waveforms must be separated by delay() or delay_auto()")
        else:
            offset = int(round(pending_delay.duration_s * sample_rate_hz))
            anchor = previous_start if pending_delay.reference == "start" else previous_stop
            start = anchor + offset

        group_stop = start
        for pulse in group:
            stop = start + pulse.envelope.size
            scheduled.append((pulse, start))
            group_stop = max(group_stop, stop)
        previous_start = start
        previous_stop = group_stop
        pending_delay = None

    if pending_delay is not None:
        raise ValueError("timeline cannot end with a delay")
    return scheduled


def _render_channels(
    timeline: Timeline | Waveform | Parallel,
    sample_rate_hz: float,
    total_duration_s: float,
    minimum_samples: int,
    modulate: bool,
    phase_reference: str = "global",
) -> dict[int, FloatArray]:
    if phase_reference not in {"global", "pulse"}:
        raise ValueError("phase_reference must be 'global' or 'pulse'")
    if total_duration_s <= 0:
        raise ValueError("total_duration_s must be positive")
    if isinstance(timeline, (Waveform, Parallel)):
        timeline = Timeline((timeline,))
    scheduled = _scheduled_waveforms(timeline, sample_rate_hz)
    for pulse, _ in scheduled:
        if pulse.fc > sample_rate_hz / 2:
            raise ValueError("waveform fc exceeds the Nyquist frequency")

    total_samples = max(
        minimum_samples,
        int(round(total_duration_s * sample_rate_hz)),
        max(start + pulse.envelope.size for pulse, start in scheduled),
    )
    channels = {
        channel: np.zeros(total_samples, dtype=np.float64)
        for channel in sorted({pulse.ch for pulse, _ in scheduled})
    }

    for pulse, start in scheduled:
        values = (
            modulate_envelope(
                pulse.envelope,
                sample_rate_hz,
                pulse.fc,
                pulse.phase_radians,
                start / sample_rate_hz if phase_reference == "global" else 0.0,
            )
            if modulate
            else pulse.envelope
        )
        channels[pulse.ch][start : start + pulse.envelope.size] += values
    return channels


def align_channels(
    timeline: Timeline | Waveform | Parallel,
    sample_rate_hz: float,
    total_duration_s: float = 5e-6,
    minimum_samples: int = MIN_WAVEFORM_SAMPLES,
    phase_reference: str = "global",
) -> dict[int, FloatArray]:
    """Render modulated channels onto one sample-aligned time axis.

    ``phase_reference="global"`` keeps each carrier phase-continuous across
    pulses (phase measured from sample 0); ``"pulse"`` restarts the carrier
    at every pulse start.
    """
    return _render_channels(
        timeline,
        sample_rate_hz,
        total_duration_s,
        minimum_samples,
        modulate=True,
        phase_reference=phase_reference,
    )


def align_channel_envelopes(
    timeline: Timeline | Waveform | Parallel,
    sample_rate_hz: float,
    total_duration_s: float = 5e-6,
    minimum_samples: int = MIN_WAVEFORM_SAMPLES,
) -> dict[int, FloatArray]:
    """Render unmodulated envelopes on the waveform time axis."""
    return _render_channels(
        timeline,
        sample_rate_hz,
        total_duration_s,
        minimum_samples,
        modulate=False,
    )


def channel_names(timeline: Timeline | Waveform | Parallel) -> dict[int, str]:
    """Return explicit names for channels that have one unambiguous name."""
    if isinstance(timeline, (Waveform, Parallel)):
        timeline = Timeline((timeline,))
    names: dict[int, set[str]] = {}
    for item in timeline.items:
        group = item.waveforms if isinstance(item, Parallel) else (item,)
        for pulse in group:
            if isinstance(pulse, Waveform) and pulse.name:
                names.setdefault(pulse.ch, set()).add(pulse.name)
    return {
        channel: next(iter(channel_names))
        for channel, channel_names in names.items()
        if len(channel_names) == 1
    }
