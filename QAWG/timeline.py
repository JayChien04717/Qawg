"""Compatibility exports for timeline helpers.

The implementation lives in :mod:`QAWG.awg5200.timeline` so the AWG package
owns waveform and timeline scheduling without depending on the parent package.
"""

from .awg5200.timeline import (
    Delay,
    Parallel,
    Timeline,
    Waveform,
    align_channel_envelopes,
    align_channels,
    channel_names,
    delay,
    delay_auto,
    parallel,
    waveform,
)

__all__ = [
    "Delay",
    "Parallel",
    "Timeline",
    "Waveform",
    "align_channel_envelopes",
    "align_channels",
    "channel_names",
    "delay",
    "delay_auto",
    "parallel",
    "waveform",
]
