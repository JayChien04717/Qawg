"""Hardware-facing waveform and sequence plans for the AWG5208."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy.typing as npt


@dataclass(frozen=True)
class AWGStep:
    """Waveforms and marker bits emitted together for one logical step."""

    waveforms: Mapping[int, npt.ArrayLike]
    markers: Mapping[int, tuple[npt.ArrayLike, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class AWGProgram:
    """A hardware-ready program; one step is a waveform, many are a sequence."""

    name: str
    steps: tuple[AWGStep, ...]
    amplitudes_vpp: Mapping[int, float]
    marker_levels: Mapping[tuple[int, int], tuple[float, float]] = field(
        default_factory=dict
    )
    repetitions: int | tuple[int, ...] = 1
    goto_step: int | None = 1


@dataclass(frozen=True)
class UploadedProgram:
    """Identity of the program currently materialized on the AWG."""

    name: str
    is_sequence: bool
    waveform_names: Mapping[int, tuple[str, ...]]

