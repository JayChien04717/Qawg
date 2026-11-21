"""Algebraic and temporal waveform builder (ported and modernized from QuantumCompilerV2).

All shapes follow the QAWG time-axis convention:
``N = int(round(span * sample_rate))`` samples on ``t = np.arange(N) / sample_rate``.
Because :class:`Shape` implements ``__array__``, instances can be passed directly
to :func:`QAWG.awg5200.timeline.waveform`, :func:`modulate_envelope`, or
:meth:`QAWG.compiler.ExperimentProgram.add_pulse` (with ``style="custom"``).
"""

from __future__ import annotations

from collections.abc import Callable
from numbers import Real
from sys import float_info
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from .waveforms import (
    FloatArray,
    modulate_envelope,
    sample_count,
    time_axis,
)

ShapeFunc = Callable[..., npt.ArrayLike]
AlignMode = Literal["head_to_head", "tail_to_tail", "tail_to_head", "head_to_tail"]


# ---------------------------------------------------------------------------
# Built-in unit-amplitude shape primitives
# Ported from QuantumCompilerV2 (function.py)
# ---------------------------------------------------------------------------


def _fn_const(x: FloatArray, lv: float = 0.0, level: float | None = None) -> FloatArray:
    val = lv if level is None else level
    return np.full(x.size, float(val), dtype=np.float64)


def _fn_gaussian(
    x: FloatArray,
    peak_x: float | None = None,
    sigma: float = 10e-9,
    center: float | None = None,
) -> FloatArray:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    mu = (x[-1] / 2.0 if x.size else 0.0) if (peak_x is None and center is None) else (
        center if peak_x is None else peak_x
    )
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _fn_exp_rising(x: FloatArray, peak_x: float, tau: float) -> FloatArray:
    if tau <= 0:
        raise ValueError("tau must be positive")
    out = np.zeros(x.size, dtype=np.float64)
    mask = x <= peak_x
    out[mask] = np.exp((x[mask] - peak_x) / tau)
    return out


def _fn_exp_falling(x: FloatArray, peak_x: float, tau: float) -> FloatArray:
    if tau <= 0:
        raise ValueError("tau must be positive")
    out = np.zeros(x.size, dtype=np.float64)
    mask = x >= peak_x
    out[mask] = np.exp((peak_x - x[mask]) / tau)
    return out


def _fn_exp_rise(x: FloatArray, tau: float) -> FloatArray:
    if tau <= 0:
        raise ValueError("tau must be positive")
    return 1.0 - np.exp(-x / tau)


def _fn_exp_decay(x: FloatArray, tau: float) -> FloatArray:
    if tau <= 0:
        raise ValueError("tau must be positive")
    return np.exp(-x / tau)


def _fn_gaussian_square(
    x: FloatArray,
    first_peak_x: float,
    flat: float,
    sigma: float,
) -> FloatArray:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if flat < 0:
        raise ValueError("flat cannot be negative")
    second_peak_x = first_peak_x + flat
    out = np.ones(x.size, dtype=np.float64)
    rising = x < first_peak_x
    falling = x > second_peak_x
    out[rising] = np.exp(-0.5 * ((x[rising] - first_peak_x) / sigma) ** 2)
    out[falling] = np.exp(-0.5 * ((x[falling] - second_peak_x) / sigma) ** 2)
    return out


def _fn_cosine_square(
    x: FloatArray,
    edge_length: float,
) -> FloatArray:
    if edge_length <= 0:
        raise ValueError("edge_length must be positive")
    if x.size == 0:
        return np.zeros(0, dtype=np.float64)
    span = x[-1] - x[0]
    if 2.0 * edge_length > span + 1e-15:
        raise ValueError("span is too short for both cosine edges")
    out = np.ones(x.size, dtype=np.float64)
    rising = x < edge_length
    falling = x > span - edge_length
    out[rising] = 0.5 * (1.0 - np.cos(np.pi * x[rising] / edge_length))
    out[falling] = 0.5 * (1.0 - np.cos(np.pi * (span - x[falling]) / edge_length))
    return out


def _fn_exp_square(
    x: FloatArray,
    first_peak_x: float,
    flat: float,
    tau: float,
) -> FloatArray:
    if tau <= 0:
        raise ValueError("tau must be positive")
    if flat < 0:
        raise ValueError("flat cannot be negative")
    second_peak_x = first_peak_x + flat
    out = np.ones(x.size, dtype=np.float64)
    rising = x < first_peak_x
    falling = x > second_peak_x
    out[rising] = np.exp((x[rising] - first_peak_x) / tau)
    out[falling] = np.exp((second_peak_x - x[falling]) / tau)
    return out


def _fn_sine(x: FloatArray, period: float, start_phase: float = 0.0) -> FloatArray:
    if period <= 0:
        raise ValueError("period must be positive")
    return np.sin(2.0 * np.pi * (x / period) + start_phase)


def _fn_sine2(x: FloatArray, frequency: float, start_phase: float = 0.0) -> FloatArray:
    return np.sin(2.0 * np.pi * frequency * x + start_phase)


def _fn_cosine(x: FloatArray, period: float, start_phase: float = 0.0) -> FloatArray:
    if period <= 0:
        raise ValueError("period must be positive")
    return np.cos(2.0 * np.pi * (x / period) + start_phase)


def _fn_cosine2(x: FloatArray, frequency: float, start_phase: float = 0.0) -> FloatArray:
    return np.cos(2.0 * np.pi * frequency * x + start_phase)


def _fn_square(x: FloatArray, start: float, flat: float) -> FloatArray:
    if flat < 0:
        raise ValueError("flat cannot be negative")
    return np.where((x >= start) & (x <= start + flat), 1.0, 0.0).astype(np.float64)


def _fn_triangle(
    x: FloatArray,
    half_width: float,
    center: float | None = None,
) -> FloatArray:
    if half_width <= 0:
        raise ValueError("half_width must be positive")
    mu = (x[-1] / 2.0 if x.size else 0.0) if center is None else center
    return np.maximum(0.0, 1.0 - np.abs(x - mu) / half_width)


def _fn_pwm(
    x: FloatArray,
    period: float,
    duty_cycle: float = 0.5,
    delay: float = 0.0,
    onval: float = 1.0,
    offval: float = 0.0,
) -> FloatArray:
    if period <= 0:
        raise ValueError("period must be positive")
    if not 0.0 <= duty_cycle <= 1.0:
        raise ValueError("duty_cycle must be between 0 and 1")
    return np.where(
        np.mod(x - delay, period) <= duty_cycle * period, onval, offval
    ).astype(np.float64)


def _fn_pwm2(
    x: FloatArray,
    period: float,
    on_span: float,
    delay: float = 0.0,
    onval: float = 1.0,
    offval: float = 0.0,
) -> FloatArray:
    if period <= 0:
        raise ValueError("period must be positive")
    if on_span < 0 or on_span > period:
        raise ValueError("on_span must be between 0 and period")
    return np.where(np.mod(x - delay, period) <= on_span, onval, offval).astype(
        np.float64
    )


SHAPE_REGISTRY: dict[str, ShapeFunc] = {
    "const": _fn_const,
    "gaussian": _fn_gaussian,
    "exp_rising": _fn_exp_rising,
    "exp_falling": _fn_exp_falling,
    "exp_rise": _fn_exp_rise,
    "exp_decay": _fn_exp_decay,
    "gaussian_square": _fn_gaussian_square,
    "cosine_square": _fn_cosine_square,
    "exp_square": _fn_exp_square,
    "sine": _fn_sine,
    "sine2": _fn_sine2,
    "cosine": _fn_cosine,
    "cosine2": _fn_cosine2,
    "square": _fn_square,
    "triangle": _fn_triangle,
    "pwm": _fn_pwm,
    "pwm2": _fn_pwm2,
}


def register_shape(
    name: str,
    func: ShapeFunc | None = None,
) -> Callable[[ShapeFunc], ShapeFunc] | ShapeFunc:
    """Register a custom ``func(x, **kwargs)`` shape function by name.

    Can be called directly ``register_shape("my_fn", my_fn)`` or used as a
    decorator ``@register_shape("my_fn")``.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("shape name must be a non-empty string")

    def _decorator(fn: ShapeFunc) -> ShapeFunc:
        if not callable(fn):
            raise TypeError("registered shape must be callable")
        SHAPE_REGISTRY[name] = fn
        return fn

    if func is not None:
        return _decorator(func)
    return _decorator


# ---------------------------------------------------------------------------
# Shape class (algebraic & temporal waveform composition, FFT, plotting)
# Ported from QuantumCompilerV2 (waveform.py: GenericWave, Shape, Waveform)
# ---------------------------------------------------------------------------


def _offset_to_samples(offset: int | float, sample_rate_hz: float) -> int:
    if isinstance(offset, (int, np.integer)) and not isinstance(offset, bool):
        return int(offset)
    return int(round(float(offset) * sample_rate_hz))


class Shape:
    """Composable waveform envelope supporting algebraic and temporal operations.

    Examples
    --------
    >>> g = Shape("gaussian", {"peak_x": 50e-9, "sigma": 10e-9}, span=100e-9, sample_rate=1e9)
    >>> drag_q = 0.2 * g.diff()
    >>> seq = (0.3 * g) | (0.15 * g >> 20e-9)
    """

    def __init__(
        self,
        func: str | Callable[[FloatArray], npt.ArrayLike],
        args: dict[str, Any] | None = None,
        span: float = 1e-6,
        sample_rate: float | None = None,
        name: str = "",
        *,
        sampling_rate: float | None = None,
    ) -> None:
        if sample_rate is not None and sampling_rate is not None:
            raise ValueError("Specify only one of sample_rate or sampling_rate")
        rate = 1e9 if (sample_rate is None and sampling_rate is None) else float(
            sample_rate if sample_rate is not None else sampling_rate
        )
        if rate <= 0:
            raise ValueError("sample_rate must be positive")

        n_samples = sample_count(span, rate)
        x = time_axis(n_samples, rate)

        if callable(func):
            raw = func(x, **(args or {})) if args else func(x)
        elif isinstance(func, str):
            if func not in SHAPE_REGISTRY:
                known = ", ".join(sorted(SHAPE_REGISTRY))
                raise KeyError(f"Unknown shape {func!r}. Available: {known}")
            raw = SHAPE_REGISTRY[func](x, **(args or {}))
        else:
            raise TypeError("func must be a registered shape name or callable")

        values = np.asarray(raw, dtype=np.float64).reshape(-1)
        if values.size != n_samples:
            raise ValueError(
                f"Shape function returned {values.size} samples, expected {n_samples}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("Shape values must be finite")

        self._values: FloatArray = values.copy()
        self._sample_rate: float = rate
        self.name: str = str(name)

    @classmethod
    def from_array(
        cls,
        values: npt.ArrayLike,
        sample_rate: float = 1e9,
        name: str = "",
    ) -> "Shape":
        """Wrap an existing 1-D array as a :class:`Shape`."""
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size < 1:
            raise ValueError("values cannot be empty")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not np.all(np.isfinite(arr)):
            raise ValueError("values must be finite")
        obj = cls.__new__(cls)
        obj._values = arr.copy()
        obj._sample_rate = float(sample_rate)
        obj.name = str(name)
        return obj

    # -- Properties ---------------------------------------------------------

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    @property
    def df(self) -> float:
        """Alias for ``sample_rate`` (V2 compatibility)."""
        return self._sample_rate

    @property
    def dt(self) -> float:
        return 1.0 / self._sample_rate

    @property
    def dx(self) -> float:
        """Alias for ``dt`` (V2 compatibility)."""
        return self.dt

    @property
    def n_samples(self) -> int:
        return int(self._values.size)

    @property
    def point_num(self) -> int:
        """Alias for ``n_samples`` (V2 compatibility)."""
        return self.n_samples

    @property
    def span(self) -> float:
        return self._values.size / self._sample_rate

    @property
    def x(self) -> FloatArray:
        return time_axis(self._values.size, self._sample_rate)

    @property
    def t(self) -> FloatArray:
        return self.x

    @property
    def y(self) -> FloatArray:
        return self._values.copy()

    @property
    def values(self) -> FloatArray:
        return self._values.copy()

    @property
    def output(self) -> FloatArray:
        """Alias for ``values`` (QAWG uses standard N-sample convention)."""
        return self._values.copy()

    @property
    def f(self) -> FloatArray:
        return np.fft.fftfreq(self._values.size, d=self.dt)

    @property
    def yf(self) -> npt.NDArray[np.complex128]:
        return np.fft.fft(self._values)

    def __len__(self) -> int:
        return int(self._values.size)

    def __array__(
        self,
        dtype: npt.DTypeLike | None = None,
        copy: bool | None = None,
    ) -> np.ndarray:
        arr = self._values if copy is False else self._values.copy()
        if dtype is not None:
            return arr.astype(dtype, copy=False)
        return arr

    def __repr__(self) -> str:
        return (
            f"Shape(name={self.name!r}, n_samples={self.n_samples}, "
            f"span={self.span:.6g}s, sample_rate={self._sample_rate:.6g}Hz)"
        )

    # -- Algebraic operations -----------------------------------------------

    def _align_binary_arrays(
        self, other: "Shape | npt.ArrayLike"
    ) -> tuple[FloatArray, FloatArray]:
        if isinstance(other, Shape):
            if not np.isclose(self._sample_rate, other._sample_rate, rtol=1e-9):
                raise ValueError(
                    "Cannot combine Shapes with different sample rates "
                    f"({self._sample_rate} vs {other._sample_rate})"
                )
            b = other._values
        else:
            b = np.asarray(other, dtype=np.float64).reshape(-1)
        a = self._values
        if a.size == b.size:
            return a, b
        n = max(a.size, b.size)
        a_pad = np.zeros(n, dtype=np.float64)
        b_pad = np.zeros(n, dtype=np.float64)
        a_pad[: a.size] = a
        b_pad[: b.size] = b
        return a_pad, b_pad

    def __neg__(self) -> "Shape":
        return Shape.from_array(-self._values, self._sample_rate, self.name)

    def __pos__(self) -> "Shape":
        return Shape.from_array(self._values.copy(), self._sample_rate, self.name)

    def __abs__(self) -> "Shape":
        return Shape.from_array(np.abs(self._values), self._sample_rate, self.name)

    def __add__(self, other: Any) -> "Shape":
        if isinstance(other, Real):
            return Shape.from_array(
                self._values + float(other), self._sample_rate, self.name
            )
        a, b = self._align_binary_arrays(other)
        return Shape.from_array(a + b, self._sample_rate, self.name)

    def __radd__(self, other: Any) -> "Shape":
        return self + other

    def __sub__(self, other: Any) -> "Shape":
        if isinstance(other, Real):
            return Shape.from_array(
                self._values - float(other), self._sample_rate, self.name
            )
        a, b = self._align_binary_arrays(other)
        return Shape.from_array(a - b, self._sample_rate, self.name)

    def __rsub__(self, other: Any) -> "Shape":
        if isinstance(other, Real):
            return Shape.from_array(
                float(other) - self._values, self._sample_rate, self.name
            )
        a, b = self._align_binary_arrays(other)
        return Shape.from_array(b - a, self._sample_rate, self.name)

    def __mul__(self, other: Any) -> "Shape":
        if isinstance(other, Real):
            return Shape.from_array(
                self._values * float(other), self._sample_rate, self.name
            )
        a, b = self._align_binary_arrays(other)
        return Shape.from_array(a * b, self._sample_rate, self.name)

    def __rmul__(self, other: Any) -> "Shape":
        return self * other

    def __truediv__(self, other: Any) -> "Shape":
        if isinstance(other, Real):
            return Shape.from_array(
                self._values / float(other), self._sample_rate, self.name
            )
        raise TypeError("Divisor must be a real number")

    def __floordiv__(self, other: Any) -> "Shape":
        if isinstance(other, Real):
            return Shape.from_array(
                self._values // float(other), self._sample_rate, self.name
            )
        raise TypeError("Divisor must be a real number")

    def __pow__(self, other: Any) -> "Shape":
        if isinstance(other, Real):
            return Shape.from_array(
                self._values ** float(other), self._sample_rate, self.name
            )
        raise TypeError("Exponent must be a real number")

    def __mod__(self, other: Any) -> "Shape":
        if isinstance(other, Real):
            return Shape.from_array(
                self._values % float(other), self._sample_rate, self.name
            )
        raise TypeError("Divisor must be a real number")

    # -- Signal transforms --------------------------------------------------

    def reverse(self) -> "Shape":
        """Return a time-reversed copy of the shape."""
        return Shape.from_array(self._values[::-1], self._sample_rate, self.name)

    def roll(self, offset: int | float = 0.0) -> "Shape":
        """Circularly shift samples by ``offset`` (int samples or float seconds)."""
        shift = _offset_to_samples(offset, self._sample_rate)
        return Shape.from_array(
            np.roll(self._values, shift), self._sample_rate, self.name
        )

    def diff(self, n: int = 1) -> "Shape":
        """Return the ``n``-th numerical time derivative ``d^n y / dt^n`` (useful for DRAG)."""
        if n < 1:
            raise ValueError("Derivative order n must be >= 1")
        if self._values.size < 2:
            raise ValueError("Cannot differentiate a shape with fewer than 2 samples")
        out = self._values.copy()
        for _ in range(n):
            out = np.gradient(out, self.dt)
        return Shape.from_array(out, self._sample_rate, self.name)

    def scale(self, factor: float) -> "Shape":
        """Multiply amplitude by ``factor``."""
        return self * float(factor)

    def normalize(self, peak: float = 1.0) -> "Shape":
        """Scale amplitude so ``max(|y|) == peak``."""
        max_abs = float(np.max(np.abs(self._values)))
        if max_abs == 0.0:
            return Shape.from_array(
                np.zeros_like(self._values), self._sample_rate, self.name
            )
        return Shape.from_array(
            self._values * (float(peak) / max_abs), self._sample_rate, self.name
        )

    def modulate(
        self,
        frequency_hz: float,
        phase_radians: float = 0.0,
        start_time_s: float = 0.0,
    ) -> "Shape":
        """Apply a sine carrier to this shape envelope."""
        modulated = modulate_envelope(
            self._values,
            self._sample_rate,
            frequency_hz,
            phase_radians=phase_radians,
            start_time_s=start_time_s,
        )
        return Shape.from_array(modulated, self._sample_rate, self.name)

    # -- Temporal composition (immutable) -----------------------------------

    def concat(self, *others: "Shape | npt.ArrayLike") -> "Shape":
        """Concatenate this shape with one or more shapes/arrays in time."""
        parts: list[FloatArray] = [self._values]
        for other in others:
            if isinstance(other, Shape):
                if not np.isclose(self._sample_rate, other._sample_rate, rtol=1e-9):
                    raise ValueError(
                        "Cannot concatenate Shapes with different sample rates"
                    )
                parts.append(other._values)
            else:
                parts.append(np.asarray(other, dtype=np.float64).reshape(-1))
        return Shape.from_array(np.concatenate(parts), self._sample_rate, self.name)

    def __or__(self, other: "Shape | npt.ArrayLike") -> "Shape":
        """Temporal concatenation operator: ``shape_a | shape_b``."""
        return self.concat(other)

    def repeat(self, count: int) -> "Shape":
        """Repeat this shape ``count`` times back-to-back in time."""
        if not isinstance(count, (int, np.integer)) or count < 1:
            raise ValueError("repeat count must be a positive integer")
        return Shape.from_array(
            np.tile(self._values, int(count)), self._sample_rate, self.name
        )

    def pad_const(
        self,
        pad_width: tuple[int | float, int | float] = (0, 0),
        pad_value: tuple[float, float] = (0.0, 0.0),
    ) -> "Shape":
        """Pad constants on the ``(left, right)`` sides (int samples or float seconds)."""
        left_n = _offset_to_samples(pad_width[0], self._sample_rate)
        right_n = _offset_to_samples(pad_width[1], self._sample_rate)
        if left_n < 0 or right_n < 0:
            raise ValueError("pad_width values cannot be negative")
        parts: list[FloatArray] = []
        if left_n > 0:
            parts.append(np.full(left_n, float(pad_value[0]), dtype=np.float64))
        parts.append(self._values)
        if right_n > 0:
            parts.append(np.full(right_n, float(pad_value[1]), dtype=np.float64))
        return Shape.from_array(np.concatenate(parts), self._sample_rate, self.name)

    def __rshift__(self, offset: int | float) -> "Shape":
        """Delay by prepending zeros: ``shape >> 50e-9`` or ``shape >> 50``."""
        return self.pad_const((offset, 0), (0.0, 0.0))

    def __lshift__(self, offset: int | float) -> "Shape":
        """Extend tail by appending zeros: ``shape << 50e-9`` or ``shape << 50``."""
        return self.pad_const((0, offset), (0.0, 0.0))

    def align_to(
        self,
        other: "Shape",
        mode: AlignMode = "head_to_head",
    ) -> tuple["Shape", "Shape"]:
        """Return two equal-length Shapes aligned without mutating either operand.

        Modes
        -----
        - ``"head_to_head"``: both start at t=0, shorter is right-padded.
        - ``"tail_to_tail"``: both end together, shorter is left-padded.
        - ``"tail_to_head"``: ``self`` finishes where ``other`` starts.
        - ``"head_to_tail"``: ``self`` starts where ``other`` finishes.
        """
        if not np.isclose(self._sample_rate, other._sample_rate, rtol=1e-9):
            raise ValueError("Cannot align Shapes with different sample rates")
        if mode == "head_to_head":
            diff = abs(self.n_samples - other.n_samples)
            if self.n_samples < other.n_samples:
                return self << diff, Shape.from_array(other._values, other._sample_rate, other.name)
            return Shape.from_array(self._values, self._sample_rate, self.name), other << diff
        if mode == "tail_to_tail":
            diff = abs(self.n_samples - other.n_samples)
            if self.n_samples < other.n_samples:
                return self >> diff, Shape.from_array(other._values, other._sample_rate, other.name)
            return Shape.from_array(self._values, self._sample_rate, self.name), other >> diff
        if mode == "tail_to_head":
            return self << other.n_samples, other >> self.n_samples
        if mode == "head_to_tail":
            return self >> other.n_samples, other << self.n_samples
        raise ValueError(
            "mode must be 'head_to_head', 'tail_to_tail', 'tail_to_head', or 'head_to_tail'"
        )

    # -- Bridge to QAWG Timeline --------------------------------------------

    def to_waveform(
        self,
        fc: float = 0.0,
        ch: int = 1,
        phase_radians: float = 0.0,
        name: str | None = None,
        gain: float = 1.0,
    ) -> Any:
        """Convert this shape into a :class:`QAWG.awg5200.timeline.Waveform`."""
        from .timeline import waveform

        return waveform(
            self._values,
            fc=fc,
            ch=ch,
            phase_radians=phase_radians,
            name=name if name is not None else (self.name or None),
            gain=gain,
        )

    # -- Visualization ------------------------------------------------------

    def plot(
        self,
        *,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        show: bool = False,
        ax: Any = None,
        mode: str = "lines",
        title: str | None = None,
    ) -> Any:
        """Plot the shape in the time domain using matplotlib or plotly."""
        label = self.name or "Shape"
        if backend == "matplotlib":
            from matplotlib import pyplot as plt

            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.figure
            ax.plot(self.x * 1e9, self._values, label=label)
            ax.set_xlabel("Time (ns)")
            ax.set_ylabel("Amplitude")
            if title is not None:
                ax.set_title(title)
            elif self.name:
                ax.set_title(self.name)
            ax.grid(True, alpha=0.3)
            ax.legend()
            if show:
                plt.show()
            return fig, ax
        if backend == "plotly":
            from plotly.graph_objects import Figure, Scatter

            trace = Scatter(
                x=self.x,
                y=self._values,
                name=label,
                mode=mode,
            )
            fig = Figure(
                [trace],
                {
                    "xaxis": {"title": "Time (s)"},
                    "yaxis": {"title": "Amplitude"},
                    "title": title if title is not None else self.name,
                },
            )
            if show:
                fig.show()
            return fig
        raise ValueError("backend must be 'matplotlib' or 'plotly'")

    def plot_fft(
        self,
        *,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        show: bool = False,
        ax: Any = None,
        one_sided: bool = True,
        title: str | None = None,
    ) -> Any:
        """Plot the magnitude spectrum ``|FFT(y)|``."""
        freqs = self.f
        mag = np.abs(self.yf) / self.n_samples
        if one_sided:
            mask = freqs >= 0
            freqs = freqs[mask]
            mag = mag[mask]
        else:
            order = np.argsort(freqs)
            freqs = freqs[order]
            mag = mag[order]

        label = self.name or "Shape"
        if backend == "matplotlib":
            from matplotlib import pyplot as plt

            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.figure
            ax.plot(freqs * 1e-6, mag, label=label)
            ax.set_xlabel("Frequency (MHz)")
            ax.set_ylabel("|FFT|")
            ax.set_title(title if title is not None else f"{label} Spectrum")
            ax.grid(True, alpha=0.3)
            ax.legend()
            if show:
                plt.show()
            return fig, ax
        if backend == "plotly":
            from plotly.graph_objects import Figure, Scatter

            fig = Figure(
                [Scatter(x=freqs * 1e-6, y=mag, name=label, mode="lines")],
                {
                    "xaxis": {"title": "Frequency (MHz)"},
                    "yaxis": {"title": "|FFT|"},
                    "title": title if title is not None else f"{label} Spectrum",
                },
            )
            if show:
                fig.show()
            return fig
        raise ValueError("backend must be 'matplotlib' or 'plotly'")


# ---------------------------------------------------------------------------
# IQArray RF/microwave power helper (ported from QuantumCompilerV2/support.py)
# ---------------------------------------------------------------------------


class IQArray(np.ndarray):
    """Complex IQ voltage array with RF power (W / dBm), gain, and phase helpers."""

    Z0: float
    _name: str

    def __new__(
        cls,
        values: npt.ArrayLike,
        Z0: float = 50.0,
        name: str = "",
    ) -> "IQArray":
        if Z0 <= 0:
            raise ValueError("Impedance Z0 must be positive")
        obj = np.asarray(values, dtype=np.complex128).view(cls)
        obj._name = str(name)
        obj.Z0 = float(Z0)
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        if obj is None:
            return
        self._name = getattr(obj, "_name", "")
        self.Z0 = getattr(obj, "Z0", 50.0)

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str = "") -> None:
        self._name = str(value)

    @property
    def vI(self) -> FloatArray:
        return np.asarray(self.real, dtype=np.float64)

    @property
    def vQ(self) -> FloatArray:
        return np.asarray(self.imag, dtype=np.float64)

    @property
    def amplitude(self) -> FloatArray:
        return np.asarray(np.abs(self), dtype=np.float64)

    @property
    def pI(self) -> FloatArray:
        return (self.vI ** 2) / (2.0 * self.Z0)

    @property
    def pQ(self) -> FloatArray:
        return (self.vQ ** 2) / (2.0 * self.Z0)

    @property
    def p(self) -> FloatArray:
        return self.pI + self.pQ

    @property
    def dBm(self) -> FloatArray:
        ratio = np.asarray(self.p / 1e-3, dtype=np.float64)
        ratio = np.where(ratio <= 0.0, float_info.min, ratio)
        return 10.0 * np.log10(ratio)

    @property
    def phase(self) -> FloatArray:
        return np.asarray(np.angle(self), dtype=np.float64)

    def scale_to_vp(self, vp: float) -> "IQArray":
        peak = float(np.max(np.abs(self)))
        if peak == 0.0:
            return IQArray(np.zeros_like(self), Z0=self.Z0, name=self.name)
        return IQArray(np.asarray(self) / peak * float(vp), Z0=self.Z0, name=self.name)

    def scaleToVp(self, Vp: float) -> "IQArray":
        """V2 compatibility alias for :meth:`scale_to_vp`."""
        return self.scale_to_vp(Vp)

    def scale_to_dbm(self, dbm: float) -> "IQArray":
        power_w = 1e-3 * 10.0 ** (float(dbm) / 10.0)
        vp = float(np.sqrt(2.0 * self.Z0 * power_w))
        return self.scale_to_vp(vp)

    def scaleTodBm(self, dBm: float) -> "IQArray":
        """V2 compatibility alias for :meth:`scale_to_dbm`."""
        return self.scale_to_dbm(dBm)

    def gain(self, gain_db: float) -> "IQArray":
        return IQArray(
            np.asarray(self) * (10.0 ** (float(gain_db) / 20.0)),
            Z0=self.Z0,
            name=self.name,
        )

    def phase_shift(self, phase_rad: float) -> "IQArray":
        return IQArray(
            np.asarray(self) * np.exp(1j * float(phase_rad)),
            Z0=self.Z0,
            name=self.name,
        )

    def phaseShift(self, phase: float) -> "IQArray":
        """V2 compatibility alias for :meth:`phase_shift`."""
        return self.phase_shift(phase)

    def iq_plot(self, ax: Any = None, show: bool = False) -> Any:
        from matplotlib import pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        ax.plot(self.vI, self.vQ, label=self.name or "IQ")
        ax.set_xlabel("I (V)")
        ax.set_ylabel("Q (V)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        ax.legend()
        if show:
            plt.show()
        return fig, ax

    def IQplot(self, ax: Any = None, show: bool = False) -> Any:
        """V2 compatibility alias for :meth:`iq_plot`."""
        return self.iq_plot(ax=ax, show=show)


__all__ = [
    "IQArray",
    "SHAPE_REGISTRY",
    "Shape",
    "register_shape",
]
