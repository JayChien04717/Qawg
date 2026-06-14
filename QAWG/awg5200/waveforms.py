"""Pure waveform and WFMX construction functions for Tektronix AWG5200."""

from __future__ import annotations

import datetime as dt
import xml.etree.ElementTree as ET
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

MIN_WAVEFORM_SAMPLES = 2400
MAX_MARKERS = 4

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


def sample_count(duration_s: float, sample_rate_hz: float) -> int:
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    return int(round(duration_s * sample_rate_hz))


def ns_to_samples(duration_ns: float, sample_rate_hz: float) -> int:
    if duration_ns <= 0:
        raise ValueError("duration_ns must be positive")
    return sample_count(duration_ns * 1e-9, sample_rate_hz)


def time_axis(number_of_samples: int, sample_rate_hz: float) -> FloatArray:
    if number_of_samples < 1:
        raise ValueError("number_of_samples must be positive")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    return np.arange(number_of_samples, dtype=np.float64) / sample_rate_hz


def time_axis_ns(number_of_samples: int, sample_rate_hz: float) -> FloatArray:
    return time_axis(number_of_samples, sample_rate_hz) * 1e9


def constant(number_of_samples: int, level: float = 0.0) -> FloatArray:
    if number_of_samples < 1:
        raise ValueError("number_of_samples must be positive")
    return np.full(number_of_samples, level, dtype=np.float64)


def sine(
    number_of_samples: int,
    sample_rate_hz: float,
    frequency_hz: float,
    amplitude_volts: float,
    phase_radians: float = 0.0,
    offset_volts: float = 0.0,
) -> FloatArray:
    if frequency_hz < 0 or frequency_hz > sample_rate_hz / 2:
        raise ValueError("frequency_hz must be between DC and Nyquist")
    phase = 2.0 * np.pi * frequency_hz * time_axis(
        number_of_samples, sample_rate_hz
    )
    return offset_volts + amplitude_volts * np.sin(phase + phase_radians)


def modulate_envelope(
    envelope: npt.ArrayLike,
    sample_rate_hz: float,
    frequency_hz: float,
    phase_radians: float = 0.0,
) -> FloatArray:
    """Apply a sine carrier to one real-valued waveform envelope."""
    values = np.asarray(envelope, dtype=np.float64).reshape(-1)
    if values.size < 1:
        raise ValueError("envelope cannot be empty")
    if frequency_hz < 0 or frequency_hz > sample_rate_hz / 2:
        raise ValueError("frequency_hz must be between DC and Nyquist")
    if frequency_hz == 0:
        return values.copy()
    carrier = sine(
        values.size,
        sample_rate_hz,
        frequency_hz,
        amplitude_volts=1.0,
        phase_radians=phase_radians,
    )
    return values * carrier


def gaussian(
    number_of_samples: int,
    sample_rate_hz: float,
    sigma_s: float,
    amplitude_volts: float,
    center_s: float | None = None,
) -> FloatArray:
    if sigma_s <= 0:
        raise ValueError("sigma_s must be positive")
    time = time_axis(number_of_samples, sample_rate_hz)
    center = time[-1] / 2.0 if center_s is None else center_s
    return amplitude_volts * np.exp(-0.5 * ((time - center) / sigma_s) ** 2)


def gaussian_square(
    number_of_samples: int,
    sample_rate_hz: float,
    sigma_s: float,
    amplitude_volts: float,
    edge_sigmas: float = 3.0,
) -> FloatArray:
    """Return a flat envelope with Gaussian rising and falling edges."""
    if sigma_s <= 0:
        raise ValueError("sigma_s must be positive")
    if edge_sigmas <= 0:
        raise ValueError("edge_sigmas must be positive")
    edge_samples = max(
        1,
        int(round(edge_sigmas * sigma_s * sample_rate_hz)),
    )
    if 2 * edge_samples > number_of_samples:
        raise ValueError("waveform is too short for the requested Gaussian edges")
    x = np.arange(edge_samples, dtype=np.float64)
    sigma_samples = max(1.0, edge_samples / edge_sigmas)
    rise = np.exp(
        -0.5 * ((x - (edge_samples - 1)) / sigma_samples) ** 2
    )
    envelope = constant(number_of_samples, amplitude_volts)
    envelope[:edge_samples] *= rise
    envelope[-edge_samples:] *= rise[::-1]
    return envelope


def cosine_square(
    number_of_samples: int,
    sample_rate_hz: float,
    edge_length_s: float,
    amplitude_volts: float,
) -> FloatArray:
    """Return a flat envelope with symmetric half-cosine edges."""
    if number_of_samples < 1:
        raise ValueError("number_of_samples must be positive")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if edge_length_s <= 0:
        raise ValueError("edge_length_s must be positive")

    edge_samples = int(round(edge_length_s * sample_rate_hz))
    if edge_samples < 2:
        raise ValueError("edge_length_s must span at least two samples")
    if 2 * edge_samples >= number_of_samples:
        raise ValueError(
            "waveform must contain a flat top after both cosine edges"
        )

    edge_phase = np.linspace(
        0.0,
        np.pi,
        edge_samples,
        endpoint=True,
        dtype=np.float64,
    )
    rise = 0.5 * (1.0 - np.cos(edge_phase))
    envelope = constant(number_of_samples, amplitude_volts)
    envelope[:edge_samples] *= rise
    envelope[-edge_samples:] *= rise[::-1]
    return envelope


def gaussian_cosine(
    number_of_samples: int,
    sample_rate_hz: float,
    frequency_hz: float,
    sigma_s: float,
    amplitude_volts: float,
    center_s: float | None = None,
    phase_radians: float = 0.0,
) -> FloatArray:
    envelope = gaussian(
        number_of_samples,
        sample_rate_hz,
        sigma_s,
        amplitude_volts,
        center_s,
    )
    time = time_axis(number_of_samples, sample_rate_hz)
    return envelope * np.cos(2.0 * np.pi * frequency_hz * time + phase_radians)


def gaussian_cosine_ns(
    duration_ns: float,
    sample_rate_hz: float,
    frequency_hz: float,
    sigma_ns: float,
    amplitude_volts: float,
    center_ns: float | None = None,
    phase_radians: float = 0.0,
) -> FloatArray:
    number_of_samples = ns_to_samples(duration_ns, sample_rate_hz)
    center_s = None if center_ns is None else center_ns * 1e-9
    return gaussian_cosine(
        number_of_samples=number_of_samples,
        sample_rate_hz=sample_rate_hz,
        frequency_hz=frequency_hz,
        sigma_s=sigma_ns * 1e-9,
        amplitude_volts=amplitude_volts,
        center_s=center_s,
        phase_radians=phase_radians,
    )


def gaussian_square_ns(
    duration_ns: float,
    sample_rate_hz: float,
    edge_sigma_ns: float,
    amplitude_volts: float,
    edge_sigmas: float = 4.0,
) -> FloatArray:
    """Return a flat pulse with Gaussian rising and falling edges."""
    if edge_sigma_ns <= 0:
        raise ValueError("edge_sigma_ns must be positive")
    if edge_sigmas <= 0:
        raise ValueError("edge_sigmas must be positive")
    number_of_samples = ns_to_samples(duration_ns, sample_rate_hz)
    edge_samples = int(round(edge_sigma_ns * edge_sigmas * 1e-9 * sample_rate_hz))
    if 2 * edge_samples >= number_of_samples:
        raise ValueError("duration_ns is too short for the requested Gaussian edges")

    envelope = np.full(number_of_samples, amplitude_volts, dtype=np.float64)
    sigma_samples = edge_sigma_ns * 1e-9 * sample_rate_hz
    edge_index = np.arange(edge_samples, dtype=np.float64)
    rising = np.exp(-0.5 * ((edge_index - edge_samples) / sigma_samples) ** 2)
    envelope[:edge_samples] *= rising
    envelope[-edge_samples:] *= rising[::-1]
    return envelope


def cosine_square_ns(
    duration_ns: float,
    sample_rate_hz: float,
    edge_length_ns: float,
    amplitude_volts: float,
) -> FloatArray:
    """Return a flat pulse with symmetric half-cosine edges."""
    return cosine_square(
        number_of_samples=ns_to_samples(duration_ns, sample_rate_hz),
        sample_rate_hz=sample_rate_hz,
        edge_length_s=edge_length_ns * 1e-9,
        amplitude_volts=amplitude_volts,
    )


###################################################
def triangle(
    number_of_samples: int,
    sample_rate_hz: float,
    half_width_s: float,
    amplitude_volts: float,
    center_s: float | None = None,
) -> FloatArray:
    if half_width_s <= 0:
        raise ValueError("half_width_s must be positive")
    
    time = time_axis(number_of_samples, sample_rate_hz)
    center = time[-1] / 2.0 if center_s is None else center_s
    
    # 計算與中心點的絕對距離
    distance = np.abs(time - center)
    
    # 線性遞減，超出 half_width_s 的部分強制歸零
    envelope = amplitude_volts * (1.0 - distance / half_width_s)
    return np.maximum(0.0, envelope)


def triangle_cosine(
    number_of_samples: int,
    sample_rate_hz: float,
    frequency_hz: float,
    half_width_s: float,
    amplitude_volts: float,
    center_s: float | None = None,
    phase_radians: float = 0.0,
) -> FloatArray:
    envelope = triangle(
        number_of_samples=number_of_samples,
        sample_rate_hz=sample_rate_hz,
        half_width_s=half_width_s,
        amplitude_volts=amplitude_volts,
        center_s=center_s,
    )
    time = time_axis(number_of_samples, sample_rate_hz)
    return envelope * np.cos(2.0 * np.pi * frequency_hz * time + phase_radians)


def triangle_ns(
    duration_ns: float,
    sample_rate_hz: float,
    frequency_hz: float,
    half_width_ns: float,
    amplitude_volts: float,
    center_ns: float | None = None,
    phase_radians: float = 0.0,
) -> FloatArray:
    number_of_samples = ns_to_samples(duration_ns, sample_rate_hz)
    center_s = None if center_ns is None else center_ns * 1e-9
    
    return triangle_cosine(
        number_of_samples=number_of_samples,
        sample_rate_hz=sample_rate_hz,
        frequency_hz=frequency_hz,
        half_width_s=half_width_ns * 1e-9,
        amplitude_volts=amplitude_volts,
        center_s=center_s,
        phase_radians=phase_radians,
    )

#############################################################


def concatenate(parts: Sequence[npt.ArrayLike]) -> FloatArray:
    if not parts:
        raise ValueError("parts cannot be empty")
    return np.concatenate(
        [np.asarray(part, dtype=np.float64).reshape(-1) for part in parts]
    )


def marker_window(
    number_of_samples: int,
    start_sample: int,
    stop_sample: int,
) -> BoolArray:
    if not 0 <= start_sample < stop_sample <= number_of_samples:
        raise ValueError("marker window is outside the waveform")
    marker = np.zeros(number_of_samples, dtype=np.bool_)
    marker[start_sample:stop_sample] = True
    return marker


def marker_window_ns(
    duration_ns: float,
    sample_rate_hz: float,
    start_ns: float,
    stop_ns: float,
) -> BoolArray:
    number_of_samples = ns_to_samples(duration_ns, sample_rate_hz)
    start_sample = int(round(start_ns * 1e-9 * sample_rate_hz))
    stop_sample = int(round(stop_ns * 1e-9 * sample_rate_hz))
    return marker_window(number_of_samples, start_sample, stop_sample)


def trigger_channel_for(
    reference_waveform: npt.ArrayLike,
    threshold_ratio: float = 1e-3,
    padding_samples: int = 0,
) -> tuple[FloatArray, BoolArray]:
    """Return a zero waveform and one marker enclosing the active pulse."""
    reference = np.asarray(reference_waveform, dtype=np.float64).reshape(-1)
    if reference.size < 1:
        raise ValueError("reference_waveform cannot be empty")
    if not 0 < threshold_ratio <= 1:
        raise ValueError("threshold_ratio must be between 0 and 1")
    if padding_samples < 0:
        raise ValueError("padding_samples cannot be negative")

    peak = float(np.max(np.abs(reference)))
    if peak == 0:
        raise ValueError("reference_waveform has no active pulse")
    active = np.flatnonzero(np.abs(reference) >= peak * threshold_ratio)
    start_sample = max(0, int(active[0]) - padding_samples)
    stop_sample = min(reference.size, int(active[-1]) + 1 + padding_samples)
    trigger_waveform = constant(reference.size)
    trigger_marker = marker_window(reference.size, start_sample, stop_sample)
    return trigger_waveform, trigger_marker


def validate_waveform(
    waveform_volts: npt.ArrayLike,
    amplitude_vpp: float,
) -> FloatArray:
    waveform = np.asarray(waveform_volts, dtype=np.float64).reshape(-1)
    if waveform.size < MIN_WAVEFORM_SAMPLES:
        raise ValueError(
            f"AWG5200 waveforms require at least {MIN_WAVEFORM_SAMPLES} samples"
        )
    if not np.all(np.isfinite(waveform)):
        raise ValueError("waveform must contain only finite values")
    if amplitude_vpp <= 0:
        raise ValueError("amplitude_vpp must be positive")
    peak_volts = float(np.max(np.abs(waveform)))
    allowed_peak_volts = amplitude_vpp / 2
    if peak_volts > allowed_peak_volts + 1e-12:
        raise ValueError(
            f"waveform peak {peak_volts:.6g} V exceeds the allowed "
            f"{allowed_peak_volts:.6g} V for amplitude_vpp={amplitude_vpp:.6g} V; "
            f"use amplitude_vpp >= {2 * peak_volts:.6g} V"
        )
    return waveform


def pack_markers(
    markers: Sequence[npt.ArrayLike],
    number_of_samples: int,
) -> npt.NDArray[np.uint8]:
    if len(markers) > MAX_MARKERS:
        raise ValueError(f"AWG5208 supports at most {MAX_MARKERS} markers")
    packed = np.zeros(number_of_samples, dtype=np.uint8)
    for index, marker in enumerate(markers):
        values = np.asarray(marker, dtype=np.bool_).reshape(-1)
        if values.size != number_of_samples:
            raise ValueError("every marker must match the waveform length")
        packed |= values.astype(np.uint8) << index
    return packed


def waveform_binary(
    waveform_volts: npt.ArrayLike,
    amplitude_vpp: float,
    markers: Sequence[npt.ArrayLike] = (),
) -> bytes:
    waveform = validate_waveform(waveform_volts, amplitude_vpp)
    normalized = (2.0 * waveform / amplitude_vpp).astype("<f4")
    marker_bytes = (
        pack_markers(markers, waveform.size).tobytes() if markers else b""
    )
    return normalized.tobytes() + marker_bytes


def wfmx_header(
    number_of_samples: int,
    markers_included: bool,
    timestamp: dt.datetime | None = None,
) -> bytes:
    if number_of_samples < MIN_WAVEFORM_SAMPLES:
        raise ValueError(
            f"number_of_samples must be at least {MIN_WAVEFORM_SAMPLES}"
        )
    created_at = timestamp or dt.datetime.now().astimezone()
    timestamp_text = created_at.isoformat(timespec="milliseconds")
    offset_digits = 9

    root = ET.Element(
        "DataFile", attrib={"offset": "0" * offset_digits, "version": "0.1"}
    )
    collection = ET.SubElement(root, "DataSetsCollection")
    collection.set("xmlns", "http://www.tektronix.com")
    collection.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    collection.set(
        "xsi:schemaLocation",
        "http://www.tektronix.com file:///C:\\Program%20Files\\Tektronix"
        "\\AWG70000\\AWG\\Schemas\\awgDataSets.xsd",
    )
    datasets = ET.SubElement(collection, "DataSets")
    datasets.set("version", "1")
    datasets.set("xmlns", "http://www.tektronix.com")
    description = ET.SubElement(datasets, "DataDescription")
    fields = {
        "NumberSamples": str(number_of_samples),
        "SamplesType": "AWGWaveformSample",
        "MarkersIncluded": str(markers_included).lower(),
        "NumberFormat": "Single",
        "Endian": "Little",
        "Timestamp": timestamp_text,
    }
    for name, value in fields.items():
        ET.SubElement(description, name).text = value

    product = ET.SubElement(datasets, "ProductSpecific", attrib={"name": ""})
    for name, value in (
        ("ReccSamplingRate", "NaN"),
        ("ReccAmplitude", "NaN"),
        ("ReccOffset", "NaN"),
        ("SerialNumber", None),
        ("SoftwareVersion", "1.0.0917"),
        ("UserNotes", None),
        ("OriginalBitDepth", "Floating"),
        ("Thumbnail", None),
    ):
        element = ET.SubElement(product, name)
        element.text = value
        if name in {"ReccSamplingRate", "ReccAmplitude", "ReccOffset"}:
            element.set("units", "Hz" if name == "ReccSamplingRate" else "Volts")
    ET.SubElement(product, "CreatorProperties", attrib={"name": ""})
    ET.SubElement(root, "Setup")

    text = ET.tostring(root, encoding="unicode").replace("><", ">\r\n<")
    text = text.replace(
        "0" * offset_digits, f"{len(text):0{offset_digits}d}", 1
    )
    return text.encode("ascii")


def make_wfmx(
    waveform_volts: npt.ArrayLike,
    amplitude_vpp: float,
    markers: Sequence[npt.ArrayLike] = (),
) -> bytes:
    waveform = validate_waveform(waveform_volts, amplitude_vpp)
    header = wfmx_header(waveform.size, bool(markers))
    return header + waveform_binary(waveform, amplitude_vpp, markers)
