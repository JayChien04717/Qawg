from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


notebook = nbf.v4.new_notebook()
notebook["metadata"] = {
    "kernelspec": {
        "display_name": "scqenv",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3"},
}

notebook["cells"] = [
    markdown(
        r"""
# 3D cavity ring-down with `ExperimentProgram`

Physical sequence:

```text
AWG IF + microwave LO -> IQ mixer -> 3D cavity
AWG cavity-fill pulse starts
        |
        +-- marker rising edge triggers ATS9371
        |
AWG cavity-fill pulse stops
        |
        +-- optional guard time
        |
ATS acquisition begins and records free cavity ring-down
3D cavity output -> IQ mixer -> Alazar IF input
```

The compiler keeps every record. Tomography processing must use
`result.raw[:, 0, :]`, demodulate every shot separately, and project every
shot onto the exponential temporal mode

\[
f(t) \propto \exp[-t/(2T_{\rm cav})].
\]

The trace average is used only for timing and fitting diagnostics.
"""
    ),
    code(
        """
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from QAWG import AWGAlazar, CavityRingdownProgram, ns, us
from QAWG.alazar import digital_downconvert
from QAWG.tomography import project_temporal_mode, temporal_mode_weights
"""
    ),
    markdown("## Hardware and cavity parameters"),
    code(
        """
AWG_RESOURCE = "TCPIP0::192.168.10.171::inst0::INSTR"

AWG_SAMPLE_RATE_HZ = 2.5e9
ALAZAR_SAMPLE_RATE_HZ = 1e9

# Physical 3D-cavity resonance used for Q. Adjust to the VNA result.
CAVITY_RESONANCE_HZ = 5.0e9

# AWG/Alazar intermediate frequency after IQ up/downconversion.
IF_FREQUENCY_HZ = 50e6

AWG_CHANNEL = 3
MARKER_CHANNEL = 1
ADC_CHANNEL = "CHA"
CHANNEL_AMPLITUDE_VPP = 0.5

# Fill for several expected cavity lifetimes.
DRIVE_LENGTH = 2.0 * us
DRIVE_GAIN = 0.0002
EDGE_SIGMA = 20 * ns

# ATS starts this long after the fill pulse stops.
RINGDOWN_GUARD = 40 * ns
ACQUIRE_LENGTH = 1.5 * us
NUM_SHOTS = 5000

# Initial guess used by the matched filter. Fit the averaged ring-down below
# and replace this value with the measured energy lifetime.
T_CAVITY_GUESS = 250 * ns

ALAZAR_TIMEOUT_MS = 60_000
"""
    ),
    markdown("## Connect AWG5208 and ATS9371"),
    code(
        """
experiment = AWGAlazar.connect(
    AWG_RESOURCE,
    awg_sample_rate_hz=AWG_SAMPLE_RATE_HZ,
    alazar_sample_rate_hz=ALAZAR_SAMPLE_RATE_HZ,
    acquire_window_s=ACQUIRE_LENGTH,
    tone_frequency_hz=IF_FREQUENCY_HZ,
    adc_channel=ADC_CHANNEL,
    timeout_ms=ALAZAR_TIMEOUT_MS,
    use_external_10mhz_reference=True,
)

print("AWG:", experiment.awg.identify())
print("ATS record samples:", experiment.acquire_window_cycles)
"""
    ),
    markdown("## Define and compile the declarative ring-down program"),
    code(
        """
cfg = {
    "frequency": IF_FREQUENCY_HZ,
    "awg_ch": AWG_CHANNEL,
    "marker_ch": MARKER_CHANNEL,
    "adc_channel": ADC_CHANNEL,
    "channel_amplitude_vpp": CHANNEL_AMPLITUDE_VPP,
    "drive_length": DRIVE_LENGTH,
    "drive_gain": DRIVE_GAIN,
    "edge_sigma": EDGE_SIGMA,
    "ringdown_guard": RINGDOWN_GUARD,
    "acquire_length": ACQUIRE_LENGTH,
    "integrate_time": ACQUIRE_LENGTH,
}

program = CavityRingdownProgram(cfg)
compiled = program.compile(hardware=experiment)

print("Sequence steps:", compiled.number_of_sequence_steps)
print("ATS trigger delay:", compiled.trigger_delay_s / ns, "ns")
print("Expected:", (DRIVE_LENGTH + RINGDOWN_GUARD) / ns, "ns")
"""
    ),
    markdown("## Preview drive waveform and marker before touching hardware"),
    code(
        """
drive = compiled.preview(AWG_CHANNEL)[0]
marker = compiled.marker_waveforms[0]
awg_time_us = np.arange(drive.size) / AWG_SAMPLE_RATE_HZ / us

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
axes[0].plot(awg_time_us, drive * 1e3)
axes[0].set_ylabel("AWG voltage (mV)")
axes[0].grid(True, alpha=0.3)
axes[1].plot(awg_time_us, marker.astype(float))
axes[1].set_xlabel("AWG sequence time (us)")
axes[1].set_ylabel("Marker")
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""
    ),
    markdown(
        """
## Upload once, then acquire reference and cavity signal

The reference keeps the marker channel enabled and disables only the cavity
drive output. The same compiled sequence and ATS timing are used for both
captures.
"""
    ),
    code(
        """
compiled.upload()

experiment.awg.set_output(AWG_CHANNEL, False)
reference_result = compiled.acquire(NUM_SHOTS)

experiment.awg.set_output(AWG_CHANNEL, True)
signal_result = compiled.acquire(NUM_SHOTS)

reference_records = reference_result.raw[:, 0, :]
signal_records = signal_result.raw[:, 0, :]
time_s = signal_result.raw_time_s

print("Reference records:", reference_records.shape)
print("Signal records:", signal_records.shape)
"""
    ),
    markdown("## Demodulate every shot and inspect the averaged envelope"),
    code(
        """
reference_baseband = digital_downconvert(
    reference_records,
    ALAZAR_SAMPLE_RATE_HZ,
    IF_FREQUENCY_HZ,
)
signal_baseband = digital_downconvert(
    signal_records,
    ALAZAR_SAMPLE_RATE_HZ,
    IF_FREQUENCY_HZ,
)

# Coherent reference subtraction removes stable leakage/offset only.
reference_mean_trace = np.mean(reference_baseband, axis=0)
ringdown_baseband = signal_baseband - reference_mean_trace[None, :]
average_ringdown = np.mean(ringdown_baseband, axis=0)

# One carrier-period moving average is for visualization only.
period_samples = max(1, round(ALAZAR_SAMPLE_RATE_HZ / IF_FREQUENCY_HZ))
kernel = np.ones(period_samples) / period_samples
average_envelope = np.convolve(
    np.abs(average_ringdown),
    kernel,
    mode="same",
)

plt.figure(figsize=(11, 4))
plt.plot(time_s / ns, average_envelope * 1e3)
plt.xlabel("Time after ATS delayed trigger (ns)")
plt.ylabel("Coherent ring-down amplitude (mV)")
plt.title("Cavity free decay after drive turn-off")
plt.grid(True, alpha=0.3)
plt.show()
"""
    ),
    markdown(
        r"""
## Fit the cavity lifetime

The field amplitude follows

\[
|a(t)| = A\exp[-t/(2T_{\rm cav})] + C,
\]

while cavity energy follows \(\exp[-t/T_{\rm cav}]\).
"""
    ),
    code(
        """
FIT_START = 30 * ns
FIT_STOP = 1.2 * us
fit_mask = (time_s >= FIT_START) & (time_s <= FIT_STOP)

def field_decay(t, amplitude, t_cavity, offset):
    return amplitude * np.exp(-t / (2.0 * t_cavity)) + offset

popt, pcov = curve_fit(
    field_decay,
    time_s[fit_mask],
    average_envelope[fit_mask],
    p0=(np.max(average_envelope), T_CAVITY_GUESS, 0.0),
    bounds=([0.0, 1 * ns, 0.0], [np.inf, 100 * us, np.inf]),
)
amplitude_fit, t_cavity_fit, offset_fit = popt
fit_std = np.sqrt(np.diag(pcov))

print(f"Energy lifetime T_cavity = {t_cavity_fit / ns:.3f} ns")
print(f"Fit uncertainty = {fit_std[1] / ns:.3f} ns")
print(
    f"Loaded Q = "
    f"{2 * np.pi * CAVITY_RESONANCE_HZ * t_cavity_fit:.6g}"
)

plt.figure(figsize=(11, 4))
plt.plot(time_s / ns, average_envelope * 1e3, label="measured")
plt.plot(
    time_s / ns,
    field_decay(time_s, *popt) * 1e3,
    "--",
    label="exponential fit",
)
plt.xlabel("Time (ns)")
plt.ylabel("Amplitude (mV)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
"""
    ),
    markdown("## Per-shot exponential matched-filter projection"),
    code(
        """
MODE_START = FIT_START
MODE_STOP = min(ACQUIRE_LENGTH, 5.0 * t_cavity_fit)
mode_start = round(MODE_START * ALAZAR_SAMPLE_RATE_HZ)
mode_stop = round(MODE_STOP * ALAZAR_SAMPLE_RATE_HZ)
mode_samples = mode_stop - mode_start

mode = temporal_mode_weights(
    mode_samples,
    kind="exponential",
    decay_samples=t_cavity_fit * ALAZAR_SAMPLE_RATE_HZ,
)
reference_mode = project_temporal_mode(
    reference_baseband,
    mode,
    start_sample=mode_start,
)
signal_mode = project_temporal_mode(
    signal_baseband,
    mode,
    start_sample=mode_start,
)
ringdown_mode = signal_mode - np.mean(reference_mode)

print("Mode window:", MODE_START / ns, "to", MODE_STOP / ns, "ns")
print("One complex IQ point per shot:", ringdown_mode.shape)
print("Mean matched-filter IQ:", np.mean(ringdown_mode))
print("Electrical SNR:", abs(np.mean(ringdown_mode)) / np.std(reference_mode))

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(reference_mode.real, reference_mode.imag, s=5, alpha=0.15, label="reference")
ax.scatter(ringdown_mode.real, ringdown_mode.imag, s=5, alpha=0.15, label="ring-down")
ax.set_xlabel("Matched-filter I")
ax.set_ylabel("Matched-filter Q")
ax.axis("equal")
ax.grid(True, alpha=0.3)
ax.legend()
plt.show()
"""
    ),
    markdown(
        """
## Interpretation

- This room-temperature experiment validates trigger timing, cavity lifetime,
  per-shot demodulation, leakage subtraction, and exponential mode matching.
- Do not average records before matched-filter projection.
- Digital subtraction works only if the receiver is not saturated.
- Photon population requires calibrated attenuation, receiver gain and added
  noise. The room-temperature IQ cloud is not yet an absolute photon scale.
"""
    ),
    code(
        """
np.savez_compressed(
    "cavity_ringdown_capture.npz",
    reference_records=reference_records,
    signal_records=signal_records,
    time_s=time_s,
    reference_mode=reference_mode,
    ringdown_mode=ringdown_mode,
    temporal_mode=mode,
    t_cavity_fit_s=t_cavity_fit,
    cavity_resonance_hz=CAVITY_RESONANCE_HZ,
    if_frequency_hz=IF_FREQUENCY_HZ,
)
print("Saved cavity_ringdown_capture.npz")
"""
    ),
    markdown("## Close hardware session"),
    code(
        """
experiment.awg.set_output(AWG_CHANNEL, False)
experiment.close()
print("AWG VISA session closed")
"""
    ),
]

output = Path("cavity_ringdown.ipynb")
nbf.write(notebook, output)
print(output.resolve())
