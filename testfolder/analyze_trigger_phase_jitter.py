"""Offline analysis of shot-to-shot phase jitter in the 0/180 degree loopback.

Reads ``single_shot_clock/single_shot_clock_data.npz`` (written by
``diagnose_single_shot_clock.py``); no instruments are needed.

Reports, per drive phase:
  * phase std, white-jitter estimate, block-averaged std, linear trend
  * the phase spread of one ADC sample at the IF (uniform quantization std)
  * dominant periodicities of the phase series
  * radial vs tangential spread (phase jitter vs additive noise)

Run from the repository root:  python testfolder/analyze_trigger_phase_jitter.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "single_shot_clock" / "single_shot_clock_data.npz"
IF_HZ = 50e6
WINDOW = (350, 950)  # samples inside the steady part of the pulse


def shot_phase_deg(raw: np.ndarray, time_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.exp(-2j * np.pi * IF_HZ * time_s[WINDOW[0] : WINDOW[1]])
    iq = 2.0 * (raw[:, WINDOW[0] : WINDOW[1]] * reference).mean(axis=-1)
    phase = np.degrees(np.angle(iq * np.exp(-1j * np.angle(iq.mean()))))
    return iq, phase


def main() -> None:
    data = np.load(DATA)
    raw, time_s = data["raw_volts"], data["raw_time_s"]
    shots = raw.shape[0]
    sample_deg = 360.0 * IF_HZ * (time_s[1] - time_s[0])
    print(f"{shots} shots; one ADC sample = {sample_deg:.2f} deg at {IF_HZ/1e6:g} MHz")
    print(f"uniform-quantization std = {sample_deg/np.sqrt(12):.3f} deg\n")

    for index in range(raw.shape[1]):
        iq, phase = shot_phase_deg(raw[:, index, :], time_s)
        print(f"drive phase index {index}: |IQ| = {np.abs(iq).mean()*1e3:.3f} mV")
        print(f"  phase std            {phase.std():.3f} deg")
        print(f"  white-jitter est.    {np.diff(phase).std()/np.sqrt(2):.3f} deg")
        for block in (10, 100, 500):
            count = shots // block
            means = phase[: count * block].reshape(count, block).mean(axis=1)
            print(
                f"  block({block:3d}) mean std {means.std():.3f} deg "
                f"(white expectation {phase.std()/np.sqrt(block):.3f})"
            )
        trend = np.polyfit(np.arange(shots), phase, 1)[0] * 1000
        print(f"  linear trend         {trend:.3f} deg / 1000 shots")
        spectrum = np.abs(np.fft.rfft(phase - phase.mean())) ** 2
        frequency = np.fft.rfftfreq(shots)
        top = np.argsort(spectrum[1:])[::-1][:3] + 1
        print("  top periods (shots) ", np.round(1 / frequency[top], 2).tolist())
        radial = np.abs(iq) - np.abs(iq).mean()
        tangential = np.abs(iq).mean() * np.radians(phase)
        print(
            f"  radial std {radial.std()*1e3:.4f} mV vs "
            f"tangential std {tangential.std()*1e3:.4f} mV\n"
        )


if __name__ == "__main__":
    main()
