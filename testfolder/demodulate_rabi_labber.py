"""Process Rabi_Script.hdf5 while preserving its native Labber layout."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import h5py
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from QAWG.labber_demod import (
    demodulate_labber_file,
)
from QAWG.labber_waveform_writer import read_trace


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Rabi_Script.hdf5"
OUTPUT_DIR = ROOT / "outputlabber"
OUTPUT = OUTPUT_DIR / "Rabi_Script_FIR_demod.hdf5"
PLOT = OUTPUT_DIR / "Rabi_Script_FIR_demod.png"


def main():
    output, new_iq, _, input_channel, output_channel = demodulate_labber_file(
        SOURCE, OUTPUT
    )
    raw_time_s, raw = read_trace(SOURCE, input_channel)
    old_name = "Digitizer - Channel A - Average piecewise demodulated values"
    old_time_s, old_iq = read_trace(SOURCE, old_name)
    new_time_s, saved_new_iq = read_trace(output, output_channel)
    np.testing.assert_allclose(saved_new_iq, new_iq)

    figure, axes = plt.subplots(3, 1, figsize=(13, 10), constrained_layout=True)
    axes[0].plot(raw_time_s * 1e9, np.real(raw) * 1e3, lw=0.8)
    axes[0].set(title="Labber raw average record", ylabel="ADC (mV)")
    axes[1].plot(old_time_s * 1e9, np.abs(old_iq) * 1e3, label="existing |IQ|")
    axes[1].set(title="Existing Labber 20 ns piecewise demod", ylabel="|IQ| (mV)")
    axes[2].plot(new_time_s * 1e9, saved_new_iq.real * 1e3, label="I")
    axes[2].plot(new_time_s * 1e9, saved_new_iq.imag * 1e3, label="Q")
    axes[2].plot(new_time_s * 1e9, np.abs(saved_new_iq) * 1e3, label="|IQ|")
    axes[2].set(
        title="QAWG 50 MHz DDC + 20 MHz, 81-tap FIR LPF",
        xlabel="Time (ns)",
        ylabel="IQ (mV)",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(frameon=False)
    figure.savefig(PLOT, dpi=160)
    plt.close(figure)
    print(f"Saved exact-format Labber clone: {output}")
    print(f"Saved demod comparison: {PLOT}")
    print("Log channels:")
    with h5py.File(output, "r") as h5:
        for row in h5["Log list"]:
            print(f"  {row['channel_name'].decode()}")


if __name__ == "__main__":
    main()
