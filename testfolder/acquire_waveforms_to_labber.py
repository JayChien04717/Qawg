"""Acquire three envelope shapes and save raw ADC plus FIR IQ to Labber."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from QAWG import AWGAlazar, ExperimentProgram, ValuesSweep, ns, us
from QAWG.alazar import design_lowpass_fir
from QAWG.labber_waveform_writer import save_waveform_labber
from QAWG.instrument import RohdeSchwarzSGS100A


AWG_RESOURCE = "TCPIP0::192.168.10.171::inst0::INSTR"
SGS100A_ADDRESS = "192.168.10.90"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputlabber"
WAVEFORM_NAMES = (
    "gaussian_square",
    "exponential_rise",
    "exponential_decay_80ns",
)
SAMPLE_RATE_HZ = 1e9
IF_FREQUENCY_HZ = 50e6
FIR_CUTOFF_HZ = 20e6
FIR_TAPS = 81
FIR_BETA = 8.0
DECAY_S = 80 * ns
N_AVERAGE = 1000


def read_sgs100a_settings():
    """Read the live SGS100A state without changing RF or LO outputs."""
    sgs = RohdeSchwarzSGS100A(SGS100A_ADDRESS)
    try:
        identity = sgs.idn()
        snapshot = sgs.snapshot()
        return {
            "hardware": "Rohde&Schwarz RF Source",
            "version": "1.2",
            "model": "SGS100A",
            "name": "RF - qubit",
            "interface": "TCPIP",
            "address": SGS100A_ADDRESS,
            "timeout_s": 5.0,
            "id": (
                "Rohde&Schwarz RF Source - IP: 192.168.10.90, "
                "RF - qubit at localhost"
            ),
            "settings": {
                "Identity": identity,
                "Frequency": snapshot["frequency"],
                "Power": snapshot["power"],
                "Output": snapshot["output"],
                "I/Q modulation": snapshot["iq"],
                "Pulse modulation": snapshot["pulsemod"],
                "LO Source": snapshot["reference_source"],
                "LO Frequency": snapshot["reference_external_frequency"],
                "LO Output Source": snapshot["reference_lo_output"],
            },
        }
    finally:
        sgs.close()


class WaveformShapeProgram(ExperimentProgram):
    """One sequence step per envelope shape, with identical acquisition."""

    def _initialize(self, cfg):
        self.declare_gen(
            "loopback",
            ch=cfg["awg_ch"],
            amplitude_vpp=cfg["channel_amplitude_vpp"],
        )
        self.declare_readout(
            "ro",
            adc_channel=cfg["adc_channel"],
            length=cfg["acquire_length"],
            demod_freq=cfg["frequency"],
            waveform_ch=cfg["awg_ch"],
            marker_channel=cfg["marker_ch"],
            marker_length=40 * ns,
            marker_lead=cfg["marker_lead"],
            integrate_time=cfg["integrate_time"],
        )
        self.add_sweep("waveform_index", ValuesSweep(range(len(WAVEFORM_NAMES))))
        common = {
            "gen": "loopback",
            "length": cfg["pulse_length"],
            "frequency": cfg["frequency"],
            "gain": cfg["pulse_gain"],
            "readout": True,
        }
        self.add_pulse(
            "gaussian_square",
            style="gaussian_square",
            edge_sigma=cfg["gaussian_edge_sigma"],
            **common,
        )
        self.add_pulse(
            "exponential_rise",
            style="exponential_rise",
            decay=cfg["rise_tau"],
            **common,
        )
        self.add_pulse(
            "exponential_decay",
            style="exponential_decay",
            decay=cfg["decay_tau"],
            **common,
        )

    def _body(self, cfg):
        self.play("gaussian_square", at=0, when=("waveform_index", 0))
        self.play("exponential_rise", at=0, when=("waveform_index", 1))
        self.play("exponential_decay", at=0, when=("waveform_index", 2))
        self.trigger("ro", trigger_delay=0)


def plot_result(result, output_path: Path) -> None:
    time_ns = result.raw_time_s / ns
    raw_average = np.mean(result.raw, axis=0)
    iq_average = np.mean(result.iq_traces, axis=0)
    figure, axes = plt.subplots(
        len(WAVEFORM_NAMES),
        2,
        figsize=(14, 10),
        sharex=True,
        constrained_layout=True,
    )
    for index, name in enumerate(WAVEFORM_NAMES):
        axes[index, 0].plot(time_ns, raw_average[index] * 1e3, lw=0.8)
        axes[index, 0].set_ylabel(f"{name}\nADC (mV)")
        axes[index, 1].plot(time_ns, iq_average[index].real * 1e3, label="I")
        axes[index, 1].plot(time_ns, iq_average[index].imag * 1e3, label="Q")
        axes[index, 1].plot(time_ns, np.abs(iq_average[index]) * 1e3, label="|IQ|")
        axes[index, 1].set_ylabel("IQ (mV)")
        for axis in axes[index]:
            axis.grid(alpha=0.25)
    axes[0, 0].set_title("Raw ADC before DDC/demodulation")
    axes[0, 1].set_title("DDC + 20 MHz FIR LPF")
    axes[0, 1].legend(frameon=False, ncols=3)
    axes[-1, 0].set_xlabel("Time after ATS trigger (ns)")
    axes[-1, 1].set_xlabel("Time after ATS trigger (ns)")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = {
        "awg_ch": 1,
        "marker_ch": 1,
        "adc_channel": "CHB",
        "channel_amplitude_vpp": 0.5,
        "frequency": IF_FREQUENCY_HZ,
        "marker_lead": 100 * ns,
        "pulse_length": 600 * ns,
        "gaussian_edge_sigma": 15 * ns,
        "rise_tau": 80 * ns,
        "decay_tau": DECAY_S,
        "pulse_gain": 0.01,
        "acquire_length": 1.5 * us,
        "integrate_time": 1.0 * us,
    }
    hardware = AWGAlazar.connect(
        AWG_RESOURCE,
        awg_sample_rate_hz=2.5e9,
        alazar_sample_rate_hz=SAMPLE_RATE_HZ,
        acquire_window_s=cfg["acquire_length"],
        trigger_slope="rising",
        trigger_level=140,
        mask_external_trigger_lsb=True,
        fir_cutoff_hz=FIR_CUTOFF_HZ,
        fir_number_of_taps=FIR_TAPS,
        fir_kaiser_beta=FIR_BETA,
    )
    try:
        awg_identity = hardware.awg.query("*IDN?").strip()
        awg_clock_source = hardware.awg.query("CLOCk:SOURce?").strip()
        program = WaveformShapeProgram(cfg, final_delay_s=1 * us)
        program.REMOVE_DC_OFFSET = False
        result = program.compile(hardware=hardware).acquire(n_average=N_AVERAGE)
    finally:
        hardware.close()

    instruments = [
        {
            "hardware": "Tektronix AWG5200",
            "version": "QAWG",
            "model": "AWG5208",
            "name": "AWG",
            "interface": "TCPIP",
            "address": "192.168.10.171",
            "settings": {
                "Identity": awg_identity,
                "Sample rate": 2.5e9,
                "Clock source": awg_clock_source,
                "Channel 1 amplitude Vpp": cfg["channel_amplitude_vpp"],
                "Waveform amplitude (gain)": cfg["pulse_gain"],
                "Marker channel": cfg["marker_ch"],
                "Marker lead": cfg["marker_lead"],
                "Marker length": 40 * ns,
            },
        },
        {
            "hardware": "AlazarTech ATS9371",
            "version": "QAWG",
            "model": "ATS9371",
            "name": "Digitizer",
            "interface": "Other",
            "address": "system 1, board 1",
            "settings": {
                "Acquisition mode": "NPT AutoDMA",
                "Sample rate": "1 GS/s",
                "Number of records": N_AVERAGE * len(WAVEFORM_NAMES),
                "Number of samples": result.raw.shape[-1],
                "Channel B - Coupling": "DC",
                "Channel B - Impedance": "50 Ohm",
                "Channel B - Range": "400 mV",
                "Clock source": "External 10 MHz reference",
                "Trigger source": "External",
                "Trigger slope": "Positive",
                "Trigger level code": 140,
                "External trigger LSB masked": True,
                "Demodulation frequency": IF_FREQUENCY_HZ,
                "FIR cutoff": FIR_CUTOFF_HZ,
                "FIR taps": FIR_TAPS,
            },
        },
    ]
    try:
        instruments.append(read_sgs100a_settings())
    except Exception as exc:
        print(f"Warning: SGS100A settings were not available: {exc}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = OUTPUT_DIR / f"waveform_raw_and_fir_iq_{timestamp}"
    coefficients = design_lowpass_fir(
        SAMPLE_RATE_HZ,
        FIR_CUTOFF_HZ,
        FIR_TAPS,
        FIR_BETA,
    )
    labber_path = save_waveform_labber(
        stem.with_suffix(".hdf5"),
        result.raw_time_s,
        result.raw,
        result.iq_traces,
        WAVEFORM_NAMES,
        adc_channel=cfg["adc_channel"],
        metadata={
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "if_frequency_hz": IF_FREQUENCY_HZ,
            "fir_cutoff_hz": FIR_CUTOFF_HZ,
            "fir_taps": FIR_TAPS,
            "fir_kaiser_beta": FIR_BETA,
            "decay_tau_s": DECAY_S,
            "waveform_amplitude_gain": cfg["pulse_gain"],
            "awg_channel_amplitude_vpp": cfg["channel_amplitude_vpp"],
            "n_average": N_AVERAGE,
            "raw_adc_is_before_ddc": True,
        },
        fir_coefficients=coefficients,
        instrument_settings=instruments,
    )
    plot_path = stem.with_suffix(".png")
    plot_result(result, plot_path)
    print(f"Saved Labber file: {labber_path}")
    print(f"Saved preview: {plot_path}")


if __name__ == "__main__":
    main()
