"""Complete example: using QuantumCompilerV2-style Shape & IQArray with QAWG.

Run offline from the repository root (no hardware required):
    python examples/shape_with_qawg_example.py
    python examples/shape_with_qawg_example.py --no-show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running directly as `python examples/shape_with_qawg_example.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from QAWG import (
    ExperimentProgram,
    IQArray,
    LinearSweep,
    MHz,
    Shape,
    delay_auto,
    ns,
    parallel,
    register_shape,
    us,
    waveform,
)


# 1. Optional: register a custom shape function (V2-style registry)
@register_shape("raised_cosine_ramp")
def raised_cosine_ramp(x: np.ndarray, rise_time: float = 20e-9) -> np.ndarray:
    return np.where(
        x < rise_time,
        0.5 * (1.0 - np.cos(np.pi * x / rise_time)),
        1.0,
    )


def build_shapes(sample_rate_hz: float) -> tuple[Shape, Shape, Shape]:
    """Build DRAG I/Q envelopes and a double-pulse sequence using Shape."""
    # Unit-amplitude Gaussian (normalized to [-1, 1] for ExperimentProgram)
    drive_i = Shape(
        "gaussian",
        {"peak_x": 40 * ns, "sigma": 10 * ns},
        span=80 * ns,
        sample_rate=sample_rate_hz,
        name="drive_I",
    )

    # Numerical derivative for DRAG Q component, normalized to [-1, 1]
    drive_q = drive_i.diff().normalize(1.0)
    drive_q.name = "drive_Q"

    # Temporal composition: two pulses separated by 20 ns (`|` = concat, `>>` = left zero-pad)
    double_pulse = drive_i | (0.5 * drive_i >> 20 * ns)
    double_pulse.name = "double_drive"

    return drive_i, drive_q, double_pulse


class DragRabiProgram(ExperimentProgram):
    """Power Rabi using Shape-built DRAG I/Q pulses and a V2 exp_rising_pulse."""

    def _initialize(self, cfg):
        self.declare_gen("drive_i", ch=1, amplitude_vpp=0.5)
        self.declare_gen("drive_q", ch=2, amplitude_vpp=0.5)
        self.declare_gen("ro_gen", ch=3, amplitude_vpp=0.5)

        self.declare_readout(
            "ro",
            adc_channel="CHA",
            length=600 * ns,
            demod_freq=50 * MHz,
            waveform_ch=3,
            marker_channel=1,
            marker_lead=200 * ns,
            integrate_time=500 * ns,
            marker_at_readout=True,
        )

        amp = self.add_sweep("amp", LinearSweep(0.1, 0.8, 5))

        # Pass Shape directly via style="custom" (length inferred from Shape.span)
        self.add_pulse(
            "pi_i",
            gen="drive_i",
            style="custom",
            envelope=cfg["drive_i"],
            frequency=50 * MHz,
            gain=amp,
        )
        self.add_pulse(
            "pi_q",
            gen="drive_q",
            style="custom",
            envelope=cfg["drive_q"],
            frequency=50 * MHz,
            phase=np.pi / 2,
            gain=0.15,
        )
        # V2 truncated rising exponential pulse (e.g. photon absorption envelope)
        self.add_pulse(
            "absorber",
            gen="drive_i",
            style="exp_rising_pulse",
            length=120 * ns,
            decay=30 * ns,
            frequency=50 * MHz,
            gain=0.3,
        )
        self.add_pulse(
            "readout",
            gen="ro_gen",
            style="cosine_square",
            length=600 * ns,
            edge_length=20 * ns,
            frequency=50 * MHz,
            gain=0.3,
            readout=True,
        )

    def _body(self, cfg):
        # Play DRAG I and Q simultaneously at t=0
        self.play("pi_i", at=0)
        self.play("pi_q", at=0)
        self.delay_auto(20 * ns)
        self.play("absorber")
        # Anchor readout at a fixed time across all sweep steps for marker_at_readout=True
        self.play("readout", at=300 * ns)
        self.trigger("ro")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Run offline compilation and plotting without opening GUI windows.",
    )
    args = parser.parse_args()

    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    sample_rate_hz = 2.5e9
    drive_i, drive_q, double_pulse = build_shapes(sample_rate_hz)

    # --- A. IQArray RF power conversion ---
    iq = IQArray(0.2 * drive_i.y + 1j * 0.05 * drive_q.y, Z0=50.0, name="drag_iq")
    iq_scaled = iq.scale_to_dbm(-10.0)
    print(
        f"IQArray peak voltage @ -10 dBm (50 ohm): "
        f"{np.max(iq_scaled.amplitude):.4f} V"
    )

    # --- B. Direct Timeline usage (Volts) ---
    w_i = waveform(0.2 * double_pulse, fc=50 * MHz, ch=1)
    w_q = (0.05 * drive_q).to_waveform(
        fc=50 * MHz, ch=2, phase_radians=np.pi / 2, name="drive_Q"
    )
    w_ro = Shape(
        "raised_cosine_ramp",
        {"rise_time": 20 * ns},
        span=400 * ns,
        sample_rate=sample_rate_hz,
        name="readout",
    ).to_waveform(fc=50 * MHz, ch=3, gain=0.15)

    timeline = parallel(w_i, w_q) / delay_auto(30 * ns) / w_ro
    fig_tl, _ = timeline.plot(
        sample_rate_hz, total_duration_s=1.2 * us, show=not args.no_show
    )
    fig_sch, _ = timeline.plot_schedule(sample_rate_hz, show=not args.no_show)

    # --- C. ExperimentProgram compilation & preview ---
    program = DragRabiProgram(
        {"drive_i": drive_i, "drive_q": drive_q},
        final_delay_s=500 * ns,
    )
    compiled = program.compile(sample_rate_hz=sample_rate_hz)
    print(
        f"Compiled {compiled.program_name}: "
        f"{compiled.number_of_sequence_steps} sweep steps, "
        f"step_duration={compiled.step_duration_s * 1e6:.2f} us"
    )

    fig_cmp, _ = compiled.plot(step=2, show=not args.no_show)

    if args.no_show:
        plt.close(fig_tl)
        plt.close(fig_sch)
        plt.close(fig_cmp)

    # --- D. Real hardware execution (uncomment when connected) ---
    # from QAWG import AWGAlazar
    # experiment = AWGAlazar.connect(
    #     "TCPIP0::192.168.10.171::inst0::INSTR",
    #     awg_sample_rate_hz=sample_rate_hz,
    #     acquire_window_s=1.5 * us,
    #     adc_channel="CHA",
    # )
    # result = experiment.compile(program).acquire(n_average=1000)
    # amps = result.axis("amp")
    # s21 = result.s21("ro")


if __name__ == "__main__":
    main()
