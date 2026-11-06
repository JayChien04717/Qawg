"""Hardware check for two-channel capture (CHA + CHB).

Wiring: AWG ch3 50 MHz tone -> ATS CHB only (leave CHA unconnected or
terminated); AWG marker -> TRIG IN.  Expected: CHB shows the tone, CHA only
noise.  If CHA shows the tone instead, the DMA channel order differs from the
assumed layout (record, channel, sample) and ``copy_dma_buffer`` must change.

Run from the repository root:  python testfolder/check_dual_channel.py
"""

import numpy as np

from QAWG import AWGAlazar, ExperimentProgram, MHz, ns, us

AWG_RESOURCE = "TCPIP0::192.168.10.171::inst0::INSTR"
IF_HZ = 50 * MHz


class ToneOnRefChannel(ExperimentProgram):
    def _initialize(self, cfg):
        self.declare_gen("ref", ch=3, amplitude_vpp=0.5)
        self.declare_gen("trig", ch=1, amplitude_vpp=0.5)
        self.declare_readout(
            "ro", adc_channel="CHA", length=2 * us, demod_freq=IF_HZ,
            marker_channel=1, marker_length=40 * ns, integrate_time=2 * us,
        )
        self.add_pulse("tone", gen="ref", style="const", length=2 * us,
                       frequency=IF_HZ, gain=0.5)
        self.add_pulse("idle", gen="trig", style="const", length=100 * ns,
                       frequency=IF_HZ, gain=0.0)

    def _body(self, cfg):
        self.play("tone", at=0)
        self.play("idle", at=0)
        self.trigger("ro", trigger_delay=0)


def main() -> None:
    experiment = AWGAlazar.connect(
        AWG_RESOURCE, awg_sample_rate_hz=2.5e9, acquire_window_s=2.5 * us,
        adc_channel="CHA", reference_channel="CHB",
    )
    try:
        compiled = experiment.compile(ToneOnRefChannel({}, final_delay_s=1 * us))
        compiled.acquire(n_average=200)
        cha = experiment.last_records_volts
        chb = experiment.last_reference_volts
        print(f"CHA rms {cha.std()*1e3:.3f} mV   CHB rms {chb.std()*1e3:.3f} mV")
        print("expected: CHB >> CHA")
        deviation = experiment.last_reference_deviation_rad
        print(f"CHB phase jitter std {np.degrees(deviation.std()):.2f} deg")
    finally:
        experiment.close()


if __name__ == "__main__":
    main()
