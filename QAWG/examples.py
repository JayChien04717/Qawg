"""Example programs built on the host-side experiment compiler."""

from __future__ import annotations

from typing import Any

from .compiler import (
    ExperimentProgram,
    LinearSweep,
    ValuesSweep,
    ns,
    us,
)


def declare_standard_hardware(
    program: ExperimentProgram,
    cfg: dict[str, Any],
) -> None:
    program.declare_gen(
        "qubit",
        ch=cfg.get("qubit_ch", 4),
        amplitude_vpp=cfg.get("qubit_amplitude_vpp", 0.5),
    )
    program.declare_gen(
        "res",
        ch=cfg.get("res_ch", 3),
        amplitude_vpp=cfg.get("res_amplitude_vpp", 0.5),
    )
    program.declare_readout(
        "ro",
        adc_channel=cfg.get("adc_channel", "CHA"),
        length=cfg.get("ro_len", 1 * us),
        demod_freq=cfg["f_res"],
        waveform_ch=cfg.get("res_ch", 3),
        marker_channel=cfg.get("marker_ch", 1),
        integrate_time=cfg.get("integrate_time"),
    )


class PulseProbeSpectroscopyProgram(ExperimentProgram):
    def _initialize(self, cfg: dict[str, Any]) -> None:
        declare_standard_hardware(self, cfg)
        frequency = self.add_sweep(
            "frequency",
            LinearSweep(
                cfg["frequency_start"],
                cfg["frequency_stop"],
                cfg["steps"],
            ),
        )
        self.add_pulse(
            "qubit_pulse",
            gen="qubit",
            style="const",
            length=cfg["probe_len"],
            frequency=frequency,
            gain=cfg["qubit_gain"],
        )
        self.add_pulse(
            "res_pulse",
            gen="res",
            style="const",
            length=cfg["res_len"],
            frequency=cfg["f_res"],
            phase=cfg.get("res_phase", 0.0),
            gain=cfg["res_gain"],
            readout=True,
        )

    def _body(self, cfg: dict[str, Any]) -> None:
        self.play("qubit_pulse", at=0)
        self.play("res_pulse", at=cfg.get("res_start", 0))
        self.trigger(
            "ro",
            trigger_delay=cfg.get("trigger_delay"),
        )


class PowerRabiProgram(ExperimentProgram):
    def _initialize(self, cfg: dict[str, Any]) -> None:
        declare_standard_hardware(self, cfg)
        gain = self.add_sweep(
            "gain",
            LinearSweep(
                cfg["gain_start"],
                cfg["gain_stop"],
                cfg["steps"],
            ),
        )
        self.add_pulse(
            "qubit_pulse",
            gen="qubit",
            style="gaussian",
            length=cfg["qubit_len"],
            sigma=cfg["qubit_sigma"],
            frequency=cfg["f_ge"],
            gain=gain,
        )
        self.add_pulse(
            "res_pulse",
            gen="res",
            style="const",
            length=cfg["res_len"],
            frequency=cfg["f_res"],
            gain=cfg["res_gain"],
            readout=True,
        )

    def _body(self, cfg: dict[str, Any]) -> None:
        self.play("qubit_pulse")
        self.delay_auto(cfg.get("qubit_to_readout", 40 * ns))
        self.play("res_pulse")
        self.trigger(
            "ro",
            trigger_delay=cfg.get("trigger_delay"),
        )


class T1Program(ExperimentProgram):
    def _initialize(self, cfg: dict[str, Any]) -> None:
        declare_standard_hardware(self, cfg)
        delay = self.add_sweep(
            "delay",
            LinearSweep(
                cfg["delay_start"],
                cfg["delay_stop"],
                cfg["steps"],
            ),
        )
        self.delay_sweep = delay
        self.add_pulse(
            "pi_pulse",
            gen="qubit",
            style="gaussian",
            length=cfg["pi_len"],
            sigma=cfg["pi_sigma"],
            frequency=cfg["f_ge"],
            gain=cfg["pi_gain"],
        )
        self.add_pulse(
            "res_pulse",
            gen="res",
            style="const",
            length=cfg["res_len"],
            frequency=cfg["f_res"],
            gain=cfg["res_gain"],
            readout=True,
        )

    def _body(self, cfg: dict[str, Any]) -> None:
        self.play("pi_pulse")
        self.delay_auto(self.delay_sweep)
        self.play("res_pulse")
        self.trigger(
            "ro",
            trigger_delay=cfg.get("trigger_delay"),
        )


class SingleShotProgram(ExperimentProgram):
    def _initialize(self, cfg: dict[str, Any]) -> None:
        declare_standard_hardware(self, cfg)
        state = self.add_sweep(
            "state",
            ValuesSweep(("g", "e")),
        )
        self.state_sweep = state
        self.add_pulse(
            "pi_pulse",
            gen="qubit",
            style="gaussian",
            length=cfg["pi_len"],
            sigma=cfg["pi_sigma"],
            frequency=cfg["f_ge"],
            gain=cfg["pi_gain"],
        )
        self.add_pulse(
            "res_pulse",
            gen="res",
            style="const",
            length=cfg["res_len"],
            frequency=cfg["f_res"],
            gain=cfg["res_gain"],
            readout=True,
        )

    def _body(self, cfg: dict[str, Any]) -> None:
        self.play("pi_pulse", when=("state", "e"))
        self.play("res_pulse", at=cfg["pi_len"] + cfg.get("readout_delay", 40 * ns))
        self.trigger(
            "ro",
            trigger_delay=cfg.get("trigger_delay"),
        )


class CavityRingdownProgram(ExperimentProgram):
    """Fill a cavity, then acquire its free ring-down after the drive stops."""

    def _initialize(self, cfg: dict[str, Any]) -> None:
        self.declare_gen(
            "cavity_drive",
            ch=cfg.get("awg_ch", 3),
            amplitude_vpp=cfg.get("channel_amplitude_vpp", 0.5),
        )
        self.declare_readout(
            "ro",
            adc_channel=cfg.get("adc_channel", "CHA"),
            length=cfg["acquire_length"],
            demod_freq=cfg["frequency"],
            waveform_ch=cfg.get("awg_ch", 3),
            marker_channel=cfg.get("marker_ch", 1),
            integrate_time=cfg.get(
                "integrate_time",
                cfg["acquire_length"],
            ),
        )
        self.add_pulse(
            "cavity_fill",
            gen="cavity_drive",
            style=cfg.get("pulse_style", "gaussian_square"),
            length=cfg["drive_length"],
            edge_sigma=cfg.get("edge_sigma", 20 * ns),
            frequency=cfg["frequency"],
            phase=cfg.get("phase", 0.0),
            gain=cfg["drive_gain"],
        )

    def _body(self, cfg: dict[str, Any]) -> None:
        self.play("cavity_fill", at=0)
        self.trigger(
            "ro",
            trigger_delay=(
                cfg["drive_length"] + cfg.get("ringdown_guard", 0.0)
            ),
        )
