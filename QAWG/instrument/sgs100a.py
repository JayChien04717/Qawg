from __future__ import annotations

from typing import Literal, Tuple, Union

import pyvisa

from .base import RFSourceInstrument

ON_OFF_MAP = {
    "on": "1", "1": "1", 1: "1", True: "1",
    "off": "0", "0": "0", 0: "0", False: "0",
}
ON_OFF_MAP_INV = {"1": "on", "0": "off"}

PULSE_SOURCE_VALS = {"INT", "EXT"}
REF_LO_SOURCE_VALS = {"INT", "EXT"}
REF_LO_OUT_VALS = {"REF", "LO", "OFF"}
REF_FREQ_VALS = {"10MHZ", "100MHZ", "1000MHZ"}
TRIG_MODE_VALS = {"SVAL", "SNVAL", "PVO", "PET", "PEMS"}
PULSE_MODE_VALS = {"SING", "DOUB", "SINGLE", "DOUBLE"}
POLARITY_VALS = {"NORM", "INV", "NORMAL", "INVERTED"}
IMPEDANCE_VALS = {"G50", "G10K"}
SLOPE_VALS = {"NEG", "POS", "NEGATIVE", "POSITIVE"}
TRIG_MODE_EXT_VALS = {"AUTO", "EXT", "EGAT", "EXTERNAL", "EGATE"}
OP_MODE_VALS = {"NORMAL", "BBBYPASS"}


class RohdeSchwarzSGS100A(RFSourceInstrument):
    """Pure PyVISA driver for the Rohde & Schwarz SGS100A signal generator."""

    KIND = "sgs100a"
    MODEL = "Rohde & Schwarz SGS100A"
    DEFAULT_LIMITS = {
        "frequency": (1e6, 20e9),
        "phase": (0, 360),
        "power": (-120, 25),
        "i_offset": (-10, 10),
        "q_offset": (-10, 10),
        "iq_gain_imbalance": (-1, 1),
        "iq_angle": (-8, 8),
        "pulsemod_delay": (0, 100),
    }

    def __init__(self, address: str) -> None:
        self.address = address
        if "::" not in address:
            self.resource_name = f"TCPIP::{address}::INSTR"
        else:
            self.resource_name = address
        self.rm = pyvisa.ResourceManager()
        try:
            self.instrument = self.rm.open_resource(self.resource_name)
        except pyvisa.Error as e:
            print(f"Could not connect to {self.resource_name}. Error: {e}")
            raise
        self.instrument.read_termination = "\n"
        self.instrument.write_termination = "\n"
        self.connect_message()

    def connect_message(self) -> None:
        try:
            idn = self.instrument.query("*IDN?")
            print(f"Connected to: {idn.strip()}")
        except pyvisa.Error as e:
            print(f"Could not query IDN. Error: {e}")

    def idn(self) -> str:
        return self.query("*IDN?")

    def close(self) -> None:
        print(f"Disconnecting from {self.instrument.resource_name}")
        self.instrument.close()
        self.rm.close()

    def write(self, cmd: str) -> None:
        self.instrument.write(cmd)

    def query(self, cmd: str) -> str:
        return self.instrument.query(cmd).strip()

    def reset(self) -> None:
        print("Resetting instrument...")
        self.write("*RST")

    def run_self_tests(self) -> str:
        print("Running self-tests...")
        result = self.query("*TST?")
        print(f"Self-test result: {result}")
        return result

    def check_error(self) -> str:
        err_msg = self.query("SYST:ERR?")
        print(f"Instrument Status: {err_msg}")
        return err_msg

    def get_limit(self, parameter: str) -> Tuple[float, float]:
        param_lower = parameter.lower()
        if param_lower in self.DEFAULT_LIMITS:
            return self.DEFAULT_LIMITS[param_lower]
        raise ValueError(f"Limits not defined for parameter '{parameter}'.")

    def get_limits(self) -> dict:
        return dict(self.DEFAULT_LIMITS)

    def discover_limits(self) -> dict:
        limits = self.get_limits()
        queries = {
            "frequency": ("SOUR:FREQ? MIN", "SOUR:FREQ? MAX"),
            "power": ("SOUR:POW? MIN", "SOUR:POW? MAX"),
        }
        for key, (min_query, max_query) in queries.items():
            try:
                limits[key] = (float(self.query(min_query)), float(self.query(max_query)))
            except Exception:
                pass
        return limits

    def _validate_and_write(self, cmd_template: str, value: str, valid_set: set, name: str) -> None:
        val_upper = str(value).upper()
        if val_upper not in valid_set:
            raise ValueError(f"Invalid {name} value: {value}. Allowed: {valid_set}")
        self.write(cmd_template.format(val_upper))

    def _map_and_write(self, cmd_template: str, value: Union[str, int, bool], name: str) -> None:
        try:
            lookup_value = value.lower() if isinstance(value, str) else value
            mapped_val = ON_OFF_MAP[lookup_value]
            self.write(cmd_template.format(mapped_val))
        except KeyError:
            raise ValueError(f"Invalid {name} value: {value}. Use 'on' or 'off'.")

    def _query_and_map(self, cmd: str) -> str:
        val = self.query(cmd)
        return ON_OFF_MAP_INV.get(val, f"unknown_val_{val}")

    @property
    def frequency(self) -> float:
        return float(self.query("SOUR:FREQ?"))

    @frequency.setter
    def frequency(self, value: float) -> None:
        min_v, max_v = self.get_limit("frequency")
        if not (min_v <= value <= max_v):
            print(f"Warning: Frequency {value} Hz is outside driver's expected range ({min_v}, {max_v})")
        self.write(f"SOUR:FREQ {value:.2f}")

    @property
    def phase(self) -> float:
        return float(self.query("SOUR:PHAS?"))

    @phase.setter
    def phase(self, value: float) -> None:
        min_v, max_v = self.get_limit("phase")
        if not (min_v <= value <= max_v):
            print(f"Warning: Phase {value} deg is outside driver's expected range ({min_v}, {max_v})")
        self.write(f"SOUR:PHAS {value:.2f}")

    @property
    def power(self) -> float:
        return float(self.query("SOUR:POW?"))

    @power.setter
    def power(self, value: float) -> None:
        min_v, max_v = self.get_limit("power")
        if not (min_v <= value <= max_v):
            print(f"Warning: Power {value} dBm is outside driver's expected range ({min_v}, {max_v})")
        self.write(f"SOUR:POW {value:.2f}")

    @property
    def status(self) -> str:
        return self._query_and_map(":OUTP:STAT?")

    @status.setter
    def status(self, value: Union[str, int, bool]) -> None:
        self._map_and_write(":OUTP:STAT {}", value, "status")

    def on(self) -> None:
        self.status = "on"

    def off(self) -> None:
        self.status = "off"

    def snapshot(self) -> dict:
        return {
            "output": self.status,
            "frequency": self.frequency,
            "power": self.power,
            "iq": self.IQ_state,
            "pulsemod": self.pulsemod_state,
            "reference_source": self.ref_osc_source,
            "reference_external_frequency": self.ref_osc_external_freq,
            "reference_lo_output": self.ref_lo_output,
        }

    @property
    def IQ_state(self) -> str:
        return self._query_and_map(":IQ:STAT?")

    @IQ_state.setter
    def IQ_state(self, value: Union[str, int, bool]) -> None:
        self._map_and_write(":IQ:STAT {}", value, "IQ_state")

    @property
    def pulsemod_state(self) -> str:
        return self._query_and_map(":SOUR:PULM:STAT?")

    @pulsemod_state.setter
    def pulsemod_state(self, value: Union[str, int, bool]) -> None:
        self._map_and_write(":SOUR:PULM:STAT {}", value, "pulsemod_state")

    @property
    def pulsemod_source(self) -> str:
        return self.query("SOUR:PULM:SOUR?")

    @pulsemod_source.setter
    def pulsemod_source(self, value: Literal["INT", "EXT", "int", "ext"]) -> None:
        self._validate_and_write("SOUR:PULM:SOUR {}", value, PULSE_SOURCE_VALS, "pulsemod_source")

    @property
    def ref_osc_source(self) -> str:
        return self.query("SOUR:ROSC:SOUR?")

    @ref_osc_source.setter
    def ref_osc_source(self, value: Literal["INT", "EXT", "int", "ext"]) -> None:
        self._validate_and_write("SOUR:ROSC:SOUR {}", value, REF_LO_SOURCE_VALS, "ref_osc_source")

    @property
    def lo_source(self) -> str:
        return self.query("SOUR:LOSC:SOUR?")

    @lo_source.setter
    def lo_source(self, value: Literal["INT", "EXT", "int", "ext"]) -> None:
        self._validate_and_write("SOUR:LOSC:SOUR {}", value, REF_LO_SOURCE_VALS, "lo_source")

    @property
    def ref_lo_output(self) -> str:
        return self.query("CONN:REFL:OUTP?")

    @ref_lo_output.setter
    def ref_lo_output(self, value: Literal["REF", "LO", "OFF", "ref", "lo", "off"]) -> None:
        self._validate_and_write(
            "CONN:REFL:OUTP {}",
            value,
            REF_LO_OUT_VALS,
            "ref_lo_output",
        )

    @property
    def ref_osc_output_freq(self) -> str:
        return self.query("SOUR:ROSC:OUTP:FREQ?")

    @ref_osc_output_freq.setter
    def ref_osc_output_freq(
        self,
        value: float | Literal["10MHz", "100MHz", "1000MHz"],
    ) -> None:
        normalized = self._normalize_reference_frequency(value)
        self._validate_and_write(
            "SOUR:ROSC:OUTP:FREQ {}",
            normalized,
            REF_FREQ_VALS,
            "ref_osc_output_freq",
        )

    @property
    def ref_osc_external_freq(self) -> str:
        return self.query("SOUR:ROSC:EXT:FREQ?")

    @ref_osc_external_freq.setter
    def ref_osc_external_freq(
        self,
        value: float | Literal["10MHz", "100MHz", "1000MHz"],
    ) -> None:
        normalized = self._normalize_reference_frequency(value)
        self._validate_and_write(
            "SOUR:ROSC:EXT:FREQ {}",
            normalized,
            REF_FREQ_VALS,
            "ref_osc_external_freq",
        )

    @staticmethod
    def _normalize_reference_frequency(value: float | str) -> str:
        if isinstance(value, str):
            normalized = value.replace(" ", "").upper()
        else:
            frequencies = {
                10e6: "10MHZ",
                100e6: "100MHZ",
                1000e6: "1000MHZ",
            }
            try:
                normalized = frequencies[float(value)]
            except KeyError as exc:
                raise ValueError(
                    "Reference frequency must be 10e6, 100e6, 1000e6, "
                    "or the corresponding MHz string."
                ) from exc
        if normalized not in REF_FREQ_VALS:
            raise ValueError(
                f"Invalid reference frequency: {value}. "
                f"Allowed: {sorted(REF_FREQ_VALS)}"
            )
        return normalized

    def configure_reference_clock(
        self,
        source: Literal["INT", "EXT", "int", "ext"],
        *,
        external_frequency_hz: float | None = None,
    ) -> None:
        """
        Select the SGS100A internal or external reference oscillator.

        When ``source`` is external, provide the frequency connected to the
        REF IN connector so the instrument can lock to it.
        """

        normalized_source = str(source).upper()
        if normalized_source == "EXT":
            if external_frequency_hz is None:
                raise ValueError(
                    "external_frequency_hz is required for an external reference"
                )
            self.ref_osc_external_freq = external_frequency_hz
        elif external_frequency_hz is not None:
            raise ValueError(
                "external_frequency_hz can only be used with source='EXT'"
            )
        self.ref_osc_source = normalized_source

    def configure_lo_output(
        self,
        enabled: bool = True,
        *,
        mode: Literal["LO", "REF", "lo", "ref"] = "LO",
        reference_frequency_hz: float | None = None,
    ) -> None:
        """
        Route LO, reference clock, or no signal to the REF/LO OUT connector.

        ``reference_frequency_hz`` is only valid when ``mode='REF'``.
        """

        normalized_mode = str(mode).upper()
        if not enabled:
            if reference_frequency_hz is not None:
                raise ValueError(
                    "reference_frequency_hz cannot be set while output is disabled"
                )
            self.ref_lo_output = "OFF"
            return
        if normalized_mode == "REF":
            if reference_frequency_hz is None:
                raise ValueError(
                    "reference_frequency_hz is required for reference output"
                )
            self.ref_osc_output_freq = reference_frequency_hz
        elif reference_frequency_hz is not None:
            raise ValueError(
                "reference_frequency_hz can only be used with mode='REF'"
            )
        self.ref_lo_output = normalized_mode

    # Backwards-compatible property names used by earlier notebooks.
    LO_source = lo_source
    ref_LO_out = ref_lo_output

    @property
    def IQ_impairments(self) -> str:
        return self._query_and_map(":SOUR:IQ:IMP:STAT?")

    @IQ_impairments.setter
    def IQ_impairments(self, value: Union[str, int, bool]) -> None:
        self._map_and_write(":SOUR:IQ:IMP:STAT {}", value, "IQ_impairments")

    @property
    def I_offset(self) -> float:
        return float(self.query("SOUR:IQ:IMP:LEAK:I?"))

    @I_offset.setter
    def I_offset(self, value: float) -> None:
        min_v, max_v = self.get_limit("i_offset")
        if not (min_v <= value <= max_v):
            print(f"Warning: I offset {value}% is outside expected range ({min_v}, {max_v})")
        self.write(f"SOUR:IQ:IMP:LEAK:I {value:.2f}")

    @property
    def Q_offset(self) -> float:
        return float(self.query("SOUR:IQ:IMP:LEAK:Q?"))

    @Q_offset.setter
    def Q_offset(self, value: float) -> None:
        min_v, max_v = self.get_limit("q_offset")
        if not (min_v <= value <= max_v):
            print(f"Warning: Q offset {value}% is outside expected range ({min_v}, {max_v})")
        self.write(f"SOUR:IQ:IMP:LEAK:Q {value:.2f}")

    @property
    def IQ_gain_imbalance(self) -> float:
        return float(self.query("SOUR:IQ:IMP:IQR?"))

    @IQ_gain_imbalance.setter
    def IQ_gain_imbalance(self, value: float) -> None:
        min_v, max_v = self.get_limit("iq_gain_imbalance")
        if not (min_v <= value <= max_v):
            print(f"Warning: IQ gain imbalance {value} dB is outside expected range ({min_v}, {max_v})")
        self.write(f"SOUR:IQ:IMP:IQR {value:.2f}")

    @property
    def IQ_angle(self) -> float:
        return float(self.query("SOUR:IQ:IMP:QUAD?"))

    @IQ_angle.setter
    def IQ_angle(self, value: float) -> None:
        min_v, max_v = self.get_limit("iq_angle")
        if not (min_v <= value <= max_v):
            print(f"Warning: IQ angle {value} deg is outside expected range ({min_v}, {max_v})")
        self.write(f"SOUR:IQ:IMP:QUAD {value:.2f}")

    @property
    def trigger_connector_mode(self) -> str:
        return self.query("CONN:TRIG:OMOD?")

    @trigger_connector_mode.setter
    def trigger_connector_mode(self, value: str) -> None:
        self._validate_and_write("CONN:TRIG:OMOD {}", value, TRIG_MODE_VALS, "trigger_connector_mode")

    @property
    def pulsemod_delay(self) -> float:
        return float(self.query("SOUR:PULM:DEL?"))

    @pulsemod_delay.setter
    def pulsemod_delay(self, value: float) -> None:
        min_v, max_v = self.get_limit("pulsemod_delay")
        if not (min_v <= value <= max_v):
            print(f"Warning: Pulse modulation delay {value} s is outside expected range ({min_v}, {max_v})")
        self.write(f"SOUR:PULM:DEL {value:g}")


class RohdeSchwarz_SGS100A(RohdeSchwarzSGS100A):
    """Alias for backwards compatibility."""
    pass
