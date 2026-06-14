"""Notebook-friendly instrument registry and safety helper."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable, Mapping
from typing import Any

from .base import InstrumentSpec, LimitMap, merge_limits


class TextReport:
    """Plain-text report that displays cleanly in notebooks and REPLs."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return self.text

    def _repr_pretty_(self, printer, cycle: bool) -> None:
        printer.text("..." if cycle else self.text)


class BaseInstrumentManager:
    """
    Register lab instruments under friendly names and expose one control API.

    Examples
    --------
    >>> inst = BaseInstrumentManager()
    >>> inst.add_yoko("q1_flux", "USB0::...", limits={"current": (-3e-3, 3e-3)})
    >>> inst.add_sgs100a("pump", "TCPIP::192.168.0.10::INSTR")
    >>> inst.status
    """

    def __init__(self) -> None:
        self._instruments: dict[str, InstrumentSpec] = {}

    def add_instrument(
        self,
        name: str,
        driver: type | Callable[..., Any],
        address: str,
        *,
        kind: str | None = None,
        limits: Mapping[str, tuple[float, float]] | None = None,
        auto_limits: bool = True,
        notes: str = "",
        **driver_kwargs: Any,
    ) -> Any:
        """Create a driver and register it by name."""

        if name in self._instruments:
            raise ValueError(f"Instrument name already exists: {name!r}")
        obj = driver(address, **driver_kwargs)
        driver_limits = self._driver_limits(obj, auto_limits=auto_limits)
        spec = InstrumentSpec(
            name=name,
            kind=kind or getattr(obj, "KIND", obj.__class__.__name__),
            address=self._driver_address(obj, address),
            driver=obj,
            limits=merge_limits(driver_limits, limits),
            notes=notes,
        )
        self._instruments[name] = spec
        return obj

    def add_yoko(
        self,
        name: str,
        address: str,
        *,
        limits: Mapping[str, tuple[float, float]] | None = None,
        auto_limits: bool = True,
        voltage_ramp_step: float | None = None,
        current_ramp_step: float | None = None,
        ramp_interval: float | None = None,
        notes: str = "",
    ) -> Any:
        """Add a Yokogawa GS200 DC source."""

        import pyvisa as visa

        from .yoko import YOKOGS200

        rm = visa.ResourceManager()
        yoko = self.add_instrument(
            name,
            lambda addr: YOKOGS200(addr, rm),
            address,
            kind="yoko",
            limits=limits,
            auto_limits=auto_limits,
            notes=notes,
        )
        if voltage_ramp_step is not None:
            yoko.voltage_ramp_step = voltage_ramp_step
        if current_ramp_step is not None:
            yoko.current_ramp_step = current_ramp_step
        if ramp_interval is not None:
            yoko.ramp_interval = ramp_interval
        return yoko

    def add_sgs100a(
        self,
        name: str,
        address: str,
        *,
        limits: Mapping[str, tuple[float, float]] | None = None,
        auto_limits: bool = True,
        notes: str = "",
    ) -> Any:
        """Add a Rohde & Schwarz SGS100A RF source."""

        from .sgs100a import RohdeSchwarzSGS100A

        return self.add_instrument(
            name,
            RohdeSchwarzSGS100A,
            address,
            kind="sgs100a",
            limits=limits,
            auto_limits=auto_limits,
            notes=notes,
        )

    def configure_sgs100a(
        self,
        name: str,
        *,
        reference_source: str = "EXT",
        reference_frequency_hz: float | None = 10e6,
        lo_output: bool = False,
        iq_modulation: bool | None = None,
    ) -> None:
        """Configure SGS100A clock locking, LO output, and optional IQ input."""

        driver = self.get(name)
        configure_reference = getattr(driver, "configure_reference_clock", None)
        configure_lo_output = getattr(driver, "configure_lo_output", None)
        if not callable(configure_reference) or not callable(configure_lo_output):
            raise TypeError(f"{name!r} is not an SGS100A-compatible instrument")

        external_frequency = (
            reference_frequency_hz
            if str(reference_source).upper() == "EXT"
            else None
        )
        configure_reference(
            reference_source,
            external_frequency_hz=external_frequency,
        )
        configure_lo_output(lo_output)
        if iq_modulation is not None:
            driver.IQ_state = iq_modulation

    def add_mg3692(
        self,
        name: str,
        address: str,
        *,
        limits: Mapping[str, tuple[float, float]] | None = None,
        auto_limits: bool = True,
        notes: str = "",
    ) -> Any:
        """Add an Anritsu MG3692 RF source."""

        from .mg3692 import AnritsuMG3692

        return self.add_instrument(
            name,
            AnritsuMG3692,
            address,
            kind="mg3692",
            limits=limits,
            auto_limits=auto_limits,
            notes=notes,
        )

    # Notebook aliases matching the compact style often used at the bench.
    addyoko = add_yoko
    addsgs100a = add_sgs100a
    addsgs100 = add_sgs100a
    addmg3692 = add_mg3692

    def get(self, name: str) -> Any:
        return self._instruments[name].driver

    def spec(self, name: str) -> InstrumentSpec:
        return self._instruments[name]

    @property
    def names(self) -> list[str]:
        return list(self._instruments)

    @property
    def status(self) -> TextReport:
        """Return a printable status table for all registered instruments."""

        if not self._instruments:
            return TextReport("No instruments registered.")
        return TextReport(
            "\n".join(self._format_status_line(spec) for spec in self._instruments.values())
        )

    def help(self, name: str | None = None) -> TextReport:
        """Return output ranges and common commands."""

        if name is None:
            if not self._instruments:
                return TextReport("No instruments registered.")
            return TextReport("\n\n".join(str(self.help(item)) for item in self._instruments))

        spec = self.spec(name)
        lines = [
            f"{spec.kind}: {spec.name}",
            f"address: {spec.address}",
            "ranges:",
        ]
        if spec.limits:
            for param, (low, high) in spec.limits.items():
                lines.append(f"  {param}: {low:g} to {high:g}")
        else:
            lines.append("  no limits registered")
        snapshot = self._safe_snapshot(spec.driver)
        if snapshot:
            lines.append("snapshot:")
            for key, value in snapshot.items():
                lines.append(f"  {key}: {value}")
        if spec.notes:
            lines.append(f"notes: {spec.notes}")
        lines.append(f"common: inst.get({name!r}), inst.on({name!r}), inst.off({name!r})")
        return TextReport("\n".join(lines))

    def limits(self, name: str | None = None) -> dict[str, LimitMap] | LimitMap:
        if name is not None:
            return dict(self.spec(name).limits)
        return {item: dict(spec.limits) for item, spec in self._instruments.items()}

    def set(self, name: str, parameter: str, value: Any) -> None:
        """Set a driver property after checking registered limits."""

        spec = self.spec(name)
        self._validate_range(spec, parameter, value)
        if not hasattr(spec.driver, parameter):
            raise AttributeError(f"{name!r} has no settable parameter {parameter!r}")
        setattr(spec.driver, parameter, value)

    def set_yoko(self, name: str, value: float, mode: str = "current") -> None:
        spec = self.spec(name)
        if mode not in {"current", "voltage"}:
            raise ValueError("mode must be 'current' or 'voltage'")
        self._validate_range(spec, mode, value)
        driver = spec.driver
        if hasattr(driver, "mode"):
            driver.mode = mode
        setattr(driver, mode, value)

    def value(self, name: str) -> dict[str, Any]:
        """Return the primary output value for one instrument."""

        spec = self.spec(name)
        return self._primary_value(spec)

    def set_value(self, name: str, value: float, *, mode: str | None = None) -> None:
        """
        Set the primary output value.

        For Yoko/DC sources this sets current or voltage. For RF sources this
        sets power. Use ``set(name, "frequency", value)`` for RF frequency.
        """

        spec = self.spec(name)
        if spec.kind == "yoko" or self._is_dc_source(spec.driver):
            self.set_yoko(name, value, mode=mode or "current")
            return
        if hasattr(spec.driver, "power"):
            self.set(name, "power", value)
            return
        raise AttributeError(f"{name!r} has no known primary output value.")

    def set_values(
        self,
        targets: Mapping[str, float | Mapping[str, Any]],
        *,
        mode: str | None = None,
        parallel: bool = True,
        max_workers: int | None = None,
    ) -> TextReport:
        """
        Set multiple Yoko/DC source output values.

        Examples
        --------
        >>> inst.set_values({
        ...     "q1_flux": {"value": 0.5e-3, "mode": "current"},
        ...     "q2_flux": {"value": -0.2e-3, "mode": "current"},
        ... })

        If a target value is a plain number, the shared ``mode`` argument is
        used. Parallel mode uses threads so independent Yoko ramps can happen
        at the same time from one notebook cell.

        This helper is intentionally limited to DC/Yoko sources, because those
        ramps are slow enough to benefit from parallel execution.
        """

        normalized = self._normalize_targets(targets, mode=mode)
        for name, spec in normalized.items():
            inst_spec = self.spec(name)
            if not (inst_spec.kind == "yoko" or self._is_dc_source(inst_spec.driver)):
                raise TypeError(
                    f"{name!r} is not a Yoko/DC source. Use set_value() or set() "
                    "for RF sources."
                )
            self._validate_set_value(name, spec["value"], spec.get("mode"))

        if parallel and len(normalized) > 1:
            return self._set_values_parallel(normalized, max_workers=max_workers)

        lines = []
        for name, spec in normalized.items():
            self.set_value(name, spec["value"], mode=spec.get("mode"))
            lines.append(f"{name}: set {self.value(name)}")
        return TextReport("\n".join(lines))

    def configure_ramp(
        self,
        name: str,
        *,
        voltage_step: float | None = None,
        current_step: float | None = None,
        interval: float | None = None,
    ) -> None:
        """Configure ramp settings for a Yoko/DC source."""

        driver = self.get(name)
        configure_ramp = getattr(driver, "configure_ramp", None)
        if callable(configure_ramp):
            configure_ramp(
                voltage_step=voltage_step,
                current_step=current_step,
                interval=interval,
            )
            return
        if voltage_step is not None and hasattr(driver, "voltage_ramp_step"):
            driver.voltage_ramp_step = voltage_step
        if current_step is not None and hasattr(driver, "current_ramp_step"):
            driver.current_ramp_step = current_step
        if interval is not None and hasattr(driver, "ramp_interval"):
            driver.ramp_interval = interval

    def ramp(self, name: str) -> dict[str, Any]:
        """Return ramp settings for a Yoko/DC source."""

        driver = self.get(name)
        ramp_rate = getattr(driver, "ramp_rate", None)
        if ramp_rate is not None:
            return dict(ramp_rate)
        return {
            "voltage_step": getattr(driver, "voltage_ramp_step", None),
            "current_step": getattr(driver, "current_ramp_step", None),
            "interval": getattr(driver, "ramp_interval", None),
        }

    def on(self, name: str) -> None:
        self.get(name).on()

    def off(self, name: str) -> None:
        self.get(name).off()

    def close(self, name: str | None = None) -> None:
        """Close one instrument or every registered instrument."""

        names = [name] if name is not None else list(self._instruments)
        for item in names:
            driver = self.get(item)
            close = getattr(driver, "close", None)
            if callable(close):
                close()

    def _validate_range(self, spec: InstrumentSpec, parameter: str, value: Any) -> None:
        if parameter not in spec.limits:
            return
        low, high = spec.limits[parameter]
        if not (low <= float(value) <= high):
            raise ValueError(
                f"{spec.name}.{parameter}={value:g} is outside allowed range "
                f"{low:g} to {high:g}"
            )

    def _validate_set_value(self, name: str, value: float, mode: str | None) -> None:
        spec = self.spec(name)
        if spec.kind == "yoko" or self._is_dc_source(spec.driver):
            target_mode = mode or "current"
            if target_mode not in {"current", "voltage"}:
                raise ValueError(f"{name}: mode must be 'current' or 'voltage'")
            self._validate_range(spec, target_mode, value)
            return
        if hasattr(spec.driver, "power"):
            self._validate_range(spec, "power", value)
            return
        raise AttributeError(f"{name!r} has no known primary output value.")

    def _normalize_targets(
        self,
        targets: Mapping[str, float | Mapping[str, Any]],
        *,
        mode: str | None,
    ) -> dict[str, dict[str, Any]]:
        normalized = {}
        for name, target in targets.items():
            if name in normalized:
                raise ValueError(f"Duplicate target instrument: {name!r}")
            self.spec(name)
            if isinstance(target, Mapping):
                if "value" not in target:
                    raise ValueError(f"{name}: target mapping must contain 'value'")
                normalized[name] = {
                    "value": target["value"],
                    "mode": target.get("mode", mode),
                }
            else:
                normalized[name] = {"value": target, "mode": mode}
        return normalized

    def _set_values_parallel(
        self,
        targets: Mapping[str, Mapping[str, Any]],
        *,
        max_workers: int | None,
    ) -> TextReport:
        workers = max_workers or len(targets)
        lines = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self.set_value, name, spec["value"], mode=spec.get("mode")): name
                for name, spec in targets.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                future.result()
                lines.append(f"{name}: set {self.value(name)}")
        lines.sort()
        return TextReport("\n".join(lines))

    def _format_status_line(self, spec: InstrumentSpec) -> str:
        driver = spec.driver
        output = self._safe_get(driver, "output")
        if output is None:
            output = self._safe_get(driver, "status")
        value = self._status_value(spec)
        return (
            f"{spec.kind}: {spec.name} address: {spec.address} | "
            f"output: {output if output is not None else 'unknown'} | "
            f"value: {value}"
        )

    def _status_value(self, spec: InstrumentSpec) -> str:
        primary = self._primary_value(spec)
        pieces = []
        if primary:
            label = primary.get("parameter", "value")
            value = primary.get("value")
            unit = primary.get("unit", "")
            pieces.append(f"{label}={value} {unit}".strip())

        driver = spec.driver
        frequency = self._safe_get(driver, "frequency")
        if frequency is not None:
            pieces.append(f"frequency={frequency} Hz")
        if spec.kind == "yoko" or self._is_dc_source(driver):
            ramp = self.ramp(spec.name)
            if any(value is not None for value in ramp.values()):
                pieces.append(
                    "ramp="
                    f"I_step={ramp.get('current_step')}, "
                    f"V_step={ramp.get('voltage_step')}, "
                    f"interval={ramp.get('interval')}"
                )
        return ", ".join(pieces) if pieces else "unknown"

    def _primary_value(self, spec: InstrumentSpec) -> dict[str, Any]:
        driver = spec.driver
        get_value = getattr(driver, "GetValue", None)
        if callable(get_value):
            try:
                info = get_value()
                mode = self._safe_get(driver, "mode")
                parameter = mode if mode in {"current", "voltage"} else "value"
                return {
                    "parameter": parameter,
                    "value": info.get("value"),
                    "unit": info.get("unit"),
                }
            except Exception as exc:
                return {"parameter": "value", "value": f"error({exc})", "unit": ""}

        power = self._safe_get(driver, "power")
        if power is not None:
            return {"parameter": "power", "value": power, "unit": "dBm"}
        return {}

    def _safe_snapshot(self, driver: Any) -> dict[str, Any]:
        snapshot = getattr(driver, "snapshot", None)
        if callable(snapshot):
            try:
                return dict(snapshot())
            except Exception:
                return {}
        return {}

    def _safe_get(self, driver: Any, attr: str) -> Any:
        try:
            value = getattr(driver, attr)
        except Exception:
            return None
        if callable(value):
            try:
                return value()
            except TypeError:
                return None
            except Exception:
                return None
        return value

    def _driver_limits(self, driver: Any, *, auto_limits: bool = True) -> LimitMap:
        if auto_limits:
            discover_limits = getattr(driver, "discover_limits", None)
            if callable(discover_limits):
                try:
                    discovered = discover_limits()
                    if discovered:
                        return dict(discovered)
                except Exception as exc:
                    print(
                        f"[InstrumentManager] Could not auto-discover limits for "
                        f"{driver.__class__.__name__}: {exc}. Using defaults."
                    )
        get_limits = getattr(driver, "get_limits", None)
        if callable(get_limits):
            return dict(get_limits())
        return dict(getattr(driver, "DEFAULT_LIMITS", {}))

    def _driver_address(self, driver: Any, fallback: str) -> str:
        for attr in ("resource_name", "VISAaddress", "address"):
            value = getattr(driver, attr, None)
            if value:
                return str(value)
        instrument = getattr(driver, "instrument", None)
        value = getattr(instrument, "resource_name", None)
        return str(value or fallback)

    def _is_dc_source(self, driver: Any) -> bool:
        return all(hasattr(driver, attr) for attr in ("current", "voltage"))


InstrumentManager = BaseInstrumentManager
