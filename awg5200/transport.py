"""Small transport abstraction for SCPI instruments."""

from __future__ import annotations

from typing import Any, Protocol


class ScpiTransport(Protocol):
    def write(self, command: str) -> Any: ...

    def query(self, command: str) -> str: ...

    def write_raw(self, message: bytes) -> Any: ...

    def close(self) -> None: ...


def open_visa_transport(
    resource_name: str,
    timeout_ms: int = 60_000,
    backend: str | None = None,
) -> ScpiTransport:
    """Open a VISA resource without making PyVISA a package import dependency."""
    try:
        import pyvisa
    except ImportError as exc:
        raise RuntimeError(
            "PyVISA is required for hardware access: pip install pyvisa pyvisa-py"
        ) from exc

    manager = pyvisa.ResourceManager(backend) if backend else pyvisa.ResourceManager()
    resource = manager.open_resource(resource_name)
    resource.timeout = timeout_ms
    resource.read_termination = "\n"
    resource.write_termination = "\n"
    return resource
