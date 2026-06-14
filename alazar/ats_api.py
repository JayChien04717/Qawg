"""Minimal ctypes bindings for the official AlazarTech ATS API.

This module has no dependency on QCoDeS. The method names are Python names;
each method directly calls the official ``Alazar...`` function shown inside it.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

from .constants import API_SUCCESS

U8 = ctypes.c_uint8
U32 = ctypes.c_uint32
C_LONG = ctypes.c_long
HANDLE = ctypes.c_void_p
RETURN_CODE = ctypes.c_uint


class AlazarError(RuntimeError):
    """Raised when an ATS-SDK function returns a failure code."""


def _set_signature(
    dll: ctypes.CDLL,
    name: str,
    argument_types: list[type[Any]],
    return_type: type[Any] = RETURN_CODE,
) -> None:
    function = getattr(dll, name)
    function.argtypes = argument_types
    function.restype = return_type


def _apply_signatures(dll: ctypes.CDLL) -> None:
    _set_signature(dll, "AlazarErrorToText", [U32], ctypes.c_char_p)
    _set_signature(dll, "AlazarGetBoardBySystemID", [U32, U32], HANDLE)
    _set_signature(dll, "AlazarGetBoardKind", [HANDLE], U32)
    _set_signature(
        dll, "AlazarGetChannelInfo", [HANDLE, ctypes.POINTER(U32), ctypes.POINTER(U8)]
    )
    _set_signature(dll, "AlazarSetCaptureClock", [HANDLE, U32, U32, U32, U32])
    _set_signature(dll, "AlazarInputControl", [HANDLE, U8, U32, U32, U32])
    _set_signature(dll, "AlazarSetBWLimit", [HANDLE, U32, U32])
    _set_signature(
        dll,
        "AlazarSetTriggerOperation",
        [HANDLE, U32, U32, U32, U32, U32, U32, U32, U32, U32],
    )
    _set_signature(dll, "AlazarSetExternalTrigger", [HANDLE, U32, U32])
    _set_signature(dll, "AlazarSetTriggerDelay", [HANDLE, U32])
    _set_signature(dll, "AlazarSetTriggerTimeOut", [HANDLE, U32])
    _set_signature(dll, "AlazarSetRecordSize", [HANDLE, U32, U32])
    _set_signature(
        dll,
        "AlazarBeforeAsyncRead",
        [HANDLE, U32, C_LONG, U32, U32, U32, U32],
    )
    _set_signature(dll, "AlazarPostAsyncBuffer", [HANDLE, ctypes.c_void_p, U32])
    _set_signature(dll, "AlazarStartCapture", [HANDLE])
    _set_signature(
        dll, "AlazarWaitAsyncBufferComplete", [HANDLE, ctypes.c_void_p, U32]
    )
    _set_signature(dll, "AlazarAbortAsyncRead", [HANDLE])


class ATSApi:
    """Thin wrapper around the official ATSApi DLL."""

    def __init__(self, dll_path: str | Path = r"C:\Windows\System32\ATSApi.dll"):
        self.dll = ctypes.cdll.LoadLibrary(str(dll_path))
        _apply_signatures(self.dll)

    def error_text(self, return_code: int) -> str:
        value = self.dll.AlazarErrorToText(return_code)
        return value.decode(errors="replace") if value else "Unknown ATS error"

    def check(self, return_code: int, function_name: str) -> None:
        if return_code != API_SUCCESS:
            text = self.error_text(return_code)
            raise AlazarError(f"{function_name} failed: {return_code} ({text})")

    def get_board(self, system_id: int, board_id: int) -> int:
        handle = self.dll.AlazarGetBoardBySystemID(system_id, board_id)
        if not handle:
            raise AlazarError(
                f"No Alazar board at system_id={system_id}, board_id={board_id}"
            )
        return handle

    def get_board_kind(self, handle: int) -> int:
        return int(self.dll.AlazarGetBoardKind(handle))

    def get_channel_info(self, handle: int) -> tuple[int, int]:
        memory_samples = U32()
        bits_per_sample = U8()
        code = self.dll.AlazarGetChannelInfo(
            handle, ctypes.byref(memory_samples), ctypes.byref(bits_per_sample)
        )
        self.check(code, "AlazarGetChannelInfo")
        return memory_samples.value, bits_per_sample.value

    def set_capture_clock(
        self, handle: int, source: int, sample_rate: int, edge: int, decimation: int
    ) -> None:
        code = self.dll.AlazarSetCaptureClock(
            handle, source, sample_rate, edge, decimation
        )
        self.check(code, "AlazarSetCaptureClock")

    def input_control(
        self, handle: int, channel: int, coupling: int, input_range: int, impedance: int
    ) -> None:
        code = self.dll.AlazarInputControl(
            handle, channel, coupling, input_range, impedance
        )
        self.check(code, "AlazarInputControl")

    def set_bw_limit(self, handle: int, channel: int, enabled: int) -> None:
        code = self.dll.AlazarSetBWLimit(handle, channel, enabled)
        self.check(code, "AlazarSetBWLimit")

    def set_trigger_operation(
        self,
        handle: int,
        operation: int,
        engine_1: int,
        source_1: int,
        slope_1: int,
        level_1: int,
        engine_2: int,
        source_2: int,
        slope_2: int,
        level_2: int,
    ) -> None:
        code = self.dll.AlazarSetTriggerOperation(
            handle,
            operation,
            engine_1,
            source_1,
            slope_1,
            level_1,
            engine_2,
            source_2,
            slope_2,
            level_2,
        )
        self.check(code, "AlazarSetTriggerOperation")

    def set_external_trigger(self, handle: int, coupling: int, trigger_range: int) -> None:
        code = self.dll.AlazarSetExternalTrigger(handle, coupling, trigger_range)
        self.check(code, "AlazarSetExternalTrigger")

    def set_trigger_delay(self, handle: int, delay_samples: int) -> None:
        code = self.dll.AlazarSetTriggerDelay(handle, delay_samples)
        self.check(code, "AlazarSetTriggerDelay")

    def set_trigger_timeout(self, handle: int, timeout_ticks: int) -> None:
        code = self.dll.AlazarSetTriggerTimeOut(handle, timeout_ticks)
        self.check(code, "AlazarSetTriggerTimeOut")

    def set_record_size(
        self, handle: int, pre_trigger_samples: int, post_trigger_samples: int
    ) -> None:
        code = self.dll.AlazarSetRecordSize(
            handle, pre_trigger_samples, post_trigger_samples
        )
        self.check(code, "AlazarSetRecordSize")

    def before_async_read(
        self,
        handle: int,
        channels: int,
        transfer_offset: int,
        samples_per_record: int,
        records_per_buffer: int,
        records_per_acquisition: int,
        flags: int,
    ) -> None:
        code = self.dll.AlazarBeforeAsyncRead(
            handle,
            channels,
            transfer_offset,
            samples_per_record,
            records_per_buffer,
            records_per_acquisition,
            flags,
        )
        self.check(code, "AlazarBeforeAsyncRead")

    def post_async_buffer(
        self, handle: int, buffer_address: int, buffer_size_bytes: int
    ) -> None:
        code = self.dll.AlazarPostAsyncBuffer(
            handle, ctypes.c_void_p(buffer_address), buffer_size_bytes
        )
        self.check(code, "AlazarPostAsyncBuffer")

    def start_capture(self, handle: int) -> None:
        self.check(self.dll.AlazarStartCapture(handle), "AlazarStartCapture")

    def wait_async_buffer(
        self, handle: int, buffer_address: int, timeout_ms: int
    ) -> None:
        code = self.dll.AlazarWaitAsyncBufferComplete(
            handle, ctypes.c_void_p(buffer_address), timeout_ms
        )
        self.check(code, "AlazarWaitAsyncBufferComplete")

    def abort_async_read(self, handle: int) -> None:
        self.check(self.dll.AlazarAbortAsyncRead(handle), "AlazarAbortAsyncRead")
