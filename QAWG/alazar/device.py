"""High-level, independently usable ATS9371 capture boundary."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType

import numpy as np
import numpy.typing as npt

from .ats_api import ATSApi
from .ats9371 import (
    BoardInfo,
    CaptureConfig,
    CaptureSession,
    TriggerConfig,
    abort_capture,
    adc_codes_to_volts,
    arm_capture,
    configure_ats9371,
    free_capture,
    open_ats9371,
    start_capture,
    wait_for_capture,
)


@dataclass(frozen=True)
class CaptureData:
    raw_codes: npt.NDArray[np.uint16]
    records_volts: npt.NDArray[np.float64]
    sample_rate_hz: float
    channels: tuple[int, ...] = (1,)

    def channel_records(self, channel: int) -> npt.NDArray[np.float64]:
        """Return (shots, samples); dual capture axes are (shots, A/B, samples)."""
        if channel not in self.channels:
            raise ValueError("requested channel was not acquired")
        if self.records_volts.ndim == 2:
            return self.records_volts
        return self.records_volts[:, self.channels.index(channel), :]

    @property
    def time_s(self) -> npt.NDArray[np.float64]:
        return np.arange(self.records_volts.shape[-1]) / self.sample_rate_hz


class ArmedCapture:
    """Own one DMA session and guarantee deterministic cleanup."""

    def __init__(
        self,
        api: ATSApi,
        board: BoardInfo,
        config: CaptureConfig,
        session: CaptureSession,
    ) -> None:
        self._api = api
        self._board = board
        self.config = config
        self._session = session
        self._closed = False
        self._collected = False

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("capture session is closed")
        if self._session.started:
            raise RuntimeError("capture session has already started")
        start_capture(self._api, self._session)

    def collect(self) -> CaptureData:
        if self._closed:
            raise RuntimeError("capture session is closed")
        if self._collected:
            raise RuntimeError("capture data has already been collected")
        raw_codes = wait_for_capture(
            self._api,
            self._session,
            self.config.timeout_ms,
        )
        self._collected = True
        records_volts = adc_codes_to_volts(
            raw_codes,
            self._board.bits_per_sample,
            self.config.input_range_volts,
            mask_external_trigger_lsb=(
                self.config.mask_external_trigger_lsb
            ),
        )
        return CaptureData(
            raw_codes=raw_codes,
            records_volts=records_volts,
            sample_rate_hz=self.config.sample_rate_hz,
            channels=self.config.channels,
        )

    def close(self) -> None:
        if self._closed:
            return
        abort_capture(self._api, self._session)
        free_capture(self._session)
        self._closed = True

    def __enter__(self) -> "ArmedCapture":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class Alazar9371:
    """Configure and capture ATS9371 records without requiring an AWG."""

    def __init__(self, api: ATSApi, board: BoardInfo) -> None:
        self.api = api
        self.board = board

    @classmethod
    def connect(
        cls,
        *,
        system_id: int = 1,
        board_id: int = 1,
        dll_path: str = r"C:\Windows\System32\ATSApi.dll",
    ) -> "Alazar9371":
        api = ATSApi(dll_path)
        return cls(api, open_ats9371(api, system_id, board_id))

    def configure(
        self,
        *,
        trigger: TriggerConfig = TriggerConfig(),
        use_external_10mhz_reference: bool = False,
        channel: int,
    ) -> None:
        configure_ats9371(
            self.api,
            self.board,
            trigger,
            use_external_10mhz_reference,
            channel,
        )

    def arm(self, config: CaptureConfig) -> ArmedCapture:
        """Allocate and post DMA buffers without starting the trigger engine."""
        session = arm_capture(self.api, self.board, config)
        return ArmedCapture(self.api, self.board, config, session)

    def capture(
        self,
        config: CaptureConfig,
        *,
        start_callback: Callable[[], None] | None = None,
    ) -> CaptureData:
        """Capture records, optionally starting a trigger source after arming."""
        with self.arm(config) as armed:
            armed.start()
            if start_callback is not None:
                start_callback()
            return armed.collect()

    def __enter__(self) -> "Alazar9371":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # The ATS API exposes board handles owned by the SDK and has no close
        # call.  Individual DMA sessions are owned by ArmedCapture.
        return None
