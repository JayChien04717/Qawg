# QAWG Alazar ATS9371

`QAWG.alazar` 是不依賴 QCoDeS 的 ATS9371 acquisition 與 data processing
package。它直接呼叫官方 `ATSApi.dll`，負責 DMA acquisition、ADC code
conversion、digital downconversion、filter、integration 與 multiplex DSP。

這一層不建立 AWG waveform 或 sequence。AWG 與 Alazar 的執行順序由
`QAWG.awg_alazar.AWGAlazar` 協調。

概念上分成兩個時間範圍：

- **Acquire window**：ATS 收到 marker 後立即擷取的完整 ADC record，用來 debug
  raw trace 與 demodulated trace。
- **Integration window**：在 acquire window 內選一段 `[delay, integration_time]`
  計算每 shot IQ。

## 模組責任

```text
alazar/
├── ats_api.py       ATSApi.dll ctypes bindings
├── device.py        Alazar9371 capture façade 與 capture contracts
├── ats9371.py       low-level board configuration 與 DMA primitives
├── dsp.py           pure NumPy DSP API
└── constants.py     Alazar SDK constants
```

## Hardware assumptions

目前 ATS9371 path 固定使用：

- Windows 與官方 AlazarTech ATS-SDK
- `C:\Windows\System32\ATSApi.dll`
- 1 GS/s
- DC coupling
- 50 ohm input
- +/-400 mV input range
- External TTL trigger
- NPT AutoDMA
- 每個 AWG marker 對應一筆 triggered record

Record length 會向上對齊為至少 256 samples，並符合 128-sample alignment。

典型 wiring：

```text
Readout or mixer IF output -> ATS9371 CHA or CHB
AWG marker output          -> ATS9371 TRIG IN
AWG and ATS reference      -> shared 10 MHz reference
```

連接前請確認 trigger voltage、termination 與 input signal 不會超過
ATS9371 input range。

## 推薦 acquisition 方式

一般使用者應透過 `AWGAlazar` 連接與擷取：

```python
from QAWG import AWGAlazar, us

experiment = AWGAlazar.connect(
    "TCPIP0::192.168.10.171::inst0::INSTR",
    awg_sample_rate_hz=2.5e9,
    alazar_sample_rate_hz=1e9,
    acquire_window_s=1.5 * us,
    trigger_slope="rising",
    trigger_level=140,
    adc_channel="CHB",
)
```

`n_average` 是 acquisition 參數，不是 `connect()` 參數。

`AWGAlazar` 只保留兩個直接 acquisition entry points：

- `acquire_decimate()`：trace/debug mode，回傳 integration window 內的 raw 與
  FIR low-pass demodulated traces。`AWGAlazar` 預設使用 81-tap、20 MHz
  Kaiser FIR。舊的 boxcar/filter-mode branch 已移除。
- `acquire()`：QM-style weighted full demodulation，直接由 raw DDC window 回傳
  per-shot IQ 和 mean IQ，不先對 time-resolved trace 做 moving average。

`acquire_decimate()` 保留完整 ADC sample grid，不做額外降採樣。

預設 rectangular integration weights；若要使用 Tukey、matched filter 或 IQ
rotation weights，可在 `AWGAlazar.connect()` 傳入與 integration window 等長的
real/complex `integration_weights`：

```python
experiment = AWGAlazar.connect(
    awg_resource,
    awg_sample_rate_hz=2.5e9,
    alazar_sample_rate_hz=1e9,
    acquire_window_s=1.5e-6,
    fir_cutoff_hz=20e6,
    fir_number_of_taps=41,
    fir_kaiser_beta=6.0,
    integration_weights=weights,
)
```

底層 pure NumPy helpers 為 `design_lowpass_fir()`、`fir_lowpass_iq()`、
`weighted_integrate_iq()` 與 `weighted_full_demodulate()`。

兩者都會保留 full acquire-window debug data：

```python
experiment.last_records_volts       # (shot, acquire_sample)
experiment.last_downconverted_iq    # (shot, acquire_sample)
experiment.last_time_s              # (acquire_sample,)
```

### Time-resolved traces

```python
trace_time_s, raw_window, downconverted_window = experiment.acquire_decimate(
    n_average=1000,
)
```

Shape：

```text
trace_time_s:         (integration_sample,)
raw_window:           (n_average, integration_sample)
downconverted_window: (n_average, integration_sample)
```

Full acquire-window raw records and downconverted records are retained on the
experiment object as `last_records_volts` and `last_downconverted_iq`.

### Single-frequency integrated IQ

```python
shots, mean_iq = experiment.acquire(n_average=1000)
```

Shape：

```text
shots:  (n_average,)
mean_iq: scalar complex
```

## 獨立使用 Alazar 與 DSP

Alazar 不需要 AWG 或 QAWG coordinator。外部 trigger source 必須在 DMA start
之後啟動，因此可透過 `start_callback` 接入任意 pulse source：

```python
from QAWG.alazar import (
    Alazar9371,
    CaptureConfig,
    downconvert,
    fir_filter,
    integrate,
)

adc = Alazar9371.connect()
adc.configure(channel=1, use_external_10mhz_reference=True)
capture = adc.capture(
    CaptureConfig(
        samples_per_record=2048,
        records_per_acquisition=1000,
        channel=1,
    ),
    start_callback=start_external_trigger_source,
)
baseband = downconvert(capture.records_volts, capture.sample_rate_hz, 50e6)
trace = fir_filter(baseband, capture.sample_rate_hz, cutoff_hz=20e6)
shots = integrate(baseband, start_sample=100, stop_sample=1100)
```

若需自行控制 arm 與 trigger 時序，可使用：

```python
with adc.arm(capture_config) as armed:
    armed.start()
    start_external_trigger_source()
    capture = armed.collect()
```

Canonical DSP API 只有 `downconvert()`、`fir_filter()` 與 `integrate()`。

`correct_interleaving_offsets()` 保留作為 ATS ADC-core offset correction。

## Low-level DMA API

只有需要自訂 acquisition lifecycle 時才直接使用：

```python
from QAWG.alazar import (
    ATSApi,
    CaptureConfig,
    abort_capture,
    arm_capture,
    free_capture,
    open_ats9371,
    start_capture,
    wait_for_capture,
)

api = ATSApi()
board = open_ats9371(api, system_id=1, board_id=1)
config = CaptureConfig(
    sample_rate_hz=1e9,
    samples_per_record=1500,
    records_per_acquisition=1000,
    records_per_buffer=100,
    dma_buffer_count=4,
    channel=1,
)

session = arm_capture(api, board, config)
try:
    start_capture(api, session)
    raw_codes = wait_for_capture(api, session, timeout_ms=5000)
finally:
    abort_capture(api, session)
    free_capture(session)
```

實際流程必須在 `start_capture()` 前先完成 AWG waveform/sequence 準備，並在
ATS arm 後才啟動 AWG。一般情況下由 `AWGAlazar` 處理這個順序較安全。

## Retained acquisition data

最近一次 `acquire()` 或 `acquire_decimate()` 會保留：

```python
experiment.last_raw_codes
experiment.last_records_volts
experiment.last_downconverted_iq
experiment.last_shot_iq
experiment.last_time_s
```

`last_records_volts` 和 `last_downconverted_iq` 都是 full acquire window，用於
debug timing、TOF、ring-up/ring-down 和 unexpected trigger alignment。
`last_shot_iq` 則是 integration window 內積分後的 per-shot IQ。

ATS9371/ATS9373 使用 External Trigger 時會以 trigger state 取代 ADC sample
的 LSB。`AWGAlazar` 預設以有效 11-bit quantization-bin midpoint 重建該位元，
避免 marker high/low 在 voltage trace 形成一個 LSB 的背景階梯。只有在檢查
原始硬體行為時才設定 `mask_external_trigger_lsb=False`。

## Data ownership

- ATS DMA 與 ADC conversion：`ats9371.py`
- DSP primitives 與 pipelines：`dsp.py`
- Result averaging：`ExperimentResult` methods
- AWG start/stop 與 acquisition coordination：`QAWG.awg_alazar.AWGAlazar`
- Experiment sweep 與 sequence record layout：`QAWG.compiler`
