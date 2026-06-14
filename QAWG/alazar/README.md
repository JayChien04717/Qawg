# QAWG Alazar ATS9371

`QAWG.alazar` 是不依賴 QCoDeS 的 ATS9371 acquisition 與 data processing
package。它直接呼叫官方 `ATSApi.dll`，負責 DMA acquisition、ADC code
conversion、digital downconversion、filter、integration 與 multiplex DSP。

這一層不建立 AWG waveform 或 sequence。AWG 與 Alazar 的執行順序由
`QAWG.awg_alazar.AWGAlazar` 協調。

## 模組責任

```text
alazar/
├── ats_api.py       ATSApi.dll ctypes bindings
├── ats9371.py       board configuration、DMA buffers 與 raw acquisition
├── demodulation.py  pure NumPy signal-processing functions
├── processor.py     AlazarProcessor DSP interface
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

### Raw records

自訂 DSP 或 multiplex readout 應先取得未處理 voltage records：

```python
raw_time_s, records = experiment.acquire_records(n_average=1000)
```

Shape：

```text
raw_time_s: (adc_sample,)
records:    (n_average, adc_sample)
```

### Single-frequency integrated IQ

```python
average_iq, baseband = experiment.acquire(n_average=1000)
```

Shape：

```text
average_iq: scalar complex
baseband:   (n_average, adc_sample)
```

### Time-resolved IQ

```python
iq_time_s, average_iq_trace = experiment.acquire_decimate(
    n_average=1000,
    filter_type="boxcar",
)
```

可用 filter：

- `"boxcar"`
- `"butterworth"` 或 `"butter"`
- `"elliptic"` 或 `"ellip"`
- `"notch"` 或 `"adaptive_notch"`

SciPy filters 需要安裝：

```powershell
pip install scipy
```

## AlazarProcessor

`AlazarProcessor` 只處理已轉成 volts 的 records，不操作硬體：

```python
from QAWG.alazar import AlazarProcessor

processor = AlazarProcessor(sample_rate_hz=1e9)
```

### Digital downconversion 與 integration

```python
baseband, shot_iq, average_iq = processor.process_integrate(
    records_volts=records,
    tone_frequency_hz=50e6,
    reference_phase_radians=0.0,
    integrate_start=100,
    integrate_stop=1100,
)
```

Shape：

```text
baseband:   (shot, adc_sample)
shot_iq:    (shot,)
average_iq: scalar complex
```

Digital downconversion 使用：

```text
baseband(t) = 2 * voltage(t) * exp(-i*(2*pi*f*t + reference_phase))
```

對 real cosine input 而言，factor 2 會保留原始 tone amplitude。

### Time-resolved filtering

```python
baseband, shot_iq_trace, average_iq_trace = processor.process_decimate(
    records_volts=records,
    tone_frequency_hz=50e6,
    reference_phase_radians=0.0,
    moving_average_samples=20,
    filter_type="boxcar",
)
```

Boxcar 使用 valid moving average，因此 output time axis 比 raw record 短：

```text
output_samples = adc_samples - moving_average_samples + 1
```

### Multiplex integration

同一組 raw records 可在多個 frequency 分別 downconvert：

```python
results = processor.process_multiplex_integrate(
    records_volts=records,
    tone_frequencies_hz={
        "q0": 50e6,
        "q1": 150e6,
    },
    reference_phases_radians={
        "q0": 0.0,
        "q1": 0.0,
    },
    integrate_start=100,
    integrate_stop=1100,
)

q0_baseband, q0_shots, q0_average = results["q0"]
q1_baseband, q1_shots, q1_average = results["q1"]
```

每個 frequency 都從相同的 triggered records 計算，不需要重複 acquisition。
完整 hardware 範例位於 repository 根目錄的
[multiplex.ipynb](../../multiplex.ipynb)。

## Baseline 與 phase processing

Pure NumPy helpers：

- `subtract_baseline()`：每個 shot 減去 baseline window 的平均值。
- `correct_interleaving_offsets()`：校正 ADC interleaving core offsets。
- `phase_align_iq()`：依 reference window 將每個 shot 旋轉到共同 phase。
- `recover_coherent_envelope()`：baseline correction、shot phase alignment 與 filtering。
- `recover_clock_referenced_envelope()`：共用 reference clock 時，不以 noisy
  shot 自身估計 phase。

若 AWG 與 ATS9371 共用穩定 10 MHz reference，優先考慮
`recover_clock_referenced_envelope()`，避免逐 shot phase estimation 造成
正向 magnitude bias。

## Low-level DMA API

只有需要自訂 acquisition lifecycle 時才直接使用：

```python
from QAWG.alazar import (
    ATSApi,
    AcquisitionConfig,
    abort_capture,
    arm_capture,
    free_capture,
    open_ats9371,
    start_capture,
    wait_for_capture,
)

api = ATSApi()
board = open_ats9371(api, system_id=1, board_id=1)
config = AcquisitionConfig(
    sample_rate_hz=1e9,
    tone_frequency_hz=50e6,
    samples_per_record=1500,
    num_averages=1000,
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

## Diagnostics

完成 acquisition 後：

```python
diagnostics = experiment.capture_diagnostics()
```

內容包括：

- ADC channel 與 resolution
- LSB voltage
- Raw code minimum/maximum
- Mean DC offset
- Averaged trace peak-to-peak
- Shot noise standard deviation

最近一次 acquisition 也保留在：

```python
experiment.last_raw_codes
experiment.last_records_volts
experiment.last_downconverted_iq
experiment.last_shot_iq
experiment.last_time_s
```

## Data ownership

- ATS DMA 與 ADC conversion：`ats9371.py`
- DSP：`demodulation.py`、`processor.py`
- AWG start/stop 與 acquisition coordination：`AWGAlazar`
- Experiment sweep 與 sequence record layout：`QAWG.compiler`
