# QAWG 中文教學

這份教學分成三層：

1. 只使用 AWG5208：產生 waveform、timeline、marker、sequence。
2. 只使用 Alazar ATS9371：擷取 acquire window，做 demodulation 和 integration。
3. 使用 QAWG：用 `ExperimentProgram` 描述實驗，compile 後由 `AWGAlazar` 協調 AWG 與 Alazar。

建議一般實驗優先使用第 3 種 QAWG flow；第 1、2 種比較適合 debug、手動測試或特殊流程。

## 基本概念

### Acquire Window

`acquire_window_s` 是 Alazar 在每次 trigger 後擷取的完整 ADC record 長度。

這段資料會保留在：

```python
experiment.last_records_volts
experiment.last_downconverted_iq
experiment.last_time_s
```

用途是 debug timing、TOF、ring-up/ring-down、trigger alignment。

### Integration Window

Integration window 是 acquire window 裡面真正拿來算每 shot IQ 的區間。

概念上是：

```text
acquire window:     trigger 後完整擷取資料
integration window: acquire window 內 [delay, delay + integration_time]
```

`acquire()` 會回傳 integration window 內的 per-shot IQ 和 mean IQ。

`acquire_decimate()` 會回傳 integration window 內的 raw trace 和 low-pass demodulated trace，同時保留完整 acquire-window debug data。

## 1. 使用 AWG5208

AWG5208 底層控制位於 `QAWG.awg5200`。Timeline helper 位於上層 `QAWG.timeline`，也可以直接從 `QAWG` 匯入。

### 連接 AWG

```python
from QAWG.awg5200 import AWG5208

awg = AWG5208.connect(
    "TCPIP0::192.168.10.171::inst0::INSTR",
    timeout_ms=60_000,
)

awg.set_awg_mode()
awg.use_external_10mhz_reference()
awg.set_sample_rate(2.5e9)
```

### 建立 envelope

```python
from QAWG.awg5200 import gaussian_square_ns

envelope = gaussian_square_ns(
    duration_ns=600,
    sample_rate_hz=2.5e9,
    edge_sigma_ns=20,
    amplitude_volts=1.0,
)
```

### 建立 timeline

```python
from QAWG import waveform, delay_auto, parallel

readout = waveform(
    envelope,
    fc=50e6,
    ch=1,
    phase_radians=0.0,
    gain=1.0,
    name="readout",
)

timeline = readout
```

如果有多個 channel：

```python
drive = waveform(envelope, fc=100e6, ch=2, gain=0.1, name="drive")
readout = waveform(envelope, fc=50e6, ch=1, gain=1.0, name="readout")

timeline = drive / delay_auto(40e-9) / readout
```

### 上傳 timeline

```python
uploaded = awg.upload_timeline(
    timeline,
    amplitude_vpp={1: 0.5, 2: 0.5},
    name_prefix="manual_test",
    total_duration_s=2e-6,
)

awg.run()
```

### 手動建立 marker

如果你想手動讓 marker 對齊某個 waveform：

```python
marker_name = awg.marker(
    waveform_ch=1,
    marker_ch=1,
    marker_number=1,
    low_volts=0.0,
    high_volts=1.2,
)
```

一般 QAWG 實驗不建議手動處理 marker，交給 compiler 比較不容易出錯。

## 2. 使用 Alazar ATS9371

低階 Alazar 功能位於 `QAWG.alazar`。一般使用者不需要直接操作 DMA lifecycle，建議透過 `AWGAlazar`。

### 使用 AWGAlazar 連接 Alazar + AWG

```python
from QAWG import AWGAlazar, ns, us

experiment = AWGAlazar.connect(
    "TCPIP0::192.168.10.171::inst0::INSTR",
    awg_sample_rate_hz=2.5e9,
    alazar_sample_rate_hz=1e9,
    acquire_window_s=1.5 * us,
    trigger_slope="rising",
    trigger_level=140,
    tone_frequency_hz=50e6,
    trigger_delay_s=0.0,
    integrate_window_ns=(0.0, 600.0),
    adc_channel="CHB",
    moving_average_time_s=20 * ns,
    baseline_time_s=100 * ns,
    timeout_ms=30_000,
)
```

### Trace / TOF 模式：acquire_decimate

`acquire_decimate()` 用來看 integration window 內的 trace。

```python
time_s, raw_window, demod_window = experiment.acquire_decimate(
    n_average=1000,
)
```

回傳 shape：

```text
time_s:       (integration_sample,)
raw_window:   (n_average, integration_sample)
demod_window: (n_average, integration_sample)
```

畫平均 trace：

```python
import numpy as np
import matplotlib.pyplot as plt

time_ns = time_s / ns
raw_mean_mv = raw_window.mean(axis=0) * 1e3
iq_mean_mv = demod_window.mean(axis=0) * 1e3

plt.figure()
plt.plot(time_ns, raw_mean_mv)
plt.xlabel("Time in integration window (ns)")
plt.ylabel("Raw (mV)")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(time_ns, np.abs(iq_mean_mv), label="|IQ|")
plt.plot(time_ns, iq_mean_mv.real, "--", label="I")
plt.plot(time_ns, iq_mean_mv.imag, "--", label="Q")
plt.xlabel("Time in integration window (ns)")
plt.ylabel("Demodulated (mV)")
plt.legend()
plt.grid(True)
plt.show()
```

不管 `acquire_decimate()` 回傳 integration window 內資料，完整 acquire window 都會保留：

```python
full_raw = experiment.last_records_volts
full_iq = experiment.last_downconverted_iq
full_time = experiment.last_time_s
```

### Averaged IQ 模式：acquire

`acquire()` 用來取得 integration window 內的 per-shot IQ 和 mean IQ。

```python
shots, mean_iq = experiment.acquire(n_average=1000)
```

回傳 shape：

```text
shots:  (n_average,)
mean_iq: scalar complex
```

畫 single-shot IQ：

```python
plt.figure()
plt.scatter(shots.real * 1e3, shots.imag * 1e3, s=5, alpha=0.4)
plt.xlabel("I (mV)")
plt.ylabel("Q (mV)")
plt.axis("equal")
plt.grid(True)
plt.show()
```

## 3. 使用 QAWG

QAWG 的推薦流程是：

```text
ExperimentProgram
    -> add_pulse / play / trigger
    -> compile()
    -> CompiledExperiment
    -> upload() 或 acquire()
    -> AWGAlazar 協調 AWG + Alazar
    -> ExperimentResult
```

### 定義 ExperimentProgram

```python
from QAWG import ExperimentProgram, ns, us


class ReadoutProgram(ExperimentProgram):
    def _initialize(self, cfg):
        self.declare_gen(
            "readout",
            ch=cfg["awg_ch"],
            amplitude_vpp=cfg["channel_amplitude_vpp"],
        )
        self.declare_readout(
            "ro",
            adc_channel=cfg["adc_channel"],
            length=cfg["readout_length"],
            demod_freq=cfg["if_frequency_hz"],
            waveform_ch=cfg["awg_ch"],
            marker_channel=cfg["marker_ch"],
            marker_padding=cfg["marker_padding"],
            integrate_time=cfg["integrate_time"],
        )
        self.add_pulse(
            "readout_pulse",
            gen="readout",
            style="gaussian_square",
            length=cfg["readout_length"],
            edge_sigma=cfg["edge_sigma"],
            frequency=cfg["if_frequency_hz"],
            gain=cfg["readout_gain"],
            readout=True,
        )

    def _body(self, cfg):
        self.play("readout_pulse", at=0.0)
        self.trigger("ro", trigger_delay=cfg["trigger_delay"])
```

### Compile

```python
cfg = {
    "awg_ch": 1,
    "marker_ch": 1,
    "adc_channel": "CHB",
    "channel_amplitude_vpp": 0.5,
    "if_frequency_hz": 50e6,
    "readout_length": 600 * ns,
    "integrate_time": 600 * ns,
    "trigger_delay": 600 * ns,
    "edge_sigma": 20 * ns,
    "readout_gain": 1.0,
    "marker_padding": 500 * ns,
}

program = ReadoutProgram(cfg, final_delay_s=1 * us)
compiled = program.compile(hardware=experiment)
```

`compile()` 會：

1. 展開 sweep。
2. 排出 pulse timeline。
3. 產生 AWG waveform。
4. 產生 marker waveform。
5. 記錄 readout window、trigger delay、record layout。

注意：`add_pulse()` 不會立刻呼叫 `waveform()`。`add_pulse()` 只是登記 pulse definition；真正 render waveform 是在 `compile()`。

### 上傳一次，多次 acquisition

如果只改外部儀器，例如掃 SGS frequency，可以先 upload 一次：

```python
compiled.upload()

for f in freqlist:
    sgs.frequency = float(f)
    result = compiled.acquire(n_average=1000)
    iq = result.iq_average("ro")
```

同一個 `compiled` 物件重複 `acquire()` 時，`AWGAlazar` 會快取已上傳的 compiled plan，不會每次重新 upload AWG waveform。

### 取得結果

```python
result = compiled.acquire(n_average=1000)
```

如果 sequence 有 `P` 個 step：

```text
result.raw.shape       = (n_average, P, adc_sample)
result.iq_traces.shape = (n_average, P, adc_sample)
result.shots().shape   = (n_average, P)
```

常用 reduction：

```python
raw_avg = result.trace_average("ro")
iq_trace_avg = result.iq_trace_average("ro")
iq_avg = result.iq_average("ro")
shots = result.shots("ro")
```

### TOF / timing calibration

TOF 建議用 `acquire_decimate()`，因為它是 trace/debug mode。

```python
compiled.upload()
experiment.configure_experiment(
    tone_frequency_hz=compiled.readout.demod_frequency_hz,
    trigger_delay_s=compiled.trigger_delay_s,
    integrate_time_s=compiled.readout.integrate_time_s
    or compiled.readout.length_s,
    adc_channel=compiled.readout.adc_channel,
)

time_s, raw_window, demod_window = experiment.acquire_decimate(
    n_average=1000,
)
```

如果要使用 `calculate_window()`，可以用完整 acquire-window debug data 組 `ExperimentResult`，或參考 `Qawgdemo.ipynb` 的 TOF cell。

### Resonator spectroscopy

如果 resonator spectroscopy 只掃 SGS frequency，AWG waveform 不需要每點重傳：

```python
compiled.upload()

resonator_iq = []
for f in freqlist:
    sgs.frequency = float(f)
    result = compiled.acquire(n_average=1000)
    resonator_iq.append(result.iq_average("ro")[0])
```

## 常見問題

### 為什麼我設定 trigger_delay 了，圖上還像沒 delay？

如果你只呼叫：

```python
compiled.upload()
experiment.acquire_decimate(...)
```

那只會上傳 AWG waveform，不會自動把 compiled readout settings 套到 Alazar。

TOF/debug mode 應該在 `acquire_decimate()` 前呼叫：

```python
experiment.configure_experiment(
    tone_frequency_hz=compiled.readout.demod_frequency_hz,
    trigger_delay_s=compiled.trigger_delay_s,
    integrate_time_s=compiled.readout.integrate_time_s
    or compiled.readout.length_s,
    adc_channel=compiled.readout.adc_channel,
)
```

如果用：

```python
compiled.acquire(n_average=...)
```

則 `AWGAlazar.acquire_compiled_experiment()` 會自動套用這些設定。

### raw trace 和 demodulated trace 差在哪？

`last_records_volts` 是 ADC 原始電壓，通常會看到 IF carrier。

`last_downconverted_iq` 是 downconversion 後的 low-pass IQ trace，比較適合看 envelope、ring-up 和 integration window。

### 什麼時候用 acquire，什麼時候用 acquire_decimate？

用 `acquire_decimate()`：

- TOF
- timing debug
- 看 raw/demodulated trace
- 決定 integration delay 和 integration time

用 `acquire()`：

- resonator spectroscopy
- single-shot IQ
- tomography
- 已經決定好 integration window 後的正式量測

