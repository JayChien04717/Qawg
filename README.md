# QAWG

QAWG 是用於 Tektronix AWG5208 與 AlazarTech ATS9371 的 host-side
實驗控制套件。它將實驗規則編譯成 AWG waveform 與 sequence，透過 marker
觸發 Alazar acquisition，並保留每次 shot 的 raw data 與 IQ data。

## 架構

```text
QAWG/
├── timeline.py    hardware-independent waveform timing helpers
├── awg5200/       AWG5208 waveform upload、marker、sequence 與 SCPI driver
├── alazar/        ATS9371 acquisition、ADC conversion 與 signal processing
├── compiler.py    sweep、pulse timing 與實驗規則
├── awg_alazar.py  AWG 與 Alazar coordination entry point
└── tomography.py  heterodyne tomography helpers
```

各模組的責任如下：

- **Timeline**：手動 waveform timing、parallel waveform 與 delay composition。
- **Alazar**：擷取與處理 digitized data，包括 demodulation、filter 與
  low-level DMA。
- **Result**：integration window 內的 shot IQ、mean IQ 與 result reduction。
- **AWG**：上傳 waveform、marker 與 sequence，並管理 AWG5208 hardware。
- **Compiler**：展開 sweep、驗證 timing，產生不依賴硬體的 compiled plan。
- **AWGAlazar**：設定兩台儀器、上傳 compiled plan、啟動 acquisition 並組裝結果。

三層可分開使用：

```python
from QAWG.awg5200 import AWG5208, AWGProgram, AWGStep
from QAWG.alazar import Alazar9371, CaptureConfig, downconvert, fir_filter, integrate
from QAWG import AWGAlazar
```

- `AWG5208.upload_program()` 依 step 數量自行選擇 waveform 或 sequence。
- `Alazar9371.capture()` 負責 DMA 與 ADC conversion，`alazar.dsp` 負責處理。
- `AWGAlazar.acquire()` 回傳 integration-window IQ。
- `AWGAlazar.acquire_decimate()` 回傳 time-resolved raw/IQ waveform。

## 環境需求

- Windows
- Python 3.11
- NumPy
- SciPy，部分 tomography routines 需要
- PyVISA 與可用的 VISA backend
- AlazarTech ATS-SDK，以及可載入的 `ATSApi.dll`

安裝常用 Python 套件：

```powershell
pip install numpy scipy matplotlib nbformat pytest pyvisa pyvisa-py
```

目前 repository 沒有 packaging metadata，請從 repository 根目錄執行
notebook、script 與測試。

## 快速開始

### 1. 連接硬體

```python
from QAWG import AWGAlazar, us

experiment = AWGAlazar.connect(
    "TCPIP0::192.168.10.171::inst0::INSTR",
    awg_sample_rate_hz=2.5e9,
    alazar_sample_rate_hz=1e9,
    acquire_window_s=1.5 * us,
    trigger_slope="rising",
    trigger_level=140,
)
```

`acquire_window_s` 是每次 trigger 後擷取的 raw record 長度。IQ integration
時間由各實驗的 readout declaration 決定。

### 2. 定義實驗

```python
from QAWG import ExperimentProgram, LinearSweep, MHz, ns, us


class SpectroscopyProgram(ExperimentProgram):
    def _initialize(self, cfg):
        self.declare_gen("qubit", ch=4, amplitude_vpp=0.5)
        self.declare_gen("res", ch=3, amplitude_vpp=0.5)

        self.declare_readout(
            "ro",
            adc_channel="CHA",
            length=1 * us,
            demod_freq=cfg["f_res"],
            waveform_ch=3,
            marker_channel=1,
            integrate_time=800 * ns,
        )

        frequency = self.add_sweep(
            "frequency",
            LinearSweep(2920 * MHz, 3000 * MHz, 101),
        )
        self.add_pulse(
            "probe",
            gen="qubit",
            style="const",
            length=15 * us,
            frequency=frequency,
            gain=0.01,
        )
        self.add_pulse(
            "readout",
            gen="res",
            style="const",
            length=1 * us,
            frequency=cfg["f_res"],
            gain=0.02,
        )

    def _body(self, cfg):
        self.play("probe", at=0)
        self.play("readout", at=0)
        self.trigger("ro", trigger_delay=0)
```

具有 half-cosine rise、flat top、half-cosine fall 的 pulse 可使用：

```python
self.add_pulse(
    "readout",
    gen="res",
    style="cosine_square",
    length=1 * us,
    edge_length=20 * ns,
    frequency=250 * MHz,
    phase=0.0,
    gain=0.02,
)
```

使用者只定義 envelope style 與時間參數。Compiler 會自動完成時間轉換與
carrier modulation：

```text
waveform(t) = gain * envelope(t) * sin(2*pi*frequency*t + phase)
```

這裡的 `t` 是 sequence step 的全域時間（step 第 0 個 sample 為 0，也就是
marker 與 ADC record 的起點），所以不同時間的 pulse 與 demodulation 共用
同一個連續載波。Ramsey 的條紋因此只來自 qubit 與 drive 的 detuning，
readout 的 IQ 相位也不會隨 readout 位置改變。需要舊行為（載波在每個
pulse 起點重置）時使用 `ExperimentProgram(cfg, phase_reference="pulse")`。

### 3. Compile 與 acquisition

不連接硬體也可以 compile 與預覽 waveform：

```python
program = SpectroscopyProgram({"f_res": 50 * MHz})
compiled = program.compile(sample_rate_hz=2.5e9)

frequency = compiled.axis("frequency")
qubit_waveforms = compiled.preview(channel=4)
```

使用已連接的 `AWGAlazar` 執行實驗：

```python
compiled = experiment.compile(program)
result = compiled.acquire(n_average=1000)
```

## Result data

假設 compiled sequence 有 `P` 個 sweep points：

```python
result.raw.shape
# (n_average, P, adc_sample)

result.iq_traces.shape
# (n_average, P, iq_sample)

result.iq_shots.shape
# (n_average, P)
```

資料預設保留 shot axis，不會自動丟棄 single-shot information。需要平均時：

```python
raw_average = result.raw_average("ro")
iq_average = result.iq_average("ro")
s21 = result.s21("ro")
```

## Timing 與 marker

- `play(..., at=...)` 指定 pulse 在 AWG step 內的位置。
- 未指定 `at` 的 `play()` 會沿 program cursor 排列。
- `delay_auto()` 從上一個 pulse 結束後加入 delay。
- `waveform_ch` 會讓 compiler 依該 channel 的 active waveform 建立 marker。
- ATS9371 收到 marker 後立即開始擷取完整 acquire window。
- `trigger_delay` 是 acquire window 內 integration window 的起點。
- 同一 sequence 內的 integration delay 必須固定。
- `declare_readout(..., marker_at_readout=True)` 讓 marker 跟著 readout（T1、Ramsey 等 readout 在 step 後段的實驗），ADC 只錄 readout 附近；所有 step 的 readout 起點必須相同。

## CHB 相位參考（waveguide QED）

trigger 對 1 GS/s 取樣時脈有 ±½ sample 的量化，50 MHz IF 下造成約 5° 的 shot 間相位抖動。
讓 AWG 另送一個參考音到 CHB，即可每 shot 修正：

```python
experiment = AWGAlazar.connect(
    "TCPIP0::192.168.10.171::inst0::INSTR",
    awg_sample_rate_hz=2.5e9,
    acquire_window_s=3 * us,
    adc_channel="CHA",          # 訊號
    reference_channel="CHB",    # 參考音
)
```

接線、參考音 program、emission 與 Wigner 流程請見 [QAWG_TUTORIAL_ZH.md](QAWG_TUTORIAL_ZH.md) 第 5 節。

## Demo

[demo.ipynb](demo.ipynb) 包含：

- Time-of-flight 與 readout timing
- Pulse length sweep
- `delay_auto` sweep
- Phase single-shot acquisition
- Raw record、IQ trajectory 與結果視覺化

## 測試

```powershell
python -m pytest -q
```

測試使用 mock hardware，不需要連接實際儀器。

## 目前限制

- 目前只支援一個名為 `"ro"` 的 readout。
- Sweep 在 compile time 展開為 AWG sequence steps，不是 FPGA runtime loop。
- 多個 sweep axes 使用 Cartesian product 展開。
- Conditional playback 目前支援 equality condition，例如
  `when=("state", "e")`。
- AWG waveform upload caching 與 sequence chunking 尚未實作。
- ATS9371 acquisition path 目前固定為 1 GS/s、50 ohm、DC coupling 與
  +/-400 mV input range。

從零開始的教學（waveform、readout、one/two-tone、Rabi、T1、Wigner）請見 [QAWG_TUTORIAL_ZH.md](QAWG_TUTORIAL_ZH.md)。
更完整的 compiler 使用說明請參考
[QAWG/README.md](QAWG/README.md)。Timeline helpers 位於
[QAWG/timeline.py](QAWG/timeline.py)，AWG5208 底層 driver 說明位於
[QAWG/awg5200](QAWG/awg5200)，Alazar acquisition 與 DSP 位於
[QAWG/alazar](QAWG/alazar)。
# Single-HEMT temporal-mode tomography

For a transmon/waveguide setup near 5.9 GHz with a 50 MHz IF:

```powershell
python tomo.py hemt-stream `
  --device-frequency 5.9e9 `
  --if-frequency 50e6 `
  --total-shots 1000000 `
  --chunk-shots 2000
```

The sequence alternates `reference` (zero AWG drive) and `signal`, projects
each IQ trace onto one temporal mode, and discards the raw chunk. Check
`stream_mode_calibration.png` before trusting the run.

For continuously driven fluorescence, use the default boxcar mode and
calibrate `--tomo-mode-start-ns` and `--tomo-mode-ns`. For a released pulse:

```powershell
python tomo.py hemt-stream --tomo-mode decay --tomo-decay-ns 60
```

`--tomo-decay-ns` is the **amplitude** lifetime. Calibrate IQ leakage, vacuum
reference, input loss, thermal occupation, and volts per sqrt(photon).
Only after calibrating the last quantity should Wigner reconstruction run:

```powershell
python tomo.py analyze-hemt `
  --data-dir tomodata\YYYYMMDD_HHMMSS `
  --photon-scale-volts 1.23e-4
```
