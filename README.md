# QAWG

QAWG 是用於 Tektronix AWG5208 與 AlazarTech ATS9371 的 host-side
實驗控制套件。它將實驗規則編譯成 AWG waveform 與 sequence，透過 marker
觸發 Alazar acquisition，並保留每次 shot 的 raw data 與 IQ data。

## 架構

```text
QAWG/
├── awg5200/       waveform、marker、sequence 與 AWG5208 SCPI driver
├── alazar/        ATS9371 acquisition、ADC conversion 與 signal processing
├── compiler.py    sweep、pulse timing 與實驗規則
├── awg_alazar.py  AWG 與 Alazar 的執行協調層
└── examples.py    spectroscopy、Power Rabi、T1 與 single-shot 範例
```

各模組的責任如下：

- **Alazar**：擷取與處理 digitized data，包括 demodulation、filter、
  integration 與 shot averaging。
- **AWG**：產生 waveform、marker 與 sequence，並管理 AWG5208 hardware。
- **Compiler**：展開 sweep、驗證 timing，產生不依賴硬體的 compiled plan。
- **AWGAlazar**：設定兩台儀器、上傳 compiled plan、啟動 acquisition 並組裝結果。

## 環境需求

- Windows
- Python 3.11
- NumPy
- SciPy，使用 Butterworth 或 elliptic filter 時需要
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
compiled = program.compile(hardware=experiment)
result = compiled.acquire(n_average=1000)
```

## Result data

假設 compiled sequence 有 `P` 個 sweep points：

```python
result.raw.shape
# (n_average, P, adc_sample)

result.iq_traces.shape
# (n_average, P, iq_sample)

result.shots("ro").shape
# (n_average, P)
```

資料預設保留 shot axis，不會自動丟棄 single-shot information。需要平均時：

```python
raw_average = result.trace_average("ro")
iq_trace_average = result.iq_trace_average("ro")
iq_average = result.iq_average("ro")
```

## Timing 與 marker

- `play(..., at=...)` 指定 pulse 在 AWG step 內的位置。
- 未指定 `at` 的 `play()` 會沿 program cursor 排列。
- `delay_auto()` 從上一個 pulse 結束後加入 delay。
- `waveform_ch` 會讓 compiler 依該 channel 的 active waveform 建立 marker。
- `trigger_delay` 是 ATS9371 收到 marker 後的 hardware acquisition delay。
- 同一 sequence 內的 ATS trigger delay 必須固定。

## Demo

[demo.ipynb](demo.ipynb) 包含：

- Time-of-flight 與 readout timing
- Pulse length sweep
- `delay_auto` sweep
- Phase single-shot acquisition
- Raw record、IQ trajectory 與結果視覺化

重新產生乾淨的 demo notebook：

```powershell
python QAWG\build_demo_notebook.py
```

這個命令只建立 notebook，不會連接或操作硬體。實際執行 notebook cells
才會連接 AWG5208 與 ATS9371。

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

更完整的 compiler 使用說明請參考
[QAWG/README.md](QAWG/README.md)。AWG timeline 與底層 driver 說明位於
[QAWG/awg5200](QAWG/awg5200)，Alazar acquisition 與 DSP 位於
[QAWG/alazar](QAWG/alazar)。
