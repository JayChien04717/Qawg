# QAWG

QAWG 是用於 **Tektronix AWG5208** 與 **AlazarTech ATS9371** 的 host-side 實驗控制與波形編譯套件。它將實驗規則編譯成 AWG waveform / sequence，透過 marker 觸發 Alazar 擷取，並完整保留每次 shot 的 raw data 與 IQ data。

---

## 文件導覽

| 文件 | 內容說明 |
| :--- | :--- |
| [QAWG_TUTORIAL_ZH.md](QAWG_TUTORIAL_ZH.md) | 從零開始的完整中文教學（AWG、Alazar、one/two-tone、Rabi、T1、Ramsey、Wigner） |
| [QAWG/README.md](QAWG/README.md) | Experiment Compiler 詳細設計、timing 規則與 Labber HDF5 存檔說明 |
| [QAWG/awg5200/README.md](QAWG/awg5200/README.md) | AWG5208 底層 driver、波形函數、`Shape` / `IQArray` 與 sequence 說明 |
| [QAWG/awg5200/TIMELINE_TUTORIAL.md](QAWG/awg5200/TIMELINE_TUTORIAL.md) | 多通道 Timeline 排程與視覺化範例 |
| [QAWG/alazar/README.md](QAWG/alazar/README.md) | ATS9371 擷取、解調、濾波與低階 DMA 說明 |
| [demo.ipynb](demo.ipynb) | 互動式 Notebook 範例（TOF、length / delay sweep、single-shot 視覺化） |

---

## 專案架構

```text
QAWG/
├── awg5200/       AWG5208 波形函數、Shape/IQArray 建構器、Timeline、Marker、Sequence 與 SCPI driver
├── alazar/        ATS9371 擷取、ADC 轉換、解調、濾波與低階 DMA
├── timeline.py    QAWG.awg5200.timeline 的相容匯出入口
├── compiler.py    高階實驗 DSL（ExperimentProgram）、Sweep 展開與排程編譯
├── awg_alazar.py  AWG 與 Alazar 協同控制入口（AWGAlazar）
├── tomography.py  Heterodyne temporal-mode tomography 工具
└── Labber_saver.py Raw ADC 與 IQ 資料輸出為 Labber HDF5
```

### 分層職責（由下而上）

1. **波形與排程層（`QAWG.awg5200` / `QAWG.timeline`）**：建立基本包絡、用 `Shape` 組合自訂波形（移植自 QuantumCompilerV2）、多通道 `Timeline` 排程與視覺化，以及 AWG5208 上傳與播放。
2. **擷取與訊號處理層（`QAWG.alazar`）**：控制 ATS9371 DMA 擷取與 ADC 電壓轉換，並由 `alazar.dsp` 執行解調、FIR 濾波與積分。
3. **實驗編譯層（`QAWG.compiler`）**：以 `ExperimentProgram` 描述實驗，在 compile time 展開 sweep 與驗證 timing，產出不依賴硬體的 `CompiledExperiment`（可離線預覽與繪圖）。
4. **協同執行層（`QAWG.awg_alazar`）**：由 `AWGAlazar` 統一設定兩台儀器、上傳 sequence、啟動擷取，並回傳保留 single-shot 軸的 `ExperimentResult`。

三層皆可獨立匯入使用，互不強制相依：

```python
from QAWG.awg5200 import AWG5208, AWGProgram, AWGStep, Shape, IQArray
from QAWG.alazar import Alazar9371, CaptureConfig, downconvert, fir_filter, integrate
from QAWG import AWGAlazar
```

---

## 環境需求

- **作業系統**：Windows
- **Python**：3.11+
- **必要套件**：`numpy`、`scipy`（tomography 需要）、`matplotlib`（繪圖）、`pyvisa` / `pyvisa-py`（連線 AWG 時需要）
- **選用套件**：`plotly`（互動式波形與甘特圖）、`h5py`（Labber HDF5 存檔）
- **硬體驅動**：可用之 VISA backend、AlazarTech ATS-SDK（`ATSApi.dll`）

```powershell
pip install numpy scipy matplotlib plotly h5py nbformat pytest pyvisa pyvisa-py
```

> 目前 repository 不含 packaging metadata，請一律從 repository 根目錄執行 notebook、script 與測試。

---

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

`acquire_window_s` 是每次 trigger 後擷取的完整 raw record 長度；實際 IQ integration 時間則由各實驗的 `declare_readout()` 決定。

### 2. 定義實驗（`ExperimentProgram`）

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
            style="cosine_square",
            length=1 * us,
            edge_length=20 * ns,
            frequency=cfg["f_res"],
            phase=0.0,
            gain=0.02,
            readout=True,
        )

    def _body(self, cfg):
        self.play("probe", at=0)
        self.play("readout", at=0)
        self.trigger("ro", trigger_delay=0)
```

#### 載波調變與全域相位（Global Phase）

Compiler 會自動將 envelope 調變為：

```text
waveform(t) = gain * envelope(t) * sin(2 * pi * frequency * t + phase)
```

預設 `t` 為 sequence step 的**全域時間**（step 第 0 個 sample 為 0，即 marker 與 ADC record 起點），讓不同時間的 pulse 與解調共用同一連續載波，因此 Ramsey 條紋只來自 detuning、readout IQ 相位也不會隨位置偏移。若需在每個 pulse 起點重置載波相位，可設定 `ExperimentProgram(cfg, phase_reference="pulse")`。

### 3. 編譯、離線預覽與上機量測

不連接硬體即可編譯並檢查各通道波形、readout 視窗與 trigger marker：

```python
program = SpectroscopyProgram({"f_res": 50 * MHz})
compiled = program.compile(sample_rate_hz=2.5e9)

frequency = compiled.axis("frequency")
qubit_waveforms = compiled.preview(channel=4)
compiled.plot(step=0, show=True)  # 支援 backend="matplotlib" 或 "plotly"
```

連接 `AWGAlazar` 後執行量測：

```python
compiled = experiment.compile(program)
result = compiled.acquire(n_average=1000)
```

---

## 自訂波形、視覺化與射頻工具（移植自 QuantumCompilerV2）

除了 QAWG 原生的 pulse style（`const`、`gaussian`、`gaussian_square`、`cosine_square`、`exponential_rise`、`exponential_decay`），本專案亦完整移植了 **QuantumCompilerV2** 的波形操作功能並無縫整合至 `Timeline` 與 `ExperimentProgram`：

| 功能模組 | 說明 |
| :--- | :--- |
| **新增波形函數** | `cosine`、`square`、`exp_rising_pulse`（截斷式指數上升波）、`exp_falling_pulse`、`exp_square`、`pwm`、`pwm_width` |
| **`Shape` 代數與微分** | 支援 `+ - * / // ** %`、`-s`、`abs(s)`、`.reverse()`、`.roll()`、**`.diff(n)`（DRAG 微分）**、`.normalize()`、`.modulate()` |
| **`Shape` 時域組合** | `s1 \| s2`（時間串接）、`.repeat(n)`、`s >> dt`（前補零延遲）、`s << dt`（後補零）、`.pad_const()`、`.align_to()`（非破壞性對齊） |
| **自訂函數註冊表** | `SHAPE_REGISTRY` 與 `@register_shape("name")` 裝飾器 |
| **波形與排程視覺化** | `Shape.plot()` / `.plot_fft()`、`Timeline.plot()` / `.plot_schedule()`（甘特圖）、`CompiledExperiment.plot()`（皆支援 Matplotlib / Plotly） |
| **`IQArray` 射頻工具** | `vI`、`vQ`、`p`、`dBm`、`phase`、`scale_to_vp()`、`scale_to_dbm()`、`gain(dB)`、`phase_shift(rad)`、`iq_plot()` |

```python
from QAWG import Shape, IQArray, register_shape, parallel, delay_auto, waveform

fs = 1e9
g = Shape("gaussian", {"peak_x": 40e-9, "sigma": 10e-9}, span=80e-9, sample_rate=fs, name="drive")
drag_q = (0.1 * g.diff()).normalize(0.05)          # 數值微分產生 DRAG Q 分量
double_pulse = (0.2 * g) | (0.2 * g >> 20e-9)      # `|` 時域串接、`>>` 前補零延遲

g.plot(show=True)                                  # 時域圖
g.plot_fft(show=True)                              # FFT 頻譜圖

# 放進 Timeline 並檢視多通道甘特排程圖
tl = parallel(
    waveform(double_pulse, fc=50e6, ch=1),
    drag_q.to_waveform(fc=50e6, ch=2),
) / delay_auto(20e-9) / waveform(g, fc=100e6, ch=3, name="ro")
tl.plot_schedule(fs, backend="plotly", show=True)

# 註冊自訂波形函數
@register_shape("my_ramp")
def my_ramp(x, slope=1e7):
    return slope * x

# 射頻電壓 / dBm 換算
iq = IQArray(g.y + 1j * drag_q.y, Z0=50.0, name="drive_iq").scale_to_dbm(-10.0)
```

> 在 `ExperimentProgram.add_pulse(..., style="custom", envelope=my_shape)` 中可直接傳入 `Shape`、NumPy 陣列或 callable；`envelope` 建議維持 `[-1, 1]`，由 `gain` 控制實際幅度比例。

---

## 資料格式（`ExperimentResult`）

假設 compiled sequence 有 `P` 個 sweep points，結果一律保留 shot 維度以支援 single-shot 分析：

```python
result.raw.shape        # (n_average, P, adc_sample)
result.iq_traces.shape  # (n_average, P, iq_sample)
result.iq_shots.shape   # (n_average, P)
```

需要平均結果時再明確呼叫 reduction helpers：

```python
raw_average = result.raw_average("ro")
iq_average = result.iq_average("ro")
s21 = result.s21("ro")
```

---

## Timing、Marker 與 CHB 相位參考

### Timing 與 Marker 規則

- `play(..., at=...)` 指定 pulse 在 AWG step 內的絕對時間；省略 `at` 則接在 cursor 後面。
- `delay_auto(dt)` 從前一個 pulse 結束後加入延遲。
- `declare_readout(..., marker_at_readout=True)` 適用於 T1、Ramsey 等 readout 位於 step 後段的實驗，讓 marker 跟著 readout 走（所有 step 的 readout 起點需固定），ADC 只需擷取 readout 附近。
- `trigger_delay` 為 acquire window 內 integration window 的起點，且同一 sequence 內必須固定。

### CHB 相位參考（Waveguide QED）

ATS 外部觸發對 1 GS/s 取樣時脈有 ±½ sample 量化抖動（50 MHz IF 下約 ±5°）。讓 AWG 另送一路參考音至 CHB 即可逐 shot 校正相位：

```python
experiment = AWGAlazar.connect(
    "TCPIP0::192.168.10.171::inst0::INSTR",
    awg_sample_rate_hz=2.5e9,
    acquire_window_s=3 * us,
    adc_channel="CHA",          # 訊號通道
    reference_channel="CHB",    # 參考音通道
)
```

詳細接線、emission 與 Wigner 流程請見 [QAWG_TUTORIAL_ZH.md](QAWG_TUTORIAL_ZH.md) 第 5 節。

---

## Single-HEMT Temporal-Mode Tomography

針對 5.9 GHz 附近、50 MHz IF 的 transmon / waveguide 系統，可使用 [tomo.py](tomo.py) 進行串流式時間模態量測與 Wigner 重建：

```powershell
python tomo.py hemt-stream `
  --device-frequency 5.9e9 `
  --if-frequency 50e6 `
  --total-shots 1000000 `
  --chunk-shots 2000
```

- 序列會交替量測 `reference`（無 AWG drive）與 `signal`，將每條 IQ trace 投影至單一時間模態後丟棄原始 chunk；執行前請先確認 `stream_mode_calibration.png`。
- 連續驅動螢光使用預設 boxcar 模式（校正 `--tomo-mode-start-ns` 與 `--tomo-mode-ns`）；釋放脈衝則使用指數衰減模式（`--tomo-decay-ns` 為**振幅**壽命）：

```powershell
python tomo.py hemt-stream --tomo-mode decay --tomo-decay-ns 60
```

- 完成 IQ leakage、真空參考、輸入損耗、熱光子數與 volts per sqrt(photon) 校正後，再執行 Wigner 重建：

```powershell
python tomo.py analyze-hemt `
  --data-dir tomodata\YYYYMMDD_HHMMSS `
  --photon-scale-volts 1.23e-4
```

---

## 測試與目前限制

### 執行測試

```powershell
python -m pytest -q
```

`tests/` 依用途分為兩個子資料夾：
- `tests/unit/`：自動化單元測試（使用 mock hardware，不需連接實體儀器，`pytest` 自動執行）。
- `tests/hardware/`：實體硬體診斷腳本與歷史實測數據（手動執行，不影響 `pytest`）。

### 目前限制

- 目前僅支援單一名為 `"ro"` 的 readout。
- Sweep 於 compile time 展開為 AWG sequence steps（非 FPGA runtime loop），多軸 sweep 採 Cartesian product 展開。
- Conditional playback 目前支援等值判斷，例如 `when=("state", "e")`。
- AWG waveform upload caching 與 sequence chunking 尚未實作。
- ATS9371 acquisition path 目前固定為 1 GS/s、50 ohm、DC coupling 與 ±400 mV input range。
