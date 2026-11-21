# AWG5208 Timeline 教學

這份教學示範：

- CH2：Qubit 1
- CH4：Qubit 2
- CH3：Readout
- CH1 Marker 1：跟隨 Readout
- 所有 channel 自動補零並對齊

## 時序語法

`delay()` 以「前一組 waveform 的開始時間」為基準：

```python
qubit / delay(10e-9) / readout
```

代表 Readout 在 Qubit 開始後 10 ns 開始，兩者可能重疊。

`delay_auto()` 以「前一組 waveform 的結束時間」為基準：

```python
qubit / delay_auto(10e-9) / readout
```

代表 Qubit 完全結束後等待 10 ns，再開始 Readout。

`parallel()` 讓多個 waveform 同時開始：

```python
parallel(qubit1, qubit2) / delay_auto(10e-9) / readout
```

`delay_auto()` 會等待 parallel group 中最晚結束的 waveform。

## 完整範例

```python
from QAWG.awg5200 import (
    AWG5208,
    delay_auto,
    gaussian_square_ns,
    parallel,
    waveform,
)

resource = "TCPIP0::192.168.10.171::inst0::INSTR"
sample_rate_hz = 2.5e9

qubit1_envelope = gaussian_square_ns(
    duration_ns=100,
    sample_rate_hz=sample_rate_hz,
    edge_sigma_ns=10,
    amplitude_volts=0.2,
)

qubit2_envelope = gaussian_square_ns(
    duration_ns=150,
    sample_rate_hz=sample_rate_hz,
    edge_sigma_ns=10,
    amplitude_volts=0.2,
)

readout_envelope = gaussian_square_ns(
    duration_ns=1000,
    sample_rate_hz=sample_rate_hz,
    edge_sigma_ns=10,
    amplitude_volts=0.2,
)

qubit1 = waveform(
    qubit1_envelope,
    fc=0,
    ch=2,
    name="qubit1",
)

qubit2 = waveform(
    qubit2_envelope,
    fc=0,
    ch=4,
    name="qubit2",
)

readout = waveform(
    readout_envelope,
    fc=0,
    ch=3,
    name="readout",
)

# CH2 與 CH4 同時開始。
# 等待較長的 CH4 pulse 結束，再等待 10 ns，然後播放 CH3。
timeline = (
    parallel(qubit1, qubit2)
    / delay_auto(10e-9)
    / readout
)

awg = AWG5208.connect(
    resource,
    timeout_ms=60_000,
)

print("Connected:", awg.identify())

awg.set_awg_mode()
awg.set_sample_rate(sample_rate_hz)

names = awg.upload_timeline(
    timeline,
    amplitude_vpp={
        2: 0.5,
        3: 0.5,
        4: 0.5,
    },
    total_duration_s=10e-6,
)

# CH1 使用零電壓 waveform，Marker 1 自動包住 CH3 readout。
marker_name = awg.marker(
    waveform_ch=3,
    marker_ch=1,
    marker_number=1,
    low_volts=0.0,
    high_volts=1.2,
    amplitude_vpp=0.5,
)

awg.run(wait_until_ready=True)

print("CH1 marker:", marker_name)
print("CH2 qubit1:", names[2])
print("CH3 readout:", names[3])
print("CH4 qubit2:", names[4])
print("Run state:", awg.run_state())
print("Error:", awg.query("SYSTem:ERRor?"))
```

## 時序結果

```text
CH2 Qubit 1  |---- 100 ns ----|
CH4 Qubit 2  |------ 150 ns ------|
                                  | 10 ns |
CH3 Readout                              |------ 1000 ns ------|
CH1 Marker                               |------ HIGH ---------|
```

Timeline 預設總長度為 5 us。上面的範例透過
`total_duration_s=10e-6` 將所有 channel 補零到 10 us。

## 只有一個 Qubit

```python
timeline = qubit1 / delay_auto(10e-9) / readout
```

## Readout 與 Qubit 重疊

```python
timeline = qubit1 / delay(10e-9) / readout
```

此時 Readout 在 Qubit 1 開始後 10 ns 開始。

## 搭配 `Shape` 組合複雜包絡與預覽排程（移植自 QuantumCompilerV2）

若需要對波形做四則運算、微分（例如 DRAG pulse）、時域串接或補零，可直接用 `Shape` 建立包絡再轉成 `waveform()`：

```python
from QAWG import Shape, delay_auto, parallel, waveform

fs = 2.5e9
g = Shape("gaussian", {"peak_x": 50e-9, "sigma": 10e-9}, span=100e-9, sample_rate=fs, name="qubit_I")
drag_q = (0.1 * g.diff()).normalize(0.05)          # 微分產生 DRAG Q 分量
double_drive = (0.2 * g) | (0.2 * g >> 20e-9)      # `|` 時域串接、`>>` 前補零

w_i = waveform(double_drive, fc=50e6, ch=2, name="qubit_I")
w_q = drag_q.to_waveform(fc=50e6, ch=4, phase_radians=1.5708, name="qubit_Q")
ro = Shape("const", {"lv": 0.15}, span=500e-9, sample_rate=fs, name="readout").to_waveform(fc=100e6, ch=3)

timeline = parallel(w_i, w_q) / delay_auto(20e-9) / ro

# 預覽各 channel 時域波形與排程甘特圖（支援 matplotlib / plotly）
timeline.plot(fs, total_duration_s=2e-6, show=True)
timeline.plot_schedule(fs, backend="plotly", show=True)
```
