# QAWG AWG5208

`QAWG.awg5200` 負責 Tektronix AWG5208 的 waveform upload、marker、
sequence 與 SCPI hardware control。這一層不處理 Alazar data，也不定義
實驗 sweep 規則。

Timeline helper 的實作已搬到上層 `QAWG.timeline`，因為 timeline 是
hardware-independent experiment-description layer，不是 AWG5208 driver
boundary。`QAWG.awg5200` 仍 re-export timeline helpers 以維持舊程式相容。

## 模組責任

```text
awg5200/
├── waveforms.py  envelope、carrier modulation、marker 與 WFMX encoding
├── timeline.py   compatibility re-export for QAWG.timeline
├── driver.py     AWG5208 SCPI、upload、marker、sequence 與 playback
└── transport.py  PyVISA transport boundary
```

## 安裝

只有實際連接 AWG 時才需要 PyVISA：

```powershell
pip install numpy pyvisa pyvisa-py
```

LAN VISA resource 範例：

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

使用完畢後：

```python
awg.close()
```

也可以使用 context manager：

```python
with AWG5208.connect("TCPIP0::192.168.10.171::inst0::INSTR") as awg:
    awg.set_awg_mode()
    awg.set_sample_rate(2.5e9)
```

## Envelope 與 carrier

Waveform rendering 分成兩個步驟：

```text
waveform(t) = gain * envelope(t) * sin(2*pi*frequency*t + phase)
```

使用者負責選擇或建立 envelope。後段負責 sample conversion、gain、
carrier frequency 與 phase。

內建 envelope helpers：

- `constant`
- `gaussian`
- `gaussian_square`
- `cosine_square`
- `triangle_ns`

### Cosine-square

`cosine_square` 是：

```text
half-cosine rise -> flat top -> half-cosine fall
```

Rise 與 fall 使用相同的 `edge_length`：

```python
from QAWG.awg5200 import cosine_square_ns

envelope = cosine_square_ns(
    duration_ns=1000,
    sample_rate_hz=2.5e9,
    edge_length_ns=20,
    amplitude_volts=0.02,
)
```

直接加入 carrier：

```python
from QAWG.awg5200 import modulate_envelope

waveform_volts = modulate_envelope(
    envelope,
    sample_rate_hz=2.5e9,
    frequency_hz=250e6,
    phase_radians=0.0,
)
```

在 experiment compiler 中則使用：

```python
self.add_pulse(
    "readout",
    gen="res",
    style="cosine_square",
    length=1e-6,
    edge_length=20e-9,
    frequency=250e6,
    phase=0.0,
    gain=0.02,
)
```

## Timeline

`waveform()` 接收 envelope，並記錄 carrier、channel、phase 與 gain：

```python
from QAWG import waveform
from QAWG.awg5200 import cosine_square_ns

envelope = cosine_square_ns(
    duration_ns=1000,
    sample_rate_hz=2.5e9,
    edge_length_ns=20,
    amplitude_volts=0.25,
)

readout = waveform(
    envelope,
    fc=250e6,
    ch=3,
    phase_radians=0.0,
    gain=0.08,
    name="readout",
)
```

Timing operators：

- `delay(dt)`：下一個 waveform 相對於前一個 waveform 的開始時間。
- `delay_auto(dt)`：下一個 waveform 相對於前一個 waveform 的結束時間。
- `parallel(...)`：多個 waveform 同時開始，可位於不同 channel 或同一 channel。

```python
from QAWG import delay_auto, parallel, waveform

drive_q0 = waveform(envelope, fc=50e6, ch=3, gain=0.02)
drive_q1 = waveform(envelope, fc=150e6, ch=3, gain=0.01)
control = waveform(envelope, fc=100e6, ch=4, gain=0.02)

timeline = parallel(drive_q0, drive_q1) / delay_auto(40e-9) / control
```

Render 到共同 sample axis：

```python
from QAWG import align_channel_envelopes, align_channels

channel_waveforms = align_channels(
    timeline,
    sample_rate_hz=2.5e9,
    total_duration_s=5e-6,
)
channel_envelopes = align_channel_envelopes(
    timeline,
    sample_rate_hz=2.5e9,
    total_duration_s=5e-6,
)
```

同一 channel 上的 parallel waveforms 會相加。使用者必須確保總 peak 不超過：

```text
abs(waveform peak) <= amplitude_vpp / 2
```

舊式 import 仍可使用：

```python
from QAWG.awg5200 import waveform, delay_auto, align_channels
```

但新程式建議從 `QAWG` 或 `QAWG.timeline` 匯入 timeline helpers。

## Upload waveform 與 marker

```python
uploaded = awg.upload_timeline(
    timeline,
    amplitude_vpp={3: 0.5, 4: 0.5},
    name_prefix="experiment",
    total_duration_s=5e-6,
)

marker_name = awg.marker(
    waveform_ch=3,
    marker_ch=1,
    marker_number=1,
    low_volts=0.0,
    high_volts=1.2,
)

awg.run()
```

`marker()` 依 reference waveform 的 active region 建立 marker。Marker 使用
AWG channel 的 marker bits，因此會降低該 channel 可用 DAC resolution。

## Sequence

先上傳每個 waveform asset，再建立各 channel 的 sequence track：

```python
step_0 = awg.upload_waveform_asset(
    "step_0_ch3",
    waveform_volts=channel_waveforms[3],
    amplitude_vpp=0.5,
)
step_1 = awg.upload_waveform_asset(
    "step_1_ch3",
    waveform_volts=channel_waveforms[3],
    amplitude_vpp=0.5,
)

sequence_name = awg.create_sequence(
    "readout_sequence",
    tracks={3: [step_0, step_1]},
    repetitions=1,
    goto_step=1,
)
```

一般實驗建議由 `ExperimentProgram.compile()` 與 `AWGAlazar` 管理 sequence，
不要手動重複實作 record ordering。

## Differential output 與 IQ

AWG 單一 channel 的 `+/-` outputs 是 differential pair：

```text
CH3- = -CH3+
```

它們相差 180 度，不是獨立的 I 與 Q。要產生 IQ single-sideband modulation，
通常需要兩個 AWG channels：

```text
CH3 +/- -> RF source I differential input
CH4 +/- -> RF source Q differential input
```

並令兩個 channel 的 carrier phase 相差 90 度。實際 sideband 方向取決於
RF source 的 IQ convention。

## Hardware limits

- Channel number：1 到 8。
- Sample rate：1.49 kSa/s 到 2.5 GSa/s。
- WFMX waveform 至少 2400 samples。
- Channel amplitude：0.25 到 1.5 Vpp。
- Marker number：1 到 4。
- Carrier frequency 必須介於 DC 與 Nyquist frequency。

延伸文件：

- [TIMELINE_TUTORIAL.md](TIMELINE_TUTORIAL.md)
- [PHASE_SWEEP_DEBUG.md](PHASE_SWEEP_DEBUG.md)
- [SEQUENCE_T1_DESIGN.md](SEQUENCE_T1_DESIGN.md)
