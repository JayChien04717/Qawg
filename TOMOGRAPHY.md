# QAWG Heterodyne State Tomography

本文件說明 QAWG 中 traveling microwave mode tomography 的物理原理、硬體
流程、逐 shot 計算、密度矩陣重建、Wigner function，以及 3D cavity
ring-down 的測試方法。

相關檔案：

- `QAWG/tomography.py`：temporal mode、IQ calibration、maximum-likelihood
  density matrix 與 Wigner function
- `tomography.ipynb`：AWG 直接產生測試波形，驗證 AWG-to-Alazar pipeline
- `cavity_ringdown.ipynb`：3D cavity fill-and-ring-down 測試
- `QAWG/examples.py`：`CavityRingdownProgram`

## 1. 量測目標

Alazar 實際量到的是每次 trigger 的實數電壓波形：

$$
v_k(t), \qquad k=1,\ldots,N_{\rm shot}.
$$

Tomography 的目標不是平均波形，而是從每個 shot 取得一個 complex
temporal-mode sample：

$$
S_k=I_k+iQ_k.
$$

所有 $\{S_k\}$ 的統計分布才包含 state fluctuation、高階 moments 與
density matrix 的資訊。

正確資料流程為：

```text
Alazar records: (shot, time)
        |
        v
每個 shot digital downconversion
        |
        v
complex baseband: (shot, time)
        |
        v
每個 shot temporal-mode projection
        |
        v
heterodyne samples: (shot,)
        |
        v
calibration / noise correction
        |
        v
density matrix rho
        |
        +--> photon population P(n)
        |
        +--> Wigner function W(alpha)
```

不能先執行：

```python
average_record = records.mean(axis=0)
```

再使用平均波形做 tomography。這只保留一階平均場
$\langle S\rangle$，會丟失 shot-to-shot fluctuations。

平均波形只適合用來：

- 檢查 trigger timing
- 找 signal window
- 擬合 cavity lifetime
- 檢查 phase drift、leakage 與 saturation

## 2. Digital downconversion

對 carrier 或 intermediate frequency $f_{\rm IF}$，每個 shot 做：

$$
z_k(t)=2v_k(t)
\exp[-i(2\pi f_{\rm IF}t+\phi_{\rm ref})].
$$

QAWG 對應函式：

```python
from QAWG.alazar import digital_downconvert

baseband = digital_downconvert(
    records_volts,
    sample_rate_hz=1e9,
    intermediate_frequency_hz=50e6,
)
```

建議再對每個 shot 的 complex baseband 做相同 low-pass：

```python
from QAWG.alazar import AlazarProcessor

processor = AlazarProcessor(sample_rate_hz=1e9)
baseband = processor.apply_butterworth_lpf(
    baseband,
    cutoff_hz=20e6,
    order=4,
)
```

資料 shape：

```text
records_volts: (number_of_shots, adc_samples)
baseband:      (number_of_shots, adc_samples)
```

因為 ADC 輸入是 real signal，混頻後除了 DC/baseband，也會出現
$2f_{\rm IF}$ image：

$$
2\cos(2\pi f_{\rm IF}t)e^{-i2\pi f_{\rm IF}t}
=1+e^{-i4\pi f_{\rm IF}t}.
$$

實驗資料應在 temporal-mode projection 前低通，或確保 matched-filter
window 對 $2f_{\rm IF}$ 有足夠抑制。若結果會隨 window 起點移動一兩個
carrier sample 而大幅變化，通常代表 image suppression 不足。

## 3. Temporal mode

Traveling field 是連續時間訊號。要將它視為一個 bosonic mode，必須定義
normalized temporal mode：

$$
\hat a_f=\int dt\,f^*(t)\hat a_{\rm out}(t),
\qquad
\int dt\,|f(t)|^2=1.
$$

離散資料中：

$$
S_k=\sum_n f_n^*z_{k,n},
\qquad
\sum_n|f_n|^2=1.
$$

這是 matched-filter inner product，不需要先產生 sliding convolution。

QAWG 使用：

```python
from QAWG.tomography import (
    project_temporal_mode,
    temporal_mode_weights,
)

mode = temporal_mode_weights(
    mode_samples,
    kind="exponential",
    decay_samples=t_cavity * sample_rate_hz,
)

shot_iq = project_temporal_mode(
    baseband,
    mode,
    start_sample=mode_start,
)
```

輸出：

```text
shot_iq: (number_of_shots,)
```

### 3.1 Boxcar mode

$$
f(t)=\text{constant}
$$

適合 steady-state tone 或 resonance fluorescence 的固定時間窗。

### 3.2 Gaussian mode

$$
f(t)\propto
\exp\left[-\frac{(t-t_0)^2}{2\sigma^2}\right].
$$

適合 Gaussian pulse 或從資料最佳化出的近似 Gaussian mode。

### 3.3 Exponential mode

若 cavity energy 或 qubit excited-state population 為：

$$
n(t)\propto e^{-t/T_1},
$$

則場振幅 mode 為：

$$
f(t)\propto e^{-t/(2T_1)}.
$$

注意 factor 2：

- field amplitude：$e^{-t/(2T_1)}$
- power/photon flux：$e^{-t/T_1}$

若 window 長度是 $T$，理想 exponential photon flux 被收集的比例為：

$$
\eta_{\rm window}=1-e^{-T/T_1}.
$$

常見數值：

| Window | Collected energy |
|---|---:|
| $1T_1$ | 63.2% |
| $3T_1$ | 95.0% |
| $5T_1$ | 99.3% |

因此 window 不需要包含無限長 decay，但 mode 必須在實際 window 內重新
normalization。

## 4. Reference 與 input pulse cancellation

接收場通常可寫成：

$$
a_{\rm measured}(t)
=\alpha_{\rm input}(t)+a_{\rm emission}(t)+h(t),
$$

其中：

- $\alpha_{\rm input}$：coherent drive leakage/reflection
- $a_{\rm emission}$：待測 emission
- $h$：amplifier 與 receiver noise

### 4.1 Digital coherent subtraction

若 receiver 未飽和，可以量相同 timing 的 reference：

- qubit detuned/off
- cavity drive channel off，但 marker 保留
- 或沒有待測 emission 的狀態

再扣除 reference 的 ensemble mean：

$$
z_k^{\rm corrected}(t)
=z_k^{\rm signal}(t)
-\langle z^{\rm reference}(t)\rangle.
$$

程式：

```python
reference_mean = reference_baseband.mean(axis=0)
corrected = signal_baseband - reference_mean[None, :]
```

也可以投影後做 displacement：

```python
corrected_iq = signal_iq - reference_iq.mean()
```

只能扣除 reference 的 ensemble average。不要為每個 signal shot 單獨
fit 並扣除自己的波形，否則可能同時刪除 tomography 所需的 fluctuation。

### 4.2 Physical cancellation

如果 input pulse 造成以下問題，必須考慮實體 cancellation path：

- amplifier、mixer 或 ADC saturation
- pulse ring-down 蓋住 emission
- receiver dynamic range 不足
- amplitude/phase drift 造成巨大 subtraction residual

Digital subtraction 無法恢復已經 clipped 或 saturated 的資料。

## 5. 3D cavity ring-down 實現

建議硬體架構：

```text
AWG IF --------\
                IQ mixer + microwave LO --> cavity input
AWG Q --------/

cavity output --> amplifier/downconversion --> Alazar signal input
AWG marker ---------------------------------> Alazar trigger input

AWG, LO and Alazar share a frequency reference
```

實驗 sequence：

```text
cavity fill pulse starts
marker rising edge triggers ATS
cavity fill pulse stops
optional guard time
ATS delayed acquisition begins
cavity free ring-down is recorded
```

QAWG declarative program：

```python
from QAWG import CavityRingdownProgram, ns, us

cfg = {
    "frequency": 50e6,          # AWG/Alazar IF, not cavity RF
    "awg_ch": 3,
    "marker_ch": 1,
    "adc_channel": "CHA",
    "channel_amplitude_vpp": 0.5,
    "drive_length": 2 * us,
    "drive_gain": 0.0002,
    "edge_sigma": 20 * ns,
    "ringdown_guard": 40 * ns,
    "acquire_length": 1.5 * us,
}

program = CavityRingdownProgram(cfg)
compiled = program.compile(hardware=experiment)
result = compiled.acquire(n_average=5000)
```

其 trigger delay 為：

$$
t_{\rm trigger\,delay}
=t_{\rm drive}+t_{\rm guard}.
$$

未平均 records：

```python
records = result.raw[:, 0, :]
```

shape：

```text
(number_of_shots, adc_samples)
```

不要使用 `result.trace_average()` 當作 tomography input。

### 5.1 Lifetime 與 loaded Q

自由 ring-down 的 field amplitude：

$$
|a(t)|=A e^{-t/(2T_{\rm cav})}+C.
$$

擬合得到的 $T_{\rm cav}$ 是 energy lifetime。Loaded quality factor：

$$
Q_L=\omega_c T_{\rm cav}
=2\pi f_cT_{\rm cav}.
$$

這裡 $f_c$ 必須是 cavity 的實際 GHz resonance，不是 50 MHz IF。

## 6. IQ calibration

Temporal-mode projection 後得到的原始資料單位通常是 volt-weighted sample，
不是 dimensionless field amplitude $\alpha$。

### 6.1 室溫 pipeline normalization

`normalize_heterodyne_reference()` 將 reference 設為：

$$
\langle \alpha_{\rm ref}\rangle=0,
\qquad
\langle|\alpha_{\rm ref}|^2\rangle=1.
$$

```python
from QAWG.tomography import normalize_heterodyne_reference

alpha_reference, (alpha_signal,), offset, scale = (
    normalize_heterodyne_reference(reference_iq, signal_iq)
)
```

這適合檢查：

- IQ cloud
- ML algorithm
- density-matrix physicality
- plotting pipeline

但室溫 electronic noise 不是 vacuum noise。因此這個 normalization 不能直接
給出真實 photon number。

### 6.2 真正 photon-unit calibration

量子 tomography 至少需要決定：

- input attenuation
- receiver power/voltage gain
- system added noise
- IQ imbalance
- phase reference
- temporal-mode normalization 對應的 physical bandwidth

常用 calibration 包括：

- calibrated coherent tone
- hot/cold load 或 variable-temperature noise source
- known thermal source
- qubit/cavity known-state calibration
- signal-on 與 signal-off moments 的 amplifier-noise deconvolution

只有 samples 已正確轉成 dimensionless $\alpha$ 後，density matrix、
photon population 與 Wigner function 才有量子物理意義。

## 7. Heterodyne probability

理想 heterodyne measurement 對應 coherent-state POVM：

$$
\Pi(\alpha)=\frac{1}{\pi}|\alpha\rangle\langle\alpha|.
$$

量測分布是 Husimi Q function：

$$
p(\alpha|\rho)
=Q(\alpha)
=\frac{1}{\pi}\langle\alpha|\rho|\alpha\rangle.
$$

截斷到 Fock cutoff $N_c$：

$$
|\alpha\rangle
=e^{-|\alpha|^2/2}
\sum_{n=0}^{N_c-1}
\frac{\alpha^n}{\sqrt{n!}}|n\rangle.
$$

QAWG 的 `heterodyne_ml_density_matrix()` 使用 diluted
$R\rho R$ iteration，尋找滿足：

$$
\rho=\rho^\dagger,\qquad
\rho\geq0,\qquad
\operatorname{Tr}\rho=1
$$

且最大化 heterodyne samples likelihood 的 density matrix。

```python
from QAWG.tomography import heterodyne_ml_density_matrix

rho = heterodyne_ml_density_matrix(
    alpha_signal,
    cutoff=8,
    iterations=200,
    dilution=0.5,
)
```

目前 implementation 假設輸入已經是 ideal heterodyne samples。它沒有自動
完成一般 phase-insensitive amplifier 的 added-noise deconvolution。

## 8. Photon population

Fock-basis photon-number population 是 density matrix 對角線：

$$
P_n=\langle n|\rho|n\rangle=\rho_{nn}.
$$

```python
population = np.real(np.diag(rho))
population = np.maximum(population, 0.0)
population /= population.sum()
```

平均 photon number：

$$
\langle n\rangle=\sum_n nP_n.
$$

零延遲二階 correlation：

$$
g^{(2)}(0)
=\frac{\langle n(n-1)\rangle}{\langle n\rangle^2}
=\frac{\sum_n n(n-1)P_n}
{\left(\sum_n nP_n\right)^2}.
$$

```python
n = np.arange(len(population))
mean_n = np.sum(n * population)
g2 = np.sum(n * (n - 1) * population) / mean_n**2
```

Wigner 圖上的紅藍環不是 photon population。Photon population 必須從
$\rho_{nn}$ 讀取。

### 8.1 Fock cutoff 檢查

如果最高保留態仍有明顯 population：

```python
population[-1] > 0.01
```

應提高 `FOCK_CUTOFF`。否則 Wigner function 邊界可能產生同心振盪等
truncation artifacts。

## 9. Wigner function

由重建的 density matrix 計算 displaced parity：

$$
W(\alpha)
=\frac{2}{\pi}
\operatorname{Tr}
\left[
D(-\alpha)\rho D^\dagger(-\alpha)\Pi
\right],
$$

其中：

$$
D(\alpha)=e^{\alpha a^\dagger-\alpha^*a},
\qquad
\Pi=(-1)^{a^\dagger a}.
$$

QAWG：

```python
from QAWG.tomography import wigner_function

axis = np.linspace(-2, 2, 81)
wigner = wigner_function(rho, axis, axis)
```

基本檢查：

$$
\int d^2\alpha\,W(\alpha)\approx1.
$$

Vacuum 在原點：

$$
W(0)=\frac{2}{\pi}.
$$

Wigner negativity 只有在以下條件成立時才可信：

- photon-unit calibration 正確
- receiver noise 已校正
- Fock cutoff 足夠
- sample 數足夠
- ML reconstruction 收斂
- 沒有 saturation 或 clipping
- negativity 對 analysis window、cutoff 與 bootstrap resampling 穩定

## 10. Moments

從 density matrix 可計算 normal-ordered moments：

$$
\left\langle(a^\dagger)^m a^n\right\rangle
=\operatorname{Tr}
\left[
\rho(a^\dagger)^m a^n
\right].
$$

例如：

$$
\langle a\rangle,\qquad
\langle a^\dagger a\rangle=\langle n\rangle,\qquad
\langle(a^\dagger)^2a^2\rangle.
$$

若直接從 amplifier output samples 推回 input-field moments，必須另外處理
gain 與 added-noise moments。單純對室溫 IQ samples 計算
$\langle(S^*)^mS^n\rangle$ 不等於 qubit/cavity emission 的量子 moments。

## 11. 建議實驗步驟

### 階段 A：AWG direct loopback

1. AWG 產生 flat-top 或 exponential carrier。
2. Marker trigger Alazar。
3. 保留每個 raw record。
4. 每 shot demodulate。
5. 每 shot temporal-mode projection。
6. 檢查 IQ cloud 與 reference normalization。
7. 驗證 ML、photon-population plot 與 Wigner plot。

結果只能解讀為 classical pipeline validation。

### 階段 B：室溫 3D cavity

1. VNA 找到 cavity resonance 與 linewidth。
2. 用長 pulse fill cavity。
3. drive 關閉後 delayed acquisition。
4. 擬合 field decay，取得 $T_{\rm cav}$。
5. 使用 $e^{-t/(2T_{\rm cav})}$ matched filter。
6. 比較 reference 與 ring-down IQ clouds。
7. 檢查 trigger jitter、phase coherence 與 receiver saturation。

結果可驗證 cavity dynamics，但室溫 cavity 是 thermal/classical system。

### 階段 C：低溫量子 tomography

1. 建立 vacuum/noise reference。
2. 校正 gain、added noise 與 IQ imbalance。
3. 檢查 input pulse 是否需要 physical cancellation。
4. 每 shot 擷取與 temporal-mode projection。
5. noise correction 或 calibrated heterodyne likelihood。
6. 重建 $\rho$。
7. 計算 $P_n$、moments 與 $W(\alpha)$。
8. 對 cutoff、window、sample count 與 calibration uncertainty 做穩定性分析。

## 12. 常見錯誤

- 先平均 waveform 再 tomography
- 把 field decay 寫成 $e^{-t/T_1}$，造成 power decay 太快
- 使用 IF frequency 計算 cavity $Q$
- 將室溫 receiver noise 當成 vacuum
- 沒有低通就直接解讀 $2f_{\rm IF}$ 振盪
- 對每個 signal shot 個別 subtract fit，刪除真正 fluctuation
- receiver saturation 後仍嘗試 digital subtraction
- Fock cutoff 太小，將 Wigner 邊界環誤認為物理結構
- 將 Wigner 色環誤認為 photon-number population

## 13. 最小分析範例

```python
import numpy as np

from QAWG.alazar import AlazarProcessor, digital_downconvert
from QAWG.tomography import (
    heterodyne_ml_density_matrix,
    normalize_heterodyne_reference,
    project_temporal_mode,
    temporal_mode_weights,
    wigner_function,
)

# records shape: (shots, samples)
reference_baseband = digital_downconvert(
    reference_records,
    sample_rate_hz,
    if_frequency_hz,
)
signal_baseband = digital_downconvert(
    signal_records,
    sample_rate_hz,
    if_frequency_hz,
)

processor = AlazarProcessor(sample_rate_hz)
reference_baseband = processor.apply_butterworth_lpf(
    reference_baseband,
    cutoff_hz=20e6,
)
signal_baseband = processor.apply_butterworth_lpf(
    signal_baseband,
    cutoff_hz=20e6,
)

mode = temporal_mode_weights(
    mode_samples,
    kind="exponential",
    decay_samples=t1_s * sample_rate_hz,
)

reference_iq = project_temporal_mode(
    reference_baseband,
    mode,
    start_sample=mode_start,
)
signal_iq = project_temporal_mode(
    signal_baseband,
    mode,
    start_sample=mode_start,
)

alpha_reference, (alpha_signal,), _, _ = (
    normalize_heterodyne_reference(reference_iq, signal_iq)
)

rho = heterodyne_ml_density_matrix(
    alpha_signal,
    cutoff=8,
    iterations=200,
)

population = np.real(np.diag(rho))
axis = np.linspace(-2, 2, 81)
wigner = wigner_function(rho, axis, axis)
```

這段程式展示完整資料方向，但 `normalize_heterodyne_reference()` 在室溫只
提供數值 normalization。真正 photon population 仍需要實際 receiver
calibration 與 noise correction。
