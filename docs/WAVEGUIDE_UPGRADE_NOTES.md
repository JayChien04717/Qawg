# QAWG waveguide upgrade notes — 2026-09-08

後續底層記憶體／DSP 優化、benchmark 與最新 45-test 結果見 [LOWLEVEL_OPTIMIZATION_NOTES.md](LOWLEVEL_OPTIMIZATION_NOTES.md)。

## 目的

支援 50 MHz IF 的 time-resolved fluorescence 與後續 tomography 資料保存。以 ADC A 為 signal、B 為獨立 pilot；不從 emission 自己估計逐 shot 相位。此記錄對應本次 code、demo 與離線驗證；尚未連線做硬體量測。

## 新增／修改

- `QAWG/alazar/ats9371.py`：`CaptureConfig(channel=3)` 啟用 A|B；buffer 容量加倍並明確設定 sample interleave。DMA 解析為 `(shots, 2, samples)`，channel axis 順序固定 A、B。單通道仍是 `(shots, samples)`。
- `CaptureData.channels`、`channel_records(1 or 2)`：取得具名通道資料，不猜軸。
- `QAWG/alazar/waveguide.py`：`IFProcessor` 驗證 real IF／Nyquist／最近 aliased image、整個通帶誤差與阻帶抑制。預設 1 GS/s、50 MHz IF、pass 30 MHz、stop 70 MHz、181 taps、beta 8。這是可驗證的 DSP 範例，不是已量測的硬體頻寬。
- `reference_phase_correct`：A/B matching shots、獨立 pilot 的固定 window、明確 amplitude threshold。弱 pilot 會拋錯，不丟棄或自行旋轉 emission。`expected_reference_phase` 可指定 scalar 或每 shot 預期相位，避免消除刻意 programming 的相位。
- `AWGAlazar.acquire_referenced_waveforms`：沿用已上傳的 AWG program，啟動 ADC 後才播放，finally 停止 AWG。回傳 `(capture, full_corrected_iq, phases)`；不積分、不平均、不自動扣整段 DC，不重排 sweep labels。依舊使用既有 trigger/clock 配置。
- `save_mode_shots`：以 HDF5 exclusive create 保存完整 IQ、每 shot mode IQ、time、weights、FIR coefficients、processor settings、condition metadata、可選 phase/raw codes。mode 必須 unit L2，且 FIR guard 內 weights 為零。輸出明確標註電壓投影，不是 photon units。既有檔案不覆蓋。
- `QAWG/Labber_saver.py`：延後 type annotations 求值，修正 Python 3.8 labberenv 在 import 時遇到 `dict[str, bytes]` 的相容性問題。

## 使用入口

先開專案根目錄的 `Waveguide_demo.ipynb`。預設合成資料可 Run All，不連線、不輸出 RF。`SAVE=False`、`RUN_HARDWARE=False`。

```python
from QAWG.alazar import IFProcessor

processor = IFProcessor(passband_hz=30e6, stopband_hz=70e6, taps=181)
# experiment 已完成 clock/trigger 配置與 pulse upload。
capture, iq_shots, phase = experiment.acquire_referenced_waveforms(
    1000, processor=processor,
    reference_window_samples=(200, 600),
    min_reference_amplitude=0.01,  # volts；必須以實測 pilot 調整
)
raw_a = capture.channel_records(1)
raw_b = capture.channel_records(2)
```

低階入口也可 `Alazar9371.configure(channel=3, ...)`，搭配 `CaptureConfig(..., channel=3)`。只呼叫 arm/capture 不會替你決定時脈或接線；完整 record 必須包含 FIR 邊界緩衝和獨立 pilot window。

`demo.ipynb`、`Qawgdemo.ipynb`、`multiplex.ipynb` 已加入新流程入口與濾波／tomography 說明。既有硬體 cells 沒有自動執行，原本單通道預設未改，以免讓既有量測靜默改變。`Qawgdemo` 的 ideal ML/RMS-normalization 範例仍是理想模型示範，不得視為 HEMT-added-noise 的實驗校正。

## 相容性與資料語義

- 新入口專用同步 A/B。既有 `run_compiled`、`acquire`、`acquire_decimate` 仍是原本單通道 API，不接受 `adc_channel=3`；要用新入口或低階 capture。
- 新入口的 `last_records_volts` 是 3D（shot/channel/sample）。不要拿它直接餵需要 2D 的舊 analysis，請用 `capture.channel_records(...)`。
- samples_per_record 仍按 ATS9371 對齊。time axis 從 record 第一 sample 開始，trigger delay／channel gain／RF frequency 等由 caller metadata 保存。
- IQ 是 full rate、centered FIR 輸出；181 taps 頭尾各 90 samples 不適合模式投影。沒有用 boxcar 再平均 waveform。
- B 要追 RF source/receive LO phase，必須共享相關 RF/LO 路徑；單純 AWG 50 MHz 直通不能追 SGS 間的全部漂移。B 也不是第二條 noise-independent receiver。
- `save_mode_shots` 是研究用 HDF5 schema，不宣稱為 Labber Log Browser 格式。正式量測按 condition/block 保存；可另外輸出 Labber 平均圖。

## 本次驗證

新增 tests 包含：不同 channel/record 值的交錯 DMA 解析與複製、buffer allocation/flags/順序、單通道格式、30 MHz envelope 恢復、無法達標的 FIR 拒絕、pilot 修正保留 intrinsic phase、弱 pilot 拒絕、HDF5 mode roundtrip、不覆蓋檔案／guard 拒絕，以及 AWG 啟動順序與擷取錯誤停止。

全套 pytest 在 Python 3.8 `labberenv` 執行：**37 passed**；pytest 安裝在專案 `.test-deps`，未修改 labberenv 套件。Waveguide demo 的離線 cells 已實際執行並保存輸出；另外三個 demo 已驗證 notebook 結構，未操作儀器。

硬體驗收仍需：A/B 分別送不同已知 tone，確認 SDK/firmware 的 interleaving 和 channel phase；pilot on/off 查串音；buffer overflow/長時間同步；loopback 幅相響應；真實 trigger/clock 設定。數值/mock tests 不是硬體驗證。

## 尚未完成的量子校正

本次完成的是「擷取 → 寬頻 IQ → 獨立 phase reference → 保存模式 shots」。沒有憑空新增一個通用 HEMT noise model，也沒有宣稱輸出真實 photon rho/Wigner。

現有 `heterodyne_ml_density_matrix` 仍為 ideal heterodyne；實驗重建下一步需你提供 signal/vacuum/coherent calibration blocks、gain/reference plane、loss/noise 穩定性，再選 noise-aware likelihood 或 corrected moments。硬體 absence of TWPA、增益與 mode mismatch 不能用任意 RMS normalization 消去。

## DMA 格式來源

[AlazarBeforeAsyncRead 官方 SDK：ADMA_INTERLEAVE_SAMPLES](https://docs.alazartech.com/ats-sdk-user-guide/latest/reference/AlazarBeforeAsyncRead.html) 定義交錯格式 `R0[A0 B0 A1 B1 ...], R1[...]`；本次明確啟用該 flag，不依賴 generic NPT 的非交錯 buffer 假設。此前實驗研究見 `waveguide_signal_research/實驗與訊號處理研究.md`。
