# QAWG 底層優化 — 2026-09-08

## 改動

1. **DDC**：合併 factor 2 與 LO reference，避免 `2 * records` 的整批實數暫存；短 record 的 LO reference 使用最多 16 組唯讀快取，超過 65536 samples 的 reference 不快取。回傳陣列不共用快取記憶體。舊 `demodulation.digital_downconvert` 改委派同一實作。
2. **FIR**：`fir_lowpass_iq(..., batch_records=128)` 以小批次計算 FFT、頻域乘法原地完成。回傳為獨立且連續的精確大小陣列，不再讓 cropped output 仍持有整塊 padded FFT allocation。接受 strided／多維／空 batch 輸入，保留最後一軸時間、centered convolution 與相同邊界慣例。輸入和全部輸出仍在 RAM，這不是 acquisition streaming。
3. **Weighted integration**：使用不共軛的 einsum contraction，減少整批乘權重暫存；維持既有 sum(abs(weights)) normalization。
4. **DMA**：`copy_dma_buffer(..., out=...)` 直接複製到最終 record array，並在 repost buffer 前完成 copy；雙通道從 interleaved view 解包，不多做一份中間副本。原呼叫不傳 out 時仍回傳獨立副本。
5. **資源管理**：free_capture 嘗試釋放所有 buffers，若其中一個 OS free 失敗，保留失敗項目供重試，不重複釋放已成功項目。active DMA 仍禁止 free。拒絕 double-start、released-session start、非 16-bit transfer containers，以及超過 SDK U32 byte-count 的 buffer。

沒有改動 FIR cutoff、IF frequency、shot averaging、相位校正物理模型或既有單／雙通道軸順序。數值重排會帶來浮點 roundoff 差異，測試以明確 tolerance 驗證，不宣稱 bit-for-bit 一致。

## 離線 benchmark

執行：`python diagnostics/benchmark_lowlevel.py`（labberenv）。腳本內保存本次優化前的 DDC/FIR 公式作參考，先驗證結果一致，再 warmup，計時五次取 median。

Python 3.8.20、NumPy 1.24.4；1024 shots × 2048 samples；181 taps。

| 項目 | 原實作 | 優化後 |
|---|---:|---:|
| DDC median | 11.72 ms | 7.76 ms |
| DDC traced peak | 48.30 MiB | 32.25 MiB |
| FIR median | 133.22 ms | 132.58 ms |
| FIR traced peak | 128.19 MiB | 64.07 MiB |
| FIR output backing allocation | 64 MiB | 32 MiB |

這台主機上 DDC 約 1.5 倍速度；FIR 主要節省記憶體，速度沒有實質提升。峰值是 tracemalloc 可追蹤的本次配置，不含原先 input，不等於完整 process RSS；不同 NumPy、CPU、record 長度或 batch size 會不同。没有實機 DMA 吞吐量結果。

原始數字：`diagnostics/lowlevel_benchmark.json`。可增加／降低 batch_records 調整 FFT 暫存，對很長 record，單筆 FFT 仍可能很大。

## 驗證

全套 **45 tests passed**。新增涵蓋 strided 多維 FIR 對直接 convolution、空 batch、kernel 比 record 長、輸入不變與 output ownership、DDC phase convention／cache mutation isolation、複數權重不共軛、釋放錯誤後安全重試、重複啟動拒絕。原本雙通道 pairing、reference-phase correction、mode HDF5 與 Wigner tests 同時通過。

Waveguide demo 使用相同公開 API，無需更改實驗設定。硬體驗證仍需在實際 ATS9371 上確認 sustained capture、buffer overflow、firmware layout 與 abort 行為。
