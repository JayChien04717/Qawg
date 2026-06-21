"""Generate the Chinese microwave-photon tomography tutorial and simulations."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "tomography_assets"
ASSET_DIR.mkdir(parents=True, exist_ok=True)

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
    "blue": "#5477C4",
    "blue_light": "#A3BEFA",
    "orange": "#CC6F47",
    "orange_light": "#F0986E",
    "gold": "#B8A037",
    "olive": "#71B436",
    "neutral": "#7A828F",
    "neutral_light": "#E2E5EA",
}


def use_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.titlecolor": TOKENS["ink"],
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft JhengHei",
                "Noto Sans CJK TC",
                "Segoe UI",
                "DejaVu Sans",
            ],
        },
    )
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["mathtext.fontset"] = "dejavusans"


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(ASSET_DIR / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def sample_heterodyne_fock(
    photon_number: int,
    shots: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample the ideal Husimi-Q distribution of |n>."""
    radius_squared = rng.gamma(shape=photon_number + 1, scale=1.0, size=shots)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=shots)
    return np.sqrt(radius_squared) * np.exp(1j * phase)


def wigner_fock_mixture(
    alpha_abs: np.ndarray,
    p0: float,
    p1: float,
) -> np.ndarray:
    radius_squared = alpha_abs**2
    return (
        2.0
        / np.pi
        * np.exp(-2.0 * radius_squared)
        * (p0 + p1 * (4.0 * radius_squared - 1.0))
    )


use_theme()
rng = np.random.default_rng(20260618)

# ---------------------------------------------------------------------------
# Figure 1: temporal-mode matching and overlap.
# ---------------------------------------------------------------------------
sample_rate_hz = 1e9
time_ns = np.arange(0, 2200, dtype=float)
time_s = time_ns * 1e-9
arrival_ns = 580.0
decay_ns = 260.0
detuning_hz = 1.2e6

true_mode = np.zeros(time_ns.size, dtype=np.complex128)
active = time_ns >= arrival_ns
true_mode[active] = np.exp(
    -(time_ns[active] - arrival_ns) / (2.0 * decay_ns)
) * np.exp(
    1j
    * 2.0
    * np.pi
    * detuning_hz
    * (time_s[active] - arrival_ns * 1e-9)
)
true_mode /= np.linalg.norm(true_mode)

matched_mode = true_mode.copy()
boxcar_mode = np.zeros_like(true_mode)
boxcar_active = (time_ns >= 580) & (time_ns < 2080)
boxcar_mode[boxcar_active] = 1.0
boxcar_mode /= np.linalg.norm(boxcar_mode)

short_mode = np.zeros_like(true_mode)
short_active = (time_ns >= 580) & (time_ns < 1100)
short_mode[short_active] = true_mode[short_active]
short_mode /= np.linalg.norm(short_mode)

overlap_matched = abs(np.vdot(matched_mode, true_mode)) ** 2
overlap_boxcar = abs(np.vdot(boxcar_mode, true_mode)) ** 2
overlap_short = abs(np.vdot(short_mode, true_mode)) ** 2
cumulative_energy = np.cumsum(abs(true_mode) ** 2)

fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.5), sharex=True)
axes[0].plot(
    time_ns,
    abs(true_mode),
    color=TOKENS["blue"],
    label="True complex temporal mode",
)
axes[0].plot(
    time_ns,
    abs(boxcar_mode),
    color=TOKENS["orange"],
    linestyle="--",
    label="Boxcar filter",
)
axes[0].axvspan(580, 1100, color=TOKENS["neutral_light"], alpha=0.5)
axes[0].set_ylabel("Normalized amplitude")
axes[0].set_title("Temporal-mode matching requires the full complex envelope")
axes[0].legend(loc="upper right")
axes[0].grid(True, alpha=0.7)

axes[1].plot(
    time_ns,
    cumulative_energy,
    color=TOKENS["blue"],
    label="Cumulative photon energy",
)
axes[1].axhline(
    overlap_boxcar,
    color=TOKENS["orange"],
    linestyle="--",
    label=f"Boxcar overlap^2 = {overlap_boxcar:.2f}",
)
axes[1].axhline(
    overlap_short,
    color=TOKENS["neutral"],
    linestyle=":",
    label=f"Truncated matched-mode overlap^2 = {overlap_short:.2f}",
)
axes[1].set_xlabel("Time after marker edge (ns)")
axes[1].set_ylabel("Energy / overlap^2")
axes[1].set_ylim(0, 1.05)
axes[1].legend(loc="lower right")
axes[1].grid(True, alpha=0.7)
save_figure(fig, "temporal_mode_matching.png")

# ---------------------------------------------------------------------------
# Figure 2: ideal heterodyne distributions.
# ---------------------------------------------------------------------------
shots = 7000
vacuum = sample_heterodyne_fock(0, shots, rng)
fock_one = sample_heterodyne_fock(1, shots, rng)
coherent_alpha = 1.6 + 0.35j
coherent = coherent_alpha + (
    rng.normal(size=shots) + 1j * rng.normal(size=shots)
) / np.sqrt(2.0)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True, sharey=True)
states = [
    ("Vacuum |0>", vacuum, TOKENS["neutral"]),
    ("Coherent |alpha>", coherent, TOKENS["blue"]),
    ("Fock |1>", fock_one, TOKENS["orange"]),
]
for axis, (title, values, color) in zip(axes, states):
    axis.scatter(
        values.real[::4],
        values.imag[::4],
        s=5,
        alpha=0.13,
        color=color,
        linewidths=0,
    )
    axis.set_title(title)
    axis.set_xlabel("Re(alpha)")
    axis.set_aspect("equal")
    axis.set_xlim(-3.5, 4.0)
    axis.set_ylim(-3.5, 3.5)
    axis.grid(True, alpha=0.6)
axes[0].set_ylabel("Im(alpha)")
fig.suptitle(
    "Ideal heterodyne samples: Fock states have zero mean but a different radial distribution",
    fontsize=13,
    fontweight="semibold",
)
save_figure(fig, "heterodyne_state_distributions.png")

# ---------------------------------------------------------------------------
# Figure 3: loss, Fock populations and Wigner negativity.
# ---------------------------------------------------------------------------
efficiency = np.linspace(0.0, 1.0, 201)
p1 = efficiency
p0 = 1.0 - efficiency
wigner_origin = 2.0 / np.pi * (p0 - p1)
radius = np.linspace(0.0, 2.0, 400)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
axes[0].plot(efficiency, p1, color=TOKENS["blue"], label="P(1)")
axes[0].plot(efficiency, p0, color=TOKENS["orange"], label="P(0)")
axes[0].plot(
    efficiency,
    wigner_origin,
    color=TOKENS["neutral"],
    linestyle="--",
    label="W(0)",
)
axes[0].axvline(0.5, color=TOKENS["ink"], linestyle=":", linewidth=1)
axes[0].axhline(0.0, color=TOKENS["axis"], linewidth=1)
axes[0].set_xlabel("Total efficiency eta")
axes[0].set_ylabel("Population / W(0)")
axes[0].set_title("Loss turns a single photon into a vacuum mixture")
axes[0].legend()

for eta, color, label in (
    (1.0, TOKENS["blue"], "eta=1.00"),
    (0.65, TOKENS["orange"], "eta=0.65"),
    (0.40, TOKENS["neutral"], "eta=0.40"),
):
    axes[1].plot(
        radius,
        wigner_fock_mixture(radius, 1.0 - eta, eta),
        color=color,
        label=label,
    )
axes[1].axhline(0.0, color=TOKENS["axis"], linewidth=1)
axes[1].set_xlabel("|alpha|")
axes[1].set_ylabel("W(alpha)")
axes[1].set_title("Wigner negativity at the origin vanishes for eta <= 0.5")
axes[1].legend()
save_figure(fig, "loss_and_wigner_negativity.png")

# ---------------------------------------------------------------------------
# Figure 4: coherent-state Fock cutoff convergence.
# ---------------------------------------------------------------------------
nbar = 6.0
numbers = np.arange(0, 25)
poisson = np.exp(-nbar) * nbar**numbers / np.asarray(
    [math.factorial(int(number)) for number in numbers],
    dtype=float,
)
cutoffs = np.arange(2, 21)
tail_probability = np.asarray(
    [1.0 - np.sum(poisson[:cutoff]) for cutoff in cutoffs]
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
axes[0].bar(
    numbers,
    poisson,
    color=TOKENS["blue_light"],
    edgecolor=TOKENS["blue"],
    linewidth=0.8,
)
axes[0].axvline(8 - 0.5, color=TOKENS["orange"], linestyle="--", label="cutoff=8")
axes[0].set_xlabel("Fock number n")
axes[0].set_ylabel("P(n)")
axes[0].set_title("Coherent state with mean photon number nbar = 6")
axes[0].legend()

axes[1].semilogy(
    cutoffs,
    tail_probability,
    marker="o",
    color=TOKENS["blue"],
)
axes[1].axhline(0.01, color=TOKENS["orange"], linestyle="--", label="1% tail")
axes[1].set_xlabel("Fock cutoff")
axes[1].set_ylabel("Population above cutoff")
axes[1].set_title("Choose the cutoff from tail convergence")
axes[1].legend()
save_figure(fig, "fock_cutoff_convergence.png")

# Save compact simulation data for reproducibility.
np.savez_compressed(
    ASSET_DIR / "tomography_simulation_data.npz",
    time_ns=time_ns,
    true_mode=true_mode,
    matched_mode=matched_mode,
    boxcar_mode=boxcar_mode,
    cumulative_energy=cumulative_energy,
    vacuum_samples=vacuum,
    coherent_samples=coherent,
    fock_one_samples=fock_one,
    efficiency=efficiency,
    p0=p0,
    p1=p1,
    wigner_origin=wigner_origin,
    fock_numbers=numbers,
    coherent_poisson=poisson,
    cutoffs=cutoffs,
    cutoff_tail_probability=tail_probability,
)

with (ASSET_DIR / "simulation_summary.csv").open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as output:
    writer = csv.writer(output)
    writer.writerow(["metric", "value"])
    writer.writerow(["matched_mode_overlap_squared", overlap_matched])
    writer.writerow(["boxcar_overlap_squared", overlap_boxcar])
    writer.writerow(["short_window_overlap_squared", overlap_short])
    writer.writerow(["wigner_negativity_efficiency_threshold", 0.5])
    writer.writerow(["coherent_mean_photon_number", nbar])
    writer.writerow(["coherent_cutoff_8_tail_probability", tail_probability[6]])

html = r"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>行進微波光子的 Heterodyne Tomography：理論、計算與實作</title>
  <style>
    :root {
      --surface: #FCFCFD; --panel: #FFFFFF; --ink: #1F2430;
      --muted: #6F768A; --grid: #E6E8F0; --axis: #D7DBE7;
      --blue: #5477C4; --blue-light: #EAF1FE;
      --orange: #CC6F47; --orange-light: #FFEDDE;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; background: var(--surface); color: var(--ink);
      font-family: "Microsoft JhengHei", "Noto Sans CJK TC", "Segoe UI", sans-serif;
    }
    main { max-width: 1040px; margin: 0 auto; padding: 44px 24px 80px; }
    header, section { margin-bottom: 42px; }
    h1 { font-size: clamp(2rem, 5vw, 3.3rem); line-height: 1.12; margin: 0 0 14px; }
    h2 { font-size: 1.65rem; line-height: 1.2; margin: 0 0 14px; }
    h3 { font-size: 1.16rem; margin-top: 28px; }
    p, li { line-height: 1.75; }
    .lede, .caption, .note { color: var(--muted); }
    .summary {
      border-left: 5px solid var(--blue); background: var(--blue-light);
      padding: 18px 22px; border-radius: 0 14px 14px 0;
    }
    .warning {
      border-left: 5px solid var(--orange); background: var(--orange-light);
      padding: 16px 20px; border-radius: 0 12px 12px 0;
    }
    .equation {
      overflow-x: auto; background: #F4F5F7; padding: 14px 18px;
      border-radius: 12px; font-family: Consolas, "DejaVu Sans Mono", monospace;
      line-height: 1.7;
    }
    pre {
      overflow-x: auto; background: #171A22; color: #EEF2FF;
      padding: 18px; border-radius: 14px; line-height: 1.55;
    }
    code { font-family: Consolas, "DejaVu Sans Mono", monospace; }
    :not(pre) > code { background: #F0F2F6; padding: 0.12em 0.35em; border-radius: 5px; }
    figure {
      margin: 24px 0; background: var(--panel); border: 1px solid var(--grid);
      border-radius: 16px; padding: 16px;
    }
    figure img { display: block; width: 100%; height: auto; border-radius: 8px; }
    figcaption { color: var(--muted); font-size: 0.92rem; margin-top: 12px; line-height: 1.55; }
    table { width: 100%; border-collapse: collapse; background: var(--panel); }
    th, td { text-align: left; padding: 11px 12px; border-bottom: 1px solid var(--grid); }
    th { color: var(--muted); font-size: 0.9rem; }
    a { color: #2E4780; }
    .two-col { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }
    .card { background: var(--panel); border: 1px solid var(--grid); border-radius: 14px; padding: 16px; }
    .small { font-size: 0.9rem; }
    @media (max-width: 720px) {
      main { padding: 28px 15px 56px; }
      .two-col { grid-template-columns: 1fr; }
      figure { padding: 8px; }
    }
  </style>
</head>
<body>
<main data-report-audience="technical">
  <header data-contract-section="title">
    <h1>行進微波光子的 Heterodyne Tomography</h1>
    <p class="lede">從 continuous IQ trace 到 temporal mode、Fock-basis density matrix 與 Wigner function 的理論、計算與 QAWG 實作。</p>
  </header>

  <section data-contract-section="technical-summary">
    <h2>技術摘要</h2>
    <div class="summary">
      <p><strong>Tomography 的對象不是單一時間點，而是一個正規化的 complex temporal mode。</strong>
      每個 shot 的 IQ trace 必須先投影成一個複數樣本，再用 reference calibration 與 heterodyne POVM 重建 density matrix。</p>
      <p><strong>Temporal-mode mismatch、傳輸損耗與有限 detection efficiency 都會把 Fock state 向 vacuum 混合。</strong>
      對單光子而言，總效率 η 直接給出 P(1)=η、P(0)=1−η；η≤0.5 時原點 Wigner negativity 消失。</p>
      <p><strong>強 coherent pulse 只應用於 signal-path 與 mode calibration。</strong>
      真正的 photon-state acquisition 必須另取資料，沿用同一 temporal mode，並在解讀 photon population 前校正 added noise、效率與 Fock cutoff。</p>
    </div>
  </section>

  <section data-contract-section="key-findings">
    <h2>完整 complex mode 比固定積分窗更可靠</h2>
    <p>Mode overlap 的平方就是 temporal-mode efficiency。只要 filter 在時間、decay constant 或 phase rotation 上不匹配，遺失的正交分量就表現為 loss。下圖的截短 window 即使使用正確 envelope，也只保留部分 photon energy。</p>
    <figure>
      <img src="tomography_assets/temporal_mode_matching.png" alt="Temporal-mode matching simulation">
      <figcaption>Simulation：1 GS/s、arrival 580 ns、field decay 260 ns、residual detuning 1.2 MHz。陰影為被截短的 window。</figcaption>
    </figure>

    <h2>Heterodyne IQ cloud 不等於 photon-number histogram</h2>
    <p>Ideal heterodyne measurement 取樣 Husimi-Q distribution。Coherent state 以平均位移表現；Fock state 的平均場為零，但 radial distribution 與 vacuum 不同。因此不能只比較平均 I/Q，也不能從單張 scatter 圖直接讀出 P(n)。</p>
    <figure>
      <img src="tomography_assets/heterodyne_state_distributions.png" alt="Vacuum coherent and Fock heterodyne distributions">
      <figcaption>每個 panel 由 7,000 個 ideal heterodyne samples 產生。圖中只顯示每四個點之一以保持可讀性。</figcaption>
    </figure>

    <h2>總效率決定單光子是否仍可見非古典性</h2>
    <p>理想 |1〉 經過 efficiency η 的 loss channel 後，density matrix 為 (1−η)|0〉〈0|+η|1〉〈1|。這個關係同時描述 propagation loss、temporal mismatch 與 detector inefficiency；但不同來源必須分開量測，才能知道該改善哪一段鏈路。</p>
    <figure>
      <img src="tomography_assets/loss_and_wigner_negativity.png" alt="Loss and Wigner negativity simulation">
      <figcaption>原點 Wigner 值 W(0)=2(1−2η)/π；η=0.5 是 negativity 的門檻。</figcaption>
    </figure>

    <h2>Fock cutoff 必須由 convergence 決定</h2>
    <p>Cutoff 太小會把真實 population 擠到最高幾個 bins，並可能在有限維 displacement/Wigner 計算中產生假 negativity。對強 coherent calibration pulse，尤其不能沿用為單光子設計的小 cutoff。</p>
    <figure>
      <img src="tomography_assets/fock_cutoff_convergence.png" alt="Fock cutoff convergence simulation">
      <figcaption>Simulation：coherent state n̄=6。cutoff=8 明顯截斷 distribution；實作時應重建多個 cutoff 並檢查最後兩個 bins。</figcaption>
    </figure>
  </section>

  <section data-contract-section="scope-data-and-metric-definitions">
    <h2>量測對象與時間定義</h2>
    <table>
      <thead><tr><th>符號 / 變數</th><th>定義</th><th>QAWG 對應</th></tr></thead>
      <tbody>
        <tr><td>a<sub>out</sub>(t)</td><td>連續時間的輸出場 operator</td><td>每個 shot 的 complex IQ trace</td></tr>
        <tr><td>f(t)</td><td>∫|f(t)|²dt=1 的 complex temporal mode</td><td><code>matched_mode</code></td></tr>
        <tr><td>a<sub>f</sub></td><td>∫f*(t)a<sub>out</sub>(t)dt</td><td><code>project_temporal_mode()</code></td></tr>
        <tr><td>trigger delay</td><td>marker-relative integration 起點</td><td><code>TOMO_MODE_START</code></td></tr>
        <tr><td>acquire window</td><td>保存的 marker-relative record 長度</td><td><code>ACQUIRE_WINDOW</code></td></tr>
        <tr><td>Fock cutoff</td><td>保留 |0〉…|N−1〉 的 Hilbert-space 維度</td><td><code>cutoff</code></td></tr>
      </tbody>
    </table>
    <div class="warning">
      <strong>目前 QAWG timing：</strong>保存的 trace 從 marker edge 開始。若 trigger delay=580 ns、要分析 1.5 µs mode，acquisition 至少要 2.08 µs；建議保留 margin，例如 2.2 µs。
    </div>
  </section>

  <section data-contract-section="methodology">
    <h2>從 IQ trace 到 density matrix</h2>

    <h3>1. 以獨立 coherent calibration 找 temporal mode</h3>
    <div class="equation">
u(t) = 〈S_signal(t)〉 − 〈S_reference(t)〉
f(t) = u(t) / √Σ|u(t)|²
    </div>
    <p>Fock state 本身有 〈a〉=0，因此不能用單光子 traces 的平均值找 envelope。應使用同一 emission path 的 coherent calibration、|0〉+|1〉 calibration，或 covariance/PCA。</p>

    <h3>2. 每個 shot 只投影一次</h3>
    <pre><code>reference_samples = project_temporal_mode(
    reference_iq,
    matched_mode,
    start_sample=mode_start_sample,
)
signal_samples = project_temporal_mode(
    signal_iq,
    matched_mode,
    start_sample=mode_start_sample,
)</code></pre>
    <p>不要先平均 traces 再做 tomography。Tomography 需要保留 shot-to-shot distribution。</p>

    <h3>3. Reference normalization</h3>
    <pre><code>reference_alpha, (signal_alpha,), iq_offset, iq_scale = (
    normalize_heterodyne_reference(reference_samples, signal_samples)
)</code></pre>
    <div class="equation">
α = [S − 〈S_ref〉] / √〈|S_ref − 〈S_ref〉|²〉
    </div>
    <p>這個 normalization 只有在 reference 可視為 measurement input 的 vacuum、且 added noise 已被正確納入 POVM 時，才是完整的 photon-unit calibration。</p>

    <h3>4. Maximum-likelihood reconstruction</h3>
    <pre><code>rho = heterodyne_ml_density_matrix(
    signal_alpha,
    cutoff=10,
    iterations=200,
    dilution=0.5,
)</code></pre>
    <p>QAWG 使用 diluted RρR iteration，維持 ρ≥0、ρ=ρ†、Trρ=1。重建應在多個 cutoff 下重複，並檢查低 photon bins、平均光子數與 tail population 是否收斂。</p>

    <h3>5. Wigner function 最後才算</h3>
    <div class="equation">
W(α) = (2/π) Tr[D(−α) ρ D(α) Π]
    </div>
    <p>只有在 temporal mode、efficiency/noise calibration、cutoff convergence 與統計誤差都通過後，Wigner negativity 才能當作非古典性的證據。</p>
  </section>

  <section data-contract-section="limitations-uncertainty-and-robustness-checks">
    <h2>限制、失敗模式與檢查方式</h2>
    <div class="two-col">
      <div class="card"><strong>Mode 被截斷</strong><p>最後 5% envelope 仍高於 peak 的 10% 時，增加 acquisition window 並重新 acquisition。</p></div>
      <div class="card"><strong>Reference / signal 順序</strong><p>gain=1 calibration 中 signal mean-trace norm 應明顯大於 zero-gain reference。</p></div>
      <div class="card"><strong>Added noise 未校正</strong><p>reference variance normalization 不是 amplifier-noise deconvolution；需要 noise moments 或 calibrated noisy POVM。</p></div>
      <div class="card"><strong>Cutoff artifact</strong><p>最高兩個 Fock bins 超過約 1% 時，不解讀 photon number 或 Wigner negativity。</p></div>
      <div class="card"><strong>有限 shots</strong><p>高階 moments 與弱 negativity 對 sampling error 敏感；使用 bootstrap 或重複 acquisition。</p></div>
      <div class="card"><strong>多 temporal modes</strong><p>單一 matched filter 只重建其投影；檢查 covariance eigenvalue spectrum 與殘差 mode。</p></div>
    </div>
  </section>

  <section data-contract-section="recommended-next-steps">
    <h2>建議實驗順序</h2>
    <ol>
      <li>以 gain=1 coherent pulse 驗證 CHB、LO/IF、sequence ordering、ADC headroom 與完整 trace。</li>
      <li>從獨立 calibration 建立 complex matched mode，固定 window 與 mode，不再用弱態資料調整。</li>
      <li>取得 zero-gain reference 與真正 photon-state shots；兩者 timing、filter、gain chain 完全一致。</li>
      <li>以多個 cutoff 重建 density matrix，先看 convergence，再看 Wigner。</li>
      <li>加入 measurement efficiency、thermal occupancy、JPA/HEMT added-noise calibration 與 bootstrap uncertainty。</li>
    </ol>
  </section>

  <section data-contract-section="further-questions">
    <h2>下一個需要回答的問題</h2>
    <ul>
      <li>目前 cavity photon release 的實際 envelope 是 exponential、shaped release，還是多 mode？</li>
      <li>JPA/HEMT chain 的 added-noise number 與總 efficiency 各是多少？</li>
      <li>真正 photon-state sequence 如何與 coherent calibration 保持相同 emission path？</li>
      <li>是否需要 joint qubit–field tomography，而不只是 field marginal state？</li>
    </ul>
    <p class="small note">理論依據：<a href="https://arxiv.org/abs/1011.6668">Experimental Tomographic State Reconstruction of Itinerant Microwave Photons</a>；<a href="https://arxiv.org/abs/1206.3405">Characterizing Quantum Microwave Radiation and Its Entanglement with Superconducting Qubits Using Linear Detectors</a>。Simulation 由本 repository 的 <code>docs/generate_tomography_tutorial.py</code> 產生。</p>
  </section>
</main>
</body>
</html>
"""

(ROOT / "tomography_tutorial_zh.html").write_text(html, encoding="utf-8")
print(ROOT / "tomography_tutorial_zh.html")
