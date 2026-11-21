"""End-to-end simulation of noisy waveguide-QED transmon emission & Wigner tomography.

Simulates a superconducting transmon decaying into a 1D waveguide (T1 = 80 ns)
measured through a noisy HEMT amplification chain (n_noise = 10 thermal photons,
collection efficiency eta = 0.90) with realistic Alazar ATS9371 dual-channel
acquisition (ChA = noisy RF emission at 50 MHz IF, ChB = phase reference tone
removing sub-sample trigger jitter and slow LO drift).

Outputs saved under ``examples/transmon_wigner_sim/``:
- ``transmon_emission_data.png``: time-domain raw/IQ traces, ChB ref stabilization,
  coherent |+> decay vs zero-mean |1> power decay, and PCA temporal mode extraction.
- ``transmon_wigner_reconstruction.png``: noisy IQ histograms, deconvolved moments,
  reconstructed Fock populations, and 2D Wigner functions W(x, p) for |0>, |+>, |1>.
- ``transmon_wigner_results.npz`` & ``summary.json``: full numerical dataset.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QAWG.alazar import DemodulationConfig, DemodulationPipeline
from QAWG.tomography import (
    complex_moment_stats,
    deconvolve_amplifier_moments,
    density_matrix_moments,
    extract_temporal_mode,
    fit_density_matrix_from_moments,
    project_temporal_mode,
    wigner_function,
)


def sample_complex_gaussian(
    rng: np.random.Generator, shape: tuple[int, ...] | int, variance: float
) -> np.ndarray:
    """Circular complex Gaussian with E[|z|^2] = variance."""
    sigma = np.sqrt(variance / 2.0)
    return rng.normal(0.0, sigma, shape) + 1j * rng.normal(0.0, sigma, shape)


def sample_husimi_q(
    rng: np.random.Generator, state: str, n_shots: int, eta: float = 0.90
) -> np.ndarray:
    """Sample exact ideal Husimi-Q amplitudes alpha for |0>, |1>, or |+> = (|0>+|1>)/sqrt(2)
    after passing through a loss channel of efficiency ``eta``.
    """
    if state == "vacuum":
        return sample_complex_gaussian(rng, n_shots, 1.0)

    if state == "fock1":
        # rho = eta |1><1| + (1 - eta) |0><0|
        is_one = rng.random(n_shots) < eta
        samples = sample_complex_gaussian(rng, n_shots, 1.0)
        n_one = int(np.sum(is_one))
        samples[is_one] = np.sqrt(rng.gamma(2.0, 1.0, n_one)) * np.exp(
            2j * np.pi * rng.random(n_one)
        )
        return samples

    if state == "plus":
        # rho = eta |+><+| + (1 - eta) |0><0|
        # Q(alpha) = exp(-|alpha|^2)/pi * (1 - eta/2 + eta/2 * |alpha|^2 + eta * Re(alpha))
        # Use exact rejection sampling from mixture proposal 0.5*Q_|0> + 0.5*Q_|1>
        # (Wait: Q(alpha) <= 2 * [ (1 - eta/2)*Q_|0> + (eta/2)*Q_|1> ] everywhere since
        #  |eta*Re(alpha)| <= eta*|alpha| <= eta*(1 + |alpha|^2)/2 <= (1-eta/2) + (eta/2)*|alpha|^2
        #  Wait: let's bound tightly with M = 2.0 against p(alpha) = 0.5*Q_|0> + 0.5*Q_|1>.)
        out = np.empty(n_shots, dtype=np.complex128)
        filled = 0
        while filled < n_shots:
            batch = max(4096, int((n_shots - filled) * 2.5))
            # Proposal: 0.5 * Q_|0> + 0.5 * Q_|1>, density = exp(-|a|^2)/pi * 0.5 * (1 + |a|^2)
            use_one = rng.random(batch) < 0.5
            cand = sample_complex_gaussian(rng, batch, 1.0)
            n_one = int(np.sum(use_one))
            cand[use_one] = np.sqrt(rng.gamma(2.0, 1.0, n_one)) * np.exp(
                2j * np.pi * rng.random(n_one)
            )
            r2 = np.abs(cand) ** 2
            q_num = (1.0 - 0.5 * eta) + 0.5 * eta * r2 + eta * cand.real
            p_num = 0.5 * (1.0 + r2)
            # Ratio q_num / p_num <= 2.0 + eta <= 3.0 everywhere
            accept_prob = q_num / (3.0 * p_num)
            keep = cand[rng.random(batch) < accept_prob]
            take = min(keep.size, n_shots - filled)
            out[filled : filled + take] = keep[:take]
            filled += take
        return out

    raise ValueError(f"Unknown state {state!r}")


def simulate_state_stream(
    rng: np.random.Generator,
    state: str,
    *,
    n_shots: int,
    chunk_shots: int,
    n_samples: int,
    sample_rate_hz: float,
    if_freq_hz: float,
    t0_sample: int,
    true_mode: np.ndarray,
    noise_photons: float,
    eta: float,
    photon_scale_volts: float,
    pipeline: DemodulationPipeline,
    keep_qa_traces: int = 200,
) -> dict[str, np.ndarray]:
    """Stream chunks of noisy RF ChA + phase-ref ChB through DemodulationPipeline."""
    time_s = np.arange(n_samples, dtype=np.float64) / sample_rate_hz
    mode_full = np.zeros(n_samples, dtype=np.complex128)
    mode_full[t0_sample : t0_sample + true_mode.size] = true_mode

    mode_samples = np.empty(n_shots, dtype=np.complex128)
    mean_iq = np.zeros(n_samples, dtype=np.complex128)
    mean_power = np.zeros(n_samples, dtype=np.float64)
    qa_raw = None
    qa_iq_uncorrected = None
    qa_iq_corrected = None
    qa_ref_dev = None

    # Pre-create an uncorrected pipeline to illustrate ChB ref phase stabilization
    uncorrected_pipeline = DemodulationPipeline(
        DemodulationConfig(
            sample_rate_hz=sample_rate_hz,
            tone_frequency_hz=if_freq_hz,
            integrate_start_sample=t0_sample,
            integrate_stop_sample=t0_sample + true_mode.size,
            remove_dc_offset=True,
        )
    )

    # For fast streaming simulation, generate RF voltage chunks & demodulate
    cursor = 0
    while cursor < n_shots:
        m = min(chunk_shots, n_shots - cursor)
        # Realistic sub-sample trigger jitter (+/- 0.5 ns) + slow LO phase drift
        shot_idx = np.arange(cursor, cursor + m, dtype=np.float64)
        jitter_s = rng.uniform(-0.5e-9, 0.5e-9, m)
        slow_drift_rad = 0.6 * np.sin(2.0 * np.pi * shot_idx / n_shots)
        total_phase_err = 2.0 * np.pi * if_freq_hz * jitter_s + slow_drift_rad

        # Quantum signal amplitude in temporal mode + white amplifier thermal noise
        alpha_q = sample_husimi_q(rng, state, m, eta=eta)
        # Baseband complex envelope in volts:
        # Taking Re(baseband * carrier) and mixing back with 2 * exp(-i w t)
        # folds the image band of white noise, doubling white-noise variance.
        # Scaling white_th by 1/sqrt(2) gives exact post-downconversion thermal
        # noise variance ``noise_photons`` in the temporal mode (SNR = eta/(1+N)).
        white_th = sample_complex_gaussian(
            rng, (m, n_samples), noise_photons / 2.0
        )
        baseband_v = photon_scale_volts * (
            alpha_q[:, None] * mode_full[None, :] + white_th
        )

        # Upconvert to real RF voltage on ChA with phase error
        carrier = np.exp(
            1j * (2.0 * np.pi * if_freq_hz * time_s[None, :] + total_phase_err[:, None])
        )
        rf_cha = np.real(baseband_v * carrier)

        # Reference tone on ChB (strong 80 mV tone sharing the exact same jitter/drift)
        rf_chb = 0.08 * np.cos(
            2.0 * np.pi * if_freq_hz * time_s[None, :]
            + total_phase_err[:, None]
            + 0.4
        ) + rng.normal(0.0, 2e-4, (m, n_samples))

        demod = pipeline.process(rf_cha, rf_chb, compute_fir=False)
        bb = demod.baseband_iq

        if qa_raw is None:
            k = min(keep_qa_traces, m)
            qa_raw = rf_cha[:k].copy()
            qa_iq_corrected = bb[:k].copy()
            qa_ref_dev = demod.reference_deviation_rad[:k].copy()
            qa_iq_uncorrected = uncorrected_pipeline.process(
                rf_cha[:k], None, compute_fir=False
            ).baseband_iq.copy()

        mean_iq += np.sum(bb, axis=0)
        mean_power += np.sum(np.abs(bb) ** 2, axis=0)
        mode_samples[cursor : cursor + m] = project_temporal_mode(
            bb, true_mode, start_sample=t0_sample
        )
        cursor += m

    mean_iq /= n_shots
    mean_power /= n_shots
    return {
        "mode_samples": mode_samples,
        "mean_iq": mean_iq,
        "mean_power": mean_power,
        "qa_raw": qa_raw,
        "qa_iq_uncorrected": qa_iq_uncorrected,
        "qa_iq_corrected": qa_iq_corrected,
        "qa_ref_dev": qa_ref_dev,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "transmon_wigner_sim"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(2026)

    # Hardware & waveguide-QED transmon parameters
    sample_rate_hz = 1e9  # 1 GS/s (Alazar ATS9371)
    if_freq_hz = 50e6     # 50 MHz IF
    n_samples = 1024      # 1024 ns acquisition window
    t0_sample = 160       # emission starts at t = 160 ns
    mode_len = 512        # 512 ns temporal mode window (160..672 ns)
    t1_ns = 80.0          # transmon energy decay T1 = 80 ns (Gamma_1/2pi ~ 2 MHz)
    tau_amp_ns = 2.0 * t1_ns  # amplitude decay time = 2 * T1 = 160 ns

    noise_photons = 3.0   # noisy waveguide chain (TWPA+HEMT: N_noise=3 -> ref var=4, SNR ~ -6.5 dB)
    eta = 0.90            # 90% collection efficiency (10% waveguide/insertion loss)
    photon_scale_volts = 0.008  # 8 mV / sqrt(photon) after amplification
    n_shots = 80_000      # 80,000 shots per state (streamed in chunks of 4,000)
    chunk_shots = 4_000

    # True normalized temporal mode f(t) proportional to exp(-t / (2 * T1))
    t_mode = np.arange(mode_len, dtype=np.float64)
    true_mode = np.exp(-t_mode / tau_amp_ns).astype(np.complex128)
    true_mode /= np.linalg.norm(true_mode)

    pipeline = DemodulationPipeline(
        DemodulationConfig(
            sample_rate_hz=sample_rate_hz,
            tone_frequency_hz=if_freq_hz,
            integrate_start_sample=t0_sample,
            integrate_stop_sample=t0_sample + mode_len,
            remove_dc_offset=True,
            reference_frequency_hz=if_freq_hz,
            reference_start_sample=64,
            reference_stop_sample=960,
            reference_center_rad=0.4,
            reference_min_amplitude=0.01,
        )
    )

    print(
        f"Simulating {n_shots:,} shots/state | T1={t1_ns} ns | "
        f"N_noise={noise_photons} | eta={eta} ..."
    )
    results = {}
    for state in ("vacuum", "plus", "fock1"):
        print(f"  Streaming state '{state}' ...")
        results[state] = simulate_state_stream(
            rng,
            state,
            n_shots=n_shots,
            chunk_shots=chunk_shots,
            n_samples=n_samples,
            sample_rate_hz=sample_rate_hz,
            if_freq_hz=if_freq_hz,
            t0_sample=t0_sample,
            true_mode=true_mode,
            noise_photons=noise_photons,
            eta=eta,
            photon_scale_volts=photon_scale_volts,
            pipeline=pipeline,
        )

    # --- Step 1: Demonstrate temporal-mode extraction from QA noisy traces ---
    # Use a small batch (4,000 shots) of full IQ traces for PCA mode extraction
    pca_shots = 4000
    t_full = np.arange(n_samples, dtype=np.float64) / sample_rate_hz
    mode_full = np.zeros(n_samples, dtype=np.complex128)
    mode_full[t0_sample : t0_sample + mode_len] = true_mode
    pca_ref_iq = photon_scale_volts * (
        sample_husimi_q(rng, "vacuum", pca_shots, eta=eta)[:, None] * mode_full[None, :]
        + sample_complex_gaussian(rng, (pca_shots, n_samples), noise_photons)
    )
    pca_fock1_iq = photon_scale_volts * (
        sample_husimi_q(rng, "fock1", pca_shots, eta=eta)[:, None] * mode_full[None, :]
        + sample_complex_gaussian(rng, (pca_shots, n_samples), noise_photons)
    )
    # Downsample to 16-ns bins (32 bins over 512 ns) to beat Marchenko-Pastur noise edge
    bin_w = 16
    def bin_traces(x: np.ndarray) -> np.ndarray:
        win = x[:, t0_sample : t0_sample + mode_len]
        return win.reshape(win.shape[0], mode_len // bin_w, bin_w).mean(axis=2)

    pca_mode_binned = extract_temporal_mode(
        bin_traces(pca_fock1_iq),
        method="pca",
        reference_traces=bin_traces(pca_ref_iq),
    )
    true_mode_binned = true_mode.reshape(mode_len // bin_w, bin_w).mean(axis=1)
    true_mode_binned /= np.linalg.norm(true_mode_binned)
    pca_overlap = float(abs(np.vdot(true_mode_binned, pca_mode_binned)))

    # --- Step 2: Calibrate to sqrt(photon) units & compute moments + Wigner ---
    ref_offset = complex(np.mean(results["vacuum"]["mode_samples"]))
    ref_alpha = (results["vacuum"]["mode_samples"] - ref_offset) / photon_scale_volts
    ref_m, ref_err = complex_moment_stats(ref_alpha, maximum_order=4)

    cutoff = 4
    wigner_axis = np.linspace(-2.5, 2.5, 81)
    reconstructed = {}
    summary: dict[str, object] = {
        "parameters": {
            "n_shots_per_state": n_shots,
            "t1_ns": t1_ns,
            "tau_amp_ns": tau_amp_ns,
            "noise_photons": noise_photons,
            "collection_efficiency_eta": eta,
            "photon_scale_volts": photon_scale_volts,
            "single_shot_snr_db": float(10.0 * np.log10(eta / (1.0 + noise_photons))),
            "pca_mode_overlap_2000_shots": pca_overlap,
        },
        "states": {},
    }

    for state in ("vacuum", "plus", "fock1"):
        sig_alpha = (results[state]["mode_samples"] - ref_offset) / photon_scale_volts
        sig_m, sig_err = complex_moment_stats(sig_alpha, maximum_order=4)
        normal_m, normal_err = deconvolve_amplifier_moments(
            sig_m, ref_m, signal_errors=sig_err, reference_errors=ref_err
        )
        rho = fit_density_matrix_from_moments(
            normal_m, cutoff=cutoff, moment_errors=normal_err
        )
        W = wigner_function(rho, wigner_axis, wigner_axis)
        reconstructed[state] = {
            "alpha": sig_alpha,
            "sig_m": sig_m,
            "normal_m": normal_m,
            "normal_err": normal_err,
            "rho": rho,
            "W": W,
        }
        w_origin = float(W[len(wigner_axis) // 2, len(wigner_axis) // 2])
        summary["states"][state] = {
            "mean_photon_number_deconvolved": float(normal_m[1, 1].real),
            "mean_photon_number_err": float(normal_err[1, 1]),
            "g2_numerator_deconvolved": float(normal_m[2, 2].real),
            "g2_numerator_err": float(normal_err[2, 2]),
            "coherence_a_deconvolved": [
                float(normal_m[0, 1].real),
                float(normal_m[0, 1].imag),
            ],
            "fock_populations": np.real(np.diag(rho)).tolist(),
            "rho_01_real": float(rho[0, 1].real),
            "rho_01_imag": float(rho[0, 1].imag),
            "wigner_origin": w_origin,
            "wigner_min": float(np.min(W)),
        }
        print(
            f"  [{state:6s}] <a^dag a>={normal_m[1,1].real:+.3f}±{normal_err[1,1]:.3f} | "
            f"P(0,1,2)={np.round(np.real(np.diag(rho))[:3], 3)} | "
            f"W(0)={w_origin:+.3f}, W_min={float(np.min(W)):+.3f}"
        )

    # --- Figure 1: Time-domain noisy waveguide emission data ---
    time_ns = t_full * 1e9
    fig1, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    # (a) Single-shot raw noisy ChA trace vs true noiseless single-photon envelope
    ax = axes[0, 0]
    ax.plot(
        time_ns,
        results["fock1"]["qa_raw"][0] * 1e3,
        color="tab:gray",
        alpha=0.75,
        lw=0.7,
        label="Single-shot raw ChA (noisy RF)",
    )
    ax.plot(
        time_ns,
        photon_scale_volts * np.abs(mode_full) * 1e3,
        color="tab:red",
        lw=2.2,
        label="True 1-photon envelope (buried in noise)",
    )
    ax.axvspan(t0_sample, t0_sample + mode_len, color="tab:orange", alpha=0.08, label="Mode window")
    ax.set_title(f"(a) Single-Shot Waveguide RF Trace (SNR = {10*np.log10(eta/(1+noise_photons)):.1f} dB)")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("ADC Voltage (mV)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) Coherent average |<S(t)>|: |+> decays exponentially, |1> averages to ~0!
    ax = axes[0, 1]
    ax.plot(
        time_ns,
        np.abs(results["plus"]["mean_iq"]) * 1e3,
        color="tab:blue",
        lw=1.4,
        label=r"$|+\rangle = (|0\rangle+|1\rangle)/\sqrt{2}$: $|\langle S(t)\rangle|$ (ChB ref on)",
    )
    ax.plot(
        time_ns,
        np.abs(results["fock1"]["mean_iq"]) * 1e3,
        color="tab:red",
        lw=1.1,
        alpha=0.85,
        label=r"$|1\rangle$: $|\langle S(t)\rangle| \approx 0$ (random optical phase)",
    )
    ax.plot(
        time_ns,
        0.5 * eta * photon_scale_volts * np.abs(mode_full) * 1e3,
        "k--",
        lw=1.5,
        label=r"Theory $\frac{\eta}{2} g |f(t)|$ ($\tau_{\rm amp}=160$ ns)",
    )
    ax.set_title(r"(b) Coherent Mean Envelope $|\langle S(t)\rangle|$ (80,000 shots)")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("|<IQ>| (mV)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (c) Excess power decay <|S(t)|^2> - <|S_ref(t)|^2>: reveals |1> decay!
    ax = axes[1, 0]
    scale_sq = photon_scale_volts ** 2
    excess_fock1 = (results["fock1"]["mean_power"] - results["vacuum"]["mean_power"]) / scale_sq
    excess_plus = (results["plus"]["mean_power"] - results["vacuum"]["mean_power"]) / scale_sq
    # Smooth with 16-ns boxcar for clear visualization
    kernel = np.ones(16) / 16.0
    ax.plot(
        time_ns,
        np.convolve(excess_fock1, kernel, mode="same"),
        color="tab:red",
        lw=1.6,
        label=r"$|1\rangle$ excess power ($\propto \eta |f(t)|^2$)",
    )
    ax.plot(
        time_ns,
        np.convolve(excess_plus, kernel, mode="same"),
        color="tab:blue",
        lw=1.4,
        label=r"$|+\rangle$ excess power ($\propto \frac{\eta}{2} |f(t)|^2$)",
    )
    ax.plot(
        time_ns,
        eta * np.abs(mode_full) ** 2,
        "k--",
        lw=1.4,
        label=r"Theory $\eta |f(t)|^2$ ($T_1 = 80$ ns)",
    )
    ax.set_title(r"(c) Noise-Subtracted Photon Power Decay $\langle|S(t)|^2\rangle - \langle|S_{\rm ref}(t)|^2\rangle$")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Photons / ns")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (d) Temporal mode f(t) extracted via PCA from zero-mean |1> traces + ChB ref check
    ax = axes[1, 1]
    t_pca = t0_sample + (np.arange(mode_len // bin_w) + 0.5) * bin_w
    ax.plot(t_pca, np.abs(true_mode_binned), "k--", lw=2.0, label="True decay mode |f(t)|")
    ax.plot(
        t_pca,
        np.abs(pca_mode_binned),
        color="tab:purple",
        lw=1.8,
        marker="o",
        ms=3.5,
        label=f"PCA-extracted mode from |1> (overlap={pca_overlap:.3f})",
    )
    ax.set_title("(d) PCA Temporal Mode Extraction (4,000 shots of zero-mean |1>)")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Normalized Mode |f(t)|")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig1.suptitle(
        f"Simulated Waveguide-QED Transmon Emission (T1={t1_ns:.0f} ns, N_noise={noise_photons:.0f}, eta={eta:.2f}, ChB Ref On)",
        fontsize=12,
        fontweight="bold",
    )
    fig1.tight_layout()
    fig1.savefig(out_dir / "transmon_emission_data.png", dpi=160)
    plt.close(fig1)

    # --- Figure 2: Noisy IQ distributions, Fock populations, and 2D Wigner functions ---
    fig2, axes2 = plt.subplots(3, 3, figsize=(13, 11))
    titles = {
        "vacuum": r"Reference Vacuum $|0\rangle$",
        "plus": r"Superposition $|+\rangle = (|0\rangle + |1\rangle)/\sqrt{2}$",
        "fock1": r"Single-Photon Fock $|1\rangle$",
    }
    for col, state in enumerate(("vacuum", "plus", "fock1")):
        rec = reconstructed[state]
        alpha = rec["alpha"]
        rho = rec["rho"]
        W = rec["W"]

        # Row 0: Raw noisy heterodyne IQ 2D histogram (dominated by N_noise=10!)
        ax = axes2[0, col]
        ax.hist2d(
            alpha.real,
            alpha.imag,
            bins=60,
            range=[[-10, 10], [-10, 10]],
            cmap="inferno",
        )
        ax.set_aspect("equal")
        ax.set_title(f"{titles[state]}\nRaw Noisy IQ (std={np.std(alpha.real):.2f})")
        ax.set_xlabel(r"$\operatorname{Re}(S)$ ($\sqrt{\mathrm{photon}}$)")
        if col == 0:
            ax.set_ylabel(r"$\operatorname{Im}(S)$ ($\sqrt{\mathrm{photon}}$)")

        # Row 1: Reconstructed Fock populations P(n) = rho[n, n]
        ax = axes2[1, col]
        pops = np.real(np.diag(rho))
        bars = ax.bar(np.arange(cutoff), pops, color="tab:blue", alpha=0.85)
        for idx, b in enumerate(bars):
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                b.get_height() + 0.02,
                f"{pops[idx]:.2f}",
                ha="center",
                fontsize=8,
            )
        ax.set_ylim(0, 1.15)
        ax.set_xticks(range(cutoff))
        ax.set_xlabel("Fock state |n>")
        if col == 0:
            ax.set_ylabel("Population P(n)")
        ax.set_title(
            rf"$\langle a^\dagger a \rangle={rec['normal_m'][1,1].real:.2f}\pm{rec['normal_err'][1,1]:.2f}$, "
            rf"$\rho_{{01}}={rho[0,1].real:+.2f}$"
        )
        ax.grid(axis="y", alpha=0.3)

        # Row 2: Reconstructed 2D Wigner function W(x, p)
        ax = axes2[2, col]
        vmax = 2.0 / np.pi
        im = ax.pcolormesh(
            wigner_axis,
            wigner_axis,
            W,
            shading="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_aspect("equal")
        w0 = W[len(wigner_axis) // 2, len(wigner_axis) // 2]
        ax.set_title(rf"Reconstructed Wigner $W(0)={w0:+.3f}$")
        ax.set_xlabel(r"$\operatorname{Re}(\alpha)$")
        if col == 0:
            ax.set_ylabel(r"$\operatorname{Im}(\alpha)$")
        fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig2.suptitle(
        f"Moment Deconvolution & Wigner Reconstruction (N_noise={noise_photons:.0f}, eta={eta:.2f}, {n_shots:,} shots/state)",
        fontsize=12,
        fontweight="bold",
    )
    fig2.tight_layout()
    fig2.savefig(out_dir / "transmon_wigner_reconstruction.png", dpi=160)
    plt.close(fig2)

    # Save dataset and summary JSON
    np.savez_compressed(
        out_dir / "transmon_wigner_results.npz",
        time_ns=time_ns,
        true_mode=true_mode,
        wigner_axis=wigner_axis,
        vacuum_alpha=reconstructed["vacuum"]["alpha"],
        plus_alpha=reconstructed["plus"]["alpha"],
        fock1_alpha=reconstructed["fock1"]["alpha"],
        rho_vacuum=reconstructed["vacuum"]["rho"],
        rho_plus=reconstructed["plus"]["rho"],
        rho_fock1=reconstructed["fock1"]["rho"],
        W_vacuum=reconstructed["vacuum"]["W"],
        W_plus=reconstructed["plus"]["W"],
        W_fock1=reconstructed["fock1"]["W"],
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Saved figures and data to {out_dir}")


if __name__ == "__main__":
    main()
