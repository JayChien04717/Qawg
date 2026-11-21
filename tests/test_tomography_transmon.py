"""Tests for noisy transmon-emission tomography and Wigner reconstruction."""

import numpy as np

from QAWG.tomography import (
    complex_moment_stats,
    deconvolve_amplifier_moments,
    extract_temporal_mode,
    fit_density_matrix_from_moments,
    heterodyne_ml_density_matrix,
    project_temporal_mode,
    temporal_mode_weights,
    wigner_function,
)


def _complex_thermal(rng: np.random.Generator, n: int, mean_photons: float) -> np.ndarray:
    """Circular complex Gaussian with E[|z|^2] = mean_photons."""
    sigma = np.sqrt(mean_photons / 2.0)
    return rng.normal(0.0, sigma, n) + 1j * rng.normal(0.0, sigma, n)


def _sample_fock_one_q(rng: np.random.Generator, n: int) -> np.ndarray:
    """Samples from ideal Husimi-Q of |1>: |alpha|^2 ~ Gamma(2, 1), uniform phase."""
    return np.sqrt(rng.gamma(2.0, 1.0, n)) * np.exp(2j * np.pi * rng.random(n))


def test_noisy_povm_ml_recovers_single_photon_wigner_negativity():
    """Noisy heterodyne (N=1.5 thermal photons) of |1> -> P(1) high, W(0) < 0."""
    rng = np.random.default_rng(42)
    n_shots = 30_000
    noise_photons = 1.5
    # S = a + h^dag -> ideal Husimi-Q sample + independent thermal noise (E[|z|^2] = N)
    samples = _sample_fock_one_q(rng, n_shots) + _complex_thermal(
        rng, n_shots, noise_photons
    )
    rho = heterodyne_ml_density_matrix(
        samples,
        cutoff=5,
        iterations=1200,
        tolerance=1e-11,
        noise_photons=noise_photons,
    )
    assert np.real(rho[1, 1]) > 0.80
    axis = np.linspace(-2.5, 2.5, 51)
    wigner = wigner_function(rho, axis, axis)
    assert wigner[25, 25] < -0.35


def test_hemt_moment_deconvolution_recovers_fock_one_under_high_noise():
    """Uncertainty-weighted moment deconvolution recovers |1> at HEMT noise N=8."""
    rng = np.random.default_rng(7)
    n_shots = 200_000
    noise_photons = 8.0
    # Reference (vacuum + amplifier noise): E[|S_ref|^2] = 1 + N
    ref = _complex_thermal(rng, n_shots, 1.0 + noise_photons)
    # Signal (|1> + amplifier noise): ideal Q(|1>) + thermal(N)
    sig = _sample_fock_one_q(rng, n_shots) + _complex_thermal(
        rng, n_shots, noise_photons
    )
    ref_m, ref_err = complex_moment_stats(ref, maximum_order=4)
    sig_m, sig_err = complex_moment_stats(sig, maximum_order=4)
    normal_m, normal_err = deconvolve_amplifier_moments(
        sig_m, ref_m, signal_errors=sig_err, reference_errors=ref_err
    )
    assert normal_err is not None
    # <a^dag a> should be within 3 sigma of 1 and <(a^dag)^2 a^2> within 2 sigma of 0
    assert abs(normal_m[1, 1].real - 1.0) < 3.0 * normal_err[1, 1]
    assert abs(normal_m[2, 2].real) < 2.0 * normal_err[2, 2]

    rho = fit_density_matrix_from_moments(
        normal_m, cutoff=4, moment_errors=normal_err
    )
    assert np.real(rho[1, 1]) > 0.80
    axis = np.linspace(-2.5, 2.5, 51)
    wigner = wigner_function(rho, axis, axis)
    assert wigner[25, 25] < -0.30


def test_extract_temporal_mode_pca_recovers_zero_mean_fock_mode():
    """PCA extracts exponential decay mode even when <a> = 0 (Fock |1>)."""
    rng = np.random.default_rng(123)
    n_shots = 4000
    n_samples = 64
    true_mode = temporal_mode_weights(
        n_samples, kind="exponential", decay_samples=12.0
    )
    # Random-phase single-photon amplitudes -> mean trace is ~0
    amplitudes = _sample_fock_one_q(rng, n_shots)
    noise_sig = _complex_thermal(rng, n_shots * n_samples, 0.5).reshape(
        n_shots, n_samples
    )
    noise_ref = _complex_thermal(rng, n_shots * n_samples, 0.5).reshape(
        n_shots, n_samples
    )
    traces = amplitudes[:, None] * true_mode[None, :] + noise_sig

    recovered = extract_temporal_mode(
        traces, method="pca", reference_traces=noise_ref
    )
    overlap = abs(np.vdot(true_mode, recovered))
    assert overlap > 0.99
    projected = project_temporal_mode(traces, recovered)
    assert projected.shape == (n_shots,)


def test_extract_temporal_mode_mean_recovers_coherent_mode():
    rng = np.random.default_rng(9)
    n_shots = 500
    n_samples = 40
    true_mode = temporal_mode_weights(n_samples, kind="gaussian", sigma_samples=6.0)
    noise = _complex_thermal(rng, n_shots * n_samples, 0.2).reshape(
        n_shots, n_samples
    )
    traces = (2.0 + 0.5j) * true_mode[None, :] + noise
    recovered = extract_temporal_mode(traces, method="mean")
    assert abs(np.vdot(true_mode, recovered)) > 0.995
