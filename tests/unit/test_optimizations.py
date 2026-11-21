import math
import numpy as np
import pytest

from QAWG import (
    coherent_density_matrix,
    fit_resonator_linewidth,
    heterodyne_ml_density_matrix,
    wigner_function,
)
from QAWG.alazar.dsp import design_lowpass_fir, integrate


def test_wigner_function_fock_states():
    # Vacuum |0><0|
    rho_vac = np.zeros((4, 4), dtype=complex)
    rho_vac[0, 0] = 1.0
    w_vac = wigner_function(rho_vac, [0.0], [0.0])
    assert np.isclose(w_vac[0, 0], 2.0 / np.pi, atol=1e-12)

    # Fock |1><1| has parity -1 at origin: W(0) = -2/pi
    rho_1 = np.zeros((4, 4), dtype=complex)
    rho_1[1, 1] = 1.0
    w_1 = wigner_function(rho_1, [0.0], [0.0])
    assert np.isclose(w_1[0, 0], -2.0 / np.pi, atol=1e-12)


def test_wigner_function_coherent_state_analytical_match():
    alpha_0 = 1.2 - 0.7j
    cutoff = 12
    rho_coh = coherent_density_matrix(alpha_0, cutoff=cutoff)

    x = np.linspace(-2.5, 2.5, 31)
    y = np.linspace(-2.5, 2.5, 31)
    w_num = wigner_function(rho_coh, x, y)

    # True analytical Wigner function of coherent state |alpha_0>:
    # W(alpha) = (2/pi) * exp(-2 * |alpha - alpha_0|^2)
    X, Y = np.meshgrid(x, y)
    alpha = X + 1j * Y
    w_exact = (2.0 / np.pi) * np.exp(-2.0 * np.abs(alpha - alpha_0) ** 2)

    np.testing.assert_allclose(w_num, w_exact, atol=1e-3)


def test_heterodyne_ml_density_matrix_physicality():
    rng = np.random.default_rng(42)
    alpha_target = 0.8 + 0.6j
    # Generate heterodyne samples from coherent state: alpha_meas ~ Normal(alpha_target, 1/2 in each quad)
    shots = 2000
    noise = (rng.normal(size=shots) + 1j * rng.normal(size=shots)) / np.sqrt(2.0)
    samples = alpha_target + noise

    cutoff = 6
    rho_est = heterodyne_ml_density_matrix(
        samples,
        cutoff=cutoff,
        iterations=100,
        dilution=0.5,
    )

    # Must be Hermitian, trace 1, positive semi-definite
    np.testing.assert_allclose(
        rho_est, np.conjugate(rho_est.T), atol=1e-12
    )
    assert np.isclose(np.trace(rho_est).real, 1.0, atol=1e-7)
    eigenvalues = np.linalg.eigvalsh(rho_est)
    assert np.min(eigenvalues) >= -1e-12

    # Population peak near |alpha|^2 = 1.0
    population = np.real(np.diag(rho_est))
    mean_n = np.sum(np.arange(cutoff) * population)
    assert 0.5 < mean_n < 1.5


def test_dsp_integrate_fast_path_and_weights():
    rng = np.random.default_rng(123)
    shots = rng.normal(size=(50, 200)) + 1j * rng.normal(size=(50, 200))

    # Fast path (None weights) vs manual uniform array
    res_fast = integrate(shots, 20, 180, integration_weights=None)
    weights_uniform = np.ones(160, dtype=complex)
    res_slow = integrate(shots, 20, 180, integration_weights=weights_uniform)
    np.testing.assert_allclose(res_fast, res_slow, atol=1e-15)

    # Non-uniform weights
    w_custom = np.exp(-np.linspace(0, 3, 160))
    res_custom = integrate(shots, 20, 180, integration_weights=w_custom)
    expected = np.sum(shots[:, 20:180] * w_custom, axis=-1) / np.sum(w_custom)
    np.testing.assert_allclose(res_custom, expected, atol=1e-15)


def test_design_lowpass_fir_caching():
    fir1 = design_lowpass_fir(1e9, 20e6, number_of_taps=81, kaiser_beta=8.0)
    fir2 = design_lowpass_fir(1e9, 20e6, number_of_taps=81, kaiser_beta=8.0)
    np.testing.assert_allclose(fir1, fir2)
    assert np.isclose(np.sum(fir1), 1.0)
    # Ensure returned array is mutable independently
    fir1[0] += 1.0
    assert not np.isclose(fir1[0], fir2[0])


def test_fit_resonator_linewidth():
    f0_true = 5.9e9
    kappa_true = 4.5e6
    freqs = f0_true + np.linspace(-15e6, 15e6, 31)

    # Ideal single-pole lorentzian circle with background
    background = 0.05 - 0.02j
    response = 0.8 + 0.3j
    clean_iq = background + response / (
        1.0 + 2j * (freqs - f0_true) / kappa_true
    )

    rng = np.random.default_rng(99)
    noisy_iq = clean_iq + (rng.normal(size=freqs.size) + 1j * rng.normal(size=freqs.size)) * 1e-4

    fit_res = fit_resonator_linewidth(freqs, noisy_iq)

    assert fit_res['success'] is True
    assert fit_res['r_squared'] > 0.999
    assert np.isclose(fit_res['f0_hz'], f0_true, rtol=1e-4)
    assert np.isclose(fit_res['linewidth_hz'], kappa_true, rtol=1e-3)
