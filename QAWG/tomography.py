"""Temporal-mode and heterodyne tomography helpers."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt


ComplexArray = npt.NDArray[np.complex128]
FloatArray = npt.NDArray[np.float64]


def temporal_mode_weights(
    number_of_samples: int,
    *,
    kind: str = "boxcar",
    sigma_samples: float | None = None,
    decay_samples: float | None = None,
) -> ComplexArray:
    """Return unit-L2 temporal-mode weights."""
    if number_of_samples < 1:
        raise ValueError("number_of_samples must be positive")
    name = kind.lower()
    if name == "boxcar":
        weights = np.ones(number_of_samples, dtype=np.float64)
    elif name in {"gaussian", "gauss"}:
        sigma = (
            number_of_samples / 6.0
            if sigma_samples is None
            else float(sigma_samples)
        )
        if sigma <= 0:
            raise ValueError("sigma_samples must be positive")
        sample = np.arange(number_of_samples, dtype=np.float64)
        center = (number_of_samples - 1) / 2.0
        weights = np.exp(-0.5 * ((sample - center) / sigma) ** 2)
    elif name in {"exponential", "exp"}:
        decay = (
            number_of_samples / 5.0
            if decay_samples is None
            else float(decay_samples)
        )
        if decay <= 0:
            raise ValueError("decay_samples must be positive")
        sample = np.arange(number_of_samples, dtype=np.float64)
        weights = np.exp(-sample / (2.0 * decay))
    else:
        raise ValueError(
            "kind must be 'boxcar', 'gaussian', or 'exponential'"
        )
    return np.asarray(weights / np.linalg.norm(weights), dtype=np.complex128)


def project_temporal_mode(
    baseband_iq: npt.ArrayLike,
    weights: npt.ArrayLike,
    *,
    start_sample: int = 0,
) -> ComplexArray:
    """Project every complex IQ trace onto one normalized temporal mode."""
    traces = np.asarray(baseband_iq, dtype=np.complex128)
    mode = np.asarray(weights, dtype=np.complex128).reshape(-1)
    if traces.ndim != 2:
        raise ValueError("baseband_iq must have shape (shots, samples)")
    if mode.size < 1:
        raise ValueError("weights cannot be empty")
    stop_sample = start_sample + mode.size
    if not 0 <= start_sample < stop_sample <= traces.shape[1]:
        raise ValueError("temporal mode is outside the acquired record")
    return traces[:, start_sample:stop_sample] @ np.conjugate(mode)


def calibrate_iq_samples(
    reference_samples: npt.ArrayLike,
    signal_samples: npt.ArrayLike,
    *,
    target_alpha: complex = 1.0 + 0.0j,
) -> tuple[ComplexArray, ComplexArray, complex, complex]:
    """Map measured IQ volts to a user-defined coherent amplitude scale.

    The reference mean becomes zero and the signal-reference displacement
    becomes ``target_alpha``. This is an electrical calibration unless
    ``target_alpha`` is independently calibrated in photon units.
    """
    reference = np.asarray(reference_samples, dtype=np.complex128).reshape(-1)
    signal = np.asarray(signal_samples, dtype=np.complex128).reshape(-1)
    if reference.size < 1 or signal.size < 1:
        raise ValueError("reference_samples and signal_samples cannot be empty")
    target = complex(target_alpha)
    if target == 0:
        raise ValueError("target_alpha cannot be zero")
    offset = complex(np.mean(reference))
    gain = complex((np.mean(signal) - offset) / target)
    if abs(gain) <= np.finfo(np.float64).eps:
        raise ValueError("signal and reference means are indistinguishable")
    return (
        np.asarray((reference - offset) / gain, dtype=np.complex128),
        np.asarray((signal - offset) / gain, dtype=np.complex128),
        offset,
        gain,
    )


def normalize_heterodyne_reference(
    reference_samples: npt.ArrayLike,
    *sample_sets: npt.ArrayLike,
) -> tuple[ComplexArray, tuple[ComplexArray, ...], complex, float]:
    """Center and scale IQ so the reference has mean |alpha|^2 equal to one.

    This matches the ideal vacuum heterodyne variance convention, but it is
    only a physical photon-unit calibration when the reference is known to be
    vacuum noise at the measurement input.
    """
    reference = np.asarray(reference_samples, dtype=np.complex128).reshape(-1)
    if reference.size < 1:
        raise ValueError("reference_samples cannot be empty")
    offset = complex(np.mean(reference))
    centered = reference - offset
    scale = float(np.sqrt(np.mean(np.abs(centered) ** 2)))
    if scale <= np.finfo(np.float64).eps:
        raise ValueError("reference_samples have zero variance")
    normalized_reference = np.asarray(centered / scale, dtype=np.complex128)
    normalized_sets = tuple(
        np.asarray(
            (np.asarray(values, dtype=np.complex128).reshape(-1) - offset)
            / scale,
            dtype=np.complex128,
        )
        for values in sample_sets
    )
    return normalized_reference, normalized_sets, offset, scale


def coherent_state_vector(alpha: complex, cutoff: int) -> ComplexArray:
    """Return a truncated coherent-state vector in the Fock basis."""
    if cutoff < 1:
        raise ValueError("cutoff must be positive")
    values = np.empty(cutoff, dtype=np.complex128)
    values[0] = np.exp(-0.5 * abs(alpha) ** 2)
    for number in range(1, cutoff):
        values[number] = values[number - 1] * alpha / math.sqrt(number)
    return values


def heterodyne_ml_density_matrix(
    samples: npt.ArrayLike,
    *,
    cutoff: int = 8,
    iterations: int = 200,
    dilution: float = 0.5,
    tolerance: float = 1e-9,
) -> ComplexArray:
    """Estimate a physical density matrix from ideal heterodyne samples.

    This uses a diluted R-rho-R iteration for the coherent-state POVM. Samples
    must already be expressed in dimensionless field-amplitude units.
    """
    alpha = np.asarray(samples, dtype=np.complex128).reshape(-1)
    if alpha.size < 1:
        raise ValueError("samples cannot be empty")
    if cutoff < 1 or iterations < 1:
        raise ValueError("cutoff and iterations must be positive")
    if not 0 < dilution <= 1:
        raise ValueError("dilution must be in (0, 1]")

    number = np.arange(cutoff, dtype=np.float64)
    factorial_sqrt = np.sqrt(
        np.asarray([math.factorial(n) for n in range(cutoff)], dtype=np.float64)
    )
    coherent_rows = (
        np.exp(-0.5 * np.abs(alpha) ** 2)[:, None]
        * alpha[:, None] ** number[None, :]
        / factorial_sqrt[None, :]
    )
    rho = np.eye(cutoff, dtype=np.complex128) / cutoff
    identity = np.eye(cutoff, dtype=np.complex128)
    previous_log_likelihood = -np.inf
    conj_coherent = np.conjugate(coherent_rows)
    inv_n_samples = 1.0 / alpha.size

    for _ in range(iterations):
        rho_coherent = coherent_rows @ rho.T
        probabilities = np.real(np.sum(rho_coherent * conj_coherent, axis=1))
        probabilities = np.maximum(probabilities, 1e-15)
        log_likelihood = float(np.mean(np.log(probabilities)))

        weighted_conj = conj_coherent / probabilities[:, None]
        r_operator = (coherent_rows.T @ weighted_conj) * inv_n_samples

        update = (1.0 - dilution) * identity + dilution * r_operator
        rho = update @ rho @ update
        rho = 0.5 * (rho + np.conjugate(rho.T))
        rho /= np.trace(rho)
        if abs(log_likelihood - previous_log_likelihood) < tolerance:
            break
        previous_log_likelihood = log_likelihood
    return np.asarray(rho, dtype=np.complex128)


def coherent_density_matrix(alpha: complex, cutoff: int = 8) -> ComplexArray:
    """Return the normalized truncated coherent-state density matrix."""
    state = coherent_state_vector(alpha, cutoff)
    state /= np.linalg.norm(state)
    return np.outer(state, np.conjugate(state))


def wigner_function(
    density_matrix: npt.ArrayLike,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
) -> FloatArray:
    r"""Evaluate W(alpha) = 2/pi Tr[D(-alpha) rho D(alpha) parity].

    Evaluated using the exact, vectorized Fock-basis Laguerre expansion:
    \[
    W(\alpha) = \frac{2}{\pi} e^{-2|\alpha|^2} \left[
        \sum_{n=0}^{C-1} \rho_{nn} (-1)^n L_n(4|\alpha|^2)
        + 2 \operatorname{Re}\left(
            \sum_{m < n} \rho_{nm} (-1)^m \sqrt{\frac{m!}{n!}}
            (2\alpha^*)^{n-m} L_m^{(n-m)}(4|\alpha|^2)
        \right)
    \right]
    \]
    """
    from scipy.special import eval_genlaguerre

    rho = np.asarray(density_matrix, dtype=np.complex128)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("density_matrix must be square")
    x_values = np.asarray(x, dtype=np.float64).reshape(-1)
    y_values = np.asarray(y, dtype=np.float64).reshape(-1)
    alpha = x_values[None, :] + 1j * y_values[:, None]
    alpha_sq = np.abs(alpha) ** 2
    r_sq_4 = 4.0 * alpha_sq
    cutoff = rho.shape[0]

    total = np.zeros((y_values.size, x_values.size), dtype=np.complex128)

    # Diagonal elements m == n
    for n in range(cutoff):
        if abs(rho[n, n]) > 1e-15:
            L_n = eval_genlaguerre(n, 0, r_sq_4)
            total += rho[n, n].real * ((-1.0) ** n) * L_n

    # Off-diagonal elements m < n
    two_alpha_conj = 2.0 * np.conjugate(alpha)
    for m in range(cutoff):
        for n in range(m + 1, cutoff):
            if abs(rho[n, m]) > 1e-15:
                k = n - m
                c = ((-1.0) ** m) * math.sqrt(
                    math.factorial(m) / math.factorial(n)
                )
                L_m_k = eval_genlaguerre(m, k, r_sq_4)
                term = rho[n, m] * c * (two_alpha_conj ** k) * L_m_k
                total += 2.0 * term

    factor = (2.0 / np.pi) * np.exp(-2.0 * alpha_sq)
    return np.asarray(np.real(factor * total), dtype=np.float64)
