"""Temporal-mode, noisy heterodyne, and Wigner tomography helpers."""

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


def extract_temporal_mode(
    traces: npt.ArrayLike,
    *,
    method: str = "mean",
    reference_traces: npt.ArrayLike | None = None,
    start_sample: int = 0,
    stop_sample: int | None = None,
) -> ComplexArray:
    """Extract a unit-L2 complex temporal mode from measured IQ traces.

    Parameters
    ----------
    traces:
        Complex baseband IQ traces with shape ``(shots, samples)``.
    method:
        - ``"mean"``: normalized coherent mean envelope (for coherent or
          superposition states where ``<a> != 0``).
        - ``"pca"``: dominant eigenvector of the noise-subtracted correlation
          matrix ``G^(1)(t, t') = <S(t) S*(t')> - <S_ref(t) S_ref*(t')>``
          (essential for zero-mean Fock states like ``|1>`` emitted by a
          transmon).
    reference_traces:
        Optional vacuum/noise-only IQ traces with the same sample axis,
        subtracted before mode extraction.
    """
    signal = np.asarray(traces, dtype=np.complex128)
    if signal.ndim != 2 or signal.shape[0] == 0 or signal.shape[1] == 0:
        raise ValueError("traces must have non-empty shape (shots, samples)")
    start = int(start_sample)
    stop = signal.shape[1] if stop_sample is None else int(stop_sample)
    if not 0 <= start < stop <= signal.shape[1]:
        raise ValueError("mode window is outside the acquired traces")

    sig_win = signal[:, start:stop]
    ref_win: ComplexArray | None = None
    if reference_traces is not None:
        ref = np.asarray(reference_traces, dtype=np.complex128)
        if ref.ndim != 2 or ref.shape[1] != signal.shape[1] or ref.shape[0] == 0:
            raise ValueError(
                "reference_traces must have shape (ref_shots, samples)"
            )
        ref_win = ref[:, start:stop]

    mode_method = method.lower()
    if mode_method == "mean":
        envelope = np.mean(sig_win, axis=0)
        if ref_win is not None:
            envelope = envelope - np.mean(ref_win, axis=0)
        norm = float(np.linalg.norm(envelope))
        if norm <= np.finfo(np.float64).eps:
            raise ValueError("extracted mean temporal mode has zero norm")
        return np.asarray(envelope / norm, dtype=np.complex128)

    if mode_method == "pca":
        sig_centered = sig_win
        if ref_win is not None:
            ref_mean = np.mean(ref_win, axis=0, keepdims=True)
            sig_centered = sig_win - ref_mean
            ref_centered = ref_win - ref_mean
            cov = (
                (sig_centered.T @ np.conjugate(sig_centered)) / sig_centered.shape[0]
                - (ref_centered.T @ np.conjugate(ref_centered)) / ref_centered.shape[0]
            )
        else:
            cov = (sig_centered.T @ np.conjugate(sig_centered)) / sig_centered.shape[0]
        cov = 0.5 * (cov + np.conjugate(cov.T))
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        if float(eigenvalues[-1]) <= 0.0:
            raise ValueError(
                "noise-subtracted correlation matrix has no positive eigenmode"
            )
        mode = eigenvectors[:, -1]
        # Fix arbitrary global phase so the peak sample is real and positive.
        peak_index = int(np.argmax(np.abs(mode)))
        if abs(mode[peak_index]) > 0:
            mode = mode * np.exp(-1j * np.angle(mode[peak_index]))
        return np.asarray(mode / np.linalg.norm(mode), dtype=np.complex128)

    raise ValueError("method must be 'mean' or 'pca'")


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
    vacuum noise at the measurement input (``noise_photons == 0``).
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


def annihilation_operator(cutoff: int) -> ComplexArray:
    """Return the truncated bosonic annihilation operator in the Fock basis."""
    if cutoff < 1:
        raise ValueError("cutoff must be positive")
    operator = np.zeros((cutoff, cutoff), dtype=np.complex128)
    for number in range(1, cutoff):
        operator[number - 1, number] = math.sqrt(number)
    return operator


def density_matrix_moments(
    density_matrix: npt.ArrayLike,
    maximum_order: int,
) -> ComplexArray:
    """Return normal-ordered moments ``M[m, n] = Tr[rho (a^dag)^m a^n]``."""
    rho = np.asarray(density_matrix, dtype=np.complex128)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("density_matrix must be square")
    if maximum_order < 0:
        raise ValueError("maximum_order cannot be negative")
    cutoff = rho.shape[0]
    annihilation = annihilation_operator(cutoff)
    creation = np.conjugate(annihilation.T)
    a_powers = [np.eye(cutoff, dtype=np.complex128)]
    adag_powers = [np.eye(cutoff, dtype=np.complex128)]
    for _ in range(maximum_order):
        a_powers.append(a_powers[-1] @ annihilation)
        adag_powers.append(adag_powers[-1] @ creation)
    moments = np.empty(
        (maximum_order + 1, maximum_order + 1),
        dtype=np.complex128,
    )
    for m in range(maximum_order + 1):
        for n in range(maximum_order + 1):
            moments[m, n] = np.trace(rho @ adag_powers[m] @ a_powers[n])
    return moments


def complex_moment_stats(
    samples: npt.ArrayLike,
    maximum_order: int,
) -> tuple[ComplexArray, FloatArray]:
    """Compute empirical moments ``<(S*)^m S^n>`` and their standard errors.

    Returns ``(moments, standard_errors)``, each with shape
    ``(maximum_order + 1, maximum_order + 1)``.
    """
    values = np.asarray(samples, dtype=np.complex128).reshape(-1)
    if values.size < 2:
        raise ValueError("at least 2 samples are required for moment statistics")
    if maximum_order < 0:
        raise ValueError("maximum_order cannot be negative")
    conj_values = np.conjugate(values)
    powers = [np.ones_like(values)]
    conj_powers = [np.ones_like(values)]
    for _ in range(maximum_order):
        powers.append(powers[-1] * values)
        conj_powers.append(conj_powers[-1] * conj_values)

    moments = np.empty((maximum_order + 1, maximum_order + 1), dtype=np.complex128)
    errors = np.zeros((maximum_order + 1, maximum_order + 1), dtype=np.float64)
    inv_sqrt_n = 1.0 / math.sqrt(values.size)
    for m in range(maximum_order + 1):
        for n in range(maximum_order + 1):
            prod = conj_powers[m] * powers[n]
            mean = complex(np.mean(prod))
            moments[m, n] = mean
            if m == 0 and n == 0:
                errors[m, n] = 0.0
            else:
                # RMS complex standard error sqrt(Var(Re) + Var(Im)) / sqrt(N).
                variance = float(np.mean(np.abs(prod - mean) ** 2))
                errors[m, n] = math.sqrt(max(variance, 0.0)) * inv_sqrt_n
    return moments, errors


def deconvolve_amplifier_moments(
    measured_signal_moments: npt.ArrayLike,
    measured_reference_moments: npt.ArrayLike,
    *,
    signal_errors: npt.ArrayLike | None = None,
    reference_errors: npt.ArrayLike | None = None,
) -> tuple[ComplexArray, FloatArray | None]:
    """Remove additive phase-insensitive amplifier noise order by order.

    Implements the exact binomial inversion of
    ``<(S*)^m S^n> = sum_{i,j} C(m,i) C(n,j) <(a^dag)^i a^j> <h^{m-i} (h^dag)^{n-j}>``
    (Eichler et al., PRL 106, 220503 (2011)) where the reference measurement
    has vacuum at the signal input.

    When ``signal_errors`` and/or ``reference_errors`` are supplied, propagates
    first-order uncorrelated standard errors to every deconvolved moment so
    downstream state reconstruction weights noisy higher-order moments
    properly.
    """
    measured = np.asarray(measured_signal_moments, dtype=np.complex128)
    noise = np.asarray(measured_reference_moments, dtype=np.complex128)
    if (
        measured.ndim != 2
        or measured.shape != noise.shape
        or measured.shape[0] != measured.shape[1]
    ):
        raise ValueError("signal and reference moment matrices must be square and matching")
    maximum_order = measured.shape[0] - 1

    sig_err = (
        None
        if signal_errors is None
        else np.asarray(signal_errors, dtype=np.float64)
    )
    ref_err = (
        None
        if reference_errors is None
        else np.asarray(reference_errors, dtype=np.float64)
    )
    propagate = sig_err is not None or ref_err is not None
    if propagate:
        if sig_err is None:
            sig_err = np.zeros_like(measured, dtype=np.float64)
        if ref_err is None:
            ref_err = np.zeros_like(noise, dtype=np.float64)
        if sig_err.shape != measured.shape or ref_err.shape != noise.shape:
            raise ValueError("error matrices must match moment matrix shapes")

    signal = np.zeros_like(measured)
    signal[0, 0] = 1.0
    out_var = np.zeros(measured.shape, dtype=np.float64) if propagate else None

    for m in range(maximum_order + 1):
        for n in range(maximum_order + 1):
            if m == 0 and n == 0:
                continue
            known = 0.0j
            var = float(sig_err[m, n] ** 2) if propagate else 0.0
            for i in range(m + 1):
                for j in range(n + 1):
                    if i == m and j == n:
                        continue
                    coeff = float(math.comb(m, i) * math.comb(n, j))
                    known += coeff * signal[i, j] * noise[m - i, n - j]
                    if propagate:
                        assert out_var is not None and ref_err is not None
                        var += (coeff ** 2) * (
                            (abs(noise[m - i, n - j]) ** 2) * out_var[i, j]
                            + (abs(signal[i, j]) ** 2) * float(ref_err[m - i, n - j] ** 2)
                        )
            signal[m, n] = measured[m, n] - known
            if propagate and out_var is not None:
                out_var[m, n] = var

    out_errors = None if out_var is None else np.sqrt(np.maximum(out_var, 0.0))
    return signal, out_errors


def fit_density_matrix_from_moments(
    target_moments: npt.ArrayLike,
    cutoff: int,
    *,
    moment_errors: npt.ArrayLike | None = None,
    max_nfev: int = 10_000,
) -> ComplexArray:
    """Find a physical density matrix matching deconvolved normal moments.

    Uses a lower-triangular Cholesky parameterization ``rho = L L^dag / Tr(L L^dag)``
    and fits unique upper-triangle moments ``m <= n`` weighted by
    ``moment_errors`` (empirical standard errors).  If ``moment_errors`` is not
    provided, falls back to uniform unit weighting rather than dividing by
    ``|target[m, n]|`` (which would catastrophically amplify zero-mean noise).
    """
    from scipy.optimize import least_squares

    target = np.asarray(target_moments, dtype=np.complex128)
    if target.ndim != 2 or target.shape[0] != target.shape[1]:
        raise ValueError("target_moments must be a square matrix")
    if cutoff < 1:
        raise ValueError("cutoff must be positive")
    maximum_order = target.shape[0] - 1

    errors: FloatArray | None = None
    if moment_errors is not None:
        errors = np.asarray(moment_errors, dtype=np.float64)
        if errors.shape != target.shape:
            raise ValueError("moment_errors must match target_moments shape")

    # Precompute normal-ordered operator matrices for unique upper-triangle
    # moments m <= n with 0 < m + n <= maximum_order.
    annihilation = annihilation_operator(cutoff)
    creation = np.conjugate(annihilation.T)
    a_powers = [np.eye(cutoff, dtype=np.complex128)]
    adag_powers = [np.eye(cutoff, dtype=np.complex128)]
    for _ in range(maximum_order):
        a_powers.append(a_powers[-1] @ annihilation)
        adag_powers.append(adag_powers[-1] @ creation)

    entries: list[tuple[int, int, ComplexArray, complex, float]] = []
    positive_errors = (
        errors[errors > 0] if errors is not None else np.array([], dtype=np.float64)
    )
    error_floor = (
        float(np.min(positive_errors)) * 0.1 if positive_errors.size else 1.0
    )
    for m in range(maximum_order + 1):
        for n in range(m, maximum_order + 1):
            if 0 < m + n <= maximum_order:
                op = adag_powers[m] @ a_powers[n]
                sigma = (
                    max(float(errors[m, n]), error_floor)
                    if errors is not None
                    else 1.0
                )
                entries.append((m, n, op, complex(target[m, n]), sigma))

    if not entries:
        return np.eye(cutoff, dtype=np.complex128) / cutoff

    # Lower-triangular Cholesky parameters:
    # - `cutoff` real diagonal entries
    # - `cutoff*(cutoff-1)//2` strictly lower-triangular complex entries
    tril_rows, tril_cols = np.tril_indices(cutoff, k=-1)
    n_off = int(tril_rows.size)
    initial = np.zeros(cutoff + 2 * n_off, dtype=np.float64)
    initial[:cutoff] = 1.0 / math.sqrt(cutoff)

    def density(parameters: FloatArray) -> ComplexArray:
        lower = np.zeros((cutoff, cutoff), dtype=np.complex128)
        np.fill_diagonal(lower, parameters[:cutoff])
        if n_off:
            lower[tril_rows, tril_cols] = (
                parameters[cutoff : cutoff + n_off]
                + 1j * parameters[cutoff + n_off :]
            )
        rho = lower @ np.conjugate(lower.T)
        trace = float(np.real(np.trace(rho)))
        if trace <= 1e-30:
            return np.eye(cutoff, dtype=np.complex128) / cutoff
        return np.asarray(rho / trace, dtype=np.complex128)

    def residual(parameters: FloatArray) -> FloatArray:
        rho = density(parameters)
        values: list[float] = []
        for m, n, op, tgt, sigma in entries:
            pred = complex(np.sum(rho * op.T))
            delta = (pred - tgt) / sigma
            if m == n:
                values.append(float(delta.real))
            else:
                values.append(float(delta.real))
                values.append(float(delta.imag))
        return np.asarray(values, dtype=np.float64)

    fit = least_squares(
        residual,
        initial,
        method="trf",
        max_nfev=max_nfev,
    )
    rho = density(fit.x)
    rho = 0.5 * (rho + np.conjugate(rho.T))
    rho /= np.trace(rho)
    return np.asarray(rho, dtype=np.complex128)


def coherent_state_vector(alpha: complex, cutoff: int) -> ComplexArray:
    """Return a truncated coherent-state vector in the Fock basis."""
    if cutoff < 1:
        raise ValueError("cutoff must be positive")
    values = np.empty(cutoff, dtype=np.complex128)
    values[0] = np.exp(-0.5 * abs(alpha) ** 2)
    for number in range(1, cutoff):
        values[number] = values[number - 1] * alpha / math.sqrt(number)
    return values


def _noisy_heterodyne_povm_batch(
    alpha: ComplexArray,
    cutoff: int,
    noise_photons: float,
) -> ComplexArray:
    """Return ``Pi[i, m, n] = <m| D(alpha_i) rho_th(N) D^dag(alpha_i) |n> / pi``.

    Uses the cancellation-free Cahill-Glauber polynomial expansion in
    ``N / (1 + N)`` and ``|alpha|^2 / (1 + N)^2``, valid for all ``N >= 0``.
    """
    n_samples = alpha.size
    one_plus_n = 1.0 + noise_photons
    abs_sq = np.abs(alpha) ** 2
    prefactor = np.exp(-abs_sq / one_plus_n) / (math.pi * one_plus_n)
    alpha_scaled = alpha / one_plus_n
    alpha_conj_scaled = np.conjugate(alpha_scaled)

    # Precompute powers of |alpha|^2, alpha/(1+N), alpha*/(1+N)
    abs_sq_pow = [np.ones(n_samples, dtype=np.float64)]
    a_pow = [np.ones(n_samples, dtype=np.complex128)]
    ac_pow = [np.ones(n_samples, dtype=np.complex128)]
    for _ in range(cutoff):
        abs_sq_pow.append(abs_sq_pow[-1] * abs_sq)
        a_pow.append(a_pow[-1] * alpha_scaled)
        ac_pow.append(ac_pow[-1] * alpha_conj_scaled)

    povm = np.zeros((n_samples, cutoff, cutoff), dtype=np.complex128)
    for p in range(cutoff):
        for q in range(p, cutoff):
            k = q - p
            # sum_{j=0}^{p} C(p+k, p-j) / j! * N^{p-j} / (1+N)^{p+j} * |alpha|^{2j}
            poly = np.zeros(n_samples, dtype=np.float64)
            for j in range(p + 1):
                weight = (
                    math.comb(p + k, p - j)
                    / math.factorial(j)
                    * (noise_photons ** (p - j))
                    / (one_plus_n ** (p + j))
                )
                if weight != 0.0:
                    poly += weight * abs_sq_pow[j]
            norm_factor = math.sqrt(math.factorial(p) / math.factorial(q))
            common = prefactor * norm_factor * poly
            if p == q:
                povm[:, p, p] = common
            else:
                # m = q > n = p -> (alpha / (1+N))^k
                val_qp = common * a_pow[k]
                povm[:, q, p] = val_qp
                povm[:, p, q] = np.conjugate(val_qp)
    return povm


def heterodyne_ml_density_matrix(
    samples: npt.ArrayLike,
    *,
    cutoff: int = 8,
    iterations: int = 200,
    dilution: float = 0.5,
    tolerance: float = 1e-9,
    noise_photons: float = 0.0,
    histogram_bins: int = 80,
) -> ComplexArray:
    """Estimate a physical density matrix from (optionally noisy) heterodyne samples.

    Uses a diluted ``R-rho-R`` iteration:
    - When ``noise_photons == 0.0`` (default), uses the ideal coherent-state
      POVM ``Pi(alpha) = |alpha><alpha| / pi``.
    - When ``noise_photons > 0.0``, uses the displaced thermal-state POVM
      ``Pi(alpha; N) = D(alpha) rho_th(N) D^dag(alpha) / pi`` corresponding to
      a linear amplifier chain adding ``N = noise_photons`` thermal photons
      (``S = a + h^dag``).  Samples must already be calibrated in
      ``sqrt(photon)`` units (e.g. vacuum reference has ``<|S_ref|^2> = 1 + N``).
      Large datasets are compressed onto a 2D ``(Re alpha, Im alpha)`` grid of
      ``histogram_bins x histogram_bins`` so memory and runtime stay bounded
      even for ``10^6`` shots, with monotonic adaptive step acceleration.
    """
    alpha = np.asarray(samples, dtype=np.complex128).reshape(-1)
    if alpha.size < 1:
        raise ValueError("samples cannot be empty")
    if cutoff < 1 or iterations < 1:
        raise ValueError("cutoff and iterations must be positive")
    if not 0 < dilution <= 1:
        raise ValueError("dilution must be in (0, 1]")
    if noise_photons < 0:
        raise ValueError("noise_photons cannot be negative")
    if histogram_bins < 8:
        raise ValueError("histogram_bins must be at least 8")

    rho = np.eye(cutoff, dtype=np.complex128) / cutoff
    identity = np.eye(cutoff, dtype=np.complex128)
    previous_log_likelihood = -np.inf
    inv_n_samples = 1.0 / alpha.size

    if noise_photons == 0.0:
        number = np.arange(cutoff, dtype=np.float64)
        factorial_sqrt = np.sqrt(
            np.asarray([math.factorial(n) for n in range(cutoff)], dtype=np.float64)
        )
        coherent_rows = (
            np.exp(-0.5 * np.abs(alpha) ** 2)[:, None]
            * alpha[:, None] ** number[None, :]
            / factorial_sqrt[None, :]
        )
        conj_coherent = np.conjugate(coherent_rows)

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

    # Compress large datasets onto a fine 2D histogram grid so POVM matrices
    # are precomputed once in O(bins^2) memory regardless of shot count.
    if alpha.size > histogram_bins * histogram_bins:
        r_max = float(np.max(np.abs(alpha))) * 1.001
        if r_max == 0.0:
            centers = np.zeros(1, dtype=np.complex128)
            weights = np.ones(1, dtype=np.float64)
        else:
            hist, x_edges, y_edges = np.histogram2d(
                alpha.real,
                alpha.imag,
                bins=histogram_bins,
                range=[[-r_max, r_max], [-r_max, r_max]],
            )
            xc = 0.5 * (x_edges[:-1] + x_edges[1:])
            yc = 0.5 * (y_edges[:-1] + y_edges[1:])
            ix, iy = np.nonzero(hist)
            centers = xc[ix] + 1j * yc[iy]
            weights = hist[ix, iy] * inv_n_samples
    else:
        centers = alpha
        weights = np.full(alpha.size, inv_n_samples, dtype=np.float64)

    povm = _noisy_heterodyne_povm_batch(centers, cutoff, float(noise_photons))
    povm_flat = povm.reshape(centers.size, cutoff * cutoff)
    povm_t_flat = np.swapaxes(povm, 1, 2).reshape(centers.size, cutoff * cutoff)

    probabilities = np.maximum(np.real(povm_t_flat @ rho.ravel()), 1e-15)
    log_likelihood = float(np.sum(weights * np.log(probabilities)))
    step = max(float(dilution), 1.0)

    for _ in range(iterations):
        r_operator = ((weights / probabilities) @ povm_flat).reshape(
            cutoff, cutoff
        )
        accepted = False
        for _ in range(15):
            update = identity + step * (r_operator - identity)
            candidate = update @ rho @ update
            candidate = 0.5 * (candidate + np.conjugate(candidate.T))
            trace = float(np.real(np.trace(candidate)))
            if trace > 0.0:
                candidate /= trace
                cand_probs = np.maximum(
                    np.real(povm_t_flat @ candidate.ravel()), 1e-15
                )
                cand_ll = float(np.sum(weights * np.log(cand_probs)))
                if cand_ll >= log_likelihood:
                    improvement = cand_ll - log_likelihood
                    rho = candidate
                    probabilities = cand_probs
                    log_likelihood = cand_ll
                    step = min(step * 1.4, 200.0)
                    accepted = True
                    if improvement < tolerance:
                        return np.asarray(rho, dtype=np.complex128)
                    break
            step *= 0.5
            if step < 1e-3:
                break
        if not accepted:
            break
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


__all__ = [
    "annihilation_operator",
    "calibrate_iq_samples",
    "coherent_density_matrix",
    "coherent_state_vector",
    "complex_moment_stats",
    "deconvolve_amplifier_moments",
    "density_matrix_moments",
    "extract_temporal_mode",
    "fit_density_matrix_from_moments",
    "heterodyne_ml_density_matrix",
    "normalize_heterodyne_reference",
    "project_temporal_mode",
    "temporal_mode_weights",
    "wigner_function",
]
