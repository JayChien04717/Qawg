import numpy as np

from QAWG import (
    heterodyne_ml_density_matrix,
    normalize_heterodyne_reference,
    wigner_function,
)


def test_single_photon_pipeline_gives_negative_wigner_at_origin():
    """Vacuum-referenced heterodyne samples of |1> -> P(1) high, W(0) < 0."""
    rng = np.random.default_rng(0)
    n = 20000
    vacuum = (rng.normal(size=n) + 1j * rng.normal(size=n)) / np.sqrt(2)
    # Husimi-Q samples of |1>: |alpha|^2 ~ Gamma(2), uniform phase.
    one = np.sqrt(rng.gamma(2.0, 1.0, n)) * np.exp(2j * np.pi * rng.random(n))
    offset = 0.3 + 0.1j  # electrical offset removed by the reference
    _, (signal,), _, _ = normalize_heterodyne_reference(
        vacuum + offset, one + offset
    )
    rho = heterodyne_ml_density_matrix(signal, cutoff=6)
    assert np.real(rho[1, 1]) > 0.9
    axis = np.linspace(-3, 3, 61)
    assert wigner_function(rho, axis, axis)[30, 30] < -0.4
