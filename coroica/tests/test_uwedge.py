import numpy as np
import pytest
from scipy import stats
from coroica.uwedge import uwedge
from coroica.utils import md_index

rs = np.random.RandomState(46463)


def orthogonal_A(n, dtype='real'):
    N = rs.normal(size=(n,n))
    if dtype == 'complex':
        N = N + 1j*rs.normal(size=(n,n))
    A, _ = np.linalg.qr(N)  # orthogonal
    return A


def sim_samples(d, M, dtype):
    """Samples generated based on VI. Simulations in UWEDGE paper"""
    A = orthogonal_A(d, dtype=dtype)
    V = np.linalg.inv(A)
    diags = [np.diag(rs.uniform(1, 2, size=d))
             for i in range(M)]
    Rxx = np.stack([A.dot(d.dot(A.T)) for d in diags])
    return A, V, diags, Rxx


@pytest.mark.parametrize("d", [3, 20])
@pytest.mark.parametrize("M", [3, 10])
@pytest.mark.parametrize("dtype", ['real', 'complex'])
def test_uwedge(d, M, dtype):
    A, V, diags, Rxx = sim_samples(d, M, dtype)
    V_, diags_, converged, iteration, meanoffdiag = uwedge(
        Rxx, return_diagonals=True, minimize_loss=True,
        condition_threshold=10000,
    )
    assert converged
    assert md_index(A, V_) < 1e-2
