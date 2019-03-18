import numpy as np
import pytest
from scipy import stats
from coroica.uwedge import uwedge

rs = np.random.RandomState(46463)


def orthogonal_A(n):
    N = rs.normal(size=(n,n))
    A, _ = np.linalg.qr(N)  # orthogonal
    return A


@pytest.mark.parametrize("d", [3, 20])
@pytest.mark.parametrize("M", [3, 10])
@pytest.mark.parametrize("dtype", ['real', 'complex'])
def test_uwedge(d, M, dtype):
    """Samples generated based on VI. Simulations in UWEDGE paper"""
    A = orthogonal_A(d)
    if dtype == 'complex':
        A = A + 1j*orthogonal_A(d)
    V = np.linalg.inv(A)
    diags = [np.diag(rs.uniform(1, 2, size=d))
             for i in range(M)]
    Rxx = np.stack([A.dot(d.dot(A.T)) for d in diags])
    V_, diags_, converged, iteration, meanoffdiag = uwedge(
        Rxx, return_diagonals=True, minimize_loss=True
    )
    assert converged
    np.testing.assert_allclose(V_, V)
    np.testing.assert_allclose(diags_, diags)
                   