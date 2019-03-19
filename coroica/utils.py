import numpy as np
from scipy.optimize import linear_sum_assignment


def autocov(X, lag=0):
    if lag == 0:
        return np.cov(X)
    else:
        A = X[:, lag:] - X[:, lag:].mean(axis=1, keepdims=True)
        B = X[:, :-lag] - X[:, :-lag].mean(axis=1, keepdims=True)
        n = A.shape[1] - 1
        return A.dot(B.T.conj()) / n


def rigidgroup(length, nosamples):
    groups = int(np.floor(length / nosamples))
    changepoints = [int(np.round(a))
                    for a in np.linspace(0, length, groups + 1)]
    changepoints = list(set(changepoints))
    changepoints.sort()
    index = np.zeros(length)
    for (i, a, b) in zip(range(groups), changepoints[:-1], changepoints[1:]):
        index[a:b] = i
    return index


def md_index(A, Vhat):
    """Minimum distance index as defined in

    Ilmonen P., Nordhausen K., Oja H., Ollila E. (2010) A New Performance Index for ICA: Properties, Computation and Asymptotic Analysis. In: Vigneron V., Zarzoso V., Moreau E., Gribonval R., Vincent E. (eds) Latent Variable Analysis and Signal Separation. LVA/ICA 2010. Lecture Notes in Computer Science, vol 6365. Springer, Berlin, Heidelberg
    """
    d = np.shape(A)[0]
    G = Vhat.dot(A)
    Gsq = np.abs(G)**2
    Gtilde = Gsq / (Gsq.sum(axis=1)).reshape((d, 1))
    costmat = 1 - 2 * Gtilde + np.tile((Gtilde**2).sum(axis=0), d).reshape((d, d))
    row_ind, col_ind = linear_sum_assignment(costmat)
    md = np.sqrt(d - np.sum(np.diag(Gtilde[row_ind, col_ind]))) / np.sqrt(d - 1)
    return md
