"""Explicit pilot MVAR/PDC and incoming directed Chebyshev operator.

Adjacency[i,j] ALWAYS means source j -> target i. Never symmetrized.
This regularized short-window estimator is a documented adaptation, not author code.
"""
import numpy as np
from scipy.linalg import solve

BANDS = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30.00001)}


def design(x, order, start=None):
    c, n = x.shape
    start = order if start is None else start
    return np.concatenate([x[:, start-k:n-k].T for k in range(1, order+1)], axis=1), x[:, start:].T


def fit_var(x, order, ridge=.01, start=None):
    x = np.asarray(x, float)
    x = x - x.mean(axis=1, keepdims=True)
    z, y = design(x, order, start)
    gram = z.T @ z
    lam = ridge * np.trace(gram) / gram.shape[0]
    if lam <= 0 or not np.isfinite(lam):
        raise ValueError('Degenerate VAR design')
    reg = gram + lam * np.eye(len(gram))
    beta = solve(reg, z.T @ y, assume_a='pos')
    residual = y - z @ beta
    cov = residual.T @ residual / len(y)
    jitter = np.trace(cov) / len(cov) * 1e-8
    sign, logdet = np.linalg.slogdet(cov + jitter * np.eye(len(cov)))
    if sign <= 0 or not np.isfinite(logdet):
        raise ValueError('Invalid residual covariance')
    df = float(np.trace(solve(reg, gram, assume_a='pos')))
    aic = logdet + 2 * (df * len(cov) + len(cov)) / len(y)
    c = len(cov)
    coefficients = beta.reshape(order, c, c).transpose(0, 2, 1)
    return coefficients, {'order': order, 'n_effective': len(y), 'parameters_per_equation': order*c,
                          'effective_df_per_equation': df, 'ridge_lambda': float(lam), 'aic': float(aic)}


def radius(coef):
    p, c, _ = coef.shape
    companion = np.zeros((p*c, p*c))
    companion[:c] = np.concatenate(coef, axis=1)
    if p > 1:
        companion[c:, :-c] = np.eye((p-1)*c)
    return float(np.max(np.abs(np.linalg.eigvals(companion))))


def pdc(coef, fs=200):
    p, c, _ = coef.shape
    freq = np.arange(1, 30.001, .5)
    phase = np.exp(-2j*np.pi*freq[:, None]*np.arange(1, p+1)[None, :]/fs)
    abar = np.eye(c)[None] - np.einsum('fp,pij->fij', phase, coef)
    amplitude = np.abs(abar)
    normalized = amplitude / np.maximum(np.sqrt((amplitude**2).sum(axis=1, keepdims=True)), 1e-12)
    np.testing.assert_allclose((normalized**2).sum(axis=1), 1., atol=1e-10)
    graphs = []
    for lo, hi in BANDS.values():
        a = normalized[(freq >= lo) & (freq < hi)].mean(axis=0)
        np.fill_diagonal(a, 0)
        graphs.append(a)
    return np.stack(graphs)


def make_graph(x, orders=(1, 2, 3, 4, 5), ridge=.01):
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite window')
    candidates = [fit_var(x, p, ridge, start=max(orders)) for p in orders]
    candidates.sort(key=lambda item: item[1]['aic'])
    for coef, info in candidates:
        rho = radius(coef)
        if rho < 1:
            a = pdc(coef)
            if not np.isfinite(a).all():
                raise ValueError('Nonfinite PDC')
            return a.astype(np.float32), {**info, 'spectral_radius': rho}
    raise ValueError('No stable VAR candidate')


def operator(adjacency):
    """Row degree normalization followed by I-P Laplacian; lambda proxy 2.

After row normalization D=I, D^-1/2 P D^-1/2=P. Thus Lhat=-P.
Asymmetry is preserved; no real-spectrum guarantee is claimed.
"""
    a = np.asarray(adjacency, float).copy()
    if np.any(a < 0) or not np.isfinite(a).all():
        raise ValueError('Invalid adjacency')
    np.fill_diagonal(a, 0)
    sums = a.sum(axis=1, keepdims=True)
    p = np.divide(a, sums, out=np.zeros_like(a), where=sums > 0)
    return -p


def synthetic_direction():
    rng = np.random.default_rng(0)
    x = np.zeros((3, 6000))
    for t in range(1, x.shape[1]):
        x[0, t] = .6*x[0, t-1] + rng.normal()
        x[1, t] = .4*x[1, t-1] + .8*x[0, t-1] + rng.normal()
        x[2, t] = .3*x[2, t-1] + rng.normal()
    coef, _ = fit_var(x[:, 1000:], 1, ridge=.001)
    a = pdc(coef)[1]
    assert a[1, 0] > 5*a[0, 1]
    assert a[1, 0] > 5*a[2, 0]
    impulse = np.array([1., 0., 0.])
    propagated = operator(a) @ impulse
    assert abs(propagated[1]) > abs(propagated[2])
    return {'source0_to_target1': float(a[1, 0]), 'reverse': float(a[0, 1]),
            'direction_test': 'passed'}
