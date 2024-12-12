import numpy as np
import pandas as pd
from scipy.stats import entropy, ks_2samp
from sklearn.neighbors import KernelDensity
from collections import namedtuple

STATS = namedtuple('STATS', ['js', 'ks', 'kl'])

__all__ = ['compute_statistics']

def _compute_js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))

def _kde_sklearn(x, x_grid, bandwidth=0.2):
    kde = KernelDensity(bandwidth=bandwidth).fit(x[:, None])
    return np.exp(kde.score_samples(x_grid[:, None]))


def compute_statistics(res1_posterior:pd.DataFrame, res2_posterior:pd.DataFrame, use_kde=False, bandwidth=0.2, bins=100):
    assert res1_posterior.shape[1] == res2_posterior.shape[1], "Dimensions of posteriors do not match."
    js_divs, ks_stats, kl_divs = [], [], []
    for col in res1_posterior.columns:
        p, q = res1_posterior[col].values, res2_posterior[col].values
        x_grid = np.linspace(min(p.min(), q.min()), max(p.max(), q.max()), 1000)
        if use_kde:
            p_density, q_density = _kde_sklearn(p, x_grid, bandwidth), _kde_sklearn(q, x_grid, bandwidth)
        else:
            # make p + q the same number of samples (the minimum of the two)
            min_len = min(len(p), len(q))
            p, q = np.random.choice(p, min_len), np.random.choice(q, min_len)
            p_density, _ = np.histogram(p, bins=bins, range=(x_grid.min(), x_grid.max()), density=True)
            q_density, _ = np.histogram(q, bins=bins, range=(x_grid.min(), x_grid.max()), density=True)
        p_density += 1e-10
        q_density += 1e-10
        p_density /= p_density.sum()
        q_density /= q_density.sum()
        js_divs.append(_compute_js_divergence(p_density, q_density))
        ks_stats.append(ks_2samp(p, q)[0])
        kl_divs.append(entropy(p_density, q_density))
    return STATS(np.mean(js_divs), np.mean(ks_stats), np.mean(kl_divs))


