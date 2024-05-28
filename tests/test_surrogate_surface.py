import numpy as np
import pandas as pd
from lnl_computer.cosmic_integration.star_formation_paramters import (
    get_star_formation_prior,
)
from scipy.stats import norm

from lnl_surrogate.surrogate import LnLSurrogate


def __generate_mock_csv(path):
    params = ["aSF", "mu_z"]
    prior = get_star_formation_prior(parameters=params)
    samples = prior.sample(100)
    samples = pd.DataFrame(samples)
    samples_means = samples.mean(axis=0)
    samples_std = samples.std(axis=0)
    # 2d Normal using provided means and stds
    lnl = np.zeros(samples.shape[0])
    for p in params:
        lnl += norm(loc=samples_means[p], scale=samples_std[p]).logpdf(
            samples[p]
        )
    samples["lnl"] = lnl
    samples.to_csv(path, index=False)
    return path


def test_csv_loader_lnl_surrogate(tmpdir):
    csv = __generate_mock_csv(f"{tmpdir}/data.csv")
    surr = LnLSurrogate.from_csv(
        csv, model_type="gp", label="test", plot=True, outdir=tmpdir
    )
