import os

import numpy as np
import pandas as pd

from lnl_surrogate.plotting import plot_overlaid_corner


def _generate_mock_posterior(n, sigma=1.0, mu=0.0, p=1):
    params = {f"p{i}": np.random.normal(mu, sigma, n) for i in range(p)}
    return pd.DataFrame(
        {
            **params,
            "log_likelihood": np.random.normal(0, 1, n),
            "log_prior": np.random.normal(0, 1, n),
        }
    )


def test_overlaid_corner(tmpdir):
    n = 1000
    rs = [
        _generate_mock_posterior(n, p=1),
        _generate_mock_posterior(n, mu=1, sigma=0.5, p=1),
    ]
    plot_overlaid_corner(
        rs,
        ["r1", "r2"],
        colors=["r", "b"],
        fname=f"{tmpdir}/corner.png",
        truths=[0.1],
    )
    rs = [
        _generate_mock_posterior(n, p=2),
        _generate_mock_posterior(n, mu=1, sigma=0.5, p=2),
    ]
    plot_overlaid_corner(
        rs,
        sample_labels=["r1", "r2"],
        colors=["r", "b"],
        fname=f"{tmpdir}/corner2.png",
        truths=[0.1, 0.1],
        label="NumPts 10",
    )


def test_plot_eval_matrix(tmpdir, mock_inout_data):
    kwgs = dict(
        in_pts=mock_inout_data.inputs,
        out_pts=mock_inout_data.outputs,
        model=mock_inout_data.model,
        search_space=mock_inout_data.search_space,
        truth=mock_inout_data.truth,
    )

    plot_evaluations(**kwgs).savefig(f"{tmpdir}/eval.png", bbox_inches="tight")
    plot_model_partial_dependence(**kwgs).savefig(f"{tmpdir}/func.png")
    assert os.path.exists(f"{tmpdir}/eval.png")
    assert os.path.exists(f"{tmpdir}/func.png")
