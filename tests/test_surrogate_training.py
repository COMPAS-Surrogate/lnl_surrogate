from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pytest
import tensorflow as tf
from conftest import MAXX, MIDX, MINX, NORM, _mock_lnl_truth
from trieste.acquisition.function import PredictiveVariance

from lnl_surrogate.surrogate import LnLSurrogate, train


def _plot_res(model, data, search_space, **kwargs):
    x = np.linspace(MINX, MAXX, 100).reshape(-1, 1)

    ref_lnl = kwargs["truth"]["lnl"]

    # model_gp = -(lnl - reference_lnl)
    true_y = -(NORM.logpdf(x) - ref_lnl)
    model_y, model_yunc = model.predict(x)

    x_obs = data.query_points
    y_obs = data.observations

    tf_to_np = lambda x: x.numpy().flatten() if hasattr(x, "numpy") else x
    model_yunc = tf_to_np(model_yunc)
    model_y = tf_to_np(model_y)

    # make new fig
    plt.figure()
    plt.plot(x, true_y, label="True", color="black")
    plt.plot(x, model_y, label="Model", color="tab:orange")
    plt.scatter(x_obs, y_obs, label="Observed", color="black")
    yup, ydown = model_y + model_yunc, model_y - model_yunc
    plt.fill_between(
        x.flatten(),
        yup.flatten(),
        ydown.flatten(),
        alpha=0.2,
        color="tab:orange",
    )
    plt.legend(loc="upper right")
    return plt.gcf()


# @pytest.mark.parametrize('model_type', ['gp', 'deepgp'])
@pytest.mark.parametrize(
    "model_type",
    [
        "gp",
        # "deepgp"
    ],
)
def test_1d(monkeypatch_lnl, mock_data, tmpdir, model_type):
    outdir = f"{tmpdir}/{model_type}"
    res = train(
        model_type=model_type,
        mcz_obs_filename=mock_data.observations_filename,
        compas_h5_filename=mock_data.compas_filename,
        acquisition_fns=["nlcb"],
        params=["aSF"],
        n_init=2,
        n_rounds=6,
        n_pts_per_round=2,
        duration=1,
        outdir=outdir,
        truth=_mock_lnl_truth(),
        model_plotter=_plot_res,
        noise_level=1e-3,
    )
    assert res is not None
    lnl_surr = LnLSurrogate.load(outdir)
    lnl_surr.parameters.update({"aSF": 0.1})
    lnl = lnl_surr.log_likelihood()
    assert isinstance(tf.squeeze(lnl).numpy(), float)
