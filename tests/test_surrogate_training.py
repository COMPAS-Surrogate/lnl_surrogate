from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pytest
import tensorflow as tf
from trieste.acquisition.function import PredictiveVariance

from lnl_surrogate.surrogate import train, load

from conftest import MINX, MAXX, MIDX, NORM, _mock_lnl_truth



def _plot_res(model, data, search_space):
    x = np.linspace(MINX, MAXX, 100).reshape(-1, 1)
    true_y = NORM.logpdf(x) * -1.0
    model_y, model_yunc = model.predict(x)
    x_obs = data.query_points
    y_obs = data.observations

    tf_to_np = lambda x: x.numpy().flatten() if hasattr(x, 'numpy') else x
    # make new fig
    plt.figure()
    plt.plot(x, true_y, label='True', color='black')
    plt.plot(x, model_y, label='Model', color="tab:orange")
    plt.scatter(x_obs, y_obs, label='Observed', color='black')
    yup, ydown = tf_to_np(model_y + model_yunc), tf_to_np(model_y - model_yunc)
    plt.fill_between(x.flatten(), yup.flatten(), ydown.flatten(), alpha=0.2, color="tab:orange")
    plt.legend(loc='upper right')
    return plt.gcf()


# @pytest.mark.parametrize('model_type', ['gp', 'deepgp'])
@pytest.mark.parametrize('model_type', ['gp', ])
def test_1d(monkeypatch_lnl, mock_data, tmpdir, model_type):
    outdir = f'{tmpdir}/{model_type}'
    res = train(
        model_type=model_type,
        mcz_obs=mock_data.observations.mcz,
        compas_h5_filename=mock_data.compas_filename,
        acquisition_fns=[PredictiveVariance()],
        params=['aSF'],
        n_init=2,
        n_rounds=1,
        n_pts_per_round=1,
        outdir=outdir,
        truth=_mock_lnl_truth(),
        model_plotter=_plot_res,
        noise_level=1e-3
    )
    assert res is not None
    model = load(outdir)
    lnl, _ = model.predict(np.array([[0.1]]))
    assert isinstance(tf.squeeze(lnl).numpy(), float)
