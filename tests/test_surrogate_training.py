import glob
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from bilby.core.result import Result
from conftest import MAXX, MIDX, MINX, NORM, _mock_lnl_truth
from lnl_computer.cosmic_integration.mcz_grid import McZGrid
from lnl_computer.cosmic_integration.star_formation_paramters import (
    get_star_formation_prior,
)
from lnl_computer.observation.mock_observation import MockObservation
from scipy.stats import norm
from trieste.acquisition.function import PredictiveVariance

from lnl_surrogate.surrogate import LnLSurrogate, train


def _plot_res(model, data, search_space, **kwargs):
    x = np.linspace(MINX, MAXX, 100).reshape(-1, 1)

    ref_lnl = kwargs["truth"]["lnl"]

    # model_gp = -(lnl - reference_lnl)
    # lnl_obj = LnLSurrogate(model, data, regret=pd.DataFrame(), reference_lnl=ref_lnl, truths=kwargs["truth"])

    true_y = -(NORM.logpdf(x) - ref_lnl)
    model_y, model_yunc = model.predict(x)

    x_obs = data.query_points
    y_obs = data.observations

    tf_to_np = lambda x: x.numpy().flatten() if hasattr(x, "numpy") else x
    model_yunc = tf_to_np(model_yunc)
    model_y = tf_to_np(model_y)
    #
    # lnls = []
    # for xi in x_obs:
    #     lnl_obj.parameters.update(dict(aSF=xi))
    #     lnls.append(lnl_obj.log_likelihood())

    # make new fig
    plt.figure()
    # plt.plot(x, true_y, label="True", color="black")
    plt.plot(x, model_y, label="Model", color="tab:orange")
    # plt.plot(x, lnls, label="lnls", color="tab:orange")
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
        duration=1,
        n_init=2,
        n_rounds=1,
        n_pts_per_round=1,
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

    # check that bilby result can be loaded
    res_paths = glob.glob(f"{outdir}/out_mcmc/*result.json")
    res = Result.from_json(res_paths[0])
    assert res.meta_data["npts"] == 3


def test_simple(mock_data, tmpdir):
    outdir = f"{tmpdir}/real_lnl"

    duration = 1
    true_aSF = 0.01
    obs = MockObservation.from_npz(mock_data.observations_filename)
    true_lnl, _ = McZGrid.lnl(
        compas_h5_path=mock_data.compas_filename,
        sf_sample={"aSF": true_aSF},
        mcz_obs=obs.mcz,
        duration=duration,
    )

    mu_1 = McZGrid.from_compas_output(
        compas_path=mock_data.compas_filename,
        cosmological_parameters={"aSF": 1},
    ).n_detections(duration)
    d = obs.n_events
    aSF_post_mu = d / mu_1
    aSF_post_sigma = aSF_post_mu / np.sqrt(d)

    # train(
    #     model_type="gp",
    #     mcz_obs_filename=mock_data.observations_filename,
    #     compas_h5_filename=mock_data.compas_filename,
    #     acquisition_fns=["pv", "ei"],
    #     params=["aSF"],
    #     duration=1,
    #     n_init=5,
    #     n_rounds=5,
    #     n_pts_per_round=5,
    #     outdir=outdir,
    #     truth=dict(aSF=true_aSF, lnl=true_lnl),
    #     model_plotter=_plot_res,
    #     noise_level=1,
    # )

    # plot expected normal distribution
    mcmc_files = glob.glob(f"{outdir}/out_mcmc/round4_*result.json")
    results = [Result.from_json(f) for f in mcmc_files]
    results = sorted(results, key=lambda x: x.meta_data["npts"])

    plt.figure()
    x = np.linspace(MINX, MAXX, 500)
    y = norm(loc=true_aSF, scale=aSF_post_sigma * 1.24).pdf(x)
    plt.plot(x, y, label="Analytical P(aSF|d)", color="black")
    # hide y ticks
    plt.gca().axes.get_yaxis().set_visible(False)
    # add ylabel
    plt.ylabel("P(aSF|d)")
    plt.xlabel("aSF")
    plt.ylim(bottom=0)
    # twin axis
    plt.twinx()
    # for each result, plot step-hist, slightly increased alpha
    for i, res in enumerate(results):
        plt.hist(
            res.posterior.aSF,
            bins=30,
            density=True,
            lw=3,
            histtype="step",
            color=f"C{i}",
            label=f"P(aSF|d) from {res.meta_data['npts']} pts {res.label}",
            alpha=0.5,
        )
    # hide y-axis labels
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.xlim(0.0085, 0.012)
    plt.xlabel("aSF")

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outdir}/aSF_posterior.png")
