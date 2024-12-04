import glob
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from bilby.core.result import Result
from conftest import MAXX, MINX
from lnl_computer.cosmic_integration.mcz_grid import McZGrid
from lnl_computer.observation.mock_observation import MockObservation
from scipy.stats import norm
from lnl_surrogate.surrogate import LnLSurrogate, train
from common import plotter_1d
import pytest


@pytest.mark.skip_on_github
def test_simple(mock_data, tmpdir):
    outdir = f"{tmpdir}/real_lnl"
    os.makedirs(outdir, exist_ok=True)

    true_aSF = 0.01
    obs = MockObservation.load(mock_data.observations_filename)
    true_lnl, _ = McZGrid.lnl(
        compas_h5_path=mock_data.compas_filename,
        sf_sample={"aSF": true_aSF},
        mcz_obs=obs,
    )

    mu_1 = McZGrid.from_compas_output(
        compas_path=mock_data.compas_filename,
        cosmological_parameters={"aSF": 1},
    ).n_detections(obs.duration)
    d = obs.n_events
    aSF_post_mu = d / mu_1
    aSF_post_sigma = aSF_post_mu / np.sqrt(d)

    train(
        model_type="gp",
        mcz_obs_filename=mock_data.observations_filename,
        compas_h5_filename=mock_data.compas_filename,
        acquisition_fns=["pv", "ei"],
        params=["aSF"],
        duration=1,
        n_init=5,
        n_rounds=5,
        n_pts_per_round=5,
        outdir=outdir,
        reference_param=dict(aSF=true_aSF, lnl=true_lnl),
        model_plotter=plotter_1d,
        noise_level=1,
    )

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
