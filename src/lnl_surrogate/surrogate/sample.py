import logging
import os

import bilby
import matplotlib.pyplot as plt
from lnl_computer.cosmic_integration.star_formation_paramters import (
    get_star_formation_prior,
)

from ..plotting import plot_overlaid_corner
from .lnl_surrogate import LnLSurrogate

ORIG_COL = "tab:blue"
VAR_COL = "tab:red"
HIGHRES_COL = "tab:green"
THRESHOLD_COL = "tab:purple"


HIGHRES_LABEL = "_highres"
VARIABLE_LABEL = "_variable_lnl"
THRESHOLD_LABEL = "_thresholded"

def sample_lnl_surrogate(
    lnl_model_path: str,
    outdir: str,
    label: str = None,
    verbose=False,
    mcmc_kwargs={},
):
    bilby_logger = logging.getLogger("bilby")

    bilby_logger.setLevel(logging.ERROR)
    if verbose:
        bilby_logger.setLevel(logging.INFO)

    lnl_surrogate = LnLSurrogate.load(
        outdir=lnl_model_path, variable_lnl=False
    )
    variable_lnl_surrogate = LnLSurrogate.load(
        lnl_model_path, variable_lnl=True
    )
    datafname = lnl_surrogate.get_datafname(outdir=lnl_model_path)
    thresholded_lnl_surr = LnLSurrogate.from_csv(
        datafname, model_type="gp", label="thresholded", lnl_threshold=10
    )

    prior = get_star_formation_prior(parameters=lnl_surrogate.param_keys)
    label = (
        label
        if label is not None
        else f"surrogate_npts{lnl_surrogate.n_training_points}"
    )
    truths = {}
    if lnl_surrogate.reference_param is not None:
        truths = {
            k: lnl_surrogate.reference_param[k]
            for k in lnl_surrogate.param_keys
        }

    mcmc_kwargs["nwalkers"] = mcmc_kwargs.get("nwalkers", 10)
    mcmc_kwargs["iterations"] = mcmc_kwargs.get("iterations", 1000)

    sampler_kwargs = dict(
        priors=prior,
        sampler="emcee",
        injection_parameters=truths,
        outdir=outdir,
        clean=True,
        verbose=verbose,
        plot=True,
        **mcmc_kwargs,
        meta_data=dict(npts=lnl_surrogate.n_training_points),
    )

    result = bilby.run_sampler(
        likelihood=lnl_surrogate,
        label=label,
        **sampler_kwargs,
        color=ORIG_COL,
    )
    thresholeded_result = bilby.run_sampler(
        likelihood=thresholded_lnl_surr,
        label=label + THRESHOLD_LABEL,
        **sampler_kwargs,
        color=THRESHOLD_COL,
    )
    variable_lnl_result = bilby.run_sampler(
        likelihood=variable_lnl_surrogate,
        label=label + VARIABLE_LABEL,
        **sampler_kwargs,
        color=VAR_COL,
    )

    sampler_kwargs["iterations"] = 3000
    result_highres = bilby.run_sampler(
        likelihood=lnl_surrogate,
        label=label + HIGHRES_LABEL,
        **sampler_kwargs,
        color=HIGHRES_COL,
    )

    plot_dir = f"{outdir}/../plots"
    os.makedirs(plot_dir, exist_ok=True)
    plot_overlaid_corner(
        [result.posterior, variable_lnl_result.posterior],
        sample_labels=["LnL surrogate", "Variable LnL surrogate"],
        axis_labels=lnl_surrogate.param_latex,
        colors=[ORIG_COL, VAR_COL],
        fname=f"{outdir}/{label}_variablecompare_corner.png",
        truths=truths,
        annotate=f"#pts: {lnl_surrogate.n_training_points}",
    )
    plot_overlaid_corner(
        [result.posterior, result_highres.posterior],
        sample_labels=["1K MCMC", "3k MCMC"],
        axis_labels=lnl_surrogate.param_latex,
        colors=[ORIG_COL, HIGHRES_COL],
        fname=f"{outdir}/{label}_mcmccompare_corner.png",
        truths=truths,
        annotate=f"#pts: {lnl_surrogate.n_training_points}",
    )
    plot_overlaid_corner(
        [thresholeded_result.posterior, result_highres.posterior],
        sample_labels=["Thresholded", "3k MCMC"],
        axis_labels=lnl_surrogate.param_latex,
        colors=[THRESHOLD_COL, HIGHRES_COL],
        fname=f"{outdir}/{label}_threshcompare_corner.png",
        truths=truths,
        annotate=f"#pts: {thresholded_lnl_surr.n_training_points}",
    )


def run_sampler(
    lnl_surrogate, outdir, label=None, verbose=False, mcmc_kwargs={}
):
    prior = get_star_formation_prior(parameters=lnl_surrogate.param_keys)
    label = (
        label
        if label is not None
        else f"surrogate_npts{lnl_surrogate.n_training_points}"
    )
    truths = {}

    if lnl_surrogate.reference_param:
        truths = {
            k: lnl_surrogate.reference_param[k]
            for k in lnl_surrogate.param_keys
        }
    else:
        lnl_surrogate.parameters.update(prior.sample())

    mcmc_kwargs["nwalkers"] = mcmc_kwargs.get("nwalkers", 10)
    mcmc_kwargs["iterations"] = mcmc_kwargs.get("iterations", 1000)
    mcmc_kwargs["color"] = ORIG_COL

    sampler_kwargs = dict(
        priors=prior,
        sampler="emcee",
        injection_parameters=truths,
        outdir=outdir,
        clean=True,
        verbose=verbose,
        plot=True,
        **mcmc_kwargs,
        meta_data=dict(npts=lnl_surrogate.n_training_points),
    )

    result = bilby.run_sampler(
        likelihood=lnl_surrogate,
        label=label,
        **sampler_kwargs,
    )
    return result
