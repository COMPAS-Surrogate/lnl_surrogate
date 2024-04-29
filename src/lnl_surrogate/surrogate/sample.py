import logging
import os

import bilby
import matplotlib.pyplot as plt
from lnl_computer.cosmic_integration.star_formation_paramters import (
    get_star_formation_prior,
)

from ..plotting import plot_overlaid_corner
from .lnl_surrogate import LnLSurrogate


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

    lnl_surrogate = LnLSurrogate.load(lnl_model_path, variable_lnl=False)
    variable_lnl_surrogate = LnLSurrogate.load(
        lnl_model_path, variable_lnl=True
    )

    prior = get_star_formation_prior(parameters=lnl_surrogate.param_keys)
    label = (
        label
        if label is not None
        else f"surrogate_npts{lnl_surrogate.n_training_points}"
    )
    truths = {}
    if lnl_surrogate.truths is not None:
        truths = {k: lnl_surrogate.truths[k] for k in lnl_surrogate.param_keys}

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
    )

    result = bilby.run_sampler(
        likelihood=lnl_surrogate,
        label=label,
        **sampler_kwargs,
    )
    variable_lnl_result = bilby.run_sampler(
        likelihood=variable_lnl_surrogate,
        label=label.replace("surrogate", "variableLnL_surrogate"),
        **sampler_kwargs,
    )

    plot_dir = f"{outdir}/../plots"
    os.makedirs(plot_dir, exist_ok=True)
    plot_overlaid_corner(
        [result.posterior, variable_lnl_result.posterior],
        sample_labels=["LnL surrogate", "Variable LnL surrogate"],
        axis_labels=lnl_surrogate.param_keys,
        colors=["tab:blue", "tab:red"],
        fname=f"{outdir}/{label}_corner.png",
        truths=truths,
        label=f"#pts: {lnl_surrogate.n_training_points}",
    )
