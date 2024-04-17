import logging

import bilby
import matplotlib.pyplot as plt
from corner import overplot_lines
from lnl_computer.cosmic_integration.star_formation_paramters import (
    get_star_formation_prior,
)

from .lnl_surrogate import LnLSurrogate


def sample_lnl_surrogate(
    lnl_model_path: str, outdir: str, label: str = None, verbose=False
):
    bilby_logger = logging.getLogger("bilby")

    bilby_logger.setLevel(logging.ERROR)
    if verbose:
        bilby_logger.setLevel(logging.INFO)

    lnl_surrogate = LnLSurrogate.load(lnl_model_path)
    prior = get_star_formation_prior(parameters=lnl_surrogate.param_keys)
    label = (
        label
        if label is not None
        else f"surrogate_npts{lnl_surrogate.n_training_points}"
    )
    truths = {}
    if lnl_surrogate.truths is not None:
        truths = {k: lnl_surrogate.truths[k] for k in lnl_surrogate.param_keys}

    result = bilby.run_sampler(
        likelihood=lnl_surrogate,
        priors=prior,
        sampler="emcee",
        iterations=1000,
        nwalkers=10,
        injection_parameters=truths,
        outdir=outdir,
        label=label,
        clean=True,
        verbose=verbose,
        plot=False,
    )
    fig = result.plot_corner(save=False)
    overplot_lines(fig, list(truths.values()), color="red")
    # add textbox on top left corner with n_training_points
    fig.text(
        0.1,
        0.9,
        f"#pts: {lnl_surrogate.n_training_points}",
        ha="center",
        va="center",
        transform=fig.transFigure,
    )
    fig.savefig(f"{outdir}/../plots/{label}_corner.png")
    plt.close(fig)
