import os

import click
import matplotlib.pyplot as plt

from .surrogate import build_surrogate
from .surrogate.lnl_surrogate import LnLSurrogate
from .surrogate.sample import sample_lnl_surrogate
from .surrogate.train import DEFAULT_DICT, train
from .plotting.plot_median_model import plot_median_model
import numpy as np
from typing import Dict, Optional, Union


@click.command(
    "train_lnl_surrogate",
    help="Train a COMPAS LnL(d|aSF, dSF, mu_z, sigma_0) surrogate model using Gaussian Processes or Deep Gaussian Processes. "
         "During training, the model will acquire the next best points to be used for training. ",
)
@click.option(
    "--compas_h5_filename",
    type=str,
    required=True,
    help="The COMPAS h5 filename",
)
@click.option(
    "--param",
    "-p",
    type=str,
    multiple=True,
    required=True,
    help="The parameters to use [aSF, dSF, sigma_0, mu_z]",
    default=["aSF", "dSF", "sigma_0", "mu_z"],
)
@click.option(
    "--mcz_obs_filename",
    type=str,
    required=False,
    help="The observed mcz (npz) filename (if None, will be generated from compas_h5_filename using default SF parameters)",
)
@click.option(
    "--duration",
    type=float,
    required=False,
    default=1.0,
)
@click.option(
    "--outdir",
    "-o",
    type=str,
    required=False,
    help="The output directory for the surrogate model",
)
@click.option(
    "--acquisition_fns",
    "-a",
    type=str,
    multiple=True,
    required=False,
    default=["pv", "ei"],
    help="The acquisition functions to use (PredictiveVariance pv, ExpectedImprovement ei)",
)
@click.option(
    "--n_init",
    type=int,
    required=False,
    default=15,
    help="The number of initial y_pts to use",
)
@click.option(
    "--n_rounds",
    type=int,
    required=False,
    default=5,
    help="The number of rounds of optimization to perform",
)
@click.option(
    "--n_pts_per_round",
    type=int,
    required=False,
    default=10,
    help="The number of y_pts to evaluate per round",
)
@click.option(
    "--save_plots",
    type=bool,
    is_flag=True,
    required=False,
    default=True,
    help="Whether to save plots",
)
@click.option(
    "--reference_param",
    type=str,
    default=None,
    required=False,
    help="The JSON file containing the reference/True parameters",
)
@click.option(
    "--max_threshold",
    type=float,
    required=False,
    default=50,
    help="The JSON file containing the reference/True parameters",
)
@click.option(
    "--aSF",
    type=float,
    required=False,
    default=None,
    help="Reference aSF",
)
@click.option(
    "--dSF",
    type=float,
    required=False,
    default=None,
    help="Reference dSF",
)
@click.option(
    "--mu_z",
    type=float,
    required=False,
    default=None,
    help="Reference mu_z",
)
def cli_train(
        compas_h5_filename,
        mcz_obs_filename,
        param,
        duration,
        outdir,
        acquisition_fns,
        n_init,
        n_rounds,
        n_pts_per_round,
        save_plots,
        reference_param,
        max_threshold,
        aSF,
        dSF=None,
        mu_z=None,
        sigma_0=None,
):

    # update reference_param dict with provided values (if not None)

    if reference_param is None:
        reference_param = DEFAULT_DICT
        # update reference_param dict with provided values (if not None)
        reference_param["aSF"] = aSF if aSF is not None else reference_param["aSF"]
        reference_param["dSF"] = dSF if dSF is not None else reference_param["dSF"]
        reference_param["mu_z"] = mu_z if mu_z is not None else reference_param["mu_z"]
        reference_param["sigma_0"] = sigma_0 if sigma_0 is not None else reference_param["sigma_0"]

    train(
        model_type="gp",
        mcz_obs_filename=mcz_obs_filename,
        compas_h5_filename=compas_h5_filename,
        params=param,
        duration=duration,
        outdir=outdir,
        acquisition_fns=acquisition_fns,
        n_init=n_init,
        n_rounds=n_rounds,
        n_pts_per_round=n_pts_per_round,
        save_plots=save_plots,
        reference_param=reference_param,
        max_threshold=max_threshold,
    )


@click.command(
    "build_surrogate",
    help="Build a surrogate model given the CSV of training data",
)
@click.option("--csv", "-c", required=True, type=str)
@click.option(
    "--model_type",
    "-m",
    type=str,
    required=False,
    help="The model type to use [gp, deepgp]",
    default="gp",
)
@click.option(
    "--label",
    "-l",
    type=str,
    required=False,
    help="The output label for the surrogate model",
    default="lnl_surrogate",
)
@click.option(
    "--outdir",
    "-o",
    type=str,
    required=False,
    help="The output directory for the surrogate model",
    default="outdir",
)
@click.option(
    "--plots/--no-plots",
    show_default=True,
    default=True,
    help="Whether to save plots",
)
@click.option(
    "--lnl-threshold",
    "-t",
    type=float,
    required=False,
    help="The threshold for the LnL",
)
@click.option(
    "--sample/--no-sample",
    show_default=True,
    default=True,
    help="Whether to run the sampler with the built surrogate",
)
def cli_build_surrogate(
        csv: str,
        model_type: str,
        label: str,
        outdir: str,
        plots: bool,
        lnl_threshold: float,
        sample: bool,
):
    build_surrogate(
        csv=csv,
        model_type=model_type,
        label=label,
        outdir=outdir,
        plots=plots,
        lnl_threshold=lnl_threshold,
        sample=sample,
    )


@click.command(
    "sample_lnl_surrogate",
    help="Sample the LNL surrogate model",
)
@click.option("--lnl-model-path", "-l", required=True, type=str)
@click.option("--outdir", "-o", required=True, type=str, help="Output directory of the sampler")
@click.option("--label", "-l", type=str, help="Label for the output")
@click.option("--verbose/--no-verbose", default=False)
@click.option("--mcmc-kwargs", type=dict, default={})
@click.option("--seed", type=int, default=None)
@click.option("--nreps", type=int, default=1)
def cli_sample_lnl_surrogate(
        lnl_model_path: str,
        outdir: str,
        label: str = None,
        verbose=False,
        mcmc_kwargs={},
        seed=None,
        nreps=1,
):
    if nreps == 1:
        if seed is not None:
            np.random.seed(seed)
        sample_lnl_surrogate(
            lnl_model_path=lnl_model_path,
            outdir=outdir,
            label=label,
            verbose=verbose,
            mcmc_kwargs=mcmc_kwargs,
        )
    else:
        for i in range(nreps):
            np.random.seed(i)
            sample_lnl_surrogate(
                lnl_model_path=lnl_model_path,
                outdir=outdir,
                label=label + f"_{i}",
                verbose=verbose,
                mcmc_kwargs=mcmc_kwargs,
            )


@click.command("plot_median_model")
@click.option("--compas-h5", type=str, help="Path to COMPAS H5 file (e.g COMPAS_output.h5)", required=True)
@click.option("--out", type=str, help="Path to output file", default="median_model.png")
@click.option("--bilby-result", type=str, help="Path to bilby result file", required=False, default=None)
@click.option("--aSF", type=float, help="aSF", default=None)
@click.option("--dSF", type=float, help="dSF", default=None)
@click.option("--mu_z", type=float, help="mu_z", default=None)
@click.option("--sigma_0", type=float, help="sigma_0", default=None)
def cli_plot_median_model(compas_h5: str, out: str, bilby_result: Optional[str] = None, aSF: Optional[float] = None,
                          dSF: Optional[float] = None, mu_z: Optional[float] = None, sigma_0: Optional[float] = None):
    cosmic_kwargs = {"aSF": aSF, "dSF": dSF, "mu_z": mu_z, "sigma_0": sigma_0}
    cosmic_kwargs = {k: v for k, v in cosmic_kwargs.items() if v is not None}
    plot_median_model(compas_h5, out, cosmic_kwargs, bilby_result)
