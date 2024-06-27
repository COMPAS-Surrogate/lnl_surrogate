import os

import click
import matplotlib.pyplot as plt

from .surrogate import build_surrogate
from .surrogate.lnl_surrogate import LnLSurrogate
from .surrogate.sample import run_sampler
from .surrogate.train import train


@click.command(
    "train_lnl_surrogate",
    help="Train a COMPAS LnL(d|aSF, dSF, muz, sigma0) surrogate model using Gaussian Processes or Deep Gaussian Processes. "
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
    help="The parameters to use [aSF, dSF, sigma0, muz]",
    default=["aSF", "dSF", "sigma0", "muz"],
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
):
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
