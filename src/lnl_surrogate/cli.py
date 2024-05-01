import os

import click
import matplotlib.pyplot as plt

from .kl_distance_computer import get_list_of_kl_distances
from .surrogate.train import train


@click.command("train_lnl_surrogate")
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
    default=["PredictiveVariance", "ExpectedImprovement"],
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
    "--truth",
    type=str,
    required=False,
    help="The JSON file containing the true parameters",
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
    truth,
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
        truth=truth,
    )


@click.command("plot_kl_distances")
@click.option(
    "--regex",
    "-r",
    type=str,
    required=True,
    help="The regex to match the result files",
)
def cli_plot_kl_distances(regex):
    npts, kl_distances = get_list_of_kl_distances(regex)
    plt.plot(npts, kl_distances)
    plt.xlabel("Number of y_pts")
    plt.ylabel("KL Divergence")
    dirname = os.path.dirname(regex)
    plt.savefig(os.path.join(dirname, "kl_distances.png"))
