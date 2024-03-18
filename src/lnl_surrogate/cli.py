import json

import click
from trieste.acquisition import PredictiveVariance, ExpectedImprovement

from lnl_computer.observation.mock_observation import MockObservation
from .surrogate.train import train


@click.command("train_lnl_surrogate")
@click.option("--compas_h5_filename", type=str, required=True, help="The COMPAS h5 filename")
@click.option("--param", '-p', type=str, multiple=True, required=True,
              help="The parameters to use [aSF, dSF, sigma0, muz]", default=["aSF", "dSF", "sigma0", "muz"])
@click.option("--mcz_obs", type=str, required=False,
              help="The observed mcz values (if None, will be generated from compas_h5_filename using default SF parameters)")
@click.option("--outdir", '-o', type=str, required=False, help="The output directory for the surrogate model")
@click.option("--acquisition_fns", '-a', type=str, multiple=True, required=False,
              default=["PredictiveVariance", "ExpectedImprovement"],
              help="The acquisition functions to use (PredictiveVariance pv, ExpectedImprovement ei)")
@click.option("--n_init", type=int, required=False, default=15, help="The number of initial points to use")
@click.option("--n_rounds", type=int, required=False, default=5, help="The number of rounds of optimization to perform")
@click.option("--n_pts_per_round", type=int, required=False, default=10,
              help="The number of points to evaluate per round")
@click.option("--save_plots", type=bool, required=False, default=True, help="Whether to save plots")
@click.option("--truth", type=str, required=False, help="The JSON file containing the true parameters")
def cli_train(
        compas_h5_filename,
        mcz_obs,
        param,
        outdir,
        acquisition_fns,
        n_init,
        n_rounds,
        n_pts_per_round,
        save_plots,
        truth,
):
    if mcz_obs is None:
        mcz_obs = MockObservation.from_compas_h5(compas_h5_filename).mcz
    elif isinstance(mcz_obs, str):
        mcz_obs = MockObservation.from_npz(mcz_obs).mcz
    else:
        raise ValueError("mcz_obs must be a npz filename or None")

    if truth is not None:
        with open(truth, 'r') as f:
            truth = json.load(f)

    _acquisition_fns = []
    for acq in acquisition_fns:
        if acq == "PredictiveVariance" or acq == "pv":
            _acquisition_fns.append(PredictiveVariance())
        elif acq == "ExpectedImprovement" or acq == "ei":
            _acquisition_fns.append(ExpectedImprovement())
        else:
            raise ValueError(f"Unknown acquisition function: {acq}")

    train(
        model_type='gp',
        mcz_obs=mcz_obs,
        compas_h5_filename=compas_h5_filename,
        params=param,
        outdir=outdir,
        acquisition_fns=_acquisition_fns,
        n_init=n_init,
        n_rounds=n_rounds,
        n_pts_per_round=n_pts_per_round,
        save_plots=save_plots,
        truth=truth,
    )
