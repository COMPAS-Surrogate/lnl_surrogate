import json
import os
import traceback
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Union

import pandas as pd
from tqdm.auto import trange
from trieste.acquisition import AcquisitionRule
from trieste.bayesian_optimizer import OptimizationResult
from trieste.data import Dataset
from trieste.models import TrainableProbabilisticModel

from ..logger import logger, set_log_verbosity
from ..plotting import save_diagnostic_plots, save_gifs
from .lnl_surrogate import LnLSurrogate
from .managers.data_manager import DataManager
from .managers.optimiser import OptimisationManager
from .sample import sample_lnl_surrogate

__all__ = ["train"]


class Trainer:
    def __init__(
        self,
        model_type: str,
        compas_h5_filename: str,
        params=None,
        mcz_obs_filename: Optional[Union[str]] = None,
        duration: Optional[float] = 1,
        acquisition_fns=None,
        n_init: Optional[int] = 5,
        n_rounds: Optional[int] = 5,
        n_pts_per_round: Optional[int] = 10,
        outdir: Optional[str] = "outdir",
        model_plotter: Optional[Callable] = None,
        truth: Optional[Union[Dict[str, float], str]] = None,
        noise_level: Optional[float] = 1e-5,
        save_plots: Optional[bool] = True,
        verbose: Optional[int] = 1,
    ):
        """
        Train a surrogate model using the given data and parameters.
        :param model_type: one of 'gp' or 'deepgp'
        :param mcz_obs: the observed MCZ values
        :param compas_h5_filename: the filename of the compas data
        :param params: the parameters to use [aSF, dSF, sigma0, muz] or dictionary of true values
        :param acquisition_fns: the acquisition functions to use
        :param n_init: the number of initial points to use
        :param n_rounds: the number of rounds of optimization to perform
        :param n_pts_per_round: the number of points to evaluate per round
        :param outdir: the output directory
        :param model_plotter: a function to plot the model
        :return: the optimization result
        """
        logger.info("Initialising surrogate trainer...")
        os.makedirs(outdir, exist_ok=True)
        self.outdir = outdir
        set_log_verbosity(verbose, outdir)

        self.data_mngr = DataManager(
            compas_h5_filename=compas_h5_filename,
            duration=duration,
            outdir=outdir,
            mcz_obs_filename=mcz_obs_filename,
            params=params,
            truths=truth,
        )
        self.opt_mngr = OptimisationManager(
            datamanager=self.data_mngr,
            acquisition_fns=acquisition_fns,
            n_init=n_init,
            model_type=model_type,
            noise_level=noise_level,
        )

        self.n_rounds: int = n_rounds
        self.n_pts_per_round: int = n_pts_per_round
        self.model_plotter: Callable = model_plotter
        self.save_plots: bool = save_plots
        self.regret_data: List[Dict[str, float]] = []

        # caching the initial data and model
        self.model, self.data = self.opt_mngr.get_optimised_model_and_data()

    def __str__(self):
        return (
            "Trainer("
            f"model_type={self.model}, "
            f"params={self.data_mngr.params}, "
            f"truths={self.data_mngr.truths}, "
            f"pts=[init{self.opt_mngr.init_data}, "
            f"rnds={self.n_rounds}x{self.n_pts_per_round}pts)"
        )

    def train_loop(self):
        for round_idx in trange(self.n_rounds, desc="Optimization round"):
            try:
                self._ith_optimization_round(round_idx)
            except Exception as e:
                logger.warning(
                    f"Error in round {round_idx}: {e} {traceback.format_exc()}"
                )

        logger.info(
            f"Optimization complete, saving result and data to {self.outdir}"
        )
        self.save_surrogate()

        if self.save_plots:
            save_gifs(self.outdir)

    def _ith_optimization_round(self, i: int):
        # optimisation stage
        self.opt_mngr.optimize(
            model=self.model,
            data=self.data,
            n_pts=self.n_pts_per_round,
            rule=self.opt_mngr.get_ith_rule(i),
        )
        self.model, self.data = self.opt_mngr.get_optimised_model_and_data()

        # saving diagnostics
        self.__update_regret_data()
        label = f"round{i}_{len(self.data)}pts"
        if self.save_plots:
            save_diagnostic_plots(
                self.data,
                self.model,
                self.opt_mngr.search_space,
                self.outdir,
                label,
                truth=self.data_mngr.truths,
                model_plotter=self.model_plotter,
                reference_lnl=self.data_mngr.reference_lnl,
            )
        self.save_surrogate(label=label)
        sample_lnl_surrogate(
            f"{self.outdir}/{label}",
            outdir=os.path.join(self.outdir, f"out_mcmc"),
            label=label,
        )

    def save_surrogate(self, label: Optional[str] = None):
        lnl_surrogate = LnLSurrogate.from_bo_result(
            bo_result=self.opt_mngr.result,
            params=self.data_mngr.params,
            truths=self.data_mngr.truths,
            outdir=self.outdir,
            reference_lnl=self.data_mngr.reference_lnl,
            label=label,
            regret=pd.DataFrame(self.regret_data),
        )
        lnl_surrogate.save(self.outdir, label)

    def __update_regret_data(self):
        min_obs = tf.reduce_min(self.data.observations).numpy()
        min_idx = tf.argmin(self.data.observations).numpy()[0]
        min_input = self.data.query_points[min_idx].numpy()
        # get the model predictions at the training points
        model_values = self.model.predict(self.data.query_points)
        # upper and lower bounds of the model predictions
        model_mean = model_values[0].numpy()
        model_std = model_values[1].numpy()
        # get the lower bound of the model predictions
        model_min = model_mean - model_std * 1.96
        # get the model lowest predicion
        min_model = tf.reduce_min(model_min).numpy()
        # get model x at the lowest prediction
        min_model_idx = tf.argmin(model_min).numpy()[0]
        min_model_input = self.data.query_points[min_model_idx].numpy()

        current_regret = dict(
            min_obs=min_obs,
            min_input=min_input,
            min_model=min_model,
            min_model_input=min_model_input,
        )
        self.regret_data.append(current_regret)


def train(*args, **kwargs) -> OptimizationResult:
    trainer = Trainer(*args, **kwargs)
    trainer.train_loop()
    return trainer.result
