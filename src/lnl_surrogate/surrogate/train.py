import json
import os
import traceback
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from lnl_computer.observation.mock_observation import MockObservation
from tqdm.auto import trange
from trieste.acquisition import AcquisitionRule
from trieste.acquisition.function import (
    ExpectedImprovement,
    PredictiveVariance,
)
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.bayesian_optimizer import OptimizationResult
from trieste.data import Dataset
from trieste.logging import set_tensorboard_writer
from trieste.models import TrainableProbabilisticModel

from ..logger import logger
from ..plotting import save_diagnostic_plots, save_gifs
from .lnl_surrogate import LnLSurrogate
from .model import get_model
from .sample import sample_lnl_surrogate
from .setup_optimizer import setup_optimizer

__all__ = ["train"]

ACQ_FUN_TYPE = List[Union[PredictiveVariance, ExpectedImprovement]]


class Trainer:
    def __init__(
        self,
        model_type: str,
        compas_h5_filename: str,
        params=None,
        mcz_obs: Optional[Union[str, np.ndarray]] = None,
        acquisition_fns=None,
        duration: Optional[float] = 1.0,
        n_init: Optional[int] = 5,
        n_rounds: Optional[int] = 5,
        n_pts_per_round: Optional[int] = 10,
        outdir: Optional[str] = "outdir",
        model_plotter: Optional[Callable] = None,
        truth: Optional[Union[Dict[str, float], str]] = None,
        noise_level: Optional[float] = 1e-5,
        save_plots: Optional[bool] = True,
    ):
        """
        Train a surrogate model using the given data and parameters.
        :param model_type: one of 'gp' or 'deepgp'
        :param mcz_obs: the observed MCZ values
        :param compas_h5_filename: the filename of the compas data
        :param params: the parameters to use [aSF, dSF, sigma0, muz]
        :param acquisition_fns: the acquisition functions to use
        :param n_init: the number of initial points to use
        :param n_rounds: the number of rounds of optimization to perform
        :param n_pts_per_round: the number of points to evaluate per round
        :param outdir: the output directory
        :param model_plotter: a function to plot the model
        :return: the optimization result
        """
        logger.info("Initialising surrogate trainer...")
        if params is None:
            params = ["aSF", "dSF", "sigma_0", "mu_z"]
        self.params = params
        self.truths = truth
        self.outdir = outdir
        self.__setup_logger()
        self.model_type = model_type
        self.compas_h5_filename = compas_h5_filename
        self.mcz_obs = mcz_obs
        self.acquisition_fns = acquisition_fns
        self.n_init = n_init
        self.n_rounds = n_rounds
        self.n_pts_per_round = n_pts_per_round

        self.model_plotter = model_plotter

        self.save_plots = save_plots
        self.noise_level = noise_level

        logger.info(f"Surrogate being built:{self}")
        self.reference_lnl = self.truths.get("lnl", 0)
        self.optimizer, self.data = setup_optimizer(
            self.mcz_obs,
            compas_h5_filename,
            duration,
            params,
            n_init,
            self.reference_lnl,
        )
        self.model = get_model(
            model_type,
            self.data,
            self.search_space,
            likelihood_variance=self.noise_level,
        )
        self.learning_rules = [
            EfficientGlobalOptimization(aq_fn)
            for aq_fn in self.acquisition_fns
        ]

        self.regret_data = []
        self.result = None

    def __str__(self):
        return f"Trainer(model_type={self.model_type}, params={self.params}, truths={self.truths}, pts=[init{self.n_init}, rnds={self.n_rounds}x{self.n_pts_per_round}pts)"

    @property
    def mcz_obs(self) -> np.ndarray:
        return self._mcz_obs

    @mcz_obs.setter
    def mcz_obs(self, mcz_obs):
        if isinstance(mcz_obs, np.ndarray):
            self._mcz_obs = mcz_obs
            return

        # if self._mcz_obs is an attribute, then it is already set
        if hasattr(self, "_mcz_obs"):
            if self._mcz_obs is not None:
                return

        mock_obs = None
        if mcz_obs is None:
            mock_obs = MockObservation.from_compas_h5(
                self.compas_h5_filename,
            )
        elif isinstance(mcz_obs, str):
            mock_obs = MockObservation.from_npz(mcz_obs)
        elif not isinstance(mcz_obs, np.ndarray):
            raise ValueError("mcz_obs must be a npz filename, ndarray or None")

        self._mcz_obs = mock_obs.mcz
        true_params = mock_obs.mcz_grid.cosmological_parameters
        true_lnl = (
            mock_obs.mcz_grid.lnl(
                mcz_obs=mock_obs.mcz,
                duration=1,
                compas_h5_path=self.compas_h5_filename,
                sf_sample=true_params,
                n_bootstraps=0,
                save_plots=True,
                outdir=self.outdir,
            )[0]
            * -1
        )
        self.truths = {"lnl": true_lnl, **true_params}

    @property
    def acquisition_fns(self) -> ACQ_FUN_TYPE:
        return self._acquisition_fns

    @acquisition_fns.setter
    def acquisition_fns(self, acquisition_fns):
        self._acquisition_fns = []

        if isinstance(acquisition_fns[0], PredictiveVariance) or isinstance(
            acquisition_fns[0], ExpectedImprovement
        ):
            self._acquisition_fns = acquisition_fns
            return

        if acquisition_fns is None:
            acquisition_fns = ["PredictiveVariance", "ExpectedImprovement"]

        for acq in acquisition_fns:
            if acq == "PredictiveVariance" or acq == "pv":
                self._acquisition_fns.append(PredictiveVariance())
            elif acq == "ExpectedImprovement" or acq == "ei":
                self._acquisition_fns.append(ExpectedImprovement())
            else:
                raise ValueError(f"Unknown acquisition function: {acq}")

    @property
    def truths(self) -> OrderedDict:
        return self._truth

    @truths.setter
    def truths(self, truth):

        self._truth = OrderedDict()
        if isinstance(truth, str):
            with open(truth, "r") as f:
                truth = json.load(f)

        if isinstance(truth, dict):

            if "muz" in truth:
                truth["mu_z"] = truth.pop("muz")
            if "sigma0" in truth:
                truth["sigma_0"] = truth.pop("sigma0")

            self._truth = OrderedDict({p: truth[p] for p in self.params})
            self._truth["lnl"] = truth["lnl"]

    @property
    def search_space(self):
        return self.optimizer._search_space

    def train_loop(self):
        for round_idx in trange(self.n_rounds, desc="Optimization round"):
            try:
                self._ith_optimization_round(round_idx)
            except Exception as e:
                # print full error message and traceback
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
        rule: AcquisitionRule = self.learning_rules[
            i % len(self.learning_rules)
        ]
        self.result: OptimizationResult = self.optimizer.optimize(
            self.n_pts_per_round,
            self.data,
            self.model,
            rule,
            track_state=False,
        )
        self.data: Dataset = self.result.try_get_final_dataset()
        self.model: TrainableProbabilisticModel = (
            self.result.try_get_final_model()
        )
        self.__update_regret_data()

        label = f"round{i}_{len(self.data)}pts"
        if self.save_plots:
            save_diagnostic_plots(
                self.data,
                self.model,
                self.search_space,
                self.outdir,
                label,
                self.truths,
                self.model_plotter,
                self.reference_lnl,
            )
        self.save_surrogate(label=label)
        sample_lnl_surrogate(
            f"{self.outdir}/{label}",
            outdir=os.path.join(self.outdir, f"out_mcmc"),
            label=label,
        )

    def save_surrogate(self, label: Optional[str] = None):
        lnl_surrogate = LnLSurrogate.from_bo_result(
            bo_result=self.result,
            params=self.params,
            truths=self.truths,
            outdir=self.outdir,
            reference_lnl=self.reference_lnl,
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

    def __setup_logger(self):
        os.makedirs(self.outdir, exist_ok=True)
        summary_writer = tf.summary.create_file_writer(self.outdir)
        set_tensorboard_writer(summary_writer)
        logger.debug(
            f"visualise optimization progress with `tensorboard --logdir={self.outdir}`"
        )


def train(*args, **kwargs) -> OptimizationResult:
    trainer = Trainer(*args, **kwargs)
    trainer.train_loop()
    return trainer.result
