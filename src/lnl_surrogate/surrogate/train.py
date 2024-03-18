import json
import os
from collections import OrderedDict
from typing import List, Callable, Dict, Union, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.auto import trange
from trieste.acquisition import AcquisitionFunctionBuilder, AcquisitionRule
from trieste.acquisition.function import PredictiveVariance, ExpectedImprovement
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.bayesian_optimizer import OptimizationResult
from trieste.data import Dataset
from trieste.logging import set_tensorboard_writer
from trieste.models import TrainableProbabilisticModel
from trieste.models.utils import get_module_with_variables

from lnl_computer.observation.mock_observation import MockObservation
from .model import get_model
from .setup_optimizer import setup_optimizer
from ..logger import logger
from ..plotting import save_diagnostic_plots, save_gifs

__all__ = ["train", "load"]

CACHED_RES_FNAME = "bo_result"

ACQ_FUN_TYPE = List[Union[PredictiveVariance, ExpectedImprovement]]


class Trainer:
    def __init__(
            self,
            model_type: str,
            compas_h5_filename: str,
            params=None,
            mcz_obs: Optional[Union[str, np.ndarray]] = None,
            acquisition_fns=None,
            n_init: Optional[int] = 5,
            n_rounds: Optional[int] = 5,
            n_pts_per_round: Optional[int] = 10,
            outdir: Optional[str] = 'outdir',
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
        if params is None:
            params = ['aSF', 'dSF', 'sigma0', 'muz']
        self.params = params
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
        self.truths = truth
        self.save_plots = save_plots
        self.noise_level = noise_level

        self.optimizer, self.data = setup_optimizer(self.mcz_obs, compas_h5_filename, params, n_init)
        self.model = get_model(model_type, self.data, self.search_space, likelihood_variance=self.noise_level)
        self.learning_rules = [EfficientGlobalOptimization(aq_fn) for aq_fn in self.acquisition_fns]

        self.regret_data = []
        self.result = None

    @property
    def mcz_obs(self) -> np.ndarray:
        return self._mcz_obs

    @mcz_obs.setter
    def mcz_obs(self, mcz_obs):
        if mcz_obs is None:
            mcz_obs = MockObservation.from_compas_h5(self.compas_h5_filename).mcz
        elif isinstance(mcz_obs, str):
            mcz_obs = MockObservation.from_npz(mcz_obs).mcz
        elif not isinstance(mcz_obs, np.ndarray):
            raise ValueError("mcz_obs must be a npz filename, ndarray or None")
        self._mcz_obs = mcz_obs

    @property
    def acquisition_fns(self) -> ACQ_FUN_TYPE:
        return self._acquisition_fns

    @acquisition_fns.setter
    def acquisition_fns(self, acquisition_fns):
        self._acquisition_fns = []

        if isinstance(acquisition_fns[0], PredictiveVariance) or isinstance(acquisition_fns[0], ExpectedImprovement):
            self._acquisition_fns = acquisition_fns
            return

        if acquisition_fns is None:
            acquisition_fns = ["PredictiveVariance",
                               "ExpectedImprovement"]

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

        self._truth = None
        if isinstance(truth, str):
            with open(truth, 'r') as f:
                truth = json.load(f)

        if isinstance(truth, dict):
            self._truth = OrderedDict({p: truth[p] for p in self.params})
            self._truth['lnl'] = truth['lnl']

    @property
    def search_space(self):
        return self.optimizer._search_space

    def train_loop(self):
        for round_idx in trange(self.n_rounds, desc='Optimization round'):
            self._ith_optimization_round(round_idx)
        logger.info(f"Optimization complete, saving result and data to {self.outdir}")
        self.save()

    def _ith_optimization_round(self, i: int):
        rule: AcquisitionRule = self.learning_rules[i % len(self.learning_rules)]
        self.result: OptimizationResult = self.optimizer.optimize(
            self.n_pts_per_round, self.data, self.model, rule,
            track_state=False, )
        self.data: Dataset = self.result.try_get_final_dataset()
        self.model: TrainableProbabilisticModel = self.result.try_get_final_model()
        self.__update_regret_data()

        if self.save_plots:
            save_diagnostic_plots(
                self.data,
                self.model,
                self.search_space,
                self.outdir,
                f"round{i}",
                self.truths,
                self.model_plotter
            )

    def save(self):
        model = self.result.try_get_final_model()
        module = get_module_with_variables(model)
        n_params = self.data.query_points.shape[1]
        module.predict = tf.function(
            model.predict,
            input_signature=[tf.TensorSpec(shape=[None, n_params], dtype=tf.float64)],
        )
        tf.saved_model.save(module, f"{self.outdir}/{CACHED_RES_FNAME}")
        inputs = self.data.query_points.numpy()
        outputs = self.data.observations.numpy()

        regret_data = pd.DataFrame(self.regret_data)
        regret_data.to_csv(f"{self.outdir}/regret.csv", index=False)

        if self.save_plots:
            save_gifs(self.outdir)

        np.savez(f"{self.outdir}/data.npz", inputs=inputs, outputs=outputs)

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
            min_model_input=min_model_input
        )
        self.regret_data.append(current_regret)

    def __setup_logger(self):
        os.makedirs(self.outdir, exist_ok=True)
        summary_writer = tf.summary.create_file_writer(self.outdir)
        set_tensorboard_writer(summary_writer)
        logger.debug(f"visualise optimization progress with `tensorboard --logdir={self.outdir}`")


def train(*args, **kwargs) -> OptimizationResult:
    trainer = Trainer(*args, **kwargs)
    trainer.train_loop()
    return trainer.result


def load(outdir: str) -> tf.keras.Model:
    """Load the cached model from the given directory"""
    return tf.saved_model.load(f"{outdir}/{CACHED_RES_FNAME}")
