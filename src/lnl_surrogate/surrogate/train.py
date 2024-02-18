from ..logger import logger
from ..plotting import save_diagnostic_plots, save_gifs
from .model import get_model
from .setup_optimizer import setup_optimizer

from typing import List, Callable, Dict

from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.bayesian_optimizer import OptimizationResult
from trieste.acquisition import AcquisitionFunctionBuilder, AcquisitionRule
from trieste.logging import set_tensorboard_writer
from trieste.models import TrainableProbabilisticModel
from trieste.data import Dataset
from collections import OrderedDict
import os
from tqdm.auto import trange
from trieste.models.utils import get_module_with_variables
import pandas as pd

import numpy as np

import tensorflow as tf

__all__ = ["train", "load"]

CACHED_RES_FNAME = "bo_result"


def train(
        model_type: str,
        mcz_obs: np.ndarray,
        compas_h5_filename: str,
        params: List[str],
        acquisition_fns: List[AcquisitionFunctionBuilder],
        n_init: int = 5,
        n_rounds: int = 5,
        n_pts_per_round: int = 10,
        outdir: str = 'outdir',
        model_plotter: Callable = None,
        truth=dict(),
        noise_level: float = 1e-5,
        save_plots: bool = True,
) -> OptimizationResult:
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

    truth = _order_truths(truth, params)

    _setup_tf_logging(outdir)
    bo, data = setup_optimizer(mcz_obs, compas_h5_filename, params, n_init)
    model = get_model(model_type, data, bo._search_space, likelihood_variance=noise_level)
    learning_rules = [EfficientGlobalOptimization(aq_fn) for aq_fn in acquisition_fns]

    regret_data = []
    result = None
    for round_idx in trange(n_rounds, desc='Optimization round'):
        rule: AcquisitionRule = learning_rules[round_idx % len(learning_rules)]
        result: OptimizationResult = bo.optimize(n_pts_per_round, data, model, rule, track_state=False, )
        data: Dataset = result.try_get_final_dataset()
        model: TrainableProbabilisticModel = result.try_get_final_model()
        regret_data.append(_collect_regret_data(model, data))

        if save_plots:
            save_diagnostic_plots(data, model, bo._search_space, outdir, f"round{round_idx}", truth, model_plotter)

    logger.info(f"Optimization complete, saving result and data to {outdir}")
    _save(result, data, outdir, save_plots, regret_data)
    return result


def load(outdir: str) -> tf.keras.Model:
    """Load the cached model from the given directory"""
    return tf.saved_model.load(f"{outdir}/{CACHED_RES_FNAME}")


def _setup_tf_logging(outdir: str):
    os.makedirs(outdir, exist_ok=True)
    summary_writer = tf.summary.create_file_writer(outdir)
    set_tensorboard_writer(summary_writer)
    logger.debug(f"visualise optimization progress with `tensorboard --logdir={outdir}`")


def _save(result: OptimizationResult, data: Dataset, outdir: str, save_plots: bool, regret_data: list):
    model = result.try_get_final_model()
    module = get_module_with_variables(model)
    n_params = data.query_points.shape[1]
    module.predict = tf.function(
        model.predict,
        input_signature=[tf.TensorSpec(shape=[None, n_params], dtype=tf.float64)],
    )
    tf.saved_model.save(module, f"{outdir}/{CACHED_RES_FNAME}")
    inputs = data.query_points.numpy()
    outputs = data.observations.numpy()


    regret_data = pd.DataFrame(regret_data)
    regret_data.to_csv(f"{outdir}/regret.csv", index=False)

    if save_plots:
        save_gifs(outdir)

    np.savez(f"{outdir}/data.npz", inputs=inputs, outputs=outputs)


def _order_truths(truth, params)->OrderedDict:
    if isinstance(truth, dict):
        _truth = OrderedDict({p: truth[p] for p in params})
        _truth['lnl'] = truth['lnl']
    return _truth


def _collect_regret_data(model: TrainableProbabilisticModel, data: Dataset)->Dict:
    # get the minimum value of the observations (and the corresponding input)
    min_obs = tf.reduce_min(data.observations).numpy()
    min_idx = tf.argmin(data.observations).numpy()[0]
    min_input = data.query_points[min_idx].numpy()
    # get the model predictions at the training points
    model_values = model.predict(data.query_points)
    # upper and lower bounds of the model predictions
    model_mean = model_values[0].numpy()
    model_std = model_values[1].numpy()
    # get the lower bound of the model predictions
    model_min = model_mean - model_std * 1.96
    # get the model lowest predicion
    min_model = tf.reduce_min(model_min).numpy()
    # get model x at the lowest prediction
    min_model_idx = tf.argmin(model_min).numpy()[0]
    min_model_input = data.query_points[min_model_idx].numpy()

    return dict(
        min_obs=min_obs,
        min_input=min_input,
        min_model=min_model,
        min_model_input=min_model_input
    )
