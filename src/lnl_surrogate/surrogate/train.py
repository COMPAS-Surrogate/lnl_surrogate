from typing import List, Callable
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.bayesian_optimizer import OptimizationResult
from trieste.acquisition import AcquisitionFunctionBuilder, AcquisitionRule
import tensorflow as tf
from trieste.logging import set_tensorboard_writer
from trieste.models import TrainableProbabilisticModel
from trieste.data import Dataset
from collections import OrderedDict
import os
from tqdm.auto import trange
from trieste.models.utils import get_module_with_variables

from ..logger import logger
from ..plotting.plot_bo_metrics import plot_bo_metrics
from ..plotting import save_diagnostic_plots

import numpy as np

from .model import get_model
from .setup_optimizer import setup_optimizer

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
        truth=dict()
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

    truth = order_truths(truth, params)

    _setup_tf_logging(outdir)
    bo, data = setup_optimizer(mcz_obs, compas_h5_filename, params, n_init)
    model = get_model(model_type, data, bo._search_space)
    learning_rules = [EfficientGlobalOptimization(aq_fn) for aq_fn in acquisition_fns]

    result = None
    for round_idx in trange(n_rounds, desc='Optimization round'):
        rule: AcquisitionRule = learning_rules[round_idx % len(learning_rules)]
        result: OptimizationResult = bo.optimize(n_pts_per_round, data, model, rule, track_state=False)
        data: Dataset = result.try_get_final_dataset()
        model: TrainableProbabilisticModel = result.try_get_final_model()
        # regret.append(result.try_get_final_regret())
        save_diagnostic_plots(data, model, bo._search_space, outdir, f"round{round_idx}", truth)
        if model_plotter:
            model_plotter(model, data, bo._search_space).savefig(f"{outdir}/round_{round_idx}.png")

    if model_plotter:
        # save gif in the save outdir
        pass

    logger.info(f"Optimization complete, saving result and data to {outdir}")
    _save(result, data, outdir)
    return result


def load(outdir: str) -> tf.keras.Model:
    """Load the cached model from the given directory"""
    return tf.saved_model.load(f"{outdir}/{CACHED_RES_FNAME}")


def _setup_tf_logging(outdir: str):
    os.makedirs(outdir, exist_ok=True)
    summary_writer = tf.summary.create_file_writer(outdir)
    set_tensorboard_writer(summary_writer)
    logger.debug(f"visualise optimization progress with `tensorboard --logdir={outdir}`")


def _save(result: OptimizationResult, data: Dataset, outdir: str):
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



    plot_bo_metrics(inputs, outputs).savefig(f"{outdir}/bo_metrics.png")
    np.savez(f"{outdir}/data.npz", inputs=inputs, outputs=outputs)


def order_truths(truth, params):
    if isinstance(truth, dict):
        _truth = OrderedDict({p: truth[p] for p in params})
        _truth['lnl'] = truth['lnl']
    return _truth
