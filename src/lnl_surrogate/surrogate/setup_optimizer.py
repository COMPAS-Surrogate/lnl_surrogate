from typing import List, Tuple

import numpy as np
import tensorflow as tf
import trieste
from lnl_computer.cosmic_integration.mcz_grid import McZGrid
from lnl_computer.cosmic_integration.star_formation_paramters import (
    get_star_formation_prior,
)
from trieste.bayesian_optimizer import BayesianOptimizer
from trieste.objectives import mk_observer
from trieste.observer import Observer
from trieste.space import SearchSpace

__all__ = ["setup_optimizer"]


def setup_optimizer(
    mcz_obs: np.ndarray,
    compas_h5_filename: str,
    params: List[str],
    n_init: int = 5,
    reference_lnl: float = 0,
) -> Tuple[BayesianOptimizer, trieste.data.Dataset]:
    search_space = _get_search_space(params)
    observer = _generate_lnl_observer(
        mcz_obs, compas_h5_filename, params, reference_lnl
    )
    x0 = search_space.sample(n_init)
    init_data = observer(x0)
    bo = BayesianOptimizer(observer, search_space)
    return bo, init_data


def _generate_lnl_observer(
    mcz_obs: np.ndarray,
    compas_h5_filename: str,
    params: List[str],
    reference_lnl: float = 0,
) -> Observer:
    def _f(x):
        if isinstance(x, tf.Tensor):
            x = x.numpy()
        lnls = [
            McZGrid.lnl(
                mcz_obs=mcz_obs,
                duration=1,
                compas_h5_path=compas_h5_filename,
                sf_sample={params[i]: _xi[i] for i in range(len(params))},
                n_bootstraps=0,
            )[0]
            * -1
            - reference_lnl
            for _xi in x
        ]
        _t = tf.convert_to_tensor(lnls, dtype=tf.float64)
        return tf.reshape(_t, (-1, 1))

    return mk_observer(_f)


def _get_search_space(params: List[str]) -> SearchSpace:
    prior = get_star_formation_prior()
    param_mins = [prior[p].minimum for p in params]
    param_maxs = [prior[p].maximum for p in params]
    return trieste.space.Box(param_mins, param_maxs)
