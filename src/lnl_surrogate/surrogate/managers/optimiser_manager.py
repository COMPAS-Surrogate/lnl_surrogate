import os
from functools import partial
from typing import Callable, List, Tuple, Union

import numpy as np
import tensorflow as tf
import trieste
from lnl_computer.cosmic_integration.mcz_grid import McZGrid
from trieste.acquisition import (
    AugmentedExpectedImprovement,
    DiscreteThompsonSampling,
    ExpectedImprovement,
    NegativeLowerConfidenceBound,
    PredictiveVariance,
)
from trieste.acquisition.function import function as acf
from trieste.acquisition.rule import AcquisitionRule as Rule
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.bayesian_optimizer import BayesianOptimizer
from trieste.bayesian_optimizer import OptimizationResult as Result
from trieste.data import Dataset
from trieste.models import TrainableProbabilisticModel
from trieste.objectives import mk_observer
from trieste.observer import Observer
from trieste.space import SearchSpace
from trieste.utils import DEFAULTS

from ..model import get_model
from ..utils import get_search_space
from .data_manager import DataManager

__all__ = ["OptimisationManager"]


class OptimisationManager:
    def __init__(
        self,
        datamanager: DataManager,
        acquisition_fns: List[Union[str, Callable]],
        n_init: int,
        model_type: str,
        noise_level: float = 1e-5,
        max_threshold: float = 500,
    ):
        self._data_mngr = datamanager
        self._p = self._data_mngr.params
        self.n_init = n_init
        self.max_threshold = max_threshold

        # setup optimisation variables
        self.search_space = get_search_space(self._p)
        self._acquisiton_fn_table = dict(
            ei=ExpectedImprovement(search_space=self.search_space),
            nlcb=NegativeLowerConfidenceBound(beta=1.96),
            pv=PredictiveVariance(jitter=DEFAULTS.JITTER),
            dts=DiscreteThompsonSampling(
                num_search_space_samples=2000, num_query_points=4
            ),
            aei=AugmentedExpectedImprovement(),
        )
        self.acquisition_fns = [
            self._acquisiton_fn_table[af] for af in acquisition_fns
        ]

        self.learning_rules = self.__load_rules()
        self._neg_rel_lnl_fn = self.__create_neg_rel_lnl_fn()
        self.__observer = self.__generate_observer()
        self.init_data = self.__observer(self.search_space.sample(n_init))
        self.bo = BayesianOptimizer(self.__observer, self.search_space)

        # init model
        self.model = get_model(
            model_type,
            self.init_data,
            self.search_space,
            likelihood_variance=noise_level,
            optimize=True,
        )

        # optimisation result
        self.result: Result = None

    def __create_neg_rel_lnl_fn(self) -> Callable:
        """Create the negative relative log likelihood function."""
        out = os.path.join(self._data_mngr.outdir, "out_mczgrids")
        os.makedirs(out, exist_ok=True)
        f = partial(
            McZGrid.lnl,
            mcz_obs=self._data_mngr.observation,
            duration=self._data_mngr.duration,
            compas_h5_path=self._data_mngr.compas_h5_filename,
            n_bootstraps=0,
            outdir=out,
            clean=False,
        )

        p = self._data_mngr.params
        ref_lnl = self._data_mngr.reference_param.get("lnl", 0)
        assert isinstance(ref_lnl, float)

        def _min_fn(_xi: np.ndarray):
            """
            We minimize gp_y = - (lnl - reference_lnl)
            (Later we can undo this by lnl = reference_lnl - gp_y)
            """
            pdict = {p[i]: _xi[i] for i in range(len(p))}
            lnl, _ = f(sf_sample=pdict)
            rel_neg_lnl = -1 * (lnl - ref_lnl)

            # Limit the upper value to MAX_REL_NEG_LNL
            if rel_neg_lnl > self.max_threshold:
                rel_neg_lnl = self.max_threshold

            return rel_neg_lnl

        return _min_fn

    def __generate_observer(self) -> Observer:
        def _f(x):
            if isinstance(x, tf.Tensor):
                x = x.numpy()
            neg_rel_lnls = np.array(
                [self._neg_rel_lnl_fn(_xi) for _xi in x]
            )  # JUST THE LNL -- not the uncertainty
            _t = tf.convert_to_tensor(neg_rel_lnls, dtype=tf.float64)
            return tf.reshape(_t, (-1, 1))

        return mk_observer(_f)

    def __load_rules(self) -> List[Rule]:
        return [EfficientGlobalOptimization(af) for af in self.acquisition_fns]

    def get_ith_rule(self, i: int) -> Rule:
        return self.learning_rules[i % len(self.learning_rules)]

    def optimize(
        self,
        model: TrainableProbabilisticModel,
        data: Dataset,
        n_pts: int,
        rule: Rule,
    ):
        self.result = self.bo.optimize(
            num_steps=n_pts,
            datasets=data,
            models=model,
            track_state=False,
            acquisition_rule=rule,
        )

    def get_optimised_model_and_data(
        self,
    ) -> Tuple[TrainableProbabilisticModel, Dataset]:
        if self.result is None:
            return self.model, self.init_data

        model = self.result.try_get_final_model()
        data = self.result.try_get_final_dataset()
        return model, data
