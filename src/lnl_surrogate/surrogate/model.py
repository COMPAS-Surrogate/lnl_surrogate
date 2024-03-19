from typing import Union

import trieste.models.optimizer
from trieste.data import Dataset
from trieste.models.gpflow import GaussianProcessRegression, build_gpr
from trieste.models.gpflux import DeepGaussianProcess, build_vanilla_deep_gp
from trieste.space import SearchSpace

__all__ = ["get_model"]

MIN_LIKELIHOOD_VARIANCE = 1e-6


def get_model(
    model_type: str,
    data: Dataset,
    search_space: SearchSpace,
    likelihood_variance: float = MIN_LIKELIHOOD_VARIANCE,
    optimize: bool = False,
) -> Union[GaussianProcessRegression, DeepGaussianProcess]:
    if model_type == "deepgp":
        model = _get_deepgp_model(data, search_space, likelihood_variance)
    elif model_type == "gp":
        model = _get_gp_model(data, search_space, likelihood_variance)
    else:
        raise ValueError(f"Model[{model_type}] not found")

    if optimize:
        bo: trieste.models.optimizer.OptimizeResult = model.optimize(data)
        model = bo.try_get_final_model()

    return model


def _get_gp_model(
    data: Dataset,
    search_space: SearchSpace,
    likelihood_variance: float = MIN_LIKELIHOOD_VARIANCE,
) -> GaussianProcessRegression:
    gpflow_model = build_gpr(
        data, search_space, likelihood_variance=likelihood_variance
    )
    model = GaussianProcessRegression(
        gpflow_model,
    )
    return model


def _get_deepgp_model(
    data: Dataset,
    search_space: SearchSpace,
    likelihood_variance: float = MIN_LIKELIHOOD_VARIANCE,
) -> DeepGaussianProcess:
    gpflow_model = build_vanilla_deep_gp(
        data,
        search_space,
        num_layers=2,
        num_inducing_points=100,
        likelihood_variance=likelihood_variance,
        trainable_likelihood=True,
    )
    model = DeepGaussianProcess(gpflow_model)
    return model
