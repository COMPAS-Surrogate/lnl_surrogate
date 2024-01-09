from typing import Union
from trieste.models.gpflow import build_gpr, GaussianProcessRegression
from trieste.models.gpflux import DeepGaussianProcess, build_vanilla_deep_gp
from trieste.data import Dataset
from trieste.space import SearchSpace

__all__ = ["get_model"]

def get_model(model_type: str, data: Dataset, search_space: SearchSpace) -> Union[
    GaussianProcessRegression, DeepGaussianProcess]:
    if model_type == "deepgp":
        model = _get_deepgp_model(data, search_space)
    elif model_type == 'gp':
        model = _get_gp_model(data, search_space)
    else:
        raise ValueError(f"Model[{model_type}] not found")
    return model


def _get_gp_model(data: Dataset, search_space: SearchSpace,
                  likelihood_variance: float = 10) -> GaussianProcessRegression:
    gpflow_model = build_gpr(data, search_space, likelihood_variance=likelihood_variance)
    model = GaussianProcessRegression(gpflow_model)
    return model


def _get_deepgp_model(data: Dataset, search_space: SearchSpace, likelihood_variance: float = 10) -> DeepGaussianProcess:
    gpflow_model = build_vanilla_deep_gp(
        data,
        search_space,
        num_layers=2,
        num_inducing_points=100,
        likelihood_variance=likelihood_variance,
        trainable_likelihood=False,
    )
    model = DeepGaussianProcess(gpflow_model)
    return model
