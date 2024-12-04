
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from bilby.core.result import Result
from conftest import MAXX, MIDX, MINX, NORM, _mock_lnl_truth
from lnl_computer.cosmic_integration.mcz_grid import McZGrid
from lnl_computer.cosmic_integration.star_formation_paramters import (
    get_star_formation_prior,
)
from lnl_computer.observation.mock_observation import MockObservation
from scipy.stats import norm
from trieste.acquisition.function import PredictiveVariance

from lnl_surrogate.surrogate import LnLSurrogate, train





def plotter_1d(model, data, search_space, **kwargs):
    x = np.linspace(MINX, MAXX, 100).reshape(-1, 1)

    ref_lnl = kwargs.get("reference_param", "lnl")

    # model_gp = -(lnl - reference_lnl)
    # lnl_obj = LnLSurrogate(model, data, regret=pd.DataFrame(), reference_lnl=ref_lnl, reference_param=kwargs["reference_param"])

    true_y = -(NORM.logpdf(x) - ref_lnl)
    model_y, model_yunc = model.predict(x)

    x_obs = data.query_points
    y_obs = data.observations

    tf_to_np = lambda x: x.numpy().flatten() if hasattr(x, "numpy") else x
    model_yunc = tf_to_np(model_yunc)
    model_y = tf_to_np(model_y)
    #
    # lnls = []
    # for xi in x_obs:
    #     lnl_obj.parameters.update(dict(aSF=xi))
    #     lnls.append(lnl_obj.log_likelihood())

    # make new fig
    plt.figure()
    # plt.plot(x, true_y, label="True", color="black")
    plt.plot(x, model_y, label="Model", color="tab:orange")
    # plt.plot(x, lnls, label="lnls", color="tab:orange")
    plt.scatter(x_obs, y_obs, label="Observed", color="black")
    yup, ydown = model_y + model_yunc, model_y - model_yunc
    plt.fill_between(
        x.flatten(),
        yup.flatten(),
        ydown.flatten(),
        alpha=0.2,
        color="tab:orange",
    )
    plt.legend(loc="upper right")
    return plt.gcf()