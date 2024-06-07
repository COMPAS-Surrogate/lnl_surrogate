import json
import os
from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from bilby.core.likelihood import Likelihood
from bilby.core.prior import Normal
from lnl_computer.cosmic_integration.star_formation_paramters import (
    get_latex_labels,
)
from trieste.data import Dataset
from trieste.models.utils import get_module_with_variables

from ..logger import logger
from ..plotting import save_diagnostic_plots
from .model import get_model
from .utils import get_search_space

MODEL_FNAME = "trained_model"
DATA_FNAME = "data.csv"
META_DATA = "meta_data.json"
REGRET_FNAME = "regret.csv"


class LnLSurrogate(Likelihood):
    def __init__(
        self,
        model: tf.keras.Model,
        data: pd.DataFrame,
        regret: pd.DataFrame,
        truths: Dict[str, float] = {},
        reference_lnl: float = 0,
        variable_lnl: bool = False,
    ):
        super().__init__()
        self.model = model
        self.data = data
        self.truths = truths
        self.regret = regret
        self.true_lnl = truths.get("lnl", None)
        self.reference_lnl = reference_lnl
        self.param_keys = list(data.columns)[:-1]  # the last column is the lnl
        self.param_latex = get_latex_labels(self.param_keys)
        self.parameters = {k: np.nan for k in self.param_keys}
        self.variable_lnl = variable_lnl

    def log_likelihood(self) -> float:
        """
        gp_y = - (lnl - reference_lnl)
        => lnl = reference_lnl - gp_y
        """
        params = np.array([[self.parameters[k] for k in self.param_keys]])
        y_mean, y_std = self.model.predict(params)
        neg_rel_lnl = y_mean.numpy().flatten()[0]
        unc_lnl = y_std.numpy().flatten()[0]

        if self.variable_lnl:
            neg_rel_lnl = Normal(mu=neg_rel_lnl, sigma=unc_lnl).sample(1)[0]

        lnl = self.reference_lnl - neg_rel_lnl

        return lnl

    @property
    def n_training_points(self) -> int:
        return len(self.data)

    @classmethod
    def from_bo_result(
        cls,
        bo_result,
        params,
        regret,
        truths={},
        outdir="outdir",
        reference_lnl=0,
        label=None,
    ):
        model = bo_result.try_get_final_model()
        data = bo_result.try_get_final_dataset()
        tf_saved_model = f"{outdir}/{MODEL_FNAME}"
        model = _tf_model_from_gpflow(model, data, tf_saved_model)

        if label is not None:
            outdir = f"{outdir}/{label}"
            os.makedirs(outdir, exist_ok=True)

        inputs = data.query_points.numpy()
        outputs = data.observations.numpy()

        # make the inputs into columns of a dataframe with the parameter names as the column names
        dataset = pd.DataFrame(inputs, columns=params)
        # add the outputs to the dataframe
        dataset["lnl"] = outputs

        regret.to_csv(f"{outdir}/{REGRET_FNAME}", index=False)

        return cls(
            model,
            dataset,
            regret=regret,
            truths=truths,
            reference_lnl=reference_lnl,
        )

    def save(self, outdir: str, label: str = None, plots=False):
        if label is not None:
            outdir = f"{outdir}/{label}"
            os.makedirs(outdir, exist_ok=True)

        tf.saved_model.save(self.model, f"{outdir}/{MODEL_FNAME}")
        self.data.to_csv(f"{outdir}/{DATA_FNAME}", index=False)
        self.regret.to_csv(f"{outdir}/{REGRET_FNAME}", index=False)
        with open(f"{outdir}/{META_DATA}", "w") as f:
            meta_data = {
                "reference_lnl": self.reference_lnl,
                **self.truths,
            }
            json.dump(meta_data, f)

        if plots:
            self.plot(outdir=outdir, label=label)

    @classmethod
    def load(cls, outdir: str, label: str = None, variable_lnl: bool = False):
        if label is not None:
            outdir = f"{outdir}/{label}"
        model = tf.saved_model.load(f"{outdir}/{MODEL_FNAME}")
        data = pd.read_csv(cls.get_datafname(outdir))
        regret = pd.read_csv(f"{outdir}/{REGRET_FNAME}")
        meta_fname = f"{outdir}/{META_DATA}"
        reference_lnl, truths = _load_metadata(meta_fname)
        return cls(model, data, regret, truths, reference_lnl, variable_lnl)

    @staticmethod
    def get_datafname(outdir: str, label: str = None):
        if label is not None:
            outdir = f"{outdir}/{label}"
        return f"{outdir}/{DATA_FNAME}"

    @classmethod
    def from_csv(
        cls,
        csv: str,
        model_type: str,
        label: str,
        outdir: str = "",
        plot: bool = False,
        lnl_threshold: float = None,
        variable_lnl: bool = False,
    ):
        data = pd.read_csv(csv)

        # keep data with LnLs that are within the threshold
        if lnl_threshold:
            n_init = len(data)
            # data = data[np.abs(data["lnl"]) < lnl_threshold]
            # set all LnL > lnl_threshold to lnl_threshold
            n_edits = len(data[data["lnl"] > lnl_threshold])

            data["lnl"][data["lnl"] > lnl_threshold] = lnl_threshold
            logger.info(
                f"Edit LnL for {n_init - n_edits} ({n_init} --> {n_edits} Training points after thresholding)"
            )

        params = _get_params_from_df(data)
        dataset = _df_to_dataset(data)
        meta_fn = csv.replace(csv.split("/")[-1], "meta_data.json")
        reference_lnl, truths = _load_metadata(meta_fn)

        model = get_model(
            model_type,
            dataset,
            get_search_space(params),
            optimize=True,
        )
        outdir = f"outdir_{label}" if outdir == "" else outdir
        model = _tf_model_from_gpflow(
            model, dataset, f"{outdir}/{MODEL_FNAME}"
        )
        surrogate = cls(
            model,
            data,
            pd.DataFrame(),
            truths,
            reference_lnl,
            variable_lnl=variable_lnl,
        )
        surrogate.save(outdir, label=label, plots=plot)
        return surrogate

    def plot(self, **kwargs):
        save_diagnostic_plots(
            data=_df_to_dataset(self.data),
            model=self.model,
            search_space=get_search_space(self.param_keys),
            outdir=kwargs.get("outdir", "outdir"),
            label=kwargs.get("label", "lnl_surrogate"),
            truths=self.truths,
            reference_lnl=self.reference_lnl,
            axis_labels=self.param_latex,
        )


def _load_metadata(fn: str):
    meta_data = {}
    if os.path.exists(fn):
        with open(fn, "r") as f:
            meta_data = json.load(f)
    reference_lnl = meta_data.pop("reference_lnl", 0)
    truths = meta_data
    return reference_lnl, truths


def _df_to_dataset(df: pd.DataFrame) -> Dataset:
    observations = df["lnl"].values
    df = df.drop(columns="lnl")
    if "lnl_unc" in df.columns:
        lnl_unc = df["lnl_unc"]
        df = df.drop(columns="lnl_unc")
        # add a new column to the observations
        observations = np.array([observations, lnl_unc]).T

    query_points = tf.convert_to_tensor(df.values, dtype=tf.float64)
    observations = tf.convert_to_tensor(observations, dtype=tf.float64)
    # ensure correct rank (at least 2)
    query_points = tf.reshape(query_points, (-1, len(df.columns)))
    observations = tf.reshape(observations, (-1, 1))

    return Dataset(
        query_points=query_points,
        observations=observations,
    )


def _get_params_from_df(df: pd.DataFrame) -> list:
    params = list(df.columns)
    for p in ["lnl", "lnl_unc"]:
        if p in params:
            params.remove(p)
    return params


def _tf_model_from_gpflow(model, dataset, save_fn: str):
    module = get_module_with_variables(model)
    n_params = dataset.query_points.shape[1]
    module.predict = tf.function(
        model.predict,
        input_signature=[
            tf.TensorSpec(shape=[None, n_params], dtype=tf.float64)
        ],
    )
    tf.saved_model.save(module, save_fn)
    return tf.saved_model.load(save_fn)
