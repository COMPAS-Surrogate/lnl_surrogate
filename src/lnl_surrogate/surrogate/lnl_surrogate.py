import json
import os
from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from bilby.core.likelihood import Likelihood
from trieste.models.utils import get_module_with_variables

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
    ):
        super().__init__()
        self.model = model
        self.data = data
        self.truths = truths
        self.regret = regret
        self.true_lnl = truths.get("lnl", None)
        self.reference_lnl = reference_lnl
        self.param_keys = list(data.columns)[:-1]  # the last column is the lnl
        self.parameters = {k: np.nan for k in self.param_keys}

    def log_likelihood(self) -> float:
        """
        gp_y = - (lnl - reference_lnl)
        => lnl = reference_lnl - gp_y
        """
        params = np.array([[self.parameters[k] for k in self.param_keys]])
        y_mean, y_std = self.model.predict(params)
        neg_rel_lnl = y_mean.numpy().flatten()[0]
        return self.reference_lnl - neg_rel_lnl

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
        module = get_module_with_variables(model)
        n_params = data.query_points.shape[1]
        module.predict = tf.function(
            model.predict,
            input_signature=[
                tf.TensorSpec(shape=[None, n_params], dtype=tf.float64)
            ],
        )

        if label is not None:
            outdir = f"{outdir}/{label}"
            os.makedirs(outdir, exist_ok=True)

        tf.saved_model.save(module, f"{outdir}/{MODEL_FNAME}")
        model = tf.saved_model.load(f"{outdir}/{MODEL_FNAME}")

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

    def save(self, outdir: str, label: str = None):
        if label is not None:
            outdir = f"{outdir}/{label}"
            os.makedirs(outdir, exist_ok=True)
        tf.saved_model.save(self.model, f"{outdir}/{MODEL_FNAME}")
        self.data.to_csv(f"{outdir}/{DATA_FNAME}", index=False)
        self.regret.to_csv(f"{outdir}/{REGRET_FNAME}", index=False)
        with open(f"{outdir}/{META_DATA}", "w") as f:
            meta_data = {"reference_lnl": self.reference_lnl, **self.truths}
            json.dump(meta_data, f)

    @classmethod
    def load(cls, outdir: str, label: str = None):
        if label is not None:
            outdir = f"{outdir}/{label}"
        model = tf.saved_model.load(f"{outdir}/{MODEL_FNAME}")
        data = pd.read_csv(f"{outdir}/{DATA_FNAME}")
        regret = pd.read_csv(f"{outdir}/{REGRET_FNAME}")
        meta_fname = f"{outdir}/{META_DATA}"
        meta_data = {}
        if os.path.exists(meta_fname):
            with open(meta_fname, "r") as f:
                meta_data = json.load(f)
        reference_lnl = meta_data.pop("reference_lnl", 0)
        truths = meta_data
        return cls(model, data, regret, truths, reference_lnl)
