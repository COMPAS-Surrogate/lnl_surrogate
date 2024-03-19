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
TRUTHS_FNAME = "truths.json"


class LnLSurrogate(Likelihood):
    def __init__(
        self,
        model: tf.keras.Model,
        data: pd.DataFrame,
        truths: Dict[str, float] = {},
    ):
        super().__init__()
        self.model = model
        self.data = data
        self.truths = truths
        self.lnl_at_true = self.truths.get("lnl", 0)
        self.param_keys = list(data.columns)[:-1]  # the last column is the lnl
        self.parameters = {k: np.nan for k in self.param_keys}

    def log_likelihood(self) -> float:
        params = np.array([[self.parameters[k] for k in self.param_keys]])
        y_mean, y_std = self.model.predict(params)
        y_mean = y_mean.numpy().flatten()[0]
        # this is the relative negative log likelihood, so we need to multiply by -1 and add the true likelihood
        return y_mean * -1 + self.lnl_at_true

    @classmethod
    def from_bo_result(cls, bo_result, params, truths={}, outdir="outdir"):
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
        tf.saved_model.save(module, f"{outdir}/{MODEL_FNAME}")
        model = tf.saved_model.load(f"{outdir}/{MODEL_FNAME}")

        inputs = data.query_points.numpy()
        outputs = data.observations.numpy()

        # make the inputs into columns of a dataframe with the parameter names as the column names
        dataset = pd.DataFrame(inputs, columns=params)
        # add the outputs to the dataframe
        dataset["lnl"] = outputs

        return cls(model, dataset, truths)

    def save(self, outdir: str):
        tf.saved_model.save(self.model, f"{outdir}/{MODEL_FNAME}")
        self.data.to_csv(f"{outdir}/{DATA_FNAME}", index=False)
        if self.truths:
            with open(f"{outdir}/{TRUTHS_FNAME}", "w") as f:
                json.dump(self.truths, f)

    @classmethod
    def load(cls, outdir: str):
        model = tf.saved_model.load(f"{outdir}/{MODEL_FNAME}")
        data = pd.read_csv(f"{outdir}/{DATA_FNAME}")
        truths_fname = f"{outdir}/{TRUTHS_FNAME}"
        truths = {}
        if os.path.exists(truths_fname):
            with open(truths_fname, "r") as f:
                truths = json.load(f)
        return cls(model, data, truths)
