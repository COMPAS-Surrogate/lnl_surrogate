import tensorflow as tf
from trieste.acquisition.function import PredictiveVariance

import numpy as np
from lnl_surrogate.surrogate import train, load


def test_1d(mock_data):
    res = train(
        model_type='gp',
        mcz_obs=mock_data.observations.mcz,
        compas_h5_filename=mock_data.compas_filename,
        acquisition_fns=[PredictiveVariance()],
        params=['aSF'],
        n_init=2,
        n_rounds=1,
        n_pts_per_round=1,
        outdir='out_test',
    )
    assert res is not None
    model = load('out_test')
    lnl, _ = model.predict(np.array([[0.1]]))
    assert isinstance(tf.squeeze(lnl).numpy(), float)

