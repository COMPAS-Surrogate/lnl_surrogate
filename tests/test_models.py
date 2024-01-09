import tensorflow as tf
from trieste.acquisition.function import PredictiveVariance, ExpectedImprovement

import numpy as np
from lnl_surrogate.surrogate import train, load
import pytest


@pytest.mark.parametrize('model_type', ['gp', 'deepgp'])
def test_1d(mock_data, tmpdir, model_type):
    outdir = f'{tmpdir}/model_type'
    res = train(
        model_type=model_type,
        mcz_obs=mock_data.observations.mcz,
        compas_h5_filename=mock_data.compas_filename,
        acquisition_fns=[ExpectedImprovement()],
        params=['aSF'],
        n_init=2,
        n_rounds=1,
        n_pts_per_round=1,
        outdir=outdir,
    )
    assert res is not None
    model = load(outdir)
    lnl, _ = model.predict(np.array([[0.1]]))
    assert isinstance(tf.squeeze(lnl).numpy(), float)
