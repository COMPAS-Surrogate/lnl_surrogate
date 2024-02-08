import tensorflow as tf
from trieste.acquisition.function import ExpectedImprovement

import numpy as np
from lnl_surrogate.surrogate import train, load
import pytest
from scipy.stats import norm
from lnl_surrogate.surrogate.setup_optimizer import McZGrid

np.random.seed(0)


@pytest.mark.parametrize('model_type', ['gp', 'deepgp'])
def test_1d(monkeypatch, mock_data, tmpdir, model_type):
    NORM = norm(loc=0, scale=1)
    def mockreturn(*args, **kwargs):
        return NORM.rvs(1), NORM.rvs(1)

    monkeypatch.setattr(McZGrid, "lnl", mockreturn)

    outdir = f'{tmpdir}/{model_type}'
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
        truth=mock_data.truth,
    )
    assert res is not None
    model = load(outdir)
    lnl, _ = model.predict(np.array([[0.1]]))
    assert isinstance(tf.squeeze(lnl).numpy(), float)
