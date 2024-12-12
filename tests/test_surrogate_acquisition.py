import glob

import pytest
import tensorflow as tf
from bilby.core.result import Result
from common import plotter_1d
from conftest import _mock_lnl_truth

from lnl_surrogate.surrogate import LnLSurrogate, train


# @pytest.mark.parametrize('model_type', ['gp', 'deepgp'])
@pytest.mark.parametrize(
    "model_type",
    [
        "gp",
        # "deepgp"
    ],
)
def test_1d(monkeypatch_lnl, mock_data, tmpdir, model_type):
    outdir = f"{tmpdir}/{model_type}"
    res = train(
        model_type=model_type,
        mcz_obs_filename=mock_data.observations_filename,
        compas_h5_filename=mock_data.compas_filename,
        acquisition_fns=["nlcb"],
        params=["aSF"],
        duration=1,
        n_init=2,
        n_rounds=1,
        n_pts_per_round=1,
        outdir=outdir,
        reference_param=_mock_lnl_truth(),
        model_plotter=plotter_1d,
        noise_level=1e-3,
    )
    assert res is not None
    lnl_surr = LnLSurrogate.load(outdir)
    lnl_surr.parameters.update({"aSF": 0.1})
    lnl = lnl_surr.log_likelihood()
    assert isinstance(tf.squeeze(lnl).numpy(), float)

    # check that bilby result can be loaded
    res_paths = glob.glob(f"{outdir}/out_mcmc/*result.json")
    res = Result.from_json(res_paths[0])
    assert res.meta_data["npts"] == 3
