import click
from click.testing import CliRunner
from lnl_surrogate.cli import cli_train
import numpy as np
from lnl_surrogate.surrogate import load
import tensorflow as tf
from conftest import _mock_lnl_truth
import json


def test_cli(monkeypatch_lnl, mock_data, tmpdir):
    outdir = f'{tmpdir}/gp_cli'
    runner = CliRunner()

    # save truth as a json in tmpdir
    truth_fname = f"{tmpdir}/truth.json"
    with open(truth_fname, 'w') as f:
        json.dump(_mock_lnl_truth(), f)


    result = runner.invoke(cli_train, [
        "--compas_h5_filename", mock_data.compas_filename,
        "--param", "aSF",
        "--outdir", outdir,
        "--n_init", "2",
        "--n_rounds", "1",
        "--n_pts_per_round", "1",
        "--save_plots", "True",
        "--truth", truth_fname
    ])
    assert result.exit_code == 0
    model = load(outdir)
    lnl, _ = model.predict(np.array([[0.1]]))
    assert isinstance(tf.squeeze(lnl).numpy(), float)