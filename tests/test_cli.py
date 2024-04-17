import json

import click
import numpy as np
import tensorflow as tf
from click.testing import CliRunner
from conftest import _mock_lnl_truth

from lnl_surrogate import LnLSurrogate
from lnl_surrogate.cli import cli_train


def test_cli(monkeypatch_lnl, mock_data, tmpdir):
    outdir = f"{tmpdir}/gp_cli"
    runner = CliRunner()

    # save truth as a json in tmpdir
    truth_fname = f"{tmpdir}/truth.json"
    with open(truth_fname, "w") as f:
        json.dump(_mock_lnl_truth(), f)

    result = runner.invoke(
        cli_train,
        [
            "--compas_h5_filename",
            mock_data.compas_filename,
            "--mcz_obs_filename",
            mock_data.observations_filename,
            "--param",
            "aSF",
            "--outdir",
            outdir,
            "--n_init",
            "2",
            "--n_rounds",
            "1",
            "--n_pts_per_round",
            "1",
            "--save_plots",
            "True",
            "--truth",
            truth_fname,
        ],
    )
    assert result.exit_code == 0
    lnl_surr = LnLSurrogate.load(outdir)
    lnl_surr.parameters.update({"aSF": 0.1})
    lnl = lnl_surr.log_likelihood()
    assert isinstance(tf.squeeze(lnl).numpy(), float)
