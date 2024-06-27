import json

import click
import numpy as np
import tensorflow as tf
from click.testing import CliRunner
from conftest import _mock_lnl_truth

from lnl_surrogate import LnLSurrogate
from lnl_surrogate.cli import cli_build_surrogate, cli_train


def test_cli(monkeypatch_lnl, mock_data, tmpdir):
    outdir = f"{tmpdir}/gp_cli"
    runner = CliRunner()

    # save reference_param as a json in tmpdir
    truth_fname = f"{tmpdir}/reference_param.json"
    with open(truth_fname, "w") as f:
        json.dump(_mock_lnl_truth(), f)

    args = [
        "--compas_h5_filename",
        mock_data.compas_filename,
        "--mcz_obs_filename",
        "LVK",
        "-a",
        "pv",
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
        "--reference_param",
        truth_fname,
        "--max_threshold",
        "500",
    ]

    cmd_str = "train_lnl_surrogate " + " ".join(args)
    result = runner.invoke(cli_train, args)
    assert (
        result.exit_code == 0
    ), f"Out: {result.output}\nError: {result.exc_info}. Command:\n\n{cmd_str}"
    lnl_surr = LnLSurrogate.load(outdir)
    lnl_surr.parameters.update({"aSF": 0.1})
    lnl = lnl_surr.log_likelihood()
    assert isinstance(tf.squeeze(lnl).numpy(), float)


def test_builder(tmpdir, training_csv):
    runner = CliRunner()
    result = runner.invoke(
        cli_build_surrogate,
        [
            "--csv",
            training_csv,
            "--model_type",
            "gp",
            "--outdir",
            tmpdir,
            "--lnl-threshold",
            "50",
            "--sample",
        ],
    )
    assert (
        result.exit_code == 0
    ), f"Out: {result.output}\nError: {result.exc_info}"
