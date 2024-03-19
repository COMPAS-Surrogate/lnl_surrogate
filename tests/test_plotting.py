import os

from lnl_surrogate.plotting import (
    plot_evaluations,
    plot_model_partial_dependence,
)


def test_plot_eval_matrix(tmpdir, mock_inout_data):
    kwgs = dict(
        in_pts=mock_inout_data.inputs,
        out_pts=mock_inout_data.outputs,
        model=mock_inout_data.model,
        search_space=mock_inout_data.search_space,
        truth=mock_inout_data.truth,
    )

    plot_evaluations(**kwgs).savefig(f"{tmpdir}/eval.png", bbox_inches="tight")
    plot_model_partial_dependence(**kwgs).savefig(f"{tmpdir}/func.png")
    assert os.path.exists(f"{tmpdir}/eval.png")
    assert os.path.exists(f"{tmpdir}/func.png")
