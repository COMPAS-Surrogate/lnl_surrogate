from lnl_surrogate.plotting.plot_evaluation_corner import plot_evaluation_corner
from lnl_surrogate.plotting.plot_func_corner import plot_func_corner


def test_plot_eval_matrix(tmpdir, mock_inout_data):
    plot_evaluation_corner(mock_inout_data.inputs).savefig(f"{tmpdir}/eval.png")


def test_plot_func_matrix(tmpdir, mock_inout_data):
    fig = plot_func_corner(mock_inout_data.inputs, mock_inout_data.func, mock_inout_data.search_space)
    fig.show()
    fig.savefig(f"{tmpdir}/func.png")
