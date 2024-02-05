from .plot_bo_metrics import plot_bo_metrics
from .plot_bo_corners import plot_evaluations, plot_model_partial_dependence
from .image_utils import make_gif

import os


def save_diagnostic_plots(data, model, search_space, outdir, label, truth={}):
    plot_out = f"{outdir}/plots"
    os.makedirs(plot_out, exist_ok=True)
    inpts, outpts = data.query_points.numpy(), data.observations.numpy()

    bo_fig = plot_bo_metrics(inpts, outpts, truth=truth.get('lnl', None))
    bo_fig.savefig(f"{plot_out}/bo_metrics_{label}.png", bbox_inches='tight')

    evl_fig = plot_evaluations(inpts, outpts, model, search_space, truth=truth)
    evl_fig.savefig(f"{plot_out}/eval_{label}.png", bbox_inches='tight')

    pd_fig = plot_model_partial_dependence(inpts, outpts, model, search_space, truth=truth)
    pd_fig.savefig(f"{plot_out}/func_{label}.png", bbox_inches='tight')


def save_gifs(outdir):
    make_gif(f"{outdir}/bo_metrics_*.png", f"{outdir}/bo_metrics.gif")
    make_gif(f"{outdir}/eval_*.png", f"{outdir}/eval.gif")
    make_gif(f"{outdir}/func_*.png", f"{outdir}/func.gif")
