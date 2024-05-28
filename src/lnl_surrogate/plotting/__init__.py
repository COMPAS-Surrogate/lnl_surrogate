import glob
import os

import matplotlib.pyplot as plt
from acquisition_plotting.trieste import (
    plot_trieste_evaluations,
    plot_trieste_objective,
)

from ..logger import logger
from .image_utils import make_gif
from .overlaid_corner import plot_overlaid_corner
from .plot_bo_metrics import plot_bo_metrics
from .plot_evaluation_corner import plot_evaluation_corner


def save_diagnostic_plots(
    data,
    model,
    search_space,
    outdir,
    label,
    truth={},
    model_plotter=None,
    reference_lnl=0,
    **kwargs,
):
    logger.info("Saving diagnostic plots")
    plot_out = f"{outdir}/plots"
    os.makedirs(plot_out, exist_ok=True)
    inpts, outpts = data.query_points.numpy(), data.observations.numpy()

    true_lnl = None
    if "lnl" in truth:
        true_lnl = truth["lnl"] - reference_lnl

    bo_fig = plot_bo_metrics(inpts, outpts, model, truth=true_lnl)
    bo_fig.savefig(f"{plot_out}/bo_metrics_{label}.png", bbox_inches="tight")

    kwgs = dict(
        in_pts=inpts,
        out_pts=outpts,
        trieste_model=model,
        trieste_space=search_space,
        truth=truth,
    )
    evl_fig = plot_trieste_evaluations(**kwgs)
    evl_fig.savefig(f"{plot_out}/eval_{label}.png", bbox_inches="tight")

    pd_fig = plot_trieste_objective(**kwgs)
    pd_fig.savefig(f"{plot_out}/func_{label}.png", bbox_inches="tight")
    plot_evaluation_corner(inpts).savefig(f"{plot_out}/corner_{label}.png")

    if model_plotter:
        model_plotter(model, data, search_space, truth=truth).savefig(
            f"{plot_out}/round_{label}.png"
        )
    plt.close("all")


def save_gifs(outdir):
    make_gif(
        f"{outdir}/plots/bo_metrics_*.png", f"{outdir}/plots/bo_metrics.gif"
    )
    make_gif(f"{outdir}/plots/eval_*.png", f"{outdir}/plots/eval.gif")
    make_gif(f"{outdir}/plots/func_*.png", f"{outdir}/plots/func.gif")
    round_imgs = glob.glob(f"{outdir}/plots/round_*.png")
    if len(round_imgs) > 0:
        make_gif(f"{outdir}/plots/round_*.png", f"{outdir}/plots/rounds.gif")
    corners = glob.glob(f"{outdir}/plots/*_corner.png")
    if len(corners) > 0:
        make_gif(f"{outdir}/plots/*_corner.png", f"{outdir}/plots/corners.gif")
    plt.close("all")
