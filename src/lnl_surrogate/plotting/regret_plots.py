from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

N_ZOOM = 3


def plot_regret(
    data: pd.DataFrame,
    color: str = "tab:blue",
    label: str = "Regret",
    true_min=None,
    fig=None,
    axes=[],
    yzoom=1,
) -> plt.Figure:
    """
    Plot the regret for the given results.
    :param data: the results
    :param outdir: the output directory
    :return: None
    """

    if true_min is not None:
        model_y = data.min_model - true_min
        obs_y = data.min_obs - true_min
        ylab = r"Relative LnL (∆ LnL)"
    else:
        model_y = data.min_model
        obs_y = data.min_obs
        ylab = "Minimum Value"

    xlim = (0, len(data) - 1)

    if fig is None:
        fig, ax = plt.subplots()
        ylim = (-yzoom, yzoom)
        xzoom = (len(data) - N_ZOOM - 1, len(data) - 1)
        axins = ax.inset_axes(
            [0.7, 0.8, 0.25, 0.15], xlim=xzoom, ylim=ylim, xticklabels=[]
        )
        ax.indicate_inset_zoom(axins, edgecolor="black")
        axins.set_xlim(*xzoom)
        axes = [ax, axins]

    if isinstance(fig, plt.Figure):
        axes[0].plot(
            data.index, obs_y, label=f"Training Sample ({label})", color=color
        )
        axes[0].plot(
            data.index,
            model_y,
            label=f"Surrogate Model ({label})",
            color=color,
            linestyle="dashed",
        )
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel(ylab)
        axes[0].axhline(0, color="black", lw=2, zorder=-10)
        axes[0].set_xlim(*xlim)

        # inset axe to show last N iterations
        idx = len(data) - N_ZOOM
        axes[1].plot(
            data.index, obs_y, label=f"Training Sample ({label})", color=color
        )
        axes[1].plot(
            data.index,
            model_y,
            label=f"Surrogate Model ({label})",
            color=color,
            linestyle="dashed",
        )
        axes[1].axhline(0, color="black", lw=2, zorder=-10)

    elif isinstance(fig, go.Figure):
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=obs_y,
                mode="lines",
                name=f"Training Sample ({label})",
                line=dict(color=color),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=model_y,
                mode="lines",
                name=f"Surrogate Model ({label})",
                line=dict(color=color, dash="dash"),
            )
        )
        fig.update_xaxes(title_text="Iteration")
        fig.update_yaxes(title_text=ylab)
        # change theme --> 'presentation'
        fig.update_layout(
            shapes=[
                dict(
                    type="line",
                    x0=min(data.index),
                    y0=0,
                    x1=max(data.index),
                    y1=0,
                    line=dict(color="black", width=2),
                )
            ],
            xaxis=dict(range=xlim),
            template="presentation",
        )
        axes = []

    return fig, axes


class RegretData:
    def __init__(self, csv: str, label: str, color: str):
        self.data = pd.read_csv(csv)
        self.label = label
        self.color = color

    def __dict__(self):
        return dict(data=self.data, label=self.label, color=self.color)


def plot_multiple_regrets(
    regret_datasets: List[RegretData],
    true_min=None,
    fname="regret.png",
    yzoom=1,
):
    # if fname ends with .html, use plotly else use matplotlib
    interactive = fname.endswith(".html")

    fig = go.Figure() if interactive else None
    axes = None
    for r in regret_datasets:
        fig, axes = plot_regret(
            **r.__dict__(), true_min=true_min, fig=fig, axes=axes, yzoom=yzoom
        )

    if interactive:

        # add plotly legend
        fig.update_layout(
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        fig.write_html(fname)
    else:
        axes[0].legend()
        plt.savefig(fname)
