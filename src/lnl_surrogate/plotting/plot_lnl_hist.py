import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

LNL_BINS = np.geomspace(10**-2, 10**2, 50)


def plot_lnl_hist(lnls, ax=None, threshold=None, fname=None, **kwargs):
    """Plot the histogram of the log likelihoods."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    else:
        fig = ax.get_figure()

    xvals = np.clip(np.abs(lnls), LNL_BINS[0], LNL_BINS[-1])
    ax.hist(xvals, bins=LNL_BINS, histtype="step", **kwargs)

    ax.set_xscale("log")

    ax.set_xlabel("Rel Abs LnL")
    if threshold is not None:
        ax.axvline(threshold, color="red", linestyle="--")
    if fname is not None:
        fig.savefig(fname, bbox_inches="tight")
    return fig


def plot_multiple_lnl_hist(lnl_regex):
    """Plot the histogram of the log likelihoods."""
    fnames = glob.glob(lnl_regex)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    print(f"Reading and ploting for {len(fnames)} files")
    lnls = []
    for fname in tqdm(fnames):
        df = pd.read_csv(fname)
        plot_lnl_hist(df["lnl"].values, ax=ax, lw=0.1, color="tab:blue")
        lnls.append(df["lnl"].values)
    all_lnls = np.concatenate(lnls)
    # twin y axis
    ax2 = ax.twinx()

    plot_lnl_hist(all_lnls, ax=ax2, lw=1, color="tab:red")
    fig.savefig("LNLS.png")
