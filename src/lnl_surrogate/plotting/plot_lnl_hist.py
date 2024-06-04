import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def plot_lnl_hist(lnls, ax=None, threshold=None, fname=None, **kwargs):
    """Plot the histogram of the log likelihoods."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    else:
        fig = ax.get_figure()
    xvals = np.abs(lnls)
    bins = np.geomspace(0.001, 10**2, 50)
    ax.hist(xvals, bins=bins, histtype="step", **kwargs, density=True)
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


plot_multiple_lnl_hist("out_surr_*/round4_650pts/data.csv")
