import glob
import os
import re
import warnings
from typing import Optional, Union

import bilby
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from tqdm.auto import tqdm

from ..logger import logger
from .kl_distance import compute_statistics


def _get_npts_from_fname(fname: str) -> np.number:
    search_res = re.search(r"round\d+_(\d+)pts", fname)
    if search_res is None:
        logger.warning(f"Filename {fname} does not match the expected format")
        return np.nan
    return int(search_res.group(1))


def _get_result_fnames(res_regex: str) -> pd.DataFrame:
    """
    Get a list of result files and the number of points in each, sorted by number of points
    """
    res_files = glob.glob(res_regex)
    npts = [_get_npts_from_fname(f) for f in res_files]
    df = pd.DataFrame({"npts": npts, "fname": res_files})
    df = df.sort_values("npts", ascending=True)
    return df


def save_distances(
    res_regex: str,
    ref_res_fname: Optional[Union[str, bilby.result.Result]] = None,
    fname: str = None,
) -> pd.DataFrame:
    """
    Get a list of KL distances for a set of results files against a reference result

    Parameters
    ----------
    res_regex: str
        A regex pattern to match the results files
        E.g. "out_surr_*/out_mcmc/*variable_lnl_result.json"
        --> [round1_100pts_variable_lnl_result.json, round2_200pts_variable_lnl_result.json, ...]

    ref_res_fname: str or bilby.result.Result
        The reference result to compare against. If a string, it should be a filename
        If a bilby.result.Result, it should be the result object
        Ideally, this should be the result with the most points (highest resolution posterior)

    fname: str
        The filename to save the distance data to.

    Returns
    -------
    pd.DataFrame
        A dataframe with the distances (kl, ks, js) for each result file with respect the reference result
    """

    fname = "distances.csv" if fname is None else fname

    if ref_res_fname is None:
        ref_res_fname, _ = _get_highest_res_result_fname(
            os.path.dirname(res_regex)
        )

    if not os.path.exists(fname):
        logger.info(
            f"Getting KLs for all results with {res_regex} + ref: {ref_res_fname}"
        )
        result_df = _get_result_fnames(res_regex)
        if ref_res_fname is None:
            ref_res_fname = result_df.fname.values[-1]
        ref_res = bilby.read_in_result(ref_res_fname)
        kl, ks, js = [], [], []
        params = list(ref_res.injection_parameters.keys())
        for i, f in enumerate(
            tqdm(result_df.fname, desc="Calculating staistics")
        ):
            r = bilby.read_in_result(f)
            stats = compute_statistics(
                r.posterior[params], ref_res.posterior[params]
            )
            kl.append(stats.kl)
            ks.append(stats.ks)
            js.append(stats.js)
        result_df["ref_res"] = ref_res_fname
        result_df["kl"] = kl
        result_df["ks"] = ks
        result_df["js"] = js
        result_df.to_csv(fname, index=False)

    result_df = pd.read_csv(fname)
    plot_kl_distances(result_df, fname.replace(".csv", ".png"))
    return result_df


def plot_kl_distances(kl_data: pd.DataFrame, fname: str) -> None:
    fig, ax = plt.subplots()
    ax.plot(kl_data["kl"], color="tab:green", label="KL Divergence")
    ax2 = ax.twinx()
    ax2.plot(kl_data["ks"], label="KS Statistic", color="tab:red")
    ax3 = ax.twinx()
    ax3.plot(kl_data["js"], label="JS Divergence", color="tab:blue")
    ax.set_xlabel("Number of y-pts")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc=0)
    fig.savefig(fname)


def _get_highest_res_result_fname(res_dir: str) -> str:
    """
    Get the filename of the result with the most points (highest resolution posterior)
    """
    high_mcmc_fnames = _get_result_fnames(
        os.path.join(res_dir, "round*_highres_result.json")
    )
    res_fname = high_mcmc_fnames.fname.values[-1]
    round_num = int(re.search(r"round(\d+)_", ref_res_fname).group(1))
    return res_fname, round_num
