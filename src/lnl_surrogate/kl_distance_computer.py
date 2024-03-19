import glob
import os
import re
import warnings

import bilby
import numpy as np
from sklearn.neighbors import NearestNeighbors


def verify_sample_shapes(s1, s2, k):
    # Expects [N, D]
    assert len(s1.shape) == len(s2.shape) == 2
    # Check dimensionality of sample is identical
    assert s1.shape[1] == s2.shape[1]


def kl_distance(s1, s2, k=5):
    """An efficient version of the scikit-learn estimator by @LoryPack
    s1: (N_1,D) Sample drawn from distribution P
    s2: (N_2,D) Sample drawn from distribution Q
    k: Number of neighbours considered (default 1)
    return: estimated D(P|Q)

    Contributed by Lorenzo Pacchiardi (@LoryPack)

    KL-Divergence estimation through K-Nearest Neighbours

    This module provides four implementations of the K-NN divergence estimator of
        Qing Wang, Sanjeev R. Kulkarni, and Sergio Verdú.
        "Divergence estimation for multidimensional densities via
        k-nearest-neighbor distances." Information Theory, IEEE Transactions on
        55.5 (2009): 2392-2405.

    https://github.com/nhartland/KL-divergence-estimators
    """
    verify_sample_shapes(s1, s2, k)

    n, m = len(s1), len(s2)
    d = float(s1.shape[1])

    s1_neighbourhood = NearestNeighbors(
        n_neighbors=k + 1, algorithm="kd_tree"
    ).fit(s1)
    s2_neighbourhood = NearestNeighbors(
        n_neighbors=k, algorithm="kd_tree"
    ).fit(s2)

    s1_distances, indices = s1_neighbourhood.kneighbors(s1, k + 1)
    s2_distances, indices = s2_neighbourhood.kneighbors(s1, k)
    rho = s1_distances[:, -1]
    nu = s2_distances[:, -1]
    if np.any(rho == 0):
        warnings.warn(
            f"The distance between an element of the first dataset and its {k}-th NN in the same dataset "
            f"is 0; this causes divergences in the code, and it is due to elements which are repeated "
            f"{k + 1} times in the first dataset. Increasing the value of k usually solves this.",
            RuntimeWarning,
        )
    D = np.sum(np.log(nu / rho))

    return (d / n) * D + np.log(
        m / (n - 1)
    )  # this second term should be enough for it to be valid for m \neq n


def bilby_kl_distance(res1, res2):
    if isinstance(res1, str):
        res1 = bilby.result.read_in_result(res1)
    if isinstance(res2, str):
        res2 = bilby.result.read_in_result(res2)

    for k in [5, 10, 15]:
        kl_div = kl_distance(
            res1.posterior.to_numpy(), res2.posterior.to_numpy(), k=k
        )
        if not np.isnan(kl_div):
            return kl_div


def get_list_of_kl_distances(res_regex):
    """
    Get a list of KL distances for a set of results files
    fnames are like round1_100pts.json, round2_200pts.json, etc
    """
    res_files = glob.glob(res_regex)

    # extract the npts from the filenames
    npts = []
    for f in res_files:
        if re.search(r"round\d+_(\d+)pts", f) is None:
            raise ValueError(
                f"Filename {f} does not match the expected format"
            )
        npts.append(int(re.search(r"round\d+_(\d+)pts", f).group(1)))
    # sort the filenames by the npts
    sorted_filenames = [f for _, f in sorted(zip(npts, res_files))]
    sorted_npts = np.array(sorted(npts))

    # largest npts file
    reference_res = bilby.read_in_result(sorted_filenames[-1])
    kl_distances = np.ones(len(sorted_filenames))
    for i, f in enumerate(sorted_filenames):
        kl_distances[i] = bilby_kl_distance(f, reference_res)

    mask = np.isfinite(kl_distances)

    return sorted_npts[mask], kl_distances[mask]
