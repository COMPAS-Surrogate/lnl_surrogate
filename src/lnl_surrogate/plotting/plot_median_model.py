import argparse
from typing import Dict, Optional, Union

from bilby.core.result import Result, read_in_result
from lnl_computer.cosmic_integration.mcz_grid import McZGrid
from lnl_computer.cosmic_integration.star_formation_paramters import (
    DEFAULT_DICT,
)

from ..logger import logger


def plot_median_model(
    compas_h5: str,
    out_fname: str,
    cosmic_kwargs: Optional[Dict[str, float]] = None,
    bilby_result: Optional[Union[str, Result]] = None,
):
    if bilby_result is not None:
        if isinstance(bilby_result, str):
            bilby_result = read_in_result(bilby_result)
        median_values = bilby_result.posterior.median()
        cosmic_kwargs = median_values.to_dict()
        cosmic_kwargs = {
            k: v for k, v in cosmic_kwargs.items() if k in DEFAULT_DICT.keys()
        }
        cosmic_kwargs = {
            k: v for k, v in cosmic_kwargs.items() if v is not None
        }
        logger.info(
            f"Making plot for {compas_h5} with bilbyResultMedian[{cosmic_kwargs}] -> {out_fname}"
        )
    else:
        logger.info(
            f"Making plot for {compas_h5} with {cosmic_kwargs} -> {out_fname}"
        )

    mcz = McZGrid.from_compas_output(
        compas_path=compas_h5,
        cosmological_parameters=cosmic_kwargs,
    )
    fig0 = mcz.plot()
    fig0.savefig(out_fname.replace(".png", "_unbinned.png"))
    fig0.clear()
    mcz.bin_data()
    fig = mcz.plot()
    fig.savefig(out_fname)
