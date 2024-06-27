import json
import os.path
from collections import OrderedDict
from typing import Dict, List, Optional, Union

from lnl_computer.cosmic_integration.mcz_grid import McZGrid
from lnl_computer.cosmic_integration.star_formation_paramters import (
    DEFAULT_DICT,
)
from lnl_computer.observation import Observation, load_observation


class DataManager:
    def __init__(
        self,
        compas_h5_filename: str,
        duration: float,
        outdir: str,
        params: Optional[List[str]] = None,
        mcz_obs_filename: Optional[str] = None,
        reference_param: Optional[Union[str, Dict]] = DEFAULT_DICT,
    ):
        self.outdir = outdir
        self.duration = duration
        self.compas_h5_filename = compas_h5_filename
        self.mcz_obs_filename = mcz_obs_filename
        self.params = params if not None else ["aSF", "dSF", "sigma_0", "mu_z"]

        # loaded attributes
        self.observation: Observation = load_observation(mcz_obs_filename)
        self.reference_param: OrderedDict = self._load_reference(
            reference_param
        )
        self.observation.plot(f"{self.outdir}/observation.png")

    def _compute_lnl_at_reference(self, sf_sample: dict = None) -> float:
        if sf_sample is None:
            sf_sample = self.observation.cosmological_parameters
        return McZGrid.lnl(
            sf_sample=sf_sample,
            mcz_obs=self.observation,
            duration=self.duration,
            compas_h5_path=self.compas_h5_filename,
            n_bootstraps=0,
            outdir=self.outdir,
        )[0]

    def _load_reference(
        self, reference: Union[Dict[str, float], str]
    ) -> OrderedDict:
        _ref = self.observation.cosmological_parameters

        if isinstance(reference, dict):
            _ref = reference
        elif isinstance(reference, str) and os.path.isfile(reference):
            with open(reference, "r") as f:
                _ref = json.load(f)

        if not isinstance(_ref, dict):
            raise ValueError(
                f"Reference parameter must be a dictionary or a file path to a json file. Got ref: {type(_ref)}, {_ref}"
            )
        if not _ref.get("lnl", 0):
            _ref["lnl"] = self._compute_lnl_at_reference(_ref)

        ordered_t = OrderedDict({p: _ref[p] for p in self.params})
        ordered_t["lnl"] = _ref["lnl"]
        return ordered_t

    @property
    def reference_lnl(self) -> float:
        return self.reference_param.get("lnl", 0)
