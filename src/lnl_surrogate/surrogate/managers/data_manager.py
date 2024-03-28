import json
import os.path
from collections import OrderedDict
from typing import Dict, List, Optional, Union

from lnl_computer.observation.mock_observation import MockObservation


class DataManager:
    def __init__(
        self,
        compas_h5_filename: str,
        duration: float,
        outdir: str,
        params: Optional[List[str]] = None,
        mcz_obs_filename: Optional[str] = None,
        truths: Optional[Union[str, Dict]] = None,
    ):
        self.outdir = outdir
        self.duration = duration
        self.compas_h5_filename = compas_h5_filename
        self.mcz_obs_filename = mcz_obs_filename
        self.params = params if not None else ["aSF", "dSF", "sigma_0", "mu_z"]

        # loaded attributes
        self.mock_observation = self._load_mock_observation()
        self.mcz_obs = self.mock_observation.mcz
        self.truths: OrderedDict = self._load_truths(truths)
        self.mock_observation.plot().savefig(
            f"{self.outdir}/mock_observation.png"
        )

    def _load_mock_observation(self) -> MockObservation:
        if self.mcz_obs_filename is None:
            return MockObservation.from_compas_h5(
                self.compas_h5_filename,
                outdir=self.outdir,
                duration=self.duration,
            )
        return MockObservation.from_npz(self.mcz_obs_filename)

    def _compute_lnl_at_true(self, sf_sample: dict) -> float:
        return self.mock_observation.mcz_grid.lnl(
            mcz_obs=self.mock_observation.mcz,
            duration=self.duration,
            compas_h5_path=self.compas_h5_filename,
            sf_sample=self.mock_observation.mcz_grid.cosmological_parameters,
            n_bootstraps=0,
            outdir=self.outdir,
        )[0]

    def _load_truths(self, truths) -> OrderedDict:
        _truths = self.mock_observation.mcz_grid.cosmological_parameters

        if isinstance(truths, dict):
            _truths = truths
        elif isinstance(truths, str) and os.path.isfile(truths):
            with open(truths, "r") as f:
                _truths = json.load(f)
        else:
            _truths["lnl"] = self._compute_lnl_at_true(_truths)

        if "muz" in _truths:
            _truths["mu_z"] = _truths.pop("muz")
        if "sigma0" in _truths:
            _truths["sigma_0"] = _truths.pop("sigma0")

        ordered_t = OrderedDict({p: _truths[p] for p in self.params})
        ordered_t["lnl"] = _truths["lnl"]
        return ordered_t

    @property
    def reference_lnl(self) -> float:
        return self.truths.get("lnl", 0)
