from typing import List

from lnl_computer.cosmic_integration.star_formation_paramters import (
    get_star_formation_prior,
)
from trieste.space import Box, SearchSpace


def get_search_space(params: List[str]) -> Box:
    prior = get_star_formation_prior()
    param_mins = [prior[p].minimum for p in params]
    param_maxs = [prior[p].maximum for p in params]
    return Box(param_mins, param_maxs)
