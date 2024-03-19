import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from lnl_surrogate.plotting.regret_plots import (
    RegretData,
    plot_multiple_regrets,
)

kwgs = dict(
    regret_datasets=[
        RegretData("regret_explore.csv", "Explore", "blue"),
        RegretData("regret_exploit.csv", "Exploit", "orange"),
    ],
    true_min=norm(0.01, 0.003).logpdf(0.01) * -1.0,
)

plot_multiple_regrets(**kwgs, fname="regret.png", yzoom=0.002)


kwgs = dict(
    regret_datasets=[
        RegretData("outdir/multi_explore/regret.csv", "Explore", "blue"),
        RegretData("outdir/multi_exploit/regret.csv", "Exploit", "orange"),
        RegretData("outdir/multi_combined/regret.csv", "Both", "green"),
    ],
    true_min=norm(0.01, 0.003).logpdf(0.01) * -1.0,
)

plot_multiple_regrets(**kwgs, fname="regret_multi.png", yzoom=0.002)
