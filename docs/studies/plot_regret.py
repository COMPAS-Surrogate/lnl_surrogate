import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from lnl_surrogate.plotting.regret_plots import plot_multiple_regrets, RegretData

kwgs = dict(
    regret_datasets=[
        RegretData('regret_explore.csv', 'Explore', 'blue'),
        RegretData('regret_exploit.csv', 'Exploit', 'orange'),
    ],
    true_min=norm(0.01, 0.003).logpdf(0.01) * -1.0,
)

plot_multiple_regrets(**kwgs, fname='regret.png', yzoom=0.002)
