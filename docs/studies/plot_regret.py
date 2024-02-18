import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm



def plot_regret(res: pd.DataFrame, color: str = 'tab:blue', label: str = 'Regret') -> plt.Figure:
    """
    Plot the regret for the given results.
    :param res: the results
    :param outdir: the output directory
    :return: None
    """
    plt.plot(res.index, res.min_obs, label=f"Training Sample ({label})", color=color)
    plt.plot(res.index, res.min_model, label=f'Model ({label})', color=color, linestyle='dashed')
    plt.xlabel('Iteration')
    plt.ylabel('Minimum Value')
    return plt.gcf()

regret_explore = pd.read_csv('regret_explore.csv')
regret_exploit = pd.read_csv('regret_exploit.csv')
fig = plot_regret(regret_explore, color='tab:blue', label='Explore')
fig = plot_regret(regret_exploit, color='tab:orange', label='Exploit')

xlim = [0, len(regret_explore)]
NORM = norm(0.01, 0.003)
true_y = NORM.logpdf(0.01) * -1.0
plt.plot(xlim, [true_y, true_y], label='True Minimum', color='black', zorder=-10)
plt.xlim(xlim)
plt.legend()
plt.show()

