import sys
from skopt.plots import plot_objective
from skopt import forest_minimize, dummy_minimize
from scipy.stats import multivariate_normal
from scipy.optimize import OptimizeResult as SciPyOptimizeResult
from skopt.space import Space
import numpy as np

np.random.seed(123)
import matplotlib.pyplot as plt

NORM = multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])



def funny_func(x):
    return -1 * NORM.pdf(x)

bounds = [(-2, 2.), ] * 2
n_calls = 150

class Dummy:
    def __init__(self, f):
        self.f = f

    def predict(self, x):
        return self.f(x)



X = NORM.rvs(100)
Y = funny_func(X)

MIN_Y_IDX = np.argmin(Y)

result = SciPyOptimizeResult(dict(
    fun = Y[MIN_Y_IDX],
    x = X[MIN_Y_IDX],
    success = True,
    func_vals = Y,
    x_iters = X,
    models = [Dummy(funny_func)],
    space = Space(bounds),

))

result1 = dummy_minimize(funny_func, bounds, n_calls=n_calls, random_state=4, verbose=True,)
result1.models = [Dummy(funny_func)]
#
#
#
# #
_ = plot_objective(result, n_points=5, dimensions=["x", "y"], n_minimum_search=100, size=2, n_samples=10)
plt.show()

_ = plot_objective(result1, n_points=5, dimensions=["x", "y"], n_minimum_search=100, size=2, n_samples=10)
plt.show()


# # plot the function
# x = np.linspace(-1, 1, 400)
# y = np.linspace(-1, 1, 400)
# X, Y = np.meshgrid(x, y)
# Z = funny_func(np.array([X, Y]).T)
# plt.imshow(Z, cmap=plt.cm.viridis, extent=[-1, 1, -1, 1])
# plt.colorbar()
# plt.axis('off')
# plt.show()
