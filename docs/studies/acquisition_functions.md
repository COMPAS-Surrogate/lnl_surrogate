# Bayesian Optimization for GPs



GPs can provide a reconstruction of an unknown function from sparse measurements

Additionally, the GP posterior predictive mean, $\mu$, and posterior predictive variance, $\sigma$, can be used to derive an _acquisition function_, $\alpha(\mu,\sigma)$. _Acquisition functions_ are useful in selecting the next point to observe from the model in optimization problems (active learning).


We can select the next point using:

$$
x_{next}= \underset{x}{\arg\max}\frac{1}{N_{\rm{mcmc}}}∑_{i=1}^{N_{\rm{mcmc}}} \alpha(\mu^i,\sigma^i)
$$

where
- $N_{\rm{mcmc}}$ is the total number of MCMC samples,
- $\mu^i$ is the _i_-th sample posterior predictive mean, and
- $\sigma^i$ is the _i_-th sample posterior predictive variance.


There are several acquisition functions that can be used.

## Acquisition functions

An acquisition function is a function that maps the posterior predictive mean and variance to a scalar value. The scalar value is used to select the next point to observe from the model.

In general, we will write the acquisition function as $\alpha(\mu,\sigma)$, where $\mu$ is the posterior predictive mean and $\sigma$ is the posterior predictive variance.

### Penalizing Recent Observations

We can add in a _penalty factor_ based on the recently observed points to avoid sampling the same region of the function space repeatedly.

This can be tacked on to _some_ acquisition function as follows:

$$
\alpha(\mu,\sigma) - \lambda\ \gamma(x, \vec{X_{\rm r}})
$$

where $\lambda$ is some penalty factor and $\gamma(x, \vec{X_{\rm r}})$ is the penalty for point $x$ based on their distance to the recently visited points $\vec{X_{\rm r}}$.


### The Upper Confidence Bound (UCB)

The upper confidence bound (UCB) acquisition function is defined as

$$
\alpha_{\rm{UCB}}^i(\mu^i, \sigma^i)= \mu^i\pm\sqrt{\beta\ \sigma^i}\qquad
$$

The coefficient $\beta$ determines an exploitation-exploration trade-off.

```{note}
The primary goal of UCB is to find a global minima/maxima rather than to reconstruct the entire function accurately.
```


This is an animation of using UCB to help choose the next points for a GP
![](https://user-images.githubusercontent.com/15642823/262271178-9d36eb7b-10b6-445c-9147-f5d0758713d0.gif)


This is the same animation but with adding in a penalty factor:
![](https://user-images.githubusercontent.com/15642823/262335925-4df4f33b-9168-4bf7-8fd2-f59b81b26060.gif)


A drawback of this method is that it requires an update of the GP model after each new observation. This can be computationally expensive.

Here are some more examples:
|     |           |
|-----|-----------|
| D1  | ![d1_ucb] |
| D2  | ![d2_ucb] |
| D3  | ![d3_ucb] |


#### Batched UCB (qUCB)

In MCMC-based GP, we obtain a separate multivariate normal posterior for each set of sampled kernel hyperparameters.
Hence, we can create a separate UCB acquisition function for each posterior, and then acquire a batch of new points.

The following are two animations made with different random seeds (no penalty factor).
|     |
|-----|
|![d1_goodqUCB] |
|![d1_badqUCB] |






### Thompson Sampling

Thompson Sampling uses the following logic:

- Fit the GP to the observations we have
- Draw one GP sample (a function) from the posterior
- Greedily choose the next point x with respect to the sample

[See wikipeda](https://en.wikipedia.org/wiki/Thompson_sampling)

As this method is probabilistic, it does not need to be updated after each new observation.

Here are some examples:

|     |          |
|-----|----------|
| D1  | ![d1_th] |
| D2  | ![d2_th] |
| D3  | ![d3_th] |


### Uncertainty-based exploration (UE)

This acquisition function aims at reducing the uncertainty of the GP model. It is defined as

$$
\alpha_{\rm{UE}}^i(\sigma^i)= \sigma^i
$$

and helps to explore the function space in regions of high uncertainty.


|     |          |
|-----|----------|
| D1  | ![d1_ue] |
| D2  | ![d2_ue] |
| D3  | ![d3_ue] |

### Expected Improvement (EI)

This also focuses on the maximisation/minimisation of the GP's posterior predictive mean.
It looks for the best value of $\mu$ so far and then looks for the next point that has a higher probability of being better than the best value so far.

$$
\alpha_{\rm{EI}}^i(\mu^i, \sigma^i)= \sigma^i\ (u^i \mathcal{N}_{\rm CDF}(u^i) \mathcal{N}_{\rm PDF}(u^i)\ ,
$$

where $u^i = \frac{\mu^i - \mu_{\rm{best}}}{\sigma^i}$ and $\mu_{\rm{best}}$ is the best value of $\mu$ so far.

|     |          |
|-----|----------|
| D1  | ![d1_ei] |
| D2  | ![d2_ei] |
| D3  | ![d3_ei] |






### Table of all anim


|     | D1         | D2        | D3        |
|-----|------------|-----------|-----------|
| EI  | ![d1_ei]   | ![d2_ei]  | ![d3_ei]  |
| TH  | ![d1_th]   | ![d2_th]  | ![d3_th]  |
| UCB | ![d1_ucb]  | ![d2_ucb] | ![d3_ucb] |
| UE  | ![d1_ue]   | ![d2_ue]  | ![d3_ue]  |
| MSE | ![d1_mse]  | ![d2_mse] | ![d3_mse] |








[d1_ei]:https://github.com/avivajpeyi/compas_ml_surrogate/assets/15642823/2cda4d1b-6cb0-4ce5-96b5-48d9cbf0790c
[d1_th]:https://github.com/avivajpeyi/compas_ml_surrogate/assets/15642823/97bc210f-c29c-4b12-8c7f-a3bb8b5abd47
[d1_ucb]:https://github.com/avivajpeyi/compas_ml_surrogate/assets/15642823/e1943785-7ea4-4079-99d3-29abb2041a1d
[d1_ue]:https://github.com/avivajpeyi/compas_ml_surrogate/assets/15642823/9f6a16cb-0be3-4a0c-8472-53e30e3bcc63

[d2_ei]:https://github.com/avivajpeyi/compas_ml_surrogate/assets/15642823/7ef729fc-8f95-40cb-b1a2-e1171eceec43
[d2_th]:https://github.com/avivajpeyi/compas_ml_surrogate/assets/15642823/7e9a36a5-d4d6-4593-a9d3-2ce706ccb5dc
[d2_ucb]:https://github.com/avivajpeyi/compas_ml_surrogate/assets/15642823/42e89e23-1e5b-44e0-a78b-259f52869c16
[d2_ue]:https://github.com/avivajpeyi/compas_ml_surrogate/assets/15642823/bd35eded-89c0-4f5c-a080-04010230e463

[d3_ei]:https://github.com/avivajpeyi/compas_ml_surrogate/assets/15642823/da7ad5b4-820b-4ce9-9680-1cf7cd50ed4f
[d3_th]:https://github.com/avivajpeyi/compas_ml_surrogate/assets/15642823/aa33ad19-e249-44d8-b5ee-f628a12266a8
[d3_ucb]:https://github.com/avivajpeyi/compas_ml_surrogate/assets/15642823/d6184215-0e05-4388-88bb-82228ea6b200
[d3_ue]:https://github.com/avivajpeyi/compas_ml_surrogate/assets/15642823/2691afbf-76d5-4168-b6e5-d5a210f31a2e






[d1_goodqUCB]:https://github.com/avivajpeyi/compas_ml_surrogate/assets/15642823/2be19245-50e4-482a-8979-e171ae33103b
[d1_badqUCB]:https://github.com/avivajpeyi/compas_ml_surrogate/assets/15642823/cb9a7e54-c71b-4d4d-8229-387f2f53450e
[d1_mse]:https://github.com/avivajpeyi/compas_ml_surrogate/assets/15642823/c1858c15-2219-40f5-975f-228e329f3f78
[d2_mse]:https://github.com/avivajpeyi/compas_ml_surrogate/assets/15642823/881896f6-42e4-4ab9-b56c-c69190a60443
[d3_mse]:https://github.com/avivajpeyi/compas_ml_surrogate/assets/15642823/58cb8d2c-315e-4276-ae2c-77a030655686
