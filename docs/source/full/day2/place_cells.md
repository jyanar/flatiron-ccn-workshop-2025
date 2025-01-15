---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.6
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
:tags: [hide-input]

%load_ext autoreload
%autoreload 2

%matplotlib inline
import warnings

warnings.filterwarnings(
    "ignore",
    message="plotting functions contained within `_documentation_utils` are intended for nemos's documentation.",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message="Ignoring cached namespace 'core'",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message=(
        "invalid value encountered in div "
    ),
    category=RuntimeWarning,
)
```

:::{admonition} Download
:class: important

This notebook can be downloaded as **{nb-download}`place_cells.ipynb`**. See the button at the top right to download as markdown or pdf.

:::

# Fit place cell

The data for this example are from [Grosmark, Andres D., and György Buzsáki. "Diversity in neural firing dynamics supports both rigid and learned hippocampal sequences." Science 351.6280 (2016): 1440-1443](https://www.science.org/doi/full/10.1126/science.aad1935).

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
from scipy.ndimage import gaussian_filter

import nemos as nmo

# some helper plotting functions
from nemos import _documentation_utils as doc_plots

# configure plots some
plt.style.use(nmo.styles.plot_style)

import workshop_utils

# shut down jax to numpy conversion warning
nap.nap_config.suppress_conversion_warnings = True
```

## Data Streaming

Here we load the data from OSF. The data is a NWB file.

```{code-cell} ipython3
path = nmo.fetch.fetch_data("Achilles_10252013.nwb")
```

## Pynapple
We are going to open the NWB file with pynapple

```{code-cell} ipython3
data = nap.load_file(path)

data
```

Let's extract the spike times, the position and the theta phase.

```{code-cell} ipython3
spikes = data["units"]
position = data["position"]
theta = data["theta_phase"]
```

The NWB file also contains the time at which the animal was traversing the linear track. We can use it to restrict the position and assign it as the `time_support` of position.

```{code-cell} ipython3
position = position.restrict(data["trials"])
```

The recording contains both inhibitory and excitatory neurons. Here we will focus of the excitatory cells. Neurons have already been labelled before.

```{code-cell} ipython3
spikes = spikes.getby_category("cell_type")["pE"]
```

We can discard the low firing neurons as well.

```{code-cell} ipython3
spikes = spikes.getby_threshold("rate", 0.3)
```

## Place fields
Let's plot some data. We start by making place fields i.e firing rate as a function of position.

```{code-cell} ipython3
pf = nap.compute_1d_tuning_curves(spikes, position, 50, position.time_support)
```

Let's do a quick sort of the place fields for display

```{code-cell} ipython3
order = pf.idxmax().sort_values().index.values
```

Here each row is one neuron

```{code-cell} ipython3
fig = plt.figure(figsize=(12, 10))
gs = plt.GridSpec(len(spikes), 1)
for i, n in enumerate(order):
    plt.subplot(gs[i, 0])
    plt.fill_between(pf.index.values, np.zeros(len(pf)), pf[n].values)
    if i < len(spikes) - 1:
        plt.xticks([])
    else:
        plt.xlabel("Position (cm)")
    plt.yticks([])
```

```{code-cell} ipython3
neurons = [92, 82, 220]
bin_size = .01
count = spikes[neurons].count(bin_size, ep=position.time_support)
position = position.interpolate(count, ep=count.time_support)
```

## Speed modulation
The speed at which the animal traverse the field is not homogeneous. Does it influence the firing rate of hippocampal neurons? We can compute tuning curves for speed as well as average speed across the maze.
In the next block, we compute the speed of the animal for each epoch (i.e. crossing of the linear track) by doing the difference of two consecutive position multiplied by the sampling rate of the position.

```{code-cell} ipython3
speed = []
for s, e in position.time_support.values: # Time support contains the epochs
    pos_ep = position.get(s, e)
    speed_ep = np.abs(np.diff(pos_ep)) # Absolute difference of two consecutive points
    speed_ep = np.pad(speed_ep, [0, 1], mode="edge") # Adding one point at the end to match the size of the position array
    speed_ep = speed_ep * position.rate # Converting to cm/s
    speed.append(speed_ep)

speed = nap.Tsd(t=position.t, d=np.hstack(speed), time_support=position.time_support)
```

Now that we have the speed of the animal, we can compute the tuning curves for speed modulation. Here we call pynapple [`compute_1d_tuning_curves`](https://pynapple.org/generated/pynapple.process.tuning_curves.html#pynapple.process.tuning_curves.compute_1d_tuning_curves):

```{code-cell} ipython3
tc_speed = nap.compute_1d_tuning_curves(spikes, speed, 20)
```

To assess the variabilty in speed when the animal is travering the linear track, we can compute the average speed and estimate the standard deviation. Here we use numpy only and put the results in a pandas [`DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html):

```{code-cell} ipython3
fig=workshop_utils.plot_position_speed(position, speed, pf, tc_speed, neurons);
```

This neurons show a strong modulation of firing rate as a function of speed but we can also notice that the animal, on average, accelerates when travering the field. Is the speed tuning we observe a true modulation or spurious correlation caused by traversing the place field at different speed and for different theta phase? We can use NeMoS to model the activity and give the position, the phase and the speed as input variable.

We will use speed, phase and position to model the activity of the neuron.
All the feature have already been brought to the same dimension thanks to `pynapple`.

```{code-cell} ipython3
print(position.shape)
print(speed.shape)
print(count.shape)
```

(basis_eval_place_cells)=
## Basis evaluation

For each feature, we will use a different set of basis :

  -   position : [`MSplineEval`](nemos.basis.MSplineEval)
  -   theta phase : [`CyclicBSplineEval`](nemos.basis.CyclicBSplineEval)
  -   speed : [`MSplineEval`](nemos.basis.MSplineEval)

```{code-cell} ipython3
position_basis = nmo.basis.MSplineEval(n_basis_funcs=10)
phase_basis = nmo.basis.CyclicBSplineEval(n_basis_funcs=12)
speed_basis = nmo.basis.MSplineEval(n_basis_funcs=15)
```

In addition, we will consider position and phase to be a joint variable. In NeMoS, we can combine basis by multiplying them and adding them. In this case the final basis object for our model can be made in one line :

```{code-cell} ipython3
basis = position_basis * phase_basis + speed_basis
```

The object basis only tell us how each basis covers the feature space. For each timestep, we need to _evaluate_ what are the features value. For that we can call NeMoS basis:

```{code-cell} ipython3
X = basis.compute_features(position, theta, speed)
```

`X` is our design matrix. For each timestamps, it contains the information about the current position,
speed and theta phase of the experiment. Notice how passing a pynapple object to the basis
also returns a `pynapple` object.

```{code-cell} ipython3
print(X)
```

## Model learning

We can now use the Poisson GLM from NeMoS to learn the model.

```{code-cell} ipython3
glm = nmo.glm.GLM(
    solver_kwargs=dict(tol=10**-12),
    solver_name="LBFGS"
)

glm.fit(X, count)
```

## Prediction

Let's check first if our model can accurately predict the different tuning curves we displayed above. We can use the [`predict`](nemos.glm.GLM.predict) function of NeMoS and then compute new tuning curves

```{code-cell} ipython3
predicted_rate = glm.predict(X) / bin_size

glm_pf = nap.compute_1d_tuning_curves_continuous(predicted_rate[:, np.newaxis], position, 50)
glm_pos_theta, xybins = nap.compute_2d_tuning_curves_continuous(
    predicted_rate[:, np.newaxis], data, 30, ep=within_ep
)
glm_speed = nap.compute_1d_tuning_curves_continuous(predicted_rate[:, np.newaxis], speed, 30)
```

Let's display both tuning curves together.

```{code-cell} ipython3
fig = doc_plots.plot_position_phase_speed_tuning(
    pf[neuron],
    glm_pf[0],
    tc_speed[neuron],
    glm_speed[0],
    tc_pos_theta[neuron],
    glm_pos_theta[0],
    xybins
    )
```

## Model selection

While this model captures nicely the features-rate relationship, it is not necessarily the simplest model. Let's construct several models and evaluate their score to determine the best model.

:::{note}

To shorten this notebook, only a few combinations are tested. Feel free to expand this list.
:::

```{code-cell} ipython3
models = {
    "position": position_basis,
    "position + speed": position_basis + speed_basis,
    "position + phase": position_basis + phase_basis,
    "position * phase + speed": position_basis * phase_basis + speed_basis,
}
features = {
    "position": (position,),
    "position + speed": (position, speed),
    "position + phase": (position, theta),
    "position * phase + speed": (position, theta, speed),
}
```

In a loop, we can (1) evaluate the basis, (2), fit the model, (3) compute the score and (4) predict the firing rate. For evaluating the score, we can define a train set of intervals and a test set of intervals.

```{code-cell} ipython3
train_iset = position.time_support[::2] # Taking every other epoch
test_iset = position.time_support[1::2]
```

Let's train all the models.

```{code-cell} ipython3
scores = {}
predicted_rates = {}

for m in models:
    print("1. Evaluating basis : ", m)
    X = models[m].compute_features(*features[m])

    print("2. Fitting model : ", m)
    glm.fit(
        X.restrict(train_iset),
        count.restrict(train_iset),
    )

    print("3. Scoring model : ", m)
    scores[m] = glm.score(
        X.restrict(test_iset),
        count.restrict(test_iset),
        score_type="pseudo-r2-McFadden",
    )

    print("4. Predicting rate")
    predicted_rates[m] = glm.predict(X.restrict(test_iset)) / bin_size


scores = pd.Series(scores)
scores = scores.sort_values()
```

Let's compute scores for each models.

```{code-cell} ipython3
plt.figure(figsize=(5, 3))
plt.barh(np.arange(len(scores)), scores)
plt.yticks(np.arange(len(scores)), scores.index)
plt.xlabel("Pseudo r2")
plt.tight_layout()
```

Some models are doing better than others.

:::{warning}
A proper model comparison should be done by scoring models repetitively on various train and test set. Here we are only doing partial models comparison for the sake of conciseness.
:::

Alternatively, we can plot some tuning curves to compare each models visually.

```{code-cell} ipython3
tuning_curves = {}

for m in models:
    tuning_curves[m] = {
        "position": nap.compute_1d_tuning_curves_continuous(
            predicted_rates[m][:, np.newaxis], position, 50, ep=test_iset
        ),
        "speed": nap.compute_1d_tuning_curves_continuous(
            predicted_rates[m][:, np.newaxis], speed, 20, ep=test_iset
        ),
    }

# recompute tuning from spikes restricting to the test-set
pf = nap.compute_1d_tuning_curves(spikes, position, 50, ep=test_iset)
tc_speed = nap.compute_1d_tuning_curves(spikes, speed, 20, ep=test_iset)


fig = plt.figure(figsize=(8, 4))
outer_grid = fig.add_gridspec(2, 2)
for i, m in enumerate(models):
    doc_plots.plot_position_speed_tuning(
        outer_grid[i // 2, i % 2],
        tuning_curves[m],
        pf[neuron],
        tc_speed[neuron],
        m)

plt.tight_layout()
plt.show()
```

## Conclusion

Various combinations of features can lead to different results. Feel free to explore more. To go beyond this notebook, you can check the following references :

  - [Hardcastle, Kiah, et al. "A multiplexed, heterogeneous, and adaptive code for navigation in medial entorhinal cortex." Neuron 94.2 (2017): 375-387](https://www.cell.com/neuron/pdf/S0896-6273(17)30237-4.pdf)

  - [McClain, Kathryn, et al. "Position–theta-phase model of hippocampal place cell activity applied to quantification of running speed modulation of firing rate." Proceedings of the National Academy of Sciences 116.52 (2019): 27035-27042](https://www.pnas.org/doi/abs/10.1073/pnas.1912792116)

  - [Peyrache, Adrien, Natalie Schieferstein, and Gyorgy Buzsáki. "Transformation of the head-direction signal into a spatial code." Nature communications 8.1 (2017): 1752.](https://www.nature.com/articles/s41467-017-01908-3)
