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
:tags: [hide-input, render-all]

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
:class: important render-all

This notebook can be downloaded as **{nb-download}`place_cells.ipynb`**. See the button at the top right to download as markdown or pdf.

:::

# Model and feature selection with scikit-learn

Data for this notebook comes from recordings in the mouse hippocampus while the mouse runs on a linear track. We explored this data [yesterday](../day1/phase_preferences.md). Today, we will see that the neurons present in this recording show both tuning for both speed and location (i.e., place fields). However, location and speed are highly correlated. We would like to know which feature is more informative for predicting neuronal firing rate --- how do we do that?

<div class="render-user">
Data for this notebook comes from recordings in the mouse hippocampus while the mouse runs on a linear track, which we [explored yesterday](../day1/phase_preferences.md).
</div>

<div class="render-all">

## Learning objectives

- Review how to use pynapple to analyze neuronal tuning
- Learn how to combine NeMoS basis objects
- Learn how to use NeMoS objects with [scikit-learn](https://scikit-learn.org/) for cross-validation
- Learn how to use NeMoS objects with scikit-learn [pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- Learn how to use cross-validation to perform model and feature selection

</div>

```{code-cell} ipython3
:tags: [render-all]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap

import nemos as nmo

# some helper plotting functions
from nemos import _documentation_utils as doc_plots
import workshop_utils

# configure plots some
plt.style.use(nmo.styles.plot_style)

import workshop_utils

from sklearn import model_selection
from sklearn import pipeline

# shut down jax to numpy conversion warning
nap.nap_config.suppress_conversion_warnings = True

# during development, set this to a lower number so everything runs faster. 
cv_folds = 2
```

## Pynapple

<div class="render-user render-presenter">
- Load the data using pynapple.
</div>

```{code-cell} ipython3
:tags: [render-all]

path = workshop_utils.fetch_data("Achilles_10252013_EEG.nwb")
data = nap.load_file(path)
data
```

<div class="render-user render-presenter">
- Extract the spike times and mouse position.
</div>

```{code-cell} ipython3
:tags: [render-all]

spikes = data["units"]
position = data["position"]
```

For today, we're only going to focus on the times when the animal was traversing the linear track. 
This is a pynapple `IntervalSet`, so we can use it to restrict our other variables:

<div class="render-user render-presenter">

- Restrict data to when animal was traversing the linear track.

</div>

```{code-cell} ipython3
:tags: [render-all]

position = position.restrict(data["forward_ep"])
spikes = spikes.restrict(data["forward_ep"])
```

The recording contains both inhibitory and excitatory neurons. Here we will focus of the excitatory cells. Neurons have already been labelled before.

<div class="render-user render-presenter">

- Restrict neurons to only excitatory neurons, discarding neurons with a low-firing rate.

</div>

```{code-cell} ipython3
:tags: [render-all]

spikes = spikes.getby_category("cell_type")["pE"]
spikes = spikes.getby_threshold("rate", 0.3)
```

### Place fields

<div class="render-user render-presenter">

- Visualize the *place fields*: neuronal firing rate as a function of position.
</div>

```{code-cell} ipython3
:tags: [render-all]

place_fields = nap.compute_1d_tuning_curves(spikes, position, 50, position.time_support)
workshop_utils.plot_place_fields(place_fields)
```

<div class="render-user render-presenter">

- For speed, we're only going to investigate the three neurons highlighted above.
- Bin spikes to counts at 100 Hz.
- Interpolate position to match spike resolution.

</div>

```{code-cell} ipython3
:tags: [render-all]
neurons = [92, 82, 220]
bin_size = .01
count = spikes[neurons].count(bin_size, ep=position.time_support)
position = position.interpolate(count, ep=count.time_support)
print(count.shape)
print(position.shape)
```

## Speed modulation


The speed at which the animal traverse the field is not homogeneous. Does it influence the firing rate of hippocampal neurons? We can compute tuning curves for speed as well as average speed across the maze. In the next block, we compute the speed of the animal for each epoch (i.e. crossing of the linear track) by doing the difference of two consecutive position multiplied by the sampling rate of the position.

<div class="render-user render-presenter">

- Compute animal's speed for each epoch.

</div>


```{code-cell} ipython3
:tags: [render-all]
speed = []
# Analyzing each epoch separately avoids edge effects.
for s, e in position.time_support.values: 
    pos_ep = position.get(s, e)
    # Absolute difference of two consecutive points
    speed_ep = np.abs(np.diff(pos_ep)) 
    # Padding the edge so that the size is the same as the position/spike counts
    speed_ep = np.pad(speed_ep, [0, 1], mode="edge") 
    # Converting to cm/s 
    speed_ep = speed_ep * position.rate
    speed.append(speed_ep)

speed = nap.Tsd(t=position.t, d=np.hstack(speed), time_support=position.time_support)
```

Now that we have the speed of the animal, we can compute the tuning curves for speed modulation. Here we call pynapple [`compute_1d_tuning_curves`](https://pynapple.org/generated/pynapple.process.tuning_curves.html#pynapple.process.tuning_curves.compute_1d_tuning_curves):

<div class="render-user render-presenter">

- Compute the tuning curve with pynapple's [`compute_1d_tuning_curves`](https://pynapple.org/generated/pynapple.process.tuning_curves.html#pynapple.process.tuning_curves.compute_1d_tuning_curves)

</div>

```{code-cell} ipython3
tc_speed = nap.compute_1d_tuning_curves(spikes, speed, 20)
```

<div class="render-user">
```{code-cell} ipython3
tc_speed = # compute tuning curve here
```

</div>


<div class="render-user render-presenter">

- Visualize the position and speed tuning for these neurons.
</div>


```{code-cell} ipython3
:tags: [render-all]
fig = workshop_utils.plot_position_speed(position, speed, place_fields, tc_speed, neurons);
```

These neurons show a strong modulation of firing rate as a function of speed but we can also notice that the animal, on average, accelerates when traversing the field. Is the speed tuning we observe a true modulation or spurious correlation caused by traversing the place field at different speeds? We can use NeMoS to model the activity and give the position and the speed as input variable.

<div class="render-user render-presenter">

These neurons all show both position and speed tuning, and we see that the animal's speed and position are highly correlated. We're going to build a GLM to predict neuronal firing rate -- which variable should we use? Is the speed tuning just epiphenomenal?

</div>

(basis_eval_place_cells)=
## Basis evaluation

For each feature, we will use a different set of basis :

  -   position : [`MSplineEval`](nemos.basis.MSplineEval)
  -   speed : [`MSplineEval`](nemos.basis.MSplineEval)

```{code-cell} ipython3
position_basis = nmo.basis.MSplineEval(n_basis_funcs=10)
speed_basis = nmo.basis.MSplineEval(n_basis_funcs=15)
```

In addition, we will consider position and phase to be a joint variable. In this case the final basis object for our model can be made in one line :

```{code-cell} ipython3
basis = position_basis + speed_basis
```

The object basis only tell us how each basis covers the feature space. For each timestep, we need to _evaluate_ what are the features value. For that we can call NeMoS basis:

```{code-cell} ipython3
X = basis.compute_features(position, speed)
```

`X` is our design matrix. For each timestamps, it contains the information about the current position,
speed and theta phase of the experiment. Notice how passing a pynapple object to the basis
also returns a `pynapple` object.

+++

## Model learning

We can now use the Poisson GLM from NeMoS to learn the model.

```{code-cell} ipython3
glm = nmo.glm.PopulationGLM(
    solver_kwargs=dict(tol=10**-12),
    solver_name="LBFGS",
)

glm.fit(X, count)
```

## Prediction

Let's check first if our model can accurately predict the different tuning curves we displayed above. We can use the [`predict`](nemos.glm.GLM.predict) function of NeMoS and then compute new tuning curves

```{code-cell} ipython3
predicted_rate = glm.predict(X) / bin_size
# recreate so we can rename columns
predicted_rate = nap.TsdFrame(
    t=predicted_rate.t,
    d=predicted_rate.d,
    columns=sorted(neurons),
)

glm_pf = nap.compute_1d_tuning_curves_continuous(predicted_rate, position, 50)
glm_speed = nap.compute_1d_tuning_curves_continuous(predicted_rate, speed, 30)
```

Let's display both tuning curves together. Does a pretty good job!

```{code-cell} ipython3
workshop_utils.plot_position_speed_tuning(place_fields, tc_speed, neurons, glm_pf, glm_speed);
```

## How to know when to regularize?

In the last session, Edoardo fit the all-to-all connectivity of the head-tuning dataset using the Ridge regularizer. In the model above, we're not using any regularization? Why is that?

We have far fewer parameters here then in the last example. However, how do you know if you need regularization or not? One thing you can do is use cross-validation to see whether model performance improves with regularization (behind the scenes, this is what we did!). We'll walk through how to do that now.

Instead of implementing our own cross-validation machinery, the developers of nemos decided that we should write the package to be compliant with scikit-learn, the canonical machine learning python library. Our models are all what scikit-learn calls "estimators", which means they have `.fit`, `.score.` and `.predict` methods. Thus, we can use them with scikit-learn's objects out of the box.

We're going to use scikit-learn's [GriSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) object, which performs a cross-validated grid search, as Edoardo explained.

This object requires an estimator, our `glm` object here, and `param_grid`, a dictionary defining what to check:

```{code-cell} ipython3
glm = nmo.glm.PopulationGLM(solver_name="LBFGS", solver_kwargs=dict(tol=10**-12))
```

```{code-cell} ipython3
param_grid = {
    "regularizer": ["UnRegularized", "Ridge"],
}
```

Anything not specified in the grid will be kept constant.

```{code-cell} ipython3
cv = model_selection.GridSearchCV(glm, param_grid, cv=cv_folds)
```

```{code-cell} ipython3
cv
```

```{code-cell} ipython3
cv.fit(X, count)
```

Describe warning above

Investigate results

```{code-cell} ipython3
cv.cv_results_
```

:::{note}
Could (and generally, should!) investigate `regularizer_strength`, but we're skipping for complexity. To do this properly, use a slightly different syntax for `param_grid` (list of dictionaries, instead of single dictionary)

```python
param_grid = [
    {"regularizer": [nmo.regularizer.UnRegularized()]},
    {"regularizer": [nmo.regularizer.Ridge()],
     "regularizer_strength": [1e-6, 1e-3, 1]}
]
```
:::

+++

## Select basis

We can do something similar to select the basis. In the above example, I just told you which basis function to use and how many of each. But, in general, you want to select those in a reasonable manner. Cross-validation to the rescue!

Unlike the glm objects, our basis objects are not scikit-learn compatible right out of the box. However, they can be made compatible by using the `.to_transformer()` method (or, equivalently, by using the `TransformerBasis` class)

```{code-cell} ipython3
position_basis = nmo.basis.MSplineEval(n_basis_funcs=10).to_transformer()
```

This gives the basis object the `transform` method. However, transformers have some limits:

```{code-cell} ipython3
:tags: [raises-exception]

position_basis.transform(position)
```

Transformers only accept 2d inputs, whereas nemos basis objects can accept inputs of any dimensionality. In order to tell nemos how to reshape the 2d matrix that is the input of `transform` to whatever the basis accepts, you need to call `set_input_shape`:

```{code-cell} ipython3
# can accept array
position_basis.set_input_shape(position)
# int
position_basis.set_input_shape(1)
# tuple
position_basis.set_input_shape(position.shape[1:])
```

```{code-cell} ipython3
# needs to be 2d
position_basis.transform(np.expand_dims(position, 1))
```

You can, equivalently, call `compute_features` *before* turning the basis into a transformer. Then we cache the shape for future use:

```{code-cell} ipython3
position_basis = nmo.basis.MSplineEval(n_basis_funcs=10)
position_basis.compute_features(position)
position_basis = position_basis.to_transformer()
speed_basis = nmo.basis.MSplineEval(n_basis_funcs=15).to_transformer().set_input_shape(1)
# could also create the additive basis and then call to transformer
basis = position_basis + speed_basis
```

Create our input:

```{code-cell} ipython3
transformer_input = nap.TsdFrame(
    t=position.t,
    d=np.stack([position.d, speed.d], 1),
    time_support=position.time_support,
    columns=["position", "speed"],
)
```

```{code-cell} ipython3
basis.transform(transformer_input)
```

One final thing: nemos basis objects can also be labeled. This will make it easier for us to keep track of which basis is handling position and which speed, so let's go ahead and do that now:

```{code-cell} ipython3
position_basis = nmo.basis.MSplineEval(n_basis_funcs=10, label="position")
position_basis.compute_features(position)
position_basis = position_basis.to_transformer()
speed_basis = nmo.basis.MSplineEval(n_basis_funcs=15, label="speed").to_transformer().set_input_shape(1)
# could also create the additive basis and then call to transformer
basis = position_basis + speed_basis
```

## Pipelines

We need one more step: scikit-learn cross-validation operates on an estimator, like our GLMs. if we want to cross-validate over the basis or its features, we need to combine our transformer basis with the estimator into a single estimator object. Luckily, scikit-learn provides tools for this: pipelines.

Pipelines are objects that accept a series of (0 or more) transformers, culminating in a final estimator. This is defined as a list of tuples, with each tuple containing a human-readable label and the object itself:

```{code-cell} ipython3
pipe = pipeline.Pipeline([
    ("basis", basis),
    ("glm", glm)
])
pipe
```

This pipeline object allows us to e.g., call fit using the *initial input*:

```{code-cell} ipython3
pipe.fit(transformer_input, count)
```

```{code-cell} ipython3
predicted_rate = pipe.predict(transformer_input) / bin_size
# recreate so we can rename columns
predicted_rate = nap.TsdFrame(
    t=predicted_rate.t,
    d=predicted_rate.d,
    columns=sorted(neurons),
)

glm_pf = nap.compute_1d_tuning_curves_continuous(predicted_rate, position, 50)
glm_speed = nap.compute_1d_tuning_curves_continuous(predicted_rate, speed, 30)

workshop_utils.plot_position_speed_tuning(place_fields, tc_speed, neurons, glm_pf, glm_speed);
```

## Cross-validating on the basis

Now that we have our pipeline estimator, we can cross-validate on any of its parameters!

```{code-cell} ipython3
pipe.steps
```

Let's cross-validate on the number of basis functions for the position basis, and the identity of the basis for the speed. That is:

```{code-cell} ipython3
print(pipe["basis"].basis1.n_basis_funcs)
print(pipe["basis"].basis2)
```

For scikit-learn parameter grids, we use `__` to stand in for `.`:

```{code-cell} ipython3
param_grid = {
    "basis__basis1__n_basis_funcs": [5, 10, 20],
    "basis__basis2": [nmo.basis.MSplineEval(15).set_input_shape(1),
                      nmo.basis.BSplineEval(15).set_input_shape(1),
                      nmo.basis.CyclicBSplineEval(15).set_input_shape(1)],
}
```

```{code-cell} ipython3
cv = model_selection.GridSearchCV(pipe, param_grid, cv=cv_folds)
```

```{code-cell} ipython3
cv.fit(transformer_input, count)
```

```{code-cell} ipython3
cv.cv_results_
```

```{code-cell} ipython3
cv_df = pd.DataFrame(cv.cv_results_)
cv_df
```

```{code-cell} ipython3
workshop_utils.plot_heatmap_cv_results(cv_df)
```

```{code-cell} ipython3
cv.best_estimator_
```

```{code-cell} ipython3
best_estim = cv.best_estimator_
```

```{code-cell} ipython3
predicted_rate = cv.best_estimator_.predict(transformer_input) / bin_size
# recreate so we can rename columns
predicted_rate = nap.TsdFrame(
    t=predicted_rate.t,
    d=predicted_rate.d,
    columns=sorted(neurons),
)

glm_pf = nap.compute_1d_tuning_curves_continuous(predicted_rate, position, 50)
glm_speed = nap.compute_1d_tuning_curves_continuous(predicted_rate, speed, 30)

workshop_utils.plot_position_speed_tuning(place_fields, tc_speed, neurons, glm_pf, glm_speed);
```

## Model selection

Now, finally, we understand almost enough about how scikit-learn works to do model selection. There's just one more thing to learn: feature masks.

```{code-cell} ipython3
pipe['glm'].feature_mask
```

```{code-cell} ipython3
workshop_utils.plot_feature_mask(pipe["glm"].feature_mask);
```

could manually edit feature mask, but have some helper functions -- these are currently being developed, so any feedback is appreciated!

```{code-cell} ipython3
m = workshop_utils.create_feature_mask(pipe["basis"], n_neurons=3)
workshop_utils.plot_feature_mask(m);
```

```{code-cell} ipython3
m = workshop_utils.create_feature_mask(pipe["basis"], ["all", "none"], n_neurons=3)
fig=workshop_utils.plot_feature_mask(m);
```

```{code-cell} ipython3
feature_masks = [
    workshop_utils.create_feature_mask(basis, "all", n_neurons=3),
    workshop_utils.create_feature_mask(basis, ["all", "none"], n_neurons=3),
    workshop_utils.create_feature_mask(basis, ["none", "all"], n_neurons=3),
]
```

```{code-cell} ipython3
workshop_utils.plot_feature_mask(feature_masks, ["All", "Position", "Speed"]);
```

One more wrinkle: the shape of this feature mask depends on the number of basis functions! (The number of features is `basis.n_basis_funcs = basis.basis1.n_basis_funcs + basis.basis2.n_basis_funcs`.) Thus we need to create a new feature mask for each possible arrangement:

```{code-cell} ipython3
param_grid = []
b1_ns = [5, 10, 20]
b2_ns = [8, 16, 32]
for b1 in b1_ns:
    basis.basis1.n_basis_funcs = b1
    basis.basis2.n_basis_funcs = b2_ns[0]
    param_grid.append({"glm__feature_mask": [workshop_utils.create_feature_mask(basis, ["all", "none"], n_neurons=3)],
                       "basis__basis1__n_basis_funcs": [b1], "basis__basis2__n_basis_funcs": [b2_ns[0]]})
for b2 in b2_ns:
    basis.basis2.n_basis_funcs = b2
    basis.basis1.n_basis_funcs = b1_ns[0]
    param_grid.append({"glm__feature_mask": [workshop_utils.create_feature_mask(basis, ["none", "all"], n_neurons=3)],
                       "basis__basis1__n_basis_funcs": [b1_ns[0]], "basis__basis2__n_basis_funcs": [b2]})
for b1 in b1_ns:
    for b2 in b2_ns:
        basis.basis1.n_basis_funcs = b1
        basis.basis2.n_basis_funcs = b2
        param_grid.append({"glm__feature_mask": [workshop_utils.create_feature_mask(basis, "all", n_neurons=3)],
                           "basis__basis1__n_basis_funcs": [b1], "basis__basis2__n_basis_funcs": [b2]})
```

```{code-cell} ipython3
cv = model_selection.GridSearchCV(best_estim, param_grid, cv=cv_folds)
```

```{code-cell} ipython3
cv
```

```{code-cell} ipython3
cv.fit(transformer_input, count)
```

```{code-cell} ipython3
cv_df = pd.DataFrame(cv.cv_results_)
cv_df
```

```{code-cell} ipython3
def label_feature_mask(x):
    mask = x.param_glm__feature_mask
    if mask.sum() / np.prod(mask.shape) == 1:
        return "all"
    elif mask[0,0] == 1:
        return "position"
    else:
        return "speed"

cv_df['feature_mask_label'] = cv_df.apply(label_feature_mask, 1)
```

```{code-cell} ipython3
workshop_utils.plot_heatmap_cv_results(cv_df, "feature_mask_label", columns="param_basis__basis2__n_basis_funcs")
```

## Conclusion

Various combinations of features can lead to different results. Feel free to explore more. To go beyond this notebook, you can check the following references :

  - [Hardcastle, Kiah, et al. "A multiplexed, heterogeneous, and adaptive code for navigation in medial entorhinal cortex." Neuron 94.2 (2017): 375-387](https://www.cell.com/neuron/pdf/S0896-6273(17)30237-4.pdf)

  - [McClain, Kathryn, et al. "Position–theta-phase model of hippocampal place cell activity applied to quantification of running speed modulation of firing rate." Proceedings of the National Academy of Sciences 116.52 (2019): 27035-27042](https://www.pnas.org/doi/abs/10.1073/pnas.1912792116)

  - [Peyrache, Adrien, Natalie Schieferstein, and Gyorgy Buzsáki. "Transformation of the head-direction signal into a spatial code." Nature communications 8.1 (2017): 1752.](https://www.nature.com/articles/s41467-017-01908-3)

## References

<div class="render-all">


The data in this tutorial comes from [Grosmark, Andres D., and György Buzsáki. "Diversity in neural firing dynamics supports both rigid and learned hippocampal sequences." Science 351.6280 (2016): 1440-1443](https://www.science.org/doi/full/10.1126/science.aad1935).

</div>
