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
Data for this notebook comes from recordings in the mouse hippocampus while the mouse runs on a linear track, which we [explored yesterday](../day1/phase_preferences-users.md).
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

neurons = [82, 92, 220]
place_fields = place_fields[neurons]
spikes = spikes[neurons]
bin_size = .01
count = spikes.count(bin_size, ep=position.time_support)
position = position.interpolate(count, ep=count.time_support)
print(count.shape)
print(position.shape)
```

### Speed modulation


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
tc_speed = nap.compute_1d_tuning_curves(spikes, speed, 20, speed.time_support)
```

<div class="render-user">

```{code-cell} ipython3
:tags: [remove-cell, skip-execution]
# compute tuning curve here
tc_speed = 
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

## NeMoS

(basis_eval_place_cells)=
### Basis evaluation

As we've seen before, we will use basis objects to represent the input values. Since we have two different inputs, we'll need two separate basis objects. We will use [`MSplineEval`](nemos.basis.MSplineEval) for both, though with different numbers of basis functions:

<div class="render-user render-presenter">

- Create a separate basis object for each model input.
- Visualize the basis objects.
</div>

```{code-cell} ipython3
position_basis = nmo.basis.MSplineEval(n_basis_funcs=10)
speed_basis = nmo.basis.MSplineEval(n_basis_funcs=15)
workshop_utils.plot_pos_speed_bases(position_basis, speed_basis)
```

However, now we have an issue: in all our previous examples, we had a single basis object, which took a single input to produce a single array which we then passed to the `GLM` object as the design matrix. What do we do when we have multiple basis objects?

We could call `basis.compute_features()` for each basis separately and then concatenated the outputs, but then we have to remember the order we concatenated them in and gets unwieldy as we add more bases.

Instead, NeMoS allows us to combine multiple basis objects into a single "additive basis", which we can pass all of our inputs to in order to produce a single design matrix:

<div class="render-user render-presenter">

- Combine the two basis objects into a single "additive basis"
</div>

```{code-cell} ipython3
basis = position_basis + speed_basis
```

<div class="render-user render-presenter">

- Create the design matrix!
- Notice that, since we passed the basis pynapple objects, we got one back, preserving the time stamps.

</div>

```{code-cell} ipython3
X = basis.compute_features(position, speed)
X
```

### Model learning

As we've done before, we can now use the Poisson GLM from NeMoS to learn the combined model:

<div class="render-user render-presenter">

- Initialize `PopulationGLM`
- Use the "LBFGS" solver and pass `{"tol": 1e-12}` to `solver_kwargs`.
- Fit the data, passing the design matrix and spike counts to the glm object.

</div>

<div class="render-user">

```{code-cell} ipython3
:tags: [skip-execution]

# initialize 
glm =

# and fit
```

</div>

```{code-cell} ipython3
glm = nmo.glm.PopulationGLM(
    solver_kwargs={"tol": 1e-12},
    solver_name="LBFGS",
)

glm.fit(X, count)
```

### Prediction

Let's check first if our model can accurately predict the tuning curves we displayed above. We can use the [`predict`](nemos.glm.GLM.predict) function of NeMoS and then compute new tuning curves

<div class="render-user render-presenter">

- Use `predict` to check whether our GLM has captured each neuron's speed and position tuning.
- Remember to convert the predicted firing rate to spikes per second!

</div>

<div class="render-user">

```{code-cell} ipython3
:tags: [skip-execution]

# predict the model's firing rate
predicted_rate = 

# compute the position and speed tuning curves using the predicted firing rate.
glm_pf = 
glm_speed = 
```

</div>

```{code-cell} ipython3
# predict the model's firing rate
predicted_rate = glm.predict(X) / bin_size

# compute the position and speed tuning curves using the predicted firing rate.
glm_pf = nap.compute_1d_tuning_curves_continuous(predicted_rate, position, 50, position.time_support)
glm_speed = nap.compute_1d_tuning_curves_continuous(predicted_rate, speed, 30, speed.time_support)
```

<div class="render-user render-presenter">

- Compare model and data tuning curves together. The model did a pretty good job!

</div>

```{code-cell} ipython3
:tags: [render-all]

workshop_utils.plot_position_speed_tuning(place_fields, tc_speed, glm_pf, glm_speed);
```

<div class="render-all">

To make our lives easier, let's create a helper function that wraps the above
lines, because we're going to be visualizing our model predictions a lot.

</div>

```{code-cell} ipython3
def visualize_model_predictions(glm, X):
    # predict the model's firing rate
    predicted_rate = glm.predict(X) / bin_size

    # compute the position and speed tuning curves using the predicted firing rate.
    glm_pf = nap.compute_1d_tuning_curves_continuous(predicted_rate, position, 50, position.time_support)
    glm_speed = nap.compute_1d_tuning_curves_continuous(predicted_rate, speed, 30, position.time_support)

    workshop_utils.plot_position_speed_tuning(place_fields, tc_speed, glm_pf, glm_speed);
```

## Scikit-learn

### How to know when to regularize?

In the last session, Edoardo fit the all-to-all connectivity of the head-tuning dataset using the Ridge regularizer. In the model above, we're not using any regularization? Why is that?

We have far fewer parameters here then in the last example. However, how do you know if you need regularization or not? One thing you can do is use cross-validation to see whether model performance improves with regularization (behind the scenes, this is what we did!). We'll walk through how to do that now.

Instead of implementing our own cross-validation machinery, the developers of nemos decided that we should write the package to be compliant with scikit-learn, the canonical machine learning python library. Our models are all what scikit-learn calls "estimators", which means they have `.fit`, `.score.` and `.predict` methods. Thus, we can use them with scikit-learn's objects out of the box.

We're going to use scikit-learn's [GriSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) object, which performs a cross-validated grid search, as Edoardo explained.

This object requires an estimator, our `glm` object here, and `param_grid`, a dictionary defining what to check:

<div class="render-user render-presenter">

- How do we decide when to use regularization?
- Cross-validation allows you to fairly compare different models on the same dataset.
- NeMoS makes use of [scikit-learn](https://scikit-learn.org/), the standard machine learning library in python.
- Define [parameter grid](https://scikit-learn.org/stable/modules/grid_search.html#grid-search) to search over.
- Anything not specified in grid will be kept constant.

</div>

```{code-cell} ipython3
param_grid = {
    "regularizer": ["UnRegularized", "Ridge"],
}
```

<div class="render-user render-presenter">

- Initialize scikit-learn's [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV) object.
</div>

```{code-cell} ipython3
cv = model_selection.GridSearchCV(glm, param_grid, cv=cv_folds)
cv
```

<div class="render-user render-presenter">

- We interact with this in a very similar way to the glm object.
- In particular, call `fit` with same arguments:
</div>

```{code-cell} ipython3
cv.fit(X, count)
```

<div class="render-user render-presenter">

- We got a warning because we didn't specify the regularizer strength, so we just fell back on default value.
- Let's investigate results:
</div>

```{code-cell} ipython3
cv.cv_results_
```

<div class="render-all">

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

</div>

### Select basis

We can do something similar to select the basis. In the above example, I just told you which basis function to use and how many of each. But, in general, you want to select those in a reasonable manner. Cross-validation to the rescue!

Unlike the glm objects, our basis objects are not scikit-learn compatible right out of the box. However, they can be made compatible by using the `.to_transformer()` method (or, equivalently, by using the `TransformerBasis` class)

<div class="render-user render-presenter">

- You can (and should) do something similar to determine how many basis functions you need for each input.
- NeMoS basis objects are not scikit-learn-compatible right out of the box.
- But we have provided a simple method to make them so:

</div>

```{code-cell} ipython3
position_basis = nmo.basis.MSplineEval(n_basis_funcs=10).to_transformer()
# or equivalently:
position_basis = nmo.basis.TransformerBasis(nmo.basis.MSplineEval(n_basis_funcs=10))
```

This gives the basis object the `transform` method, which is equivalent to `compute_features`. However, transformers have some limits:

<div class="render-user render-presenter">

- This gives the basis object the `transform` method, which is equivalent to `compute_features`.
- However, transformers have some limits:

</div>

```{code-cell} ipython3
:tags: [raises-exception]

position_basis.transform(position)
```

<div class="render-user render-presenter">

- Transformers only accept 2d inputs, whereas nemos basis objects can accept inputs of any dimensionality.
- In order to tell nemos how to reshape the 2d matrix that is the input of `transform` to whatever the basis accepts, you need to call `set_input_shape`:

</div>

Transformers only accept 2d inputs, whereas nemos basis objects can accept inputs of any dimensionality. In order to tell nemos how to reshape the 2d matrix that is the input of `transform` to whatever the basis accepts, you need to call `set_input_shape`:

```{code-cell} ipython3
# can accept array
position_basis.set_input_shape(position)
# int
position_basis.set_input_shape(1)
# tuple
position_basis.set_input_shape(position.shape[1:])
```

<div class="render-user render-presenter">

- Then you can call transform on the 2d input as expected.
</div>

```{code-cell} ipython3
# needs to be 2d
position_basis.transform(np.expand_dims(position, 1))
```

<div class="render-all">

- You can, equivalently, call `compute_features` *before* turning the basis into a transformer. Then we cache the shape for future use:

</div>

```{code-cell} ipython3
position_basis = nmo.basis.MSplineEval(n_basis_funcs=10)
position_basis.compute_features(position)
position_basis = position_basis.to_transformer()
speed_basis = nmo.basis.MSplineEval(n_basis_funcs=15).to_transformer().set_input_shape(1)
basis = position_basis + speed_basis
```

Create our input:

<div class="render-user render-presenter">

- Create a single TsdFrame to hold all our inputs:
</div>

```{code-cell} ipython3
:tags: [render-all]

transformer_input = nap.TsdFrame(
    t=position.t,
    d=np.stack([position.d, speed.d], 1),
    time_support=position.time_support,
    columns=["position", "speed"],
)
```

<div class="render-user render-presenter">

- Pass this input to our transformed additive basis:
</div>

```{code-cell} ipython3
basis.transform(transformer_input)
```

### Pipelines

We need one more step: scikit-learn cross-validation operates on an estimator, like our GLMs. if we want to cross-validate over the basis or its features, we need to combine our transformer basis with the estimator into a single estimator object. Luckily, scikit-learn provides tools for this: pipelines.

Pipelines are objects that accept a series of (0 or more) transformers, culminating in a final estimator. This is defined as a list of tuples, with each tuple containing a human-readable label and the object itself:

<div class="render-user render-presenter">

- If we want to cross-validate over the basis, we need more one more step: combining the basis and the GLM into a single scikit-learn estimator.
- [Pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) to the rescue!

</div>

```{code-cell} ipython3
pipe = pipeline.Pipeline([
    ("basis", basis),
    ("glm", glm)
])
pipe
```

This pipeline object allows us to e.g., call fit using the *initial input*:

<div class="render-user render-presenter">

- Pipeline runs `basis.transform`, then passes that output to `glm`, so we can do everything in a single line:
</div>

```{code-cell} ipython3
pipe.fit(transformer_input, count)
```

We then visualize the predictions the same as before, using `pipe` instead of `glm`.

<div class="render-user render-presenter">

- Visualize model predictions!

</div>

```{code-cell} ipython3
visualize_model_predictions(pipe, transformer_input)
```

### Cross-validating on the basis

<div class="render-all">
Now that we have our pipeline estimator, we can cross-validate on any of its parameters!

</div>

```{code-cell} ipython3
pipe.steps
```

Let's cross-validate on the number of basis functions for the position basis, and the identity of the basis for the speed. That is:

<div class="render-user render-presenter">

Let's cross-validate on:
- The number of the basis functions of the position basis
- The functional form of the basis for speed
</div>

```{code-cell} ipython3
print(pipe["basis"].basis1.n_basis_funcs)
print(pipe["basis"].basis2)
```

For scikit-learn parameter grids, we use `__` to stand in for `.`:

<div class="render-user render-presenter">

- Construct `param_grid`, using `__` to stand in for `.`
</div>

```{code-cell} ipython3
param_grid = {
    "basis__basis1__n_basis_funcs": [5, 10, 20],
    "basis__basis2": [nmo.basis.MSplineEval(15).set_input_shape(1),
                      nmo.basis.BSplineEval(15).set_input_shape(1),
                      nmo.basis.CyclicBSplineEval(15).set_input_shape(1)],
}
```

<div class="render-user render-presenter">

- Cross-validate as before:
</div>

```{code-cell} ipython3
cv = model_selection.GridSearchCV(pipe, param_grid, cv=cv_folds)
cv.fit(transformer_input, count)
```

<div class="render-user render-presenter">

- Investigate results:
</div>

```{code-cell} ipython3
cv.cv_results_
```

<div class="render-user render-presenter">

- These results are more complicated, so let's use pandas dataframe to make them a bit more understandable:
</div>

```{code-cell} ipython3
cv_df = pd.DataFrame(cv.cv_results_)
cv_df
# helper function for visualization
workshop_utils.plot_heatmap_cv_results(cv_df)
```

<div class="render-user render-presenter">

- Can easily grab the best estimator, the pipeline that did the best:
</div>

```{code-cell} ipython3
best_estim = cv.best_estimator_
best_estim
```

We then visualize the predictions of `best_estim` the same as before.

<div class="render-user render-presenter">

- Visualize model predictions!

</div>

```{code-cell} ipython3
:tags: [render-all]
visualize_model_predictions(best_estim, transformer_input)
```

### Feature selection

Now, finally, we understand almost enough about how scikit-learn works to do feature selection. There's just one more thing to learn: feature masks.

<div class="render-user render-presenter">

- Now one more thing we can do with scikit-learn!
- Each `PopulationGLM` object has a feature mask, which allows us to exclude certain parts of the input
- (By default, everything is included.)
</div>

```{code-cell} ipython3
pipe['glm'].feature_mask
```

```{code-cell} ipython3
workshop_utils.plot_feature_mask(pipe["glm"].feature_mask);
```

<div class="render-user render-presenter">
- We could manually edit feature mask, but have some helper functions -- these are currently being developed, so any feedback is appreciated!
- By default, we include all features:

</div>

```{code-cell} ipython3
m = workshop_utils.create_feature_mask(pipe["basis"], n_neurons=count.shape[1])
workshop_utils.plot_feature_mask(m);
```

<div class="render-user render-presenter">

- Make use of our additive basis to figure out the structure in the input
- Can selectively remove some of the features:
</div>

```{code-cell} ipython3
m = workshop_utils.create_feature_mask(pipe["basis"], ["all", "none"], n_neurons=count.shape[1])
fig=workshop_utils.plot_feature_mask(m);
```

<div class="render-user render-presenter">

- Can construct a set of feature masks that includes / excludes each of the sets of inputs:
</div>

```{code-cell} ipython3
feature_masks = [
    workshop_utils.create_feature_mask(basis, "all", n_neurons=count.shape[1]),
    workshop_utils.create_feature_mask(basis, ["all", "none"], n_neurons=count.shape[1]),
    workshop_utils.create_feature_mask(basis, ["none", "all"], n_neurons=count.shape[1]),
]

workshop_utils.plot_feature_mask(feature_masks, ["All", "Position", "Speed"]);
```

One more wrinkle: the shape of this feature mask depends on the number of basis functions! (The number of features is `basis.n_basis_funcs = basis.basis1.n_basis_funcs + basis.basis2.n_basis_funcs`.) Thus we need to create a new feature mask for each possible arrangement:

<div class="render-user render-presenter">

- One more wrinkle: the shape of this feature mask depends on the number of basis functions!
- Thus, must create a new feature mask for each possible arrangement:
</div>

```{code-cell} ipython3
:tags: [render-all]

param_grid = workshop_utils.create_feature_mask_paramgrid(basis, [5, 10, 20], 
                                                          [8, 16, 32], count.shape[1])
```

<div class="render-user render-presenter">

- Initialize and fit GridSearchCV
</div>

```{code-cell} ipython3
cv = model_selection.GridSearchCV(best_estim, param_grid, cv=cv_folds)
cv.fit(transformer_input, count)
```

<div class="render-user render-presenter">

- Investigate results using pandas
</div>

```{code-cell} ipython3
cv_df = pd.DataFrame(cv.cv_results_)
cv_df
```

<div class="render-user render-presenter">

- For our own sanity, let's create an easier-to-read label:
</div>

```{code-cell} ipython3
:tags: [render-all]

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

<div class="render-user render-presenter">

- And visualize:
</div>

```{code-cell} ipython3
workshop_utils.plot_heatmap_cv_results(cv_df, "feature_mask_label", columns="param_basis__basis2__n_basis_funcs")
```

<div class="render-user">

- What do we see?
</div>

<div class="render-presenter">

From the above plots, we can see that:
- Position matters more than speed.
- Number of basis functions for speed doesn't matter much.
- We don't need many basis functions to represent the position.

</div>

From the above plots, we can see that:
- Position matters more than speed.
- Number of basis functions for speed doesn't matter much.
- We don't need many basis functions to represent the position.

Let's visualize the predictions of the best estimator.

<div class="render-user render-presenter">

- Visualize model predictions!

</div>

```{code-cell} ipython3
:tags: [render-all]
visualize_model_predictions(cv.best_estimator_, transformer_input)
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
