---
jupytext:
  formats: md:myst
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

This notebook can be downloaded as **{nb-download}`model_selection.ipynb`**. See the button at the top right to download as markdown or pdf.

:::

# Model selection and comparison

In this final selection, we're going to learn a little bit more about how to do model selection with nemos. This is a very big topic, which we're only going to cover a small portion of.

nemos does not implement the types of algorithms or machinery that are used for performing model selection. Instead, to avoid re-inventing the wheel, we have written nemos objects so that they are compatible with [scikit-learn](https://scikit-learn.org/stable/index.html) (also called "sklearn"), the standard library for machine learning in python. Thus, we can take advantage of their extensive library! In this notebook, we're going to focus on using cross-validation for model selection and comparison.

:::{note}
While we've done our best, we haven't tested all possible ways of using nemos with scikit-learn. If you try to use it and discover unexpected behavior or an error, please [let us know!](https://github.com/flatironinstitute/nemos/issues) We'd really appreciate it.
:::

## Learning objectives

- Learn how to use nemos objects with [scikit-learn](https://scikit-learn.org/stable/index.html).
- Learn how to use scikit-learn's pipeline and cross-validation objects.
- Learn how to compare different models using cross-validation.

First, some imports and configuration. Most of this should look familiar by now:

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
import pandas as pd
import nemos as nmo

# some helper plotting functions
from nemos import _documentation_utils as doc_plots
import workshop_utils

# configure pynapple to ignore conversion warning
nap.nap_config.suppress_conversion_warnings = True

# configure plots some
plt.style.use(nmo.styles.plot_style)

from sklearn import model_selection, pipeline

cv_folds = 3
```

To allow us to focus on scikit-learn and model selection, we're going to use the head direction dataset that Edoardo was just analyzing. The following code loads in this dataset and fits the fully-connected model (using bases) that Edoardo walked us through:

```{code-cell} ipython3
path = nmo.fetch.fetch_data("Mouse32-140822.nwb")
data = nap.load_file(path)
data

epochs = data["epochs"]
wake_ep = epochs[epochs.tags == "wake"]
wake_ep = nap.IntervalSet(
    start=wake_ep.start[0], end=wake_ep.start[0] + 3 * 60
)

spikes = data["units"]
spikes = spikes.getby_category("location")["adn"]
spikes = spikes.restrict(wake_ep).getby_threshold("rate", 1.0)

angle = data["ry"]
angle = angle.restrict(wake_ep)

bin_size = 0.01
count = spikes.count(bin_size, ep=wake_ep)

tuning_curves = nap.compute_1d_tuning_curves(
    group=spikes, feature=angle, nb_bins=61, minmax=(0, 2 * np.pi)
)
pref_ang = tuning_curves.idxmax()
count = nap.TsdFrame(
    t=count.t,
    d=count.values[:, np.argsort(pref_ang.values)],
    time_support=count.time_support,
)

window_size_sec = 0.8
window_size = int(window_size_sec * count.rate)

basis = nmo.basis.RaisedCosineLogConv(
    n_basis_funcs=8, window_size=window_size
)
basis_time, basis_kernel = basis.evaluate_on_grid(80)
# convolve all the neurons
convolved_count = basis.compute_features(count)

conn_model = nmo.glm.PopulationGLM(
    regularizer="Ridge",
    solver_name="LBFGS",
    regularizer_strength=0.1
    ).fit(convolved_count, count)
conn_firing_rate = conn_model.predict(convolved_count) * convolved_count.rate
```

## Why regularize?

There are many types of "model selection" problems one can investigate with cross-validation. To start, let's consider something we swept under the rug in the last notebook: why did Edoardo use a regularizer?

Let's see what our solution would've looked like without regularization. Let's create and fit a fully-connected model, so it uses the design matrix we set up above, but without any regularization. You know how to do this by now:

```{code-cell} ipython3
unreg_model = nmo.glm.PopulationGLM(solver_name="LBFGS").fit(convolved_count, count)
```

Now that we've fit the data, let's generate the predictions and investigate them.

```{code-cell} ipython3
predicted_firing_rate = unreg_model.predict(convolved_count) * convolved_count.rate

workshop_utils.plot_head_direction_tuning_model(tuning_curves, spikes, angle, 
                                                predicted_firing_rate, threshold_hz=1,
                                                start=8910, end=8960, cmap_label="hsv",);
```

```{code-cell} ipython3
fig = doc_plots.plot_rates_and_smoothed_counts(
    count[:, 0],
    {"All-to-all: regularized": conn_firing_rate[:, 0],
     "All-to-all: unregularized": predicted_firing_rate[:, 0]}
)
```

```{code-cell} ipython3
tuning = nap.compute_1d_tuning_curves_continuous(predicted_firing_rate,
                                                 feature=angle,
                                                 nb_bins=61,
                                                 minmax=(0, 2 * np.pi))
```

```{code-cell} ipython3
weights = basis.split_by_feature(unreg_model.coef_, 0)['RaisedCosineLogConv']
```

```{code-cell} ipython3
responses = np.einsum("jki,tk->ijt", weights, basis_kernel)
```

```{code-cell} ipython3
fig = doc_plots.plot_coupling(responses, tuning)
```

## Sklearn

so how did we pick the regularizer Edoardo showed in the last notebook? cross-validation!

to do this, we use sklearn

```{code-cell} ipython3
t_basis = basis.to_transformer()
```

```{code-cell} ipython3
t_basis.transform(count)
```

```{code-cell} ipython3
np.array_equal(basis.compute_features(count), t_basis.transform(count), True)
```

```{code-cell} ipython3
pipe = pipeline.Pipeline([
    ("basis", t_basis),
    ("glm", nmo.glm.PopulationGLM()),
])
```

```{code-cell} ipython3
pipe.fit(count, count)
```

```{code-cell} ipython3
pipe.predict(count)
```

```{code-cell} ipython3
param_grid = {
    "glm": [nmo.glm.PopulationGLM(solver_name="LBFGS"),
            nmo.glm.PopulationGLM(regularizer_strength=.1, regularizer=nmo.regularizer.Ridge(), solver_name="LBFGS")],
}
```

```{code-cell} ipython3
cv = model_selection.GridSearchCV(pipe, param_grid, cv=cv_folds)
```

```{code-cell} ipython3
cv.fit(count, count)
```

```{code-cell} ipython3
cv.cv_results_
```

```{code-cell} ipython3
cv.best_estimator_
```

But maybe we picked the basis unfairly?

```{code-cell} ipython3
pipe['basis'].n_basis_funcs
```

```{code-cell} ipython3
t_basis.n_basis_funcs
```

```{code-cell} ipython3
param_grid = {
    "basis__n_basis_funcs": [8, 16],
    "glm": [nmo.glm.PopulationGLM(solver_name="LBFGS"),
            nmo.glm.PopulationGLM(regularizer_strength=.1, regularizer=nmo.regularizer.Ridge(), solver_name="LBFGS")],
}
```

```{code-cell} ipython3
cv = model_selection.GridSearchCV(pipe, param_grid, cv=cv_folds)
```

```{code-cell} ipython3
cv.fit(count, count)
```

```{code-cell} ipython3
cv.cv_results_
```

```{code-cell} ipython3
cv_df = pd.DataFrame(cv.cv_results_)
cv_df
```

```{code-cell} ipython3
workshop_utils.plot_heatmap_cv_results(cv_df, rows='param_basis__n_basis_funcs', columns='param_glm')
```

can see that regularization matters more as the number of basis functions and thus the number of parameters to fit increases

we can also try out different regularization strength:

```{code-cell} ipython3
# unnecessarily complex!
param_grid = {
    "basis__n_basis_funcs": [8, 16],
    "glm": [nmo.glm.PopulationGLM(solver_name="LBFGS"),
            nmo.glm.PopulationGLM(regularizer_strength=.1, regularizer=nmo.regularizer.Ridge(), solver_name="LBFGS"),
            nmo.glm.PopulationGLM(regularizer_strength=1, regularizer=nmo.regularizer.Ridge(), solver_name="LBFGS")],
}
```

```{code-cell} ipython3
# not quite!
param_grid = {
    "basis__n_basis_funcs": [8, 16],
    "glm__regularizer": [nmo.regularizer.UnRegularized(), nmo.regularizer.Ridge()],
    "glm__regularizer_strength": [.1, 10],
}
```

```{code-cell} ipython3
# much better!
param_grid = [
    {"basis__n_basis_funcs": [8, 16],
     "glm__regularizer": [nmo.regularizer.UnRegularized()],},
    {"basis__n_basis_funcs": [8, 16],
     "glm__regularizer": [nmo.regularizer.Ridge()],
     "glm__regularizer_strength": [.1, 10]}
]
```

```{code-cell} ipython3
pipe["basis"].transform(count)
```

```{code-cell} ipython3
cv = model_selection.GridSearchCV(pipe, param_grid, cv=cv_folds)
```

```{code-cell} ipython3
cv.fit(count, count)
```

```{code-cell} ipython3
cv.cv_results_
```

```{code-cell} ipython3
cv_df = pd.DataFrame(cv.cv_results_)
cv_df
```

```{code-cell} ipython3
workshop_utils.plot_heatmap_cv_results(cv_df, rows='param_basis__n_basis_funcs', columns='param_glm')
```

:::{note}
further possibilities:
- try different basis objects (explain `set_input_shape`)
- different regularizers, solvers
:::

+++

## let's add behavior

might think let's just use the angle as a predictor but that doesn't work -- that's equivalent to saying the firing rate should be exp(c*angle) for some parameter c. that's nonsensical

so instead we use a basis. cyclic b spline!

```{code-cell} ipython3
cyclic_basis = nmo.basis.CyclicBSplineEval(10, label="HeadDirection")
```

```{code-cell} ipython3
plt.plot(*cyclic_basis.evaluate_on_grid(100))
```

Explain more about basis here?

```{code-cell} ipython3
angle_interp = np.expand_dims(angle.interpolate(count), 1)
```

```{code-cell} ipython3
pipe = pipeline.Pipeline([
    ("basis", cyclic_basis.set_input_shape(1).to_transformer()),
    ("glm", nmo.glm.PopulationGLM(solver_name="LBFGS")),
])
```

```{code-cell} ipython3
pipe.fit(angle_interp, count)
```

```{code-cell} ipython3
basis_angs, cyclic_basis_kernel = cyclic_basis.evaluate_on_grid(100)
model_tuning = np.einsum("kj,tk->tj", pipe[1].coef_, cyclic_basis_kernel)
model_tuning = pd.DataFrame(data=model_tuning, index=2*np.pi * basis_angs, columns=pref_ang.sort_values().index)
```

```{code-cell} ipython3
predicted_firing_rate = pipe.predict(angle_interp) * angle_interp.rate

# use pynapple for time axis for all variables plotted for tick labels in imshow
workshop_utils.plot_head_direction_tuning_model(tuning_curves, spikes, angle, 
                                                predicted_firing_rate, threshold_hz=1,
                                                start=8910, end=8960, cmap_label="hsv",
                                                pref_ang=pref_ang,
                                                model_tuning_curves=model_tuning.clip(0));
```

looks very similar to what we saw before! 

let's use cv to compare them!

comparing models with different inputs is not straightforward -- need to build a big combined model and use feature mask

```{code-cell} ipython3
add_basis = basis + cyclic_basis
```

```{code-cell} ipython3
add_basis.compute_features(count, angle_interp)
transformed_basis = add_basis.to_transformer()
```

```{code-cell} ipython3
X = np.concat([count, angle_interp], 1)
```

```{code-cell} ipython3
pipe = pipeline.Pipeline([
    ("basis", transformed_basis),
    ("glm", nmo.glm.PopulationGLM(solver_name="LBFGS",     
                                  regularizer="Ridge",
                                  regularizer_strength=0.1))
])
```

```{code-cell} ipython3
pipe.fit(X, count)
```

```{code-cell} ipython3
pipe[1].feature_mask
```

```{code-cell} ipython3
workshop_utils.plot_feature_mask(pipe[1].feature_mask);
```

could manually edit feature mask, but have some helper functions -- these are currently being developed, so any feedback is appreciated!

```{code-cell} ipython3
m = workshop_utils.create_feature_mask(add_basis, n_neurons=19)
workshop_utils.plot_feature_mask(m);
```

```{code-cell} ipython3
m = workshop_utils.create_feature_mask(add_basis, ["all", "none"], n_neurons=19)
fig=workshop_utils.plot_feature_mask(m);
```

```{code-cell} ipython3
feature_masks = [
    workshop_utils.create_feature_mask(add_basis, "all", n_neurons=19),
    workshop_utils.create_feature_mask(add_basis, ["all", "none"], n_neurons=19),
    workshop_utils.create_feature_mask(add_basis, ["none", "all"], n_neurons=19),
]
```

```{code-cell} ipython3
workshop_utils.plot_feature_mask(feature_masks);
```

```{code-cell} ipython3
param_grid = {
    "glm__feature_mask": feature_masks,
    "glm__regularizer": ["Ridge", "UnRegularized"]
}
```

```{code-cell} ipython3
cv = model_selection.GridSearchCV(pipe, param_grid, cv=cv_folds)
```

```{code-cell} ipython3
cv.fit(X, count)
```

```{code-cell} ipython3
cv_df = pd.DataFrame(cv.cv_results_)
cv_df["feature_masks"] = ["all","all", "spikes-only","spikes-only", "behavior-only","behavior-only"]
```

```{code-cell} ipython3
workshop_utils.plot_heatmap_cv_results(cv_df, rows="param_glm__regularizer", columns="feature_masks")
```

The rest here might not be needed

```{code-cell} ipython3
cv.best_estimator_[1].coef_.shape
```

```{code-cell} ipython3
coefs = cv.best_estimator_[0].split_by_feature(cv.best_estimator_[1].coef_, 0)
```

```{code-cell} ipython3
for k, v in coefs.items():
    print(k, v.shape)
```

```{code-cell} ipython3
responses = np.einsum("jki,tk->ijt", coefs["Spikes"], basis_kernel)
print(responses.shape)
```

```{code-cell} ipython3
predicted_firing_rate = cv.best_estimator_.predict(X) * X.rate
```

```{code-cell} ipython3
tuning = nap.compute_1d_tuning_curves_continuous(predicted_firing_rate,
                                                 feature=angle,
                                                 nb_bins=61,
                                                 minmax=(0, 2 * np.pi))
```

```{code-cell} ipython3
tuning.columns = pref_ang.sort_values().index
```

```{code-cell} ipython3
# use pynapple for time axis for all variables plotted for tick labels in imshow
workshop_utils.plot_head_direction_tuning_model(tuning_curves, spikes, angle, 
                                                predicted_firing_rate, threshold_hz=1,
                                                start=8910, end=8960, cmap_label="hsv", 
                                                pref_ang=pref_ang, 
                                                );
```

```{code-cell} ipython3
fig = doc_plots.plot_coupling(responses, tuning)
```
