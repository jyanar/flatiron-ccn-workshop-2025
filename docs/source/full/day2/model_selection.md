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

# Model selection

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
import pandas as pd
import nemos as nmo
import seaborn as sns

# some helper plotting functions
from nemos import _documentation_utils as doc_plots
import workshop_utils

# configure pynapple to ignore conversion warning
nap.nap_config.suppress_conversion_warnings = True

# configure plots some
plt.style.use(nmo.styles.plot_style)

from sklearn import model_selection, pipeline
```

```{code-cell} ipython3
path = nmo.fetch.fetch_data("Mouse32-140822.nwb")
data = nap.load_file(path)
data
```

```{code-cell} ipython3
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
```

```{code-cell} ipython3
bin_size = 0.01
count = spikes.count(bin_size, ep=wake_ep)
```

```{code-cell} ipython3
tuning_curves = nap.compute_1d_tuning_curves(
    group=spikes, feature=angle, nb_bins=61, minmax=(0, 2 * np.pi)
)
pref_ang = tuning_curves.idxmax()
count = nap.TsdFrame(
    t=count.t,
    d=count.values[:, np.argsort(pref_ang.values)],
    time_support=count.time_support,
)
```

```{code-cell} ipython3
window_size_sec = 0.8
window_size = int(window_size_sec * count.rate)
```

One additional thing: we're going to add labels

```{code-cell} ipython3
basis = nmo.basis.RaisedCosineLogConv(
    n_basis_funcs=8, window_size=window_size, label="Spikes",
)

# convolve all the neurons
convolved_count = basis.compute_features(count)
```

```{code-cell} ipython3
conn_model = nmo.glm.PopulationGLM(
    regularizer="Ridge",
    solver_name="LBFGS",
    regularizer_strength=0.1
    ).fit(convolved_count, count)
```

## Sklearn

before moving to alternative models, let's talk about integration with sklearn and show to use cross-validation to determine model behavior

crossval on number of basis identity, same model as before, then number of basis functions?

show code for how to crossval on regularization, but don't do it

```{code-cell} ipython3
t_basis = basis.to_transformer()
```

```{code-cell} ipython3
pipe = pipeline.Pipeline([
    ("basis", t_basis),
    ("glm", nmo.glm.PopulationGLM(
    regularizer="Ridge",
    solver_name="LBFGS",
    regularizer_strength=0.1
    )),
])
```

```{code-cell} ipython3
pipe.fit(count, count)
```

```{code-cell} ipython3
pipe.predict(count)
```

```{code-cell} ipython3
# need to re-init because cv.fit fails if the glm has already been fit -- bc of how it reinitializes?
pipe = pipeline.Pipeline([
    ("basis", t_basis),
    ("glm", nmo.glm.PopulationGLM(
    regularizer="Ridge",
    solver_name="LBFGS",
    regularizer_strength=0.1
    )),
])
```

```{code-cell} ipython3
param_grid = {
    "basis__n_basis_funcs": [4, 8, 16, 32],
    "glm__regularizer_strength": [.01, .1, 1, 10],
}
```

```{code-cell} ipython3
cv = model_selection.GridSearchCV(pipe, param_grid, cv=5)
```

```{code-cell} ipython3
cv.fit(count, count)
```

```{code-cell} ipython3
cv_df = pd.DataFrame(cv.cv_results_).pivot(
    index="param_basis__n_basis_funcs",
    columns="param_glm__regularizer_strength",
    values="mean_test_score"
)
```

```{code-cell} ipython3
ax = sns.heatmap(cv_df, annot=True, square=True, linecolor="white", linewidth=0.5, fmt=".3f")
doc_plots.highlight_max_cell(cv_df, ax)
```

explain set input shape

```{code-cell} ipython3
param_grid["basis"] = [t_basis, nmo.basis.RaisedCosineLinearConv(8, window_size).set_input_shape(19).to_transformer()]
param_grid
```

```{code-cell} ipython3
cv = model_selection.GridSearchCV(pipe, param_grid, cv=5)
```

```{code-cell} ipython3
cv.fit(count, count)
```

```{code-cell} ipython3
# make sure colormaps are the same, only one colorbar
# clean up some
def facet_heatmap(index, columns, values, **heatmap_kwargs):
    ax = plt.gca()
    data = heatmap_kwargs.pop("data")
    data = data.pivot(index=index, columns=columns, values=values)
    sns.heatmap(data, **heatmap_kwargs)
    doc_plots.highlight_max_cell(data, ax)

def plot_heatmap_cv_results(cv_df, col=None, index="param_basis__n_basis_funcs", 
                            columns="param_glm__regularizer_strength", values="mean_test_score"):
    fig = sns.FacetGrid(cv_df, col=col, height=5)
    fig.map_dataframe(facet_heatmap, index, columns, values, annot=True, square=True, linecolor="white", linewidth=0.5, fmt=".3f")
```

```{code-cell} ipython3
cv_df = pd.DataFrame(cv.cv_results_)
cv_df["param_basis_label"] = cv_df.param_basis.apply(lambda x: x.label)
```

```{code-cell} ipython3
plot_heatmap_cv_results(cv_df, "param_basis_label")
```

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
workshop_utils.plot_head_direction_tuning_model(tuning_curves, predicted_firing_rate, spikes, angle, threshold_hz=1,
                                                start=8910, end=8960, cmap_label="hsv", pref_ang=pref_ang, model_tuning_curves=model_tuning.clip(0));
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
    ("add_basis", transformed_basis),
    ("glm", nmo.glm.PopulationGLM(solver_name="LBFGS",     
                                  regularizer="Ridge",
                                  regularizer_strength=0.1))
])
```

```{code-cell} ipython3
pipe.fit(X, count)
```

```{code-cell} ipython3
pipe[1].feature_mask.shape
```

```{code-cell} ipython3
plt.imshow(pipe[1].feature_mask)
```
