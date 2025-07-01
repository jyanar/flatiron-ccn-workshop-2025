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

This notebook can be downloaded as **{nb-download}`place_cells-users.ipynb`**. See the button at the top right to download as markdown or pdf.

:::
# Model and feature selection with scikit-learn
This notebook has had all its explanatory text removed and has not been run.
 It is intended to be downloaded and run locally (or on the provided binder)
 while listening to the presenter's explanation. In order to see the fully
 rendered of this notebook, go [here](../../full/day2/place_cells.md)

Data for this notebook comes from recordings in the mouse hippocampus while the mouse runs on a linear track, which we [explored yesterday](../day1/phase_precession-users.md).


## Learning objectives

- Review how to use pynapple to analyze neuronal tuning
- Learn how to combine NeMoS basis objects
- Learn how to use NeMoS objects with [scikit-learn](https://scikit-learn.org/) for cross-validation
- Learn how to use NeMoS objects with scikit-learn [pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- Learn how to use cross-validation to perform model and feature selection
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

- Load the data using pynapple.
```{code-cell} ipython3
:tags: [render-all]

path = workshop_utils.fetch_data("Achilles_10252013_EEG.nwb")
data = nap.load_file(path)
data
```

- Extract the spike times and mouse position.
```{code-cell} ipython3
:tags: [render-all]

spikes = data["units"]
position = data["position"]
```


- Restrict data to when animal was traversing the linear track.
```{code-cell} ipython3
:tags: [render-all]

position = position.restrict(data["forward_ep"])
spikes = spikes.restrict(data["forward_ep"])
```


- Restrict neurons to only excitatory neurons, discarding neurons with a low-firing rate.
```{code-cell} ipython3
:tags: [render-all]

spikes = spikes.getby_category("cell_type")["pE"]
spikes = spikes.getby_threshold("rate", 0.3)
```
### Place fields


- Visualize the *place fields*: neuronal firing rate as a function of position.
```{code-cell} ipython3
:tags: [render-all]

place_fields = nap.compute_1d_tuning_curves(spikes, position, 50, position.time_support)
workshop_utils.plot_place_fields(place_fields)
```


- For speed, we're only going to investigate the three neurons highlighted above.
- Bin spikes to counts at 100 Hz.
- Interpolate position to match spike resolution.
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


- Compute animal's speed for each epoch.
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
print(speed.shape)
```


- Compute the tuning curve with pynapple's [`compute_1d_tuning_curves`](https://pynapple.org/generated/pynapple.process.tuning_curves.html#pynapple.process.tuning_curves.compute_1d_tuning_curves)
```{code-cell}
# enter code here
```



```{code-cell} ipython3
:tags: [remove-cell, skip-execution]
# compute tuning curve here
tc_speed = 
```


- Visualize the position and speed tuning for these neurons.
```{code-cell} ipython3
:tags: [render-all]

fig = workshop_utils.plot_position_speed(position, speed, place_fields, tc_speed, neurons);
```


These neurons all show both position and speed tuning, and we see that the animal's speed and position are highly correlated. We're going to build a GLM to predict neuronal firing rate -- which variable should we use? Is the speed tuning just epiphenomenal?
## NeMoS
### Basis evaluation


- Create a separate basis object for each model input.
- Visualize the basis objects.
```{code-cell}
# enter code here
```



- Combine the two basis objects into a single "additive basis"
```{code-cell}
# enter code here
```



- Create the design matrix!
- Notice that, since we passed the basis pynapple objects, we got one back, preserving the time stamps.
- `X` has the same number of time points as our input position and speed, but 25 columns. The columns come from  `n_basis_funcs` from each basis (10 for position, 15 for speed).
```{code-cell}
# enter code here
```

### Model learning


- Initialize `PopulationGLM`
- Use the "LBFGS" solver and pass `{"tol": 1e-12}` to `solver_kwargs`.
- Fit the data, passing the design matrix and spike counts to the glm object.


```{code-cell} ipython3
:tags: [skip-execution]

# initialize 
glm =

# and fit
```
### Prediction


- Use `predict` to check whether our GLM has captured each neuron's speed and position tuning.
- Remember to convert the predicted firing rate to spikes per second!


```{code-cell} ipython3
:tags: [skip-execution]

# predict the model's firing rate
predicted_rate = 

# same shape as the counts we were trying to predict
print(predicted_rate.shape, count.shape)

# compute the position and speed tuning curves using the predicted firing rate.
glm_pf = 
glm_speed = 
```


- Compare model and data tuning curves together. The model did a pretty good job!
```{code-cell} ipython3
:tags: [render-all]

workshop_utils.plot_position_speed_tuning(place_fields, tc_speed, glm_pf, glm_speed);
```


To make our lives easier, let's create a helper function that wraps the above
lines, because we're going to be visualizing our model predictions a lot.
```{code-cell}
# enter code here
```

## Scikit-learn
### How to know when to regularize?


- How do we decide when to use regularization?
- Cross-validation allows you to fairly compare different models on the same dataset.
- NeMoS makes use of [scikit-learn](https://scikit-learn.org/), the standard machine learning library in python.
- Define [parameter grid](https://scikit-learn.org/stable/modules/grid_search.html#grid-search) to search over.
- Anything not specified in grid will be kept constant.
```{code-cell}
# enter code here
```



- Initialize scikit-learn's [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV) object.
```{code-cell}
# enter code here
```



- We interact with this in a very similar way to the glm object.
- In particular, call `fit` with same arguments:
```{code-cell}
# enter code here
```



- We got a warning because we didn't specify the regularizer strength, so we just fell back on default value.
- Let's investigate results:
```{code-cell}
# enter code here
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
### Select basis


- You can (and should) do something similar to determine how many basis functions you need for each input.
- NeMoS basis objects are not scikit-learn-compatible right out of the box.
- But we have provided a simple method to make them so:
```{code-cell}
# enter code here
```



- This gives the basis object the `transform` method, which is equivalent to `compute_features`.
- However, transformers have some limits:
```{code-cell}
# enter code here
```



- Transformers only accept 2d inputs, whereas nemos basis objects can accept inputs of any dimensionality.
- In order to tell nemos how to reshape the 2d matrix that is the input of `transform` to whatever the basis accepts, you need to call `set_input_shape`:
```{code-cell}
# enter code here
```



- Then you can call transform on the 2d input as expected.
```{code-cell}
# enter code here
```



- You can, equivalently, call `compute_features` *before* turning the basis into a transformer. Then we cache the shape for future use:
```{code-cell}
# enter code here
```



- Create a single TsdFrame to hold all our inputs:
```{code-cell} ipython3
:tags: [render-all]

transformer_input = nap.TsdFrame(
    t=position.t,
    d=np.stack([position.d, speed.d], 1),
    time_support=position.time_support,
    columns=["position", "speed"],
)
```


- Pass this input to our transformed additive basis:
```{code-cell}
# enter code here
```

### Pipelines


- If we want to cross-validate over the basis, we need more one more step: combining the basis and the GLM into a single scikit-learn estimator.
- [Pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) to the rescue!
```{code-cell}
# enter code here
```



- Pipeline runs `basis.transform`, then passes that output to `glm`, so we can do everything in a single line:
```{code-cell}
# enter code here
```



- Visualize model predictions!
```{code-cell}
# enter code here
```

### Cross-validating on the basis

Now that we have our pipeline estimator, we can cross-validate on any of its parameters!
```{code-cell}
# enter code here
```



Let's cross-validate on:
- The number of the basis functions of the position basis
- The functional form of the basis for speed
```{code-cell}
# enter code here
```



- Construct `param_grid`, using `__` to stand in for `.`
```{code-cell}
# enter code here
```



- Cross-validate as before:
```{code-cell}
# enter code here
```



- Investigate results:
```{code-cell}
# enter code here
```



- These results are more complicated, so let's use pandas dataframe to make them a bit more understandable:
```{code-cell}
# enter code here
```



- Can easily grab the best estimator, the pipeline that did the best:
```{code-cell}
# enter code here
```



- Visualize model predictions!
```{code-cell} ipython3
:tags: [render-all]
visualize_model_predictions(best_estim, transformer_input)
```
### Feature selection


- Now one more thing we can do with scikit-learn!
- Each `PopulationGLM` object has a feature mask, which allows us to exclude certain parts of the input
- Feature mask shape: `X.shape[1]` (number of columns in the design matrix) by `n_neurons` (number of neurons we're trying to predict)
- (By default, everything is included.)
```{code-cell}
# enter code here
```


- We could manually edit feature mask, but have some helper functions -- these are currently being developed, so any feedback is appreciated!
- By default, we include all features:
```{code-cell}
# enter code here
```



- Make use of our additive basis to figure out the structure in the input
- Can selectively remove some of the features:
```{code-cell}
# enter code here
```



- Can construct a set of feature masks that includes / excludes each of the sets of inputs:
```{code-cell}
# enter code here
```



- One more wrinkle: the shape of this feature mask depends on the number of basis functions!
- Thus, must create a new feature mask for each possible arrangement:
```{code-cell} ipython3
:tags: [render-all]

param_grid = workshop_utils.create_feature_mask_paramgrid(basis, [5, 10, 20], 
                                                          [8, 16, 32], count.shape[1])
```


- Initialize and fit GridSearchCV
```{code-cell}
# enter code here
```



- Investigate results using pandas
```{code-cell}
# enter code here
```



- For our own sanity, let's create an easier-to-read label:
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


- And visualize:
```{code-cell}
# enter code here
```



- What do we see?


- Visualize model predictions!
```{code-cell} ipython3
:tags: [render-all]
visualize_model_predictions(cv.best_estimator_, transformer_input)
```
## Conclusion
## References



The data in this tutorial comes from [Grosmark, Andres D., and György Buzsáki. "Diversity in neural firing dynamics supports both rigid and learned hippocampal sequences." Science 351.6280 (2016): 1440-1443](https://www.science.org/doi/full/10.1126/science.aad1935).