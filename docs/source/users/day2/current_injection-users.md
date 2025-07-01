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

This notebook can be downloaded as **{nb-download}`current_injection-users.ipynb`**. See the button at the top right to download as markdown or pdf.
:::
# Introduction to GLM
This notebook has had all its explanatory text removed and has not been run.
 It is intended to be downloaded and run locally (or on the provided binder)
 while listening to the presenter's explanation. In order to see the fully
 rendered of this notebook, go [here](../../full/day2/current_injection.md)

Data for this notebook is a patch clamp experiment with a mouse V1 neuron, from the [Allen Brain Atlas](https://celltypes.brain-map.org/experiment/electrophysiology/478498617)

![Allen Brain Atlas view of the data we will analyze.](../../_static/allen_data.png)


## Learning objectives 

- Learn how to explore spiking data and do basic analyses using pynapple
- Learn how to structure data for NeMoS
- Learn how to fit a basic Generalized Linear Model using NeMoS
- Learn how to retrieve the parameters and predictions from a fit GLM for
  intrepetation.
```{code-cell} ipython3
:tags: [render-all]

# Import everything
import jax
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap

import nemos as nmo

# some helper plotting functions
from nemos import _documentation_utils as doc_plots
import workshop_utils

# configure plots some
plt.style.use(nmo.styles.plot_style)
```
## Data Streaming

- Stream the data. Format is [Neurodata Without Borders (NWB) standard](https://nwb-overview.readthedocs.io/en/latest/)
```{code-cell} ipython3
:tags: [render-all]
path = workshop_utils.fetch_data("allen_478498617.nwb")
```
## Pynapple
### Data structures and preparation

- Open the NWB file with [pynapple](https://pynapple-org.github.io/pynapple/)
```{code-cell} ipython3
:tags: [render-all]
data = nap.load_file(path)
print(data)
```

![Annotated view of the data we will analyze.](../../_static/allen_data_annotated.gif)
<!-- this gif created with the following imagemagick command: convert -layers OptimizePlus -delay 100 allen_data_annotated-units.svg allen_data_annotated-epochs.svg allen_data_annotated-stimulus.svg allen_data_annotated-response.svg -loop 0 allen_data_annotated.gif -->

- `stimulus`: injected current, in Amperes, sampled at 20k Hz.
- `response`: the neuron's intracellular voltage, sampled at 20k Hz. We will not use this info in this example.
- `units`: dictionary of neurons, holding each neuron's spike timestamps.
- `epochs`: start and end times of different intervals, defining the experimental structure, specifying when each stimulation protocol began and ended.
```{code-cell} ipython3
:tags: [render-all]
trial_interval_set = data["epochs"]

current = data["stimulus"]
spikes = data["units"]
```
```{code-cell} ipython3
:tags: [render-all]
trial_interval_set
```

- `Noise 1`: epochs of random noise
```{code-cell} ipython3
:tags: [render-all]
noise_interval = trial_interval_set[trial_interval_set.tags == "Noise 1"]
noise_interval
```

- Let's focus on the first epoch.
```{code-cell} ipython3
:tags: [render-all]
noise_interval = noise_interval[0]
noise_interval
```

- `current` : Tsd (TimeSeriesData) : time index + data
```{code-cell} ipython3
:tags: [render-all]
current
```

- `restrict` : restricts a time series object to a set of time intervals delimited by an IntervalSet object
```{code-cell} ipython3
:tags: [render-all]
current = current.restrict(noise_interval)
# convert current from Ampere to pico-amperes, to match the above visualization
# and move the values to a more reasonable range.
current = current * 1e12
current
```

- `TsGroup` : a dictionary-like object holding multiple `Ts` (timeseries) objects with potentially different time indices.
```{code-cell} ipython3
:tags: [render-all]
spikes
```

We can index into the `TsGroup` to see the timestamps for this neuron's spikes:
```{code-cell} ipython3
:tags: [render-all]
spikes[0]
```

Let's restrict to the same epoch `noise_interval`:
```{code-cell} ipython3
:tags: [render-all]
spikes = spikes.restrict(noise_interval)
print(spikes)
spikes[0]
```

Let's visualize the data from this trial:
```{code-cell} ipython3
:tags: [render-all]
fig, ax = plt.subplots(1, 1, figsize=(8, 2))
ax.plot(current, "grey")
ax.plot(spikes.to_tsd([-5]), "|", color="k", ms = 10)
ax.set_ylabel("Current (pA)")
ax.set_xlabel("Time (s)")
```
### Basic analyses

The Generalized Linear Model gives a predicted firing rate. First we can use pynapple to visualize this firing rate for a single trial.

- `count` : count the number of events within `bin_size`
```{code-cell} ipython3
:tags: [render-all]
# bin size in seconds
bin_size = 0.001
# Get spikes for neuron 0
count = spikes[0].count(bin_size)
count
```

Let's convert the spike counts to firing rate :

- `smooth` : convolve with a Gaussian kernel
```{code-cell} ipython3
:tags: [render-all]
# the inputs to this function are the standard deviation of the gaussian in seconds and
# the full width of the window, in standard deviations. So std=.05 and size_factor=20
# gives a total filter size of 0.05 sec * 20 = 1 sec.
firing_rate = count.smooth(std=0.05, size_factor=20)
# convert from spikes per bin to spikes per second (Hz)
firing_rate = firing_rate / bin_size
```
```{code-cell} ipython3
:tags: [render-all]
print(type(firing_rate))
```
```{code-cell} ipython3
:tags: [render-all]
doc_plots.current_injection_plot(current, spikes, firing_rate);
```

What is the relationship between the current and the spiking activity?
[`compute_1d_tuning_curves`](https://pynapple-org.github.io/pynapple/reference/process/tuning_curves/#pynapple.process.tuning_curves.compute_1d_tuning_curves) : compute the firing rate as a function of a 1-dimensional feature.
```{code-cell} ipython3
:tags: [render-all]
tuning_curve = nap.compute_1d_tuning_curves(spikes, current, nb_bins=15)
tuning_curve
```

Let's plot the tuning curve of the neuron.
```{code-cell} ipython3
:tags: [render-all]
doc_plots.tuning_curve_plot(tuning_curve);
```
## NeMoS 
### Preparing data

Get data from pynapple to NeMoS-ready format:

- predictors and spikes must have same number of time points
```{code-cell}
# enter code here
```


- predictors must be 2d, spikes 1d
```{code-cell}
# enter code here
```

### Fitting the model

- GLM objects need regularizers and observation models
```{code-cell}
# enter code here
```


- call fit and retrieve parameters
```{code-cell}
# enter code here
```


- generate and examine model predictions.
```{code-cell}
# enter code here
```


- what do we see?
```{code-cell}
# enter code here
```


- examine tuning curve &mdash; what do we see?
```{code-cell}
# enter code here
```

### Extending the model to use injection history

  - choose a length of time over which the neuron integrates the input current
```{code-cell}
# enter code here
```

```{code-cell} ipython3
:tags: [render-all]
doc_plots.plot_basis();
```

  - define a basis object
```{code-cell}
# enter code here
```


  - create the design matrix
  - examine the features it contains
```{code-cell}
# enter code here
```

```{code-cell} ipython3
:tags: [render-all]

# in this plot, we're normalizing the amplitudes to make the comparison easier --
# the amplitude of these features will be fit by the model, so their un-scaled
# amplitudes is not informative
workshop_utils.plot_current_history_features(binned_current, current_history, basis,
                                             current_history_duration_sec)
```

  - create and fit the GLM
  - examine the parameters
```{code-cell}
# enter code here
```

```{code-cell} ipython3
:tags: [render-all]

print(f"firing_rate(t) = exp({history_model.coef_} * current(t) + {history_model.intercept_})")
```

  - compare the predicted firing rate to the data and the old model
  - what do we see?
```{code-cell} ipython3
:tags: [render-all]

# all this code is the same as above
history_pred_fr = history_model.predict(current_history)
history_pred_fr = history_pred_fr / bin_size
smooth_history_pred_fr = history_pred_fr.dropna().smooth(.05, size_factor=20)
workshop_utils.current_injection_plot(current, spikes, firing_rate,
                                      # compare against the old firing rate
                                      smooth_history_pred_fr, smooth_predicted_fr)
```

  - examine the predicted average firing rate and tuning curve
  - what do we see?
```{code-cell}
# enter code here
```

```{code-cell} ipython3
:tags: [render-all]

# Visualize tuning curve
tuning_curve_history_model = nap.compute_1d_tuning_curves_continuous(smooth_history_pred_fr, current, 15)
fig = doc_plots.tuning_curve_plot(tuning_curve)
fig.axes[0].plot(tuning_curve_history_model, color="tomato", label="glm (current history)")
fig.axes[0].plot(tuning_curve_model, color="tomato", linestyle='--', label="glm (instantaneous current)")
fig.axes[0].legend()
```

  - use log-likelihood to compare models
```{code-cell}
# enter code here
```

### Finishing up

  - what if you want to compare models across datasets?
```{code-cell}
# enter code here
```


- what about spiking?
```{code-cell}
# enter code here
```

## Further Exercises 

- what else can we do?


### Data citation

The data used in this tutorial is from the **Allen Brain Map**, with the [following citation](https://knowledge.brain-map.org/data/1HEYEW7GMUKWIQW37BO/summary):

**Contributors:** Agata Budzillo, Bosiljka Tasic, Brian R. Lee, Fahimeh Baftizadeh, Gabe Murphy, Hongkui Zeng, Jim Berg, Nathan Gouwens, Rachel Dalley, Staci A. Sorensen, Tim Jarsky, Uygar Sümbül Zizhen Yao

**Dataset:** Allen Institute for Brain Science (2020). Allen Cell Types Database -- Mouse Patch-seq [dataset]. Available from brain-map.org/explore/classes/multimodal-characterization.

**Primary publication:** Gouwens, N.W., Sorensen, S.A., et al. (2020). Integrated morphoelectric and transcriptomic classification of cortical GABAergic cells. Cell, 183(4), 935-953.E19. https://doi.org/10.1016/j.cell.2020.09.057

**Patch-seq protocol:** Lee, B. R., Budzillo, A., et al. (2021). Scaled, high fidelity electrophysiological, morphological, and transcriptomic cell characterization. eLife, 2021;10:e65482. https://doi.org/10.7554/eLife.65482

**Mouse VISp L2/3 glutamatergic neurons:** Berg, J., Sorensen, S. A., Miller, J., Ting, J., et al. (2021) Human neocortical expansion involves glutamatergic neuron diversification. Nature, 598(7879):151-158. doi: 10.1038/s41586-021-03813-8