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

This notebook can be downloaded as **{nb-download}`head_direction.ipynb`**. See the button at the top right to download as markdown or pdf.

:::

(head-direction-tutorial)=
# Fit head-direction population

## Learning objectives

- Learn how to add history-related predictors to NeMoS GLM
- Learn about NeMoS [`Basis`](nemos_basis) objects
- Learn how to use [`Basis`](nemos_basis) objects with convolution

<div class="render">
- Include history-related predictors to NeMoS GLM.
- Reduce over-fitting with `Basis`.
- Learn functional connectivity.
</div>

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap

import nemos as nmo

# some helper plotting functions
from nemos import _documentation_utils as doc_plots
import workshop_utils

# configure pynapple to ignore conversion warning
nap.nap_config.suppress_conversion_warnings = True

# configure plots some
plt.style.use(nmo.styles.plot_style)
```

## Data Streaming

Here we load the data from OSF. The data is a NWB file.

<div class="render">
- Fetch the data.
</div>

```{code-cell} ipython3
path = workshop_utils.fetch.fetch_data("Mouse32-140822.nwb")
```

## Pynapple
We are going to open the NWB file with pynapple.

<div class="render">
- `load_file` : open the NWB file and give a preview.
</div>

```{code-cell} ipython3
data = nap.load_file(path)

data
```

Get spike timings

<div class="notes">
- Load the units
</div>

```{code-cell} ipython3
spikes = data["units"]

spikes
```

Get the behavioural epochs (in this case, sleep and wakefulness)

<div class="notes">
- Load the epochs and take only wakefulness
</div>

```{code-cell} ipython3
epochs = data["epochs"]
wake_epochs = epochs[epochs.tags == "wake"]
```

Get the tracked orientation of the animal

<div class="notes">
- Load the angular head-direction of the animal (in radians)
</div>

```{code-cell} ipython3
angle = data["ry"]
```

This cell will restrict the data to what we care about i.e. the activity of head-direction neurons during wakefulness.

<div class="notes">
- Select only those units that are in ADn
- Restrict the activity to wakefulness (both the spiking activity and the angle)
</div>

```{code-cell} ipython3
spikes = spikes[spikes.location == "adn"]

spikes = spikes.restrict(wake_epochs).getby_threshold("rate", 1.0)
angle = angle.restrict(wake_epochs)
```

First let's check that they are head-direction neurons.

<div class="notes">
- Compute tuning curves as a function of head-direction
</div>

```{code-cell} ipython3
tuning_curves = nap.compute_1d_tuning_curves(
    group=spikes, feature=angle, nb_bins=61, minmax=(0, 2 * np.pi)
)
```

Each row indicates an angular bin (in radians), and each column corresponds to a single unit.
Let's plot the tuning curve of the first two neurons.

<div class="notes">
- Plot the tuning curves.
</div>

```{code-cell} ipython3
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(tuning_curves.iloc[:, 0])
ax[0].set_xlabel("Angle (rad)")
ax[0].set_ylabel("Firing rate (Hz)")
ax[1].plot(tuning_curves.iloc[:, 1])
ax[1].set_xlabel("Angle (rad)")
plt.tight_layout()
```

Before using NeMoS, let's explore the data at the population level.

Let's plot the preferred heading

<div class="notes">
- Let's visualize the data at the population level.
</div>

```{code-cell} ipython3
fig = doc_plots.plot_head_direction_tuning(
    tuning_curves, spikes, angle, threshold_hz=1, start=8910, end=8960
)
```

As we can see, the population activity tracks very well the current head-direction of the animal.
**Question : are neurons constantly tuned to head-direction and can we use it to predict the spiking activity of each neuron based only on the activity of other neurons?**

To fit the GLM faster, we will use only the first 3 min of wake

<div class="notes">
- Define a `wake_ep` IntervalSet with the first 3 minutes of wakefulness (to speed up model fitting).
</div>

```{code-cell} ipython3
wake_ep = nap.IntervalSet(
    start=wake_epochs.start[0], end=wake_epochs.start[0] + 3 * 60
)
```

To use the GLM, we need first to bin the spike trains. Here we use pynapple

<div class="notes">
- bin the spike trains in 10 ms bin (`count = ...`).
</div>

```{code-cell} ipython3
bin_size = 0.01
count = spikes.count(bin_size, ep=wake_ep)
```

Here we are going to rearrange neurons order based on their preferred directions.

<div class="notes">
- sort the neurons by their preferred direction using pandas:
    - Preferred angle:  `pref_ang = tuning_curves.idxmax()`.
    - Define a new `count` TsdFrame, sorting the columns according to `pref_ang`.
</div>

```{code-cell} ipython3
pref_ang = tuning_curves.idxmax()

count = nap.TsdFrame(
    t=count.t,
    d=count.values[:, np.argsort(pref_ang.values)],
)
```

## NeMoS 
It's time to use NeMoS. Our goal is to estimate the pairwise interaction between neurons.
This can be quantified with a GLM if we use the recent population spike history to predict the current time step.
### Self-Connected Single Neuron
To simplify our life, let's see first how we can model spike history effects in a single neuron.
The simplest approach is to use counts in fixed length window $i$, $y_{t-i}, \dots, y_{t-1}$ to predict the next
count $y_{t}$. Let's plot the count history,

<div class="notes">
- Start with modeling a self-connected single neuron.
- Select a neuron (call the variable `neuron_count`).
- Select the first 1.2 seconds for visualization. (call the epoch `epoch_one_spk`).
</div>

```{code-cell} ipython3
# select a neuron's spike count time series
neuron_count = count[:, 0]

# restrict to a smaller time interval
epoch_one_spk = nap.IntervalSet(
    start=count.time_support.start[0], end=count.time_support.start[0] + 1.2
)
```

#### Features Construction
Let's fix the spike history window size that we will use as predictor.

<div class="notes">
- Fix a history window of 800ms (0.8 seconds).
- Plot the result using `doc_plots.plot_history_window`
</div>

```{code-cell} ipython3
# KEEP-CODE

# set the size of the spike history window in seconds
window_size_sec = 0.8

doc_plots.plot_history_window(neuron_count, epoch_one_spk, window_size_sec);
```

For each time point, we shift our window one bin at the time and vertically stack the spike count history in a matrix.
Each row of the matrix will be used as the predictors for the rate in the next bin (red narrow rectangle in
the figure).

<div class="notes">
- By shifting the time window we can predict new count bins.
- Concatenating all the shifts, we form our feature matrix.
</div>

```{code-cell} ipython3
doc_plots.run_animation(neuron_count, epoch_one_spk.start[0])
```

If $t$ is smaller than the window size, we won't have a full window of spike history for estimating the rate.
One may think of padding the window (with zeros for example) but this may generate weird border artifacts.
To avoid that, we can simply restrict our analysis to times $t$ larger than the window and NaN-pad earlier
time-points;

You can construct this feature matrix with the [`HistoryConv`](https://nemos--282.org.readthedocs.build/en/282/generated/basis/nemos.basis.HistoryConv.html#nemos.basis.HistoryConv) basis.

<div class="notes">
- This is equivalent to convolving `count` with an identity matrix.
- That's what NeMoS `HistoryConv` basis is for: 
    - Convert the window size in number of bins (call it `window_size`)
    - Define an `HistoryConv` basis covering this window size (call it `history_basis`).
    - Create the feature matrix with `history_basis.compute_features` (call it `input_feature`).
</div>

```{code-cell} ipython3
# convert the prediction window to bins (by multiplying with the sampling rate)
window_size = int(window_size_sec * neuron_count.rate)

# define the history bases
history_basis = nmo.basis.HistoryConv(window_size)

# create the feature matrix
input_feature = history_basis.compute_features(neuron_count)
```

<div class="notes">
- NeMoS NaN pads if there aren't enough samples to predict the counts. 
</div>

```{code-cell} ipython3
# print the NaN indices along the time axis
print("NaN indices:\n", np.where(np.isnan(input_feature[:, 0]))[0])
```

The binned counts originally have shape "number of samples", we should check that the
dimension are matching our expectation

<div class="notes">
- Check the shape of the counts and features.
</div>

```{code-cell} ipython3
print(f"Time bins in counts: {neuron_count.shape[0]}")
print(f"Convolution window size in bins: {window_size}")
print(f"Feature shape: {input_feature.shape}")
print(f"Feature shape: {input_feature.shape}")
```

We can visualize the output for a few time bins

<div class="notes">
- Plot the convolution output with `workshop_utils.plot_features`.
</div>

```{code-cell} ipython3
# KEEP-CODE
suptitle = "Input feature: Count History"
neuron_id = 0
workshop_utils.plot_features(input_feature, count.rate, suptitle);
```

As you may see, the time axis is backward, this happens because under the hood, the basis is using the convolution operator which flips the time axis.
This is equivalent, as we can interpret the result as how much a spike will affect the future rate.
In the previous tutorial our feature was 1-dimensional (just the current), now
instead the feature dimension is 80, because our bin size was 0.01 sec and the window size is 0.8 sec.
We can learn these weights by maximum likelihood by fitting a GLM.




#### Fitting the Model

When working a real dataset, it is good practice to train your models on a chunk of the data and
use the other chunk to assess the model performance. This process is known as "cross-validation".
There is no unique strategy on how to cross-validate your model; What works best
depends on the characteristic of your data (time series or independent samples,
presence or absence of trials...), and that of your model. Here, for simplicity use the first
half of the wake epochs for training and the second half for testing. This is a reasonable
choice if the statistics of the neural activity does not change during the course of
the recording. We will learn about better cross-validation strategies with other
examples.

<div class="notes">
- Split your epochs in two for validation purposes:
    - Define two `IntervalSet`s, each with half of the `input_feature.time_support` duration
</div>

```{code-cell} ipython3
# KEEP-CODE

# construct the train and test epochs
duration = input_feature.time_support.tot_length("s")
start = input_feature.time_support["start"]
end = input_feature.time_support["end"]
first_half = nap.IntervalSet(start, start + duration / 2)
second_half = nap.IntervalSet(start + duration / 2, end)
```

Fit the glm to the first half of the recording and visualize the ML weights.

<div class="notes">
- Fit a GLM to the first half. (Call it `model`).
</div>

```{code-cell} ipython3
# define the GLM object
model = nmo.glm.GLM(solver_name="LBFGS")

# Fit over the training epochs
model.fit(
    input_feature.restrict(first_half),
    neuron_count.restrict(first_half)
)
```

<div class="notes">
- Plot the weights (`model.coef_`).
</div>

```{code-cell} ipython3
# KEEP-CODE

plt.figure()
plt.title("Spike History Weights")
plt.plot(np.arange(window_size) / count.rate, np.squeeze(model.coef_), lw=2, label="GLM raw history 1st Half")
plt.axhline(0, color="k", lw=0.5)
plt.xlabel("Time From Spike (sec)")
plt.ylabel("Kernel")
plt.legend()
```

The response in the previous figure seems noise added to a decay, therefore the response
can be described with fewer degrees of freedom. In other words, it looks like we
are using way too many weights to describe a simple response.
If we are correct, what would happen if we re-fit the weights on the other half of the data?

#### Inspecting the results

<div class="notes">
- Fit on the other half. (Call it `model_second_half`)
</div>

```{code-cell} ipython3
# fit on the test set

model_second_half = nmo.glm.GLM(solver_name="LBFGS")
model_second_half.fit(
    input_feature.restrict(second_half),
    neuron_count.restrict(second_half)
)
```

<div class="notes">
- Compare results.
</div>

```{code-cell} ipython3
# KEEP-CODE

plt.figure()
plt.title("Spike History Weights")
plt.plot(np.arange(window_size) / count.rate, np.squeeze(model.coef_),
         label="GLM raw history 1st Half", lw=2)
plt.plot(np.arange(window_size) / count.rate,  np.squeeze(model_second_half.coef_),
         color="orange", label="GLM raw history 2nd Half", lw=2)
plt.axhline(0, color="k", lw=0.5)
plt.xlabel("Time From Spike (sec)")
plt.ylabel("Kernel")
plt.legend()
```

What can we conclude?

The fast fluctuations are inconsistent across fits, indicating that
they are probably capturing noise, a phenomenon known as over-fitting;
On the other hand, the decaying trend is fairly consistent, even if
our estimate is noisy. You can imagine how things could get
worst if we needed a finer temporal resolution, such 1ms time bins
(which would require 800 coefficients instead of 80).
What can we do to mitigate over-fitting now?

(head_direction_reducing_dimensionality)=
#### Reducing feature dimensionality
Let's see how to use NeMoS' `basis` module to reduce dimensionality and avoid over-fitting!
For history-type inputs, we'll use again the raised cosine log-stretched basis,
[Pillow et al., 2005](https://www.jneurosci.org/content/25/47/11003).

<div class="notes">
- Visualize the raised cosine basis.
</div>

```{code-cell} ipython3
# KEEP-CODE

doc_plots.plot_basis();
```

:::{note}

We provide a handful of different choices for basis functions, and
selecting the proper basis function for your input is an important
analytical step. We will eventually provide guidance on this choice, but
for now we'll give you a decent choice.
:::

We can initialize the `RaisedCosineLogConv` by providing the number of basis functions and the window size for the convolution. With more basis functions, we'll be able to represent the effect of the corresponding input with the higher precision, at the cost of adding additional parameters.

<div class="notes">
- Define the basis `RaisedCosineLogConv`and name it `basis`. 
- Basis parameters:
    - 8 basis functions.
    - Window size of 0.8sec.
</div>

```{code-cell} ipython3
# a basis object can be instantiated in "conv" mode for convolving  the input.
basis = nmo.basis.RaisedCosineLogConv(
    n_basis_funcs=8, window_size=window_size
)
```

Our spike history predictor was huge: every possible 80 time point chunk of the
data, for $144 \cdot 10^4$ total numbers. By using this basis set we can instead reduce
the predictor to 8 numbers for every 80 time point window for $144 \cdot 10^3$ total
numbers, an order of magnitude less. With 1ms bins we would have
achieved 2 order of magnitude reduction in input size. This is a huge benefit
in terms of memory allocation and, computing time. As an additional benefit,
we will reduce over-fitting.

Let's see our basis in action. We can "compress" spike history feature by convolving the basis
with the counts (without creating the large spike history feature matrix).
This can be performed in NeMoS by calling the `compute_features` method of basis.

<div class="notes">
- Convolve the counts with the basis functions. (Call the output `conv_spk`)
- Print the shape of `conv_spk` and compare it to `input_feature`.
</div>

```{code-cell} ipython3
# equivalent to
# `nmo.convolve.create_convolutional_predictor(basis_kernels, neuron_count)`
conv_spk = basis.compute_features(neuron_count)

print(f"Raw count history as feature: {input_feature.shape}")
print(f"Compressed count history as feature: {conv_spk.shape}")
```

Let’s focus on two small time windows and visualize the features, which result from convolving the counts with the basis elements.

<div class="notes">
- Visualize the output.
</div>

```{code-cell} ipython3
# KEEP-CODE

# Visualize the convolution results
epoch_one_spk = nap.IntervalSet(8917.5, 8918.5)
epoch_multi_spk = nap.IntervalSet(8979.2, 8980.2)

doc_plots.plot_convolved_counts(neuron_count, conv_spk, epoch_one_spk, epoch_multi_spk);
```

Now that we have our "compressed" history feature matrix, we can fit the ML parameters for a GLM.


#### Fit and compare the models

<div class="notes">
- Fit the model using the compressed features. Call it `model_basis`.
</div>

```{code-cell} ipython3
# use restrict on interval set training
model_basis = nmo.glm.GLM(solver_name="LBFGS")
model_basis.fit(conv_spk.restrict(first_half), neuron_count.restrict(first_half))
```

We can plot the resulting response, noting that the weights we just learned needs to be "expanded" back
to the original `window_size` dimension by multiplying them with the basis kernels.
We have now 8 coefficients,

```{code-cell} ipython3
print(model_basis.coef_)
```

In order to get the response we need to multiply the coefficients by their corresponding
basis function, and sum them.

<div class="notes">
- Reconstruct the history filter:
    - Extract the basis kernels with `_, basis_kernels = basis.evaluate_on_grid(window_size)`.
    - Multiply the `basis_kernel` with the coefficient using `np.matmul` (call the result `self_connection`).
- Check the shape of `self_connection`.
</div>

```{code-cell} ipython3
# get the basis function kernels
_, basis_kernels = basis.evaluate_on_grid(window_size)

# multiply with the weights
self_connection = np.matmul(basis_kernels, model_basis.coef_)

print(self_connection.shape)
```

Let's check if our new estimate does a better job in terms of over-fitting. We can do that
by visual comparison, as we did previously. Let's fit the second half of the dataset.

<div class="notes">
- Check if with less parameter we are not over-fitting.
- Fit the other half of the data. Name it `model_basis_second_half`.
</div>

```{code-cell} ipython3
model_basis_second_half = nmo.glm.GLM(solver_name="LBFGS").fit(
    conv_spk.restrict(second_half), neuron_count.restrict(second_half)
)
```

Get the response filters.

<div class="notes">
- Get the response filters: multiply the `basis_kernels` with the weights from `model_basis_second_half`.
- Call the output `self_connection_second_half`.
</div>

```{code-cell} ipython3
self_connection_second_half = np.matmul(basis_kernels, model_basis_second_half.coef_)
```

<div class="notes">
- Plot and compare the results.
</div>

And plot the results.

```{code-cell} ipython3
# KEEP-CODE

time = np.arange(window_size) / count.rate
plt.figure()
plt.title("Spike History Weights")
plt.plot(time, np.squeeze(model.coef_), "k", alpha=0.3, label="GLM raw history 1st half")
plt.plot(time, np.squeeze(model_second_half.coef_), alpha=0.3, color="orange", label="GLM raw history 2nd half")
plt.plot(time, self_connection, "--k", lw=2, label="GLM basis 1st half")
plt.plot(time, self_connection_second_half, color="orange", lw=2, ls="--", label="GLM basis 2nd half")
plt.axhline(0, color="k", lw=0.5)
plt.xlabel("Time from spike (sec)")
plt.ylabel("Weight")
plt.legend()
```

Let's extract the firing rate.

<div class="notes">
- Predict the rates from `model` and `model_basis`. Call it `rate_history` and `rate_basis`.
- Convert the rate from spike/bin to spike/sec by multiplying with `conv_spk.rate`.
</div>

```{code-cell} ipython3
rate_basis = model_basis.predict(conv_spk) * conv_spk.rate
rate_history = model.predict(input_feature) * conv_spk.rate
```

And plot it.

<div class="notes">
- Plot the results.
</div>

```{code-cell} ipython3

ep = nap.IntervalSet(start=8819.4, end=8821)
# plot the rates
doc_plots.plot_rates_and_smoothed_counts(
    neuron_count,
    {"Self-connection raw history":rate_history, "Self-connection bsais": rate_basis}
);
```


### All-to-all Connectivity
The same approach can be applied to the whole population. Now the firing rate of a neuron
is predicted not only by its own count history, but also by the rest of the
simultaneously recorded population. We can convolve the basis with the counts of each neuron
to get an array of predictors of shape, `(num_time_points, num_neurons * num_basis_funcs)`.

#### Preparing the features

<div class="notes">
- Convolve all counts. Call the output in `convolved_count`.
- Print the output shape
</div>

Since this time we are convolving more than one neuron, we need to reset the expected input shape. This can be done by passing the population counts to the `set_input_shape` method.

```{code-cell} ipython3
# set the input shape by passing the pop. count
basis.set_input_shape(count)

# convolve all the neurons
convolved_count = basis.compute_features(count)
```

Check the dimension to make sure it make sense
Shape should be (n_samples, n_basis_func * n_neurons)

```{code-cell} ipython3
print(f"Convolved count shape: {convolved_count.shape}")
```

#### Fitting the Model
This is an all-to-all neurons model.
We are using the class [`PopulationGLM`](nemos.glm.PopulationGLM) to fit the whole population at once.

:::{note}

Once we condition on past activity, log-likelihood of the population is the sum of the log-likelihood
of individual neurons. Maximizing the sum (i.e. the population log-likelihood) is equivalent to
maximizing each individual term separately (i.e. fitting one neuron at the time).
:::

<div class="notes">
- Fit a `PopulationGLM`, call the object `model`
- Use Ridge regularization with a `regularizer_strength=0.1`
- Print the shape of the estimated coefficients.
</div>

```{code-cell} ipython3
model = nmo.glm.PopulationGLM(
    regularizer="Ridge",
    solver_name="LBFGS",
    regularizer_strength=0.1
    ).fit(convolved_count, count)

print(f"Model coefficients shape: {model.coef_.shape}")
```

#### Comparing model predictions.
Predict the rate (counts are already sorted by tuning prefs)

<div class="notes">
- Predict the firing rate of each neuron. Call it `predicted_firing_rate`.
- Convert the rate from spike/bin to spike/sec.
</div>

```{code-cell} ipython3
predicted_firing_rate = model.predict(convolved_count) * conv_spk.rate
```

Plot fit predictions over a short window not used for training.

<div class="notes">
- Visualize the predicted rate and tuning function.
</div>

```{code-cell} ipython3
# KEEP-CODE

# use pynapple for time axis for all variables plotted for tick labels in imshow
doc_plots.plot_head_direction_tuning_model(tuning_curves, predicted_firing_rate, spikes, angle, threshold_hz=1,
                                          start=8910, end=8960, cmap_label="hsv");
```

Let's see if our firing rate predictions improved and in what sense.

<div class="notes">
- Visually compare all the models.
</div>

```{code-cell} ipython3
# KEEP-CODE

fig = doc_plots.plot_rates_and_smoothed_counts(
    neuron_count,
    {"Self-connection: raw history": rate_history,
     "Self-connection: bsais": rate_basis,
     "All-to-all: basis": predicted_firing_rate[:, 0]}
)
```

#### Visualizing the connectivity
Compute the tuning curve form the predicted rates.

<div class="notes">
- Compute tuning curves from the predicted rates using pynapple.
- Store the output of pynapple in a single variable, call it `tuning `.
</div>

```{code-cell} ipython3
tuning = nap.compute_1d_tuning_curves_continuous(predicted_firing_rate,
                                                 feature=angle,
                                                 nb_bins=61,
                                                 minmax=(0, 2 * np.pi))
```

Extract the weights and store it in a `(n_neurons, n_neurons, n_basis_funcs)` array. 

You can use the `split_by_feature` method of `basis` for this. 

<div class="notes">
- Extract the weights:
    - Use `basis.split_by_feature` (returns a dictionary).
    - Get the weight array from the dictionary (and call the output `weights`). 
    - Print the weights shape.
</div>

```{code-cell} ipython3
print(f"GLM coeff: {model.coef_.shape}")
# split the coefficient vector along the feature axis (axis=0)
weights_dict = basis.split_by_feature(model.coef_, axis=0)

# the output is a dict with key the basis label, 
# and value the reshaped coefficients
weights = weights_dict["RaisedCosineLogConv"]
print(f"Re-shaped coeff: {weights.shape}")
```

<div class="notes">
- The shape is `(sender_neuron, num_basis, receiver_neuron)`.
- Multiply the weights with the kernels with: `np.einsum("jki,tk->ijt", weights, basis_kernels)`.
- Call the output `responses` and print its shape.
</div>

Multiply the weights by the basis, to get the history filters.

```{code-cell} ipython3
responses = np.einsum("jki,tk->ijt", weights, basis_kernels)

print(responses.shape)
```

Finally, we can visualize the pairwise interactions by plotting
all the coupling filters.

<div class="notes">
- Plot the connectivity map.
</div>

```{code-cell} ipython3
# KEEP-CODE
fig = doc_plots.plot_coupling(responses, tuning)
```

```{code-cell} ipython3

```
