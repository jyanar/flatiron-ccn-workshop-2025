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

Spike-phase coupling and decoding: 
Phase precession and hippocampal sequences
==========================================

In this tutorial we will learn how to apply two methods included in pynapple: filtering and decoding. We'll apply these methods to demonstrate and visualize some well-known physiological properties of hippocampal activity, specifically phase presession of place cells and sequential coordination of place cell activity during theta oscillations.

Background
----------
- hippocampus (rat)
- place cells
- LFP and theta oscillation
- phase precession
- theta sequences

```{code-cell} ipython3
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import scipy
import seaborn as sns
import tqdm
import pynapple as nap
import workshop_utils

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.5, rc=custom_params)
```

***
Fetching the data
--------------------
The data set we'll be looking at is from the manuscript [Diversity in neural firing dynamics supports both rigid and learned hippocampal sequences](https://www.science.org/doi/10.1126/science.aad1935). In this study, the authors collected electrophisiology data in rats across multiple sites in layer CA1 of hippocampus to extract the LFP alongside spiking activity of many simultaneous pyramidal units. In each recording session, data were collected while the rats explored a novel environment (a linear track), as well as during sleep before and after exploration. In our following analyses, we'll focus on the exploration period of a single rat and recording session.

The full dataset for this study can be accessed on [DANDI](https://dandiarchive.org/dandiset/000044/0.210812.1516). Since the file size of a recording session can be large from the LFP saved for each recorded channel, we'll use a smaller file that contains the spiking activity and the LFP from a single, representative channel, which is hosted on [OSF](https://osf.io/2dfvp). This smaller file, like the original data, is saved as an [NWB](https://www.nwb.org) file.

If you ran the workshop setup script, you should have this file downloaded already. If not, the function we'll use to fetch it will download it for you. This function is called `fetch_data`, and can be imported from the `workshop_utils` module.

```{code-cell} ipython3
from workshop_utils import fetch_data

path = fetch_data("Achilles_10252013_EEG.nwb")
```

This function will give us the file path to where the data is stored. We can then use the pynapple function `load_file` to load in the data, which is able to handle the NWB file type.

```{code-cell} ipython3
data = nap.load_file(path)
print(data)
```

What this gives you is a dictionary of pynapple objects that have been inferred from the NWB file. This dictionary contains the following fields:
- `units`: a `TsGroup` with each units spike times as well as metadata about each unit (i.e. location, shank, and cell type). This dataset contains 137 units all in CA1.
- `rem`: an `IntervalSet` of REM sleep epochs, with 3 occuring before exploration and 1 occuring after.
- `nrem`: an `IntervalSet` of nREM sleep epochs, with 6 occuring before exploration and 5 occuring after.
- `forward_ep`: an `IntervalSet` containing each time window when the animal crossed the linear track in one direction. There are a total of 84 traversals in this session.
- `eeg`: a `TsdFrame` containing an LFP voltage traces for a single representative channel in CA1.
- `theta_phase`: a `Tsd` with the computed theta phase of the LFP used in the study. We will be computing this ourselves.
- `position`: a `Tsd` containing the linearized position

+++

***
Filtering the data
------------------
For the following exercises, we'll only focus on the exploration epochs contained in `forward_ep`. Therefore, when extracting the LFP, spikes, and position, we can use `restrict()` with the `forward_ep` IntervalSet to subselect the data.

```{code-cell} ipython3
lfp_run = data["eeg"][:,0].restrict(data["forward_ep"])
spikes = data["units"].restrict(data["forward_ep"])
position = data["position"].restrict(data["forward_ep"])
```

For visualization, we'll look at a single run down the linear track. For a good example, we'll start by looking at run 10 (python index 9). It is encouraged, however, to repeat these exercises on additional runs!

```{code-cell} ipython3
ex_run_ep = data["forward_ep"][9]
```

***
Plotting the LFP and animal position
------------------------------------
To get a sense of what the LFP looks like while the animal runs down the linear track, we can plot each variable, `lfp_run` and `position`, side-by-side. We'll want to further restrict each variable to our run of interest stored in `ex_run_ep`.

```{code-cell} ipython3
ex_lfp_run = lfp_run.restrict(ex_run_ep)
ex_position = position.restrict(ex_run_ep)
```

By default, plotting Tsd objects will use the time index on the x-axis. However, for a more interpretable time axis, we'll subtract the first time index from each variable's time indices and pass it as the first argument in matplotlib's `plot`. This will give the relative time elapsed on the current run.

```{code-cell} ipython3
fig, axs = plt.subplots(2, 1, constrained_layout=True, figsize=(10, 6), sharex=True)

# plot LFP
axs[0].plot(ex_lfp_run.index - ex_lfp_run.index[0], ex_lfp_run)
axs[0].set_title("Local Field Potential on Linear Track")
axs[0].set_ylabel("LFP (a.u.)")

# plot animal's position
axs[1].plot(ex_position.index - ex_position.index[0], ex_position)
axs[1].set_title("Animal Position on Linear Track")
axs[1].set_ylabel("Position (cm)") # LOOK UP UNITS
axs[1].set_xlabel("time (s)")
```

***
Getting the Wavelet Decomposition
-----------------------------------
As we would expect, there is a strong theta oscillation dominating the LFP while the animal runs down the track. To illustrate this further, we'll perform a wavelet decomposition on the LFP trace during this run.

DEFINE WAVELET DECOMPOSITION

We must define the frequency set that we'd like to use for our decomposition. We can do this with the numpy function `np.geomspace`, which returns numbers evenly spaced on a log scale. We pass the lower frequency, the upper frequency, and number of samples as positional arguments.

```{code-cell} ipython3
# 25 log-spaced samples between 5Hz and 200Hz
freqs = np.geomspace(5, 200, 100)
```

We can now compute the wavelet transform on our LFP data during the example run using the pynapple function `nap.compute_wavelet_trasform`, which takes the time series and array of frequencies as positional arguments. Optionally, we can pass the keyword argument `fs` to provide the the sampling frequency, which is known to be 1250Hz from the study methods.

double check: FS back tracked to https://www.jneurosci.org/content/28/26/6731 methods

```{code-cell} ipython3
sample_freq = 1250  # We know from the methods of the paper
cwt_run = nap.compute_wavelet_transform(lfp_run.restrict(ex_run_ep), freqs, fs=sample_freq)
```

If `fs` is not provided, it can be inferred from the time series `rate` attribute, which matches what was pulled from the methods

```{code-cell} ipython3
print(ex_lfp_run.rate)
```

We can visualize the results by plotting a heat map of the calculated wavelet scalogram.

```{code-cell} ipython3
fig, axs = plt.subplots(2, 1, figsize=(10,6), constrained_layout=True, height_ratios=[1.0, 0.3], sharex=True)
fig.suptitle("Wavelet Decomposition")

t = ex_lfp_run.index - ex_lfp_run.index[0]
power = np.abs(cwt_run.values)
cax = axs[0].pcolormesh(t, freqs, power.T)
axs[0].set(ylabel="Frequency (Hz)", yscale='log', yticks=freqs[::10], yticklabels=np.rint(freqs[::10]));
axs[0].minorticks_off()
fig.colorbar(cax,label="Power")

axs[1].plot(t, ex_lfp_run)
axs[1].set(ylabel="LFP (a.u.)", xlabel="Time(s)")
axs[1].margins(0)
```

***
Filtering for theta
-------------------
We can extract the theta oscillation by applying a bandpass filter on the raw LFP. To do this, we use the pynapple function `nap.apply_bandpass_filter`, which takes the time series as the first argument and the frequency cutoffs as the second argument. Similarly to `nap.compute_wavelet_transorm`, we can optinally pass the sampling frequency keyword argument `fs`.

Conveniently, this function will recognize and handle splits in the subsampled data (i.e. applying the filtering separately to discontinuous epochs), so we can pass the LFP for all the runs together.

```{code-cell} ipython3
theta_band = nap.apply_bandpass_filter(lfp_run, (6.0, 12.0), fs=sample_freq)
```

We can visualize the output by plotting the filtered signal with the original signal.

```{code-cell} ipython3
plt.figure(constrained_layout=True, figsize=(12, 3))
plt.plot(t, ex_lfp_run, alpha=0.5, label="raw")
plt.plot(t, theta_band.restrict(ex_run_ep), label="filtered")
plt.ylabel("Time (s)")
plt.xlabel("LFP (a.u.)")
plt.title("Bandpass filter for theta oscillations (6-12 Hz)")
plt.legend();
```

***
Computing phase
---------------
In order to examine phase precession in place cells, we need to extract the phase of theta from the filtered signal. We can do this by taking the angle of the [Hilbert transform](https://en.wikipedia.org/wiki/Hilbert_transform).

The `signal` module of `scipy` includes a function to perform the Hilbert transform, after which we can use the numpy function `np.angle` to extract the angle.

```{code-cell} ipython3
from scipy import signal

phase = np.angle(signal.hilbert(theta_band)) # compute phase with hilbert transform
phase[phase < 0] += 2 * np.pi # wrap to [0,2pi]
theta_phase = nap.Tsd(t=theta_band.t, d=phase)
```

Let's plot the phase.

```{code-cell} ipython3
fig,axs = plt.subplots(2, 1, figsize=(12,4), constrained_layout=True, sharex=True, height_ratios=[2,1])

axs[0].plot(lfp_run.restrict(ex_run_ep), alpha=0.5, label="raw")
axs[0].plot(theta_band.restrict(ex_run_ep), label="filtered")
axs[0].set_ylabel("LFP (a.u.)")
axs[0].legend()

axs[1].plot(theta_phase.restrict(ex_run_ep), color='r')
axs[1].set_ylabel("Phase (rad)")
axs[1].set_xlabel("Time (s)")
```

```{code-cell} ipython3
fig,ax = plt.subplots(figsize=(12,2), constrained_layout=True) #, sharex=True, height_ratios=[2,1])

ax.plot(t, theta_phase.restrict(ex_run_ep), color='r', label="phase")
ax.set_ylabel("Phase (rad)")
ax.set_xlabel("Time (s)")
ax = ax.twinx()
ax.plot(t, theta_band.restrict(ex_run_ep), alpha=0.5, label="filtered LFP")
ax.set_ylabel("LFP (a.u.)")
fig.legend()
```

cycle "resets" at peaks

+++

***
Identifying place-selective cells
---------------------------------
Now that we have the phase of our theta wavelet, and our spike times, we can find the phase firing preferences
of each of the units using the `compute_1d_tuning_curves` function.

We will start by throwing away cells which do not have a high enough firing rate during our interval.

```{code-cell} ipython3
pyr_spikes = spikes[(spikes.rate > 1) & (spikes.rate < 10)]
```

compute place fields

```{code-cell} ipython3
from scipy.ndimage import gaussian_filter1d
place_fields = nap.compute_1d_tuning_curves(pyr_spikes, position, nb_bins=50)
# filter
place_fields[:] = gaussian_filter1d(place_fields.values, 1, axis=0)
```

```{code-cell} ipython3
fig, axs = plt.subplots(6, 10, figsize=(30, 30))
for i, (f, fields) in enumerate(place_fields.items()):
    idx = np.unravel_index(i, axs.shape)
    axs[idx].plot(fields)
    axs[idx].set_title(f)
```

```{code-cell} ipython3
plt.figure(constrained_layout=True, figsize = (12, 3))
for i in range(3):
    plt.subplot(1,3,i+1)
    plt.plot(phase_modulation.iloc[:,i])
    plt.xlabel("Phase (rad)")
    plt.ylabel("Firing rate (Hz)")
plt.show()
```

There is clearly a strong modulation for the third neuron.
Finally, we can use the function `value_from` to align each spikes to the corresponding phase position and overlay
it with the LFP.

```{code-cell} ipython3
unit = 177
spike_phase = spikes[unit].value_from(theta_phase)
# spike_position = spikes[unit].value_from(position)
```

Let's plot it.

```{code-cell} ipython3
fig,axs = plt.subplots(2,1, figsize=(12,6), constrained_layout=True, sharex=True)
axs[0].plot(lfp_run.restrict(ex_run_ep))
axs[0].plot(theta_band.restrict(ex_run_ep))
axs[1].plot(theta_phase.restrict(ex_run_ep), alpha=0.5)
axs[1].plot(spike_phase.restrict(ex_run_ep), 'o')
ax = axs[1].twinx()
ax.plot(position.restrict(ex_run_ep))
```

```{code-cell} ipython3
spike_position = spikes[unit].value_from(position)
plt.subplots(figsize=(3,3))
plt.plot(spike_phase, spike_position, 'o')
plt.xlabel("Phase (rad)")
plt.ylabel("Position (cm)")
```

```{code-cell} ipython3
# hold out trial from place field computation
run_train = data["forward_ep"].set_diff(ex_run_ep)
position_train = data["position"].restrict(run_train)
place_fields = nap.compute_1d_tuning_curves(spikes, position_train, nb_bins=50)

# filter place fields
tc = gaussian_filter1d(place_fields.values, 1, axis=0)
place_fields[:] = tc

# use moving sum of spike counts
ct = spikes.restrict(ex_run_ep).count(0.01).convolve(np.ones(4))
t = spikes.restrict(ex_run_ep).count(0.01).index
group = nap.TsdFrame(t=t, d=ct, columns=spikes.keys())

# decode
_, p = nap.decode_1d(place_fields, group, ex_run_ep, bin_size=0.04)

# plot
plt.subplots(figsize=(12, 4), constrained_layout=True)
plt.pcolormesh(p.index, p.columns, np.transpose(p))
plt.plot(position.restrict(ex_run_ep), color="r")
plt.xlabel("Time (s)")
plt.ylabel("Position (cm)")
plt.colorbar(label = "predicted probability")
```

:::{card}
Authors
^^^
Kipp Freud (https://kippfreud.com/)

Guillaume Viejo

:::
