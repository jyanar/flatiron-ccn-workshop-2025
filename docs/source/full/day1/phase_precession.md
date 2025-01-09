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
import nemos as nmo

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.5, rc=custom_params)
```

***
Downloading the data
--------------------
The data set we'll be looking at is from the manuscript [Diversity in neural firing dynamics supports both rigid and learned hippocampal sequences](https://www.science.org/doi/10.1126/science.aad1935). In this study, the authors collected electrophisiology data in rats across multiple sites in layer CA1 of hippocampus to extract the LFP alongside spiking activity of many simultaneous pyramidal units. In each recording session, data were collected while the rats explored a novel environment (a linear track), as well as during sleep before and after exploration. In our following analyses, we'll focus on the exploration period of a single rat and recording session.

First, we need to download the data and save it locally. Since the file size of a recording session can be large from the LFP saved for each recorded channel, we'll use a smaller file that contains the spiking activity and the LFP from a single, representative channel, which is hosted on [OSF](https://osf.io/2dfvp). This smaller file, like the original data, is saved as an [NWB](https://www.nwb.org) file.

Full dataset: https://dandiarchive.org/dandiset/000044/0.210812.1516

(is there a simpler way of doing this? i.e. use nemos?)
- make workshop pooch, make sure its part of the pre-workshop download

```{code-cell} ipython3
path = "Achilles_10252013.nwb"
path = nmo.fetch.fetch_data(path)
# if path not in os.listdir("."):
#     r = requests.get(f"https://osf.io/2dfvp/download", stream=True)
#     block_size = 1024 * 1024
#     with open(path, "wb") as f:
#         for data in tqdm.tqdm(
#             r.iter_content(block_size),
#             unit="MB",
#             unit_scale=True,
#             total=math.ceil(int(r.headers.get("content-length", 0)) // block_size),
#         ):
#             f.write(data)
```

***
Loading the data
------------------
With the file downloaded, we can use the pynapple function `load_file` to load in the data, which is able to handle NWB file types.

```{code-cell} ipython3
data = nap.load_file(path)
print(data)
```

***
Selecting a single run
-----------------------------------
for visualization, grab a single run down the linear track (selected in advance)

```{code-cell} ipython3
ex_run_ep = data["forward_ep"][9]
```

Restrict data to awake epochs: lfp, spikes, and position

```{code-cell} ipython3
lfp_run = data["eeg"][:,0].restrict(data["forward_ep"])
spikes = data["units"].restrict(data["forward_ep"])
position = data["position"].restrict(data["forward_ep"])
```

***
Plotting the LFP Activity
-----------------------------------
plot LFP and animal position during trial

```{code-cell} ipython3
fig, axs = plt.subplots(2, 1, constrained_layout=True, figsize=(10, 6), sharex=True)

# plot LFP
axs[0].plot(lfp_run.restrict(ex_run_ep))
axs[0].set_title("Local Field Potential on Linear Track")
axs[0].set_ylabel("LFP (a.u.)")
# axs[0].set_xlabel("time (s)")

# plot animal's position
axs[1].plot(position.restrict(ex_run_ep))
axs[1].set_title("Animal Position on Linear Track")
axs[1].set_ylabel("Position (cm)") # LOOK UP UNITS
axs[1].set_xlabel("time (s)")
```

***
Getting the Wavelet Decomposition
-----------------------------------
As we would expect, it looks like we have a very strong theta oscillation within our data
- this is a common feature of REM sleep. Let's perform a wavelet decomposition,
as we did in the last tutorial, to see get a more informative breakdown of the
frequencies present in the data.

We must define the frequency set that we'd like to use for our decomposition.

```{code-cell} ipython3
freqs = np.geomspace(5, 200, 25)
```

We compute the wavelet transform on our LFP data (only during the example interval).

double check: FS tracked to https://www.jneurosci.org/content/28/26/6731 methods

```{code-cell} ipython3
FS = 1250  # We know from the methods of the paper
cwt_run = nap.compute_wavelet_transform(lfp_run.restrict(ex_run_ep), fs=FS, freqs=freqs)
```

***
Now let's plot the calculated wavelet scalogram.

```{code-cell} ipython3
# Define wavelet decomposition plotting function
def plot_timefrequency(freqs, powers, ax=None):
    im = ax.imshow(np.abs(powers), aspect="auto")
    ax.invert_yaxis()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.get_xaxis().set_visible(False)
    ax.set(yticks=np.arange(len(freqs))[::2], yticklabels=np.rint(freqs[::2]))
    ax.grid(False)
    return im

fig, axs = plt.subplots(2, 1, figsize=(10,6), constrained_layout=True, height_ratios=[1.0, 0.3])
fig.suptitle("Wavelet Decomposition")

im = plot_timefrequency(freqs, np.transpose(cwt_run[:, :].values), ax=axs[0])
cbar = fig.colorbar(im, ax=axs[0], orientation="vertical")

axs[1].plot(lfp_run.restrict(ex_run_ep))
axs[1].set_ylabel("LFP (a.u.)")
axs[1].set_xlabel("Time (s)")
axs[1].margins(0)
```

***
Filtering Theta
---------------

As expected, there is a strong 8Hz component during REM sleep. We can filter it using the function `nap.apply_bandpass_filter`.

```{code-cell} ipython3
theta_band = nap.apply_bandpass_filter(lfp_run, cutoff=(6.0, 12.0), fs=FS)
```

We can plot the original signal and the filtered signal.

```{code-cell} ipython3
plt.figure(constrained_layout=True, figsize=(12, 3))
plt.plot(lfp_run.restrict(ex_run_ep), alpha=0.5)
plt.plot(theta_band.restrict(ex_run_ep))
plt.xlabel("Time (s)")
plt.show()
```

***
Computing phase
---------------

From the filtered signal, it is easy to get the phase using the Hilbert transform. Here we use scipy Hilbert method.

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

axs[1].plot(theta_phase.restrict(ex_run_ep), color='r')
axs[1].set_ylabel("Phase (rad)")
axs[1].set_xlabel("Time (s)")
```

***
Finding Phase of Spikes
-----------------------
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
