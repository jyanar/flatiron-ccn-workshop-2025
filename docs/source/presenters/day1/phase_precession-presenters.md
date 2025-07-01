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
# Spike-phase coupling and decoding: Phase precession and hippocampal sequences
This notebook has had all its explanatory text removed and has not been run.
 It is intended to be downloaded and run locally (or on the provided binder)
 while listening to the presenter's explanation. In order to see the fully
 rendered of this notebook, go [here](../../full/day1/phase_precession.md)

    
In this tutorial we will learn how to use more advanced applications of pynapple: signal processing and decoding. We'll apply these methods to demonstrate and visualize some well-known physiological properties of hippocampal activity, specifically phase presession of place cells and sequential coordination of place cell activity during theta oscillations.


## Objectives

    
We can break our goals of identifying phase presession and coordinated activity of place cells into the following objectives: 
1. Get a feel for the data set   
    - Load in and visualize the data
    - Restrict the data to regions of interest
2. Identify and extract theta oscillations in the LFP
    - Decompose the LFP into frequency components to identify theta oscillations
    - Filter the LFP to isolate the theta frequency band
3. Identify place cells
    - Calculate 1D tuning curves to identify place selectivity across many units
4. Visualize phase precession
    - Compare spike location to spike phase in a single unit
    - Calculate 2D tuning curves to identify place vs phase selectivity across many units
5. Reconstruct spatial sequences from population activity
    - Apply Bayesian decoding to predict position from spiking activity
    - Decode at a fast time-scale to identify spatial "sweeps" coordinated with theta oscillations
  

```{code-cell} ipython3
:tags: [render-all]

# suppress warnings
import warnings
warnings.simplefilter("ignore")

# imports
import math
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy import signal
import seaborn as sns
import tqdm
import pynapple as nap
import workshop_utils

# necessary for animation
import nemos as nmo
plt.style.use(nmo.styles.plot_style)
```
## Fetching the data

First we'll fetch the relevant data set for this exercise.

- Manuscript: [Diversity in neural firing dynamics supports both rigid and learned hippocampal sequences](https://www.science.org/doi/10.1126/science.aad1935)
- Full data source: [DANDI](https://dandiarchive.org/dandiset/000044/0.210812.1516)


```{code-cell} ipython3
# fetch file path
path = workshop_utils.fetch_data("Achilles_10252013_EEG.nwb")
# load data with pynapple
data = nap.load_file(path)
print(data)
```

```{code-cell} ipython3
# fetch file path
path = workshop_utils.fetch_data("Achilles_10252013_EEG.nwb")
```
```{code-cell} ipython3
# load data with pynapple
data = nap.load_file(path)
print(data)
```
:::{admonition} Note
:class: note render-all
We will ignore the object `theta_phase` because we will be computing this ourselves later on in the exercise.
:::
### units

    
The `units` field is a `TsGroup`: a collection of `Ts` objects containing the spike times of each unit, where the "Index" is the unit number or key. Each unit has the following metadata:
- **rate**: computed by pynapple, is the average firing rate of the neuron across all recorded time points.
- **location**, **shank**, and **cell_type**: variables saved and imported from the original data set.


```{code-cell} ipython3
:tags: [render-all]

data["units"]
```


We can access the spike times of a single unit by indexing the `TsGroup` by its unit number. For example, to access the spike times of unit 1:


```{code-cell} ipython3
:tags: [render-all]

data["units"][1]
```
### rem, nrem, and forward_ep


The next three objects; `rem`, `nrem`, and `forward_ep`; are all IntervalSets containing time windows of REM sleep, nREM sleep, and forward runs down the linear maze, respectively. 


```{code-cell} ipython3
:tags: [render-all]

data["rem"]
```
```{code-cell} ipython3
:tags: [render-all]

data["nrem"]
```
```{code-cell} ipython3
:tags: [render-all]

data["forward_ep"]
```

- sleep epochs are intertwined, forward epoch in middle


    
The following plot demonstrates how each of these labelled epochs are organized across the session.


```{code-cell} ipython3
:tags: [render-all]

t_start = data["nrem"].start[0]
fig,ax = plt.subplots(figsize=(10,2), constrained_layout=True)
sp1 = [ax.axvspan((iset.start[0]-t_start)/60, (iset.end[0]-t_start)/60, color="blue", alpha=0.1) for iset in data["rem"]];
sp2 = [ax.axvspan((iset.start[0]-t_start)/60, (iset.end[0]-t_start)/60, color="green", alpha=0.1) for iset in data["nrem"]];
sp3 = [ax.axvspan((iset.start[0]-t_start)/60, (iset.end[0]-t_start)/60, color="red", alpha=0.1) for iset in data["forward_ep"]];
ax.set(xlabel="Time within session (minutes)", title="Labelled time intervals across session", yticks=[])
ax.legend([sp1[0],sp2[0],sp3[0]], ["REM sleep","nREM sleep","forward runs"]);
```
### eeg


The `eeg` object is a `TsdFrame` containing an LFP voltage trace for a single representative channel in CA1.


```{code-cell} ipython3
:tags: [render-all]

data["eeg"]
```


Despite having a single column, this `TsdFrame` is still a 2D object. We can represent this as a 1D `Tsd` by indexing into the first column.


```{code-cell} ipython3
:tags: [render-all]

data["eeg"][:,0]
```
### position


The final object, `position`, is a `Tsd` containing the linearized position of the animal, in centimeters, recorded during the exploration window.


```{code-cell} ipython3
:tags: [render-all]

data["position"]
```


Positions that are not defined, i.e. when the animal is at rest, are filled with `NaN`.

This object additionally contains a `time_support` attribute, which gives the time interval during which positions are recorded (including points recorded as `NaN`).


```{code-cell} ipython3
:tags: [render-all]

data["position"].time_support
```


Let's visualize the first 300 seconds of position data and overlay `forward_ep` intervals.


```{code-cell} ipython3
:tags: [render-all]

pos_start = data["position"].time_support.start[0]
fig, ax = plt.subplots(figsize=(10,3))
l1 = ax.plot(data["position"])
l2 = [ax.axvspan(iset.start[0], iset.end[0], color="red", alpha=0.1) for iset in data["forward_ep"]];
ax.set(xlim=[pos_start,pos_start+300], ylabel="Position (cm)", xlabel="Time (s)", title="Tracked position along linear maze")
ax.legend([l1[0], l2[0]], ["animal position", "forward run epochs"])
```

- position only while moving
- `forward_ep` only captures forward runs

## Restricting the data

    
For the following exercises, we'll only focus on periods when the animal is awake. We'll start by pulling out `forward_ep` from the data.


```{code-cell} ipython3
:tags: [render-all]

forward_ep = data["forward_ep"]
```


Since `forward_ep` is formatted as discontinuous epochs when the animal is running down the track, we will want two additional IntervalSets to describe the exploration period:

1. An IntervalSet with a single interval for the entire awake period
2. An IntervalSet containing the intervals at which the animal is at rest.

We can derive both of these from `forward_ep`. 

For the first, we can use the `IntervalSet` method `time_span`, which will give the total epoch spanning all the intervals in `forward_ep`.


```{code-cell} ipython3
awake_ep = forward_ep.time_span()
```


For the second, we know that the animal is likely at rest when there is no recorded position (i.e. the position is NaN). We can create this `IntervalSet`, then, using the following steps.

1. Drop `NaN` values from the position to grab only points where position is defined.


```{code-cell} ipython3
# drop nan values
pos_good = data["position"].dropna()
pos_good
```


2. Extract time intervals from `pos_good` using the `find_support` method
   - The first input argument, `min_gap`, sets the minumum separation between adjacent intervals in order to be split
   - Here, use `min_gap` of 1 s


```{code-cell} ipython3
# extract time support
position_ep = pos_good.find_support(1)
position_ep
```


3. Define resting epochs as the set difference between `awake_ep` and `position_ep`, using the `set_diff` method.
   - `set_diff` should be applied to `awake_ep`, not the other way around, such that intervals in `position_ep` are subtracted out of `awake_ep`


```{code-cell} ipython3
rest_ep = awake_ep.set_diff(position_ep)
rest_ep
```
:::{admonition} Note
:class: note render-all

Performing `set_diff` between `awake_ep` and `forward_ep` will *not* give us purely resting epochs, since these intervals will also include times when the animal is moving *backwards* across the linear track.

:::

    
Now, when extracting the LFP, spikes, and position, we can use `restrict()` with `awake_ep` to restrict the data to our region of interest.


```{code-cell} ipython3
:tags: [render-all]

lfp_run = data["eeg"][:,0].restrict(awake_ep)
spikes = data["units"].restrict(awake_ep)
position = data["position"].restrict(awake_ep)
```

    
For visualization, we'll look at a single run down the linear track. For a good example, we'll start by looking at run 10 (python index 9). Furthermore, we'll add two seconds on the end of the run to additionally visualize a period of rest following the run.
    

```{code-cell} ipython3
:tags: [render-all]

ex_run_ep = nap.IntervalSet(start=forward_ep[9].start, end=forward_ep[9].end+2)
ex_run_ep
```
## Plotting the LFP and animal position


To get a sense of what the LFP looks like while the animal runs down the linear track, we can plot each variable, `lfp_run` and `position`, side-by-side. 

We'll want to further restrict each variable to our run of interest, `ex_run_ep`.


```{code-cell} ipython3
ex_lfp_run = lfp_run.restrict(ex_run_ep)
ex_position = position.restrict(ex_run_ep)
```


Let's plot the example LFP trace and anmimal position. Plotting `Tsd` objects will automatically put time on the x-axis.


```{code-cell} ipython3
:tags: [render-all]

fig, axs = plt.subplots(2, 1, constrained_layout=True, figsize=(10, 4), sharex=True)

# plot LFP
axs[0].plot(ex_lfp_run)
axs[0].set_title("Local Field Potential on Linear Track")
axs[0].set_ylabel("LFP (a.u.)")

# plot animal's position
axs[1].plot(ex_position)
axs[1].set_title("Animal Position on Linear Track")
axs[1].set_ylabel("Position (cm)") # LOOK UP UNITS
axs[1].set_xlabel("Time (s)");
```

- theta while running, weaker after run is complete

## Getting the Wavelet Decomposition


To illustrate this further, we'll perform a wavelet decomposition on the LFP trace during this run. We can do this in pynapple using the function `nap.compute_wavelet_transform`. This function takes the following inputs (in order):
- `sig`: the input signal; a `Tsd`, a `TsdFrame`, or a `TsdTensor`
- `freqs`: a 1D array of frequency values to decompose

We will also supply the following optional arguments:
- `fs`: the sampling rate of `sig`



Wavelet transform:
- continuous wavelet transform decomposes signal into Morlet wavelets spanning frequency and time
- like cross-correlation between signal and wavelets, giving similarity between the signal and a short oscillating wavelet at different points in time
- give estimate fo what frequencies are dominating a signal
- unlike FFT, gives estimate as a function of time


1. Define 100 log-spaced samples between 5 and 200 Hz using `np.geomspace`

```{code-cell} ipython3
# 100 log-spaced samples between 5Hz and 200Hz
freqs = np.geomspace(5, 200, 100)
```

2. Compute the wavelet transform, supplying the known sampling rate of 1250 Hz.

```{code-cell} ipython3
sample_rate = 1250
cwt_run = nap.compute_wavelet_transform(ex_lfp_run, freqs, fs=sample_rate)
```


If `fs` is not provided, it can be inferred from the time series `rate` attribute.


```{code-cell} ipython3
:tags: [render-all]

print(ex_lfp_run.rate)
```

- note floating point error



We can visualize the results by plotting a heat map of the calculated wavelet scalogram.


```{code-cell} ipython3
:tags: [render-all]

fig, axs = plt.subplots(2, 1, figsize=(10,4), constrained_layout=True, height_ratios=[1.0, 0.3], sharex=True)
fig.suptitle("Wavelet Decomposition")

amp = np.abs(cwt_run.values)
cax = axs[0].pcolormesh(cwt_run.t, freqs, amp.T)
axs[0].set(ylabel="Frequency (Hz)", yscale='log', yticks=freqs[::10], yticklabels=np.rint(freqs[::10]));
axs[0].minorticks_off()
fig.colorbar(cax,label="Amplitude")

p1 = axs[1].plot(ex_lfp_run)
axs[1].set(ylabel="LFP (a.u.)", xlabel="Time(s)")
axs[1].margins(0)
ax = axs[1].twinx()
p2 = ax.plot(ex_position, color="orange")
ax.set_ylabel("Position (cm)")
ax.legend([p1[0], p2[0]],["raw LFP","animal position"])
```

- strong amplitude in 6-12Hz range while animal is running, dampens after

## Bonus: Additional signal processing methods

    
- `nap.compute_fft`
  

```{code-cell} ipython3
:tags: [render-all]

fft_amp = np.abs(nap.compute_fft(lfp_run, fs=sample_rate, norm=True))
fig, ax = plt.subplots(figsize=(10,4), constrained_layout=True)
ax.plot(fft_amp[(fft_amp.index >= 1) & (fft_amp.index <= 100)])
ax.axvspan(6, 12, color="red", alpha=0.1, label = "theta band")
ax.set(xlabel="Frequency (Hz)", ylabel="Normalized Amplitude (a.u.)", title="FFT amplitude during the awake epoch")
fig.legend(loc="center")
```

    
- `nap.compute_power_spectral_density`
  

```{code-cell} ipython3
:tags: [render-all]

power = nap.compute_power_spectral_density(lfp_run, fs=sample_rate)
fig, ax = plt.subplots(figsize=(10,4), constrained_layout=True)
ax.plot(power[(power.index >= 1) & (power.index <= 100)])
ax.axvspan(6, 12, color="red", alpha=0.1, label = "theta band")
ax.set(xlabel="Frequency (Hz)", ylabel="Power/Frequency (a.u./Hz)", title="Periodogram during the awake epoch")
fig.legend(loc="center")
```
## Filtering for theta


For the remaining exercises, we'll reduce our example epoch to the portion when the animal is running down the linear track.


```{code-cell} ipython3
:tags: [render-all]

ex_run_ep = forward_ep[9]
ex_lfp_run = lfp_run.restrict(ex_run_ep)
ex_position = position.restrict(ex_run_ep)
```

We can filter our signal for theta by using `nap.apply_bandpass_filter`, which requires following arguments:
- `data`: the signal to be filtered; a `Tsd`, `TsdFrame`, or `TsdTensor`
- `cutoff`: tuple containing the frequency cutoffs, (lower frequency, upper frequency)

Same as before, we'll pass the optional argument:
- `fs`: the sampling rate of `data` in Hz

Using this function, filter `lfp_run` within a 6-12 Hz range.


- note handling of disconinuous data

```{code-cell} ipython3
theta_band = nap.apply_bandpass_filter(lfp_run, (6.0, 12.0), fs=sample_rate)
```


We can visualize the output by plotting the filtered signal with the original signal.


```{code-cell} ipython3
:tags: [render-all]

plt.figure(constrained_layout=True, figsize=(10, 3))
plt.plot(ex_lfp_run, label="raw")
plt.plot(theta_band.restrict(ex_run_ep), label="filtered")
plt.xlabel("Time (s)")
plt.ylabel("LFP (a.u.)")
plt.title("Bandpass filter for theta oscillations (6-12 Hz)")
plt.legend();
```
## Computing theta phase


In order to examine phase precession in place cells, we need to extract the phase of theta from the filtered signal. We can do this by taking the angle of the [Hilbert transform](https://en.wikipedia.org/wiki/Hilbert_transform).

The `signal` module of `scipy` includes a function to perform the Hilbert transform, after which we can use the numpy function `np.angle` to extract the angle.


```{code-cell} ipython3
phase = np.angle(signal.hilbert(theta_band)) # compute phase with hilbert transform
phase
```


The output angle will be in the range $-\pi$ to $\pi$. Converting this to a $0$ to $2\pi$ range instead, by adding $2\pi$ to negative angles, will make later visualization more interpretable.


```{code-cell} ipython3
phase[phase < 0] += 2 * np.pi # wrap to [0,2pi]
```


Finally, we need to turn this into a `Tsd` to make full use of pynapple's conveniences! Do this using the time index of `theta_band`. 


```{code-cell} ipython3
theta_phase = nap.Tsd(t=theta_band.t, d=phase)
```


Let's plot the phase on top of the filtered LFP signal.


```{code-cell} ipython3
:tags: [render-all]

fig,axs = plt.subplots(2,1,figsize=(10,4), constrained_layout=True) #, sharex=True, height_ratios=[2,1])

ax = axs[0]
ax.plot(ex_lfp_run)

ax = axs[1]
p1 = ax.plot(theta_phase.restrict(ex_run_ep), color='r')
ax.set_ylabel("Phase (rad)")
ax.set_xlabel("Time (s)")
ax = ax.twinx()
p2 = ax.plot(theta_band.restrict(ex_run_ep))
ax.set_ylabel("LFP (a.u.)")
ax.legend([p1[0],p2[0]],["theta phase","filtered LFP"])
```


Let's zoom in on a few cycles to get a better look.


```{code-cell} ipython3
:tags: [render-all]

fig,ax = plt.subplots(figsize=(10,4), constrained_layout=True) #, sharex=True, height_ratios=[2,1])

ex_run_shorter = nap.IntervalSet(ex_run_ep.start[0], ex_run_ep.start[0]+0.25)

p1 = ax.plot(theta_phase.restrict(ex_run_shorter), color='r')
ax.set_ylabel("Phase (rad)")
ax.set_xlabel("Time (s)")
ax = ax.twinx()
p2 = ax.plot(theta_band.restrict(ex_run_shorter))
ax.set_ylabel("LFP (a.u.)")
ax.legend([p1[0],p2[0]],["theta phase","filtered LFP"])
```

- note cycle reset

## Computing 1D tuning curves: place fields


In order to identify phase precession in single units, we need to know their place selectivity. We can find place firing preferences of each unit by using the function `nap.compute_1d_tuning_curves`. This function has the following required inputs:
- `group`: a `TsGroup` of units for which tuning curves are computed
- `feature`: a `Tsd` or single-column `TsdFrame` of the feature over which tuning curves are computed (e.g. position)
- `nb_bins`: the number of bins in which to split the feature values for the tuning curve

First, we'll filter for units that fire at least 1 Hz and at most 10 Hz when the animal is running forward along the linear track. This will select for units that are active during our window of interest and eliminate putative interneurons (i.e. fast-firing inhibitory neurons that don't usually have place selectivity). 



1. Restrict `spikes` to `forward_ep`


```{code-cell} ipython3
forward_spikes = spikes.restrict(forward_ep)
```


2. Select for units whose rate is at least 1 Hz and at most 10 Hz


```{code-cell} ipython3
good_units = (forward_spikes.rate >= 1) & (forward_spikes.rate <= 10)
good_spikes = forward_spikes[good_units]
```

```{code-cell} ipython3
good_spikes = spikes[(spikes.restrict(forward_ep).rate >= 1) & (spikes.restrict(forward_ep).rate <= 10)]
```


Using these units and the position data, we can compute their place fields using `nap.compute_1d_tuning_curves`. This function will return a `pandas.DataFrame`, where the index is the corresponding feature value, and the column is the unit label. Let's compute this for 50 position bins.


:::{admonition} Tip
:class: tip render-all

The reason `nap.compute_1d_tuning_curves` returns a `pandas.DataFrame` and not a Pynapple object is because the index corresponds to the *feature*, where all Pynapple objects assume the index is *time*.
:::
```{code-cell} ipython3
place_fields = nap.compute_1d_tuning_curves(good_spikes, position, 50)
```


We can use a subplot array to visualize the place fields of many units simultaneously. Let's do this for the first 50 units.


```{code-cell} ipython3
:tags: [render-all]

from scipy.ndimage import gaussian_filter1d

# smooth the place fields so they look nice
place_fields[:] = gaussian_filter1d(place_fields.values, 1, axis=0)

fig, axs = plt.subplots(10, 5, figsize=(12, 15), sharex=True, constrained_layout=True)
for i, (f, fields) in enumerate(place_fields.iloc[:,:50].items()):
    idx = np.unravel_index(i, axs.shape)
    axs[idx].plot(fields)
    axs[idx].set_title(f)

fig.supylabel("Firing rate (Hz)")
fig.supxlabel("Position (cm)")
```

- note representations cover entire track

## Visualizing phase precession within a single unit

First, let's look at how an example unit fires with respect to the filtered LFP. Using the pynapple object method `value_from`, we can find the value of `theta_band` corresponding to each spike time. Let's do this for unit 177, who's place field is cenetered on the linear track.

```{code-cell} ipython3
unit = 177
spike_theta = spikes[unit].value_from(theta_band)
```


Let's plot `spike_theta` on top of the LFP and filtered theta, as well as visualize the animal's position along the track.


```{code-cell} ipython3
:tags: [render-all]

fig,axs = plt.subplots(2, 1, figsize=(10,4), constrained_layout=True, sharex=True)
axs[0].plot(ex_lfp_run, alpha=0.5, label="raw LFP")
axs[0].plot(theta_band.restrict(ex_run_ep), color="slateblue", label="filtered theta")
axs[0].plot(spike_theta.restrict(ex_run_ep), 'o', color="orange", label="spike times")
axs[0].set(ylabel="LFP (a.u.)")
axs[0].legend()

axs[1].plot(ex_position, '--', color="green", label="animal position")
axs[1].plot(ex_position[(ex_position > 50).values & (ex_position < 130).values], color="green", lw=3, label="place field bounds")
axs[1].set(ylabel="Position (cm)", xlabel="Time (s)")
axs[1].legend()
```

As the animal runs through unit 177's place field (thick green), the unit spikes (orange dots) at specific points along the theta cycle dependent on position: starting at the rising edge, moving towards the trough, and ending at the falling edge.

We can exemplify this pattern by plotting the spike times aligned to the phase of theta. Let's compute the phase at which each spike occurs by using `value_from` with `theta_phase`. 

```{code-cell} ipython3
spike_phase = spikes[unit].value_from(theta_phase)
```


To visualize the results, we'll recreate the plot above, but instead with the theta phase.


```{code-cell} ipython3
:tags: [render-all]

fig,axs = plt.subplots(3, 1, figsize=(10,6), constrained_layout=True, sharex=True)
axs[0].plot(theta_band.restrict(ex_run_ep), color="slateblue", label="filtered theta")
axs[0].plot(spike_theta.restrict(ex_run_ep), 'o', color="orange", label="spike times")
axs[0].set(ylabel="LFP (a.u.)", title="Spike times relative to filtered theta")
axs[0].legend()

axs[1].plot(theta_phase.restrict(ex_run_ep), color="slateblue", label="theta phase")
axs[1].plot(spike_phase.restrict(ex_run_ep), 'o', color="orange", label="spike times")
axs[1].set(ylabel="Phase (rad)", title="Spike times relative to theta phase")
axs[1].legend()

axs[2].plot(ex_position, '--', color="green", label="animal position")
axs[2].plot(ex_position[(ex_position > 50).values & (ex_position < 130).values], color="green", lw=3, label="place field bounds")
axs[2].set(ylabel="Position (cm)", xlabel="Time (s)", title="Animal position")
axs[2].legend()
```

We now see a negative trend in the spike phase as the animal moves through unit 177's place field, indicative of this unit *phase precessing*. 

We can observe this phenomena on average across the session by relating the spike phase to the spike position. Try computing the spike position from what we've learned so far.

```{code-cell} ipython3
spike_position = spikes[unit].value_from(position)
```


Now we can plot the spike phase against the spike position in a scatter plot.


```{code-cell} ipython3
:tags: [render-all]

plt.subplots(figsize=(5,3))
plt.plot(spike_position, spike_phase, 'o')
plt.ylabel("Phase (rad)")
plt.xlabel("Position (cm)")
```

- note negative relationship

## Computing 2D tuning curves: position vs. phase


The scatter plot above can be similarly be represented as a 2D tuning curve over position and phase. We can compute this using the function `nap.compute_2d_tuning_curves`. This function requires the same inputs as `nap.compute_1d_tuning_curves`, except now the second input, `features`, must be a 2-column `TsdFrame` containing the two target features.

To use this function, we'll need to combine `position` and `theta_phase` into a `TsdFrame`. To do this, both variables must have the same length. We can achieve this by upsampling `position` to the length of `theta_phase` using the pynapple object method `interpolate`. This method will linearly interpolate new position samples between existing position samples at timestamps given by another pynapple object, in our case by `theta_phase`.


```{code-cell} ipython3
upsampled_pos = position.interpolate(theta_phase)
```


Let's visualize the results of the interpolation.


```{code-cell} ipython3
:tags: [render-all]

fig,axs = plt.subplots(2,1,constrained_layout=True,sharex=True,figsize=(10,4))
axs[0].plot(position.restrict(ex_run_ep),'.')
axs[0].set(ylabel="Position (cm)", title="Original position points")
axs[1].plot(upsampled_pos.restrict(ex_run_ep),'.')
axs[1].set(ylabel="Position (cm)", xlabel="Time (s)", title="Upsampled position points")
```


We can now stack `upsampled_pos` and `theta_phase` into a single array.


```{code-cell} ipython3
feats = np.stack((upsampled_pos.values, theta_phase.values))
feats.shape
```


Using `feats`, we can define a `TsdFrame` using the time index from `theta_phase` and the time support from `upsampled_pos`. Note that `feats` has the wrong shape; we want time in the first dimension, so we'll need to pass its transpose.


```{code-cell} ipython3
features = nap.TsdFrame(
    t=theta_phase.t,
    d=np.transpose(feats),
    time_support=upsampled_pos.time_support,
    columns=["position", "theta"],
)
```


Now we have what we need to compute 2D tuning curves. Let's apply `nap.compute_2d_tuning_curves` on our reduced group of units, `good_spikes`, using 20 bins for each feature. 

This function will return two outputs:
1. A dictionary of the 2D tuning curves, where dictionary keys correspond to the unit label
2. A list with length 2 containing the feature bin centers
   

```{code-cell} ipython3
tuning_curves, [pos_x, phase_y] = nap.compute_2d_tuning_curves(good_spikes, features, 20)
```


We can plot the first 50 2D tuning curves and visualize how many of these units are phase precessing.


```{code-cell} ipython3
:tags: [render-all]

fig, axs = plt.subplots(10, 5, figsize=(10, 15), sharex=True, constrained_layout=True)
for i, f in enumerate(list(tuning_curves.keys())[:50]):
    idx = np.unravel_index(i, axs.shape)
    axs[idx].pcolormesh(pos_x, phase_y, tuning_curves[f])
    axs[idx].set_title(f)

fig.supylabel("Phase (rad)")
fig.supxlabel("Position (cm)");
```

- point out some of the precessing units

## Decoding position from spiking activity


Next we'll do a popular analysis in the rat hippocampal sphere: Bayesian decoding. This analysis is an elegent application of Bayes' rule in predicting the animal's location (or other behavioral variables) from neural activity at some point in time. 


### Background

For a more in-depth background on Bayesian decoding, see the full version of this notebook online.

### ASIDE: Cross-validation
:::{admonition} Important
:class: important render-user render-presenter

Generally this method is cross-validated, which means you train the model on one set of data and test the model on a different, held-out data set. For Bayesian decoding, the "model" refers to the model *likelihood*, which is computed from the tuning curves. Run the code below if you want to use a separate training set to compute the tuning curves.

:::
```{code-cell} ipython3
:tags: [render-all]

# hold out trial from place field computation
run_train = forward_ep.set_diff(ex_run_ep)
# get position of training set
position_train = position.restrict(run_train)
# compute place fields using training set
place_fields = nap.compute_1d_tuning_curves(spikes, position_train, nb_bins=50)
# smooth place fields
place_fields[:] = gaussian_filter1d(place_fields.values, 1, axis=0)
```
### Run 1D decoder


With a single dimension in our tuning curves (position), we can apply Bayesian decoding using the function `nap.decode_1d`. This function requires the following inputs:
- `tuning_curves`: a `pandas.DataFrame`, computed by `nap.compute_1d_tuning_curves`, with the tuning curves relative to the feature being decoded
- `group`: a `TsGroup` of spike times, or a `TsdFrame` of spike counts, for each unit in `tuning_curves`.
- `ep`: an `IntervalSet` containing the epoch to be decoded
- `bin_size`: the time length, in seconds, of each decoded bin. If `group` is a `TsGroup` of spike times, this determines how the spikes are binned in time. If `group` is a `TsdFrame` of spike counts, this should be the bin size used for the counts.

This function will return two outputs:
- a `Tsd` containing the decoded feature at each decoded time point
- a `TsdFrame` containing the decoded probability of each feature value at each decoded time point, where the column names are the corresponding feature values



To increase decoder accuracy, we'll want to use the tuning curves of all the units in `spikes`. Recompute `place_fields` using `nap.compute_1d_tuning_curves` for all available units. (You can skip this if you're using the cross-validated `place_fields` from above.)


```{code-cell} ipython3
place_fields = nap.compute_1d_tuning_curves(spikes, position, nb_bins=50)
```


    
Let's decode position during `ex_run_ep` using 50 ms time bins.


```{code-cell} ipython3
decoded_position, decoded_prob = nap.decode_1d(place_fields, spikes, ex_run_ep, 0.05)
```


Let's plot decoded position with the animal's true position. We'll overlay them on a heat map of the decoded probability to visualize the confidence of the decoder.


```{code-cell} ipython3
:tags: [render-all]

fig,ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
c = ax.pcolormesh(decoded_position.index,place_fields.index,np.transpose(decoded_prob))
ax.plot(decoded_position, "--", color="red", label="decoded position")
ax.plot(ex_position, color="red", label="true position")
ax.legend()
fig.colorbar(c, label="decoded probability")
ax.set(xlabel="Time (s)", ylabel="Position (cm)", );
```

- note decoder error

### Smooth spike counts


One way to improve our decoder is to supply smoothed spike counts to `nap.decode_1d`. We can smooth the spike counts by convolving them with a kernel of ones; this is equivalent to applying a moving sum to adjacent bins, where the length of the kernel is the number of adjacent bins being added together. You can think of this as counting spikes in a *sliding window* that shifts in shorter increments than the window's width, resulting in bins that overlap. This combines the accuracy of using a wider time bin with the temporal resolution of a shorter time bin.



- example: 200ms sliding window that shifts by 40ms

```{code-cell} ipython3
:tags: [render-all]

ex_counts = spikes[unit].restrict(ex_run_ep).count(0.04)
workshop_utils.animate_1d_convolution(ex_counts, np.ones(5), tsd_label="original counts", kernel_label="moving sum", conv_label="convolved counts")
```

- convolve kernel (yellow) centered at time point with original counts (blue)
- sum counts in center bin with two before and two after (shaded green)
- result is smoothed counts (bottom green)


Let's compute the smoothed counts for all units.

1. On spike times restricted to `ep_run_ep`, count spikes in $40 ms$ bins using the pynapple object method `count`.

```{code-cell} ipython3
counts = spikes.restrict(ex_run_ep).count(0.04)
```

2. Convolve the counts with the kernel `np.ones(5)` using the pynapple object method `convolve`.

```{code-cell} ipython3
smth_counts = counts.convolve(np.ones(5))
```


Now we can use `nap.decode_1d` again with our smoothed counts in place of the raw spike times. Note that the bin size we'll want to provide is $200 ms$, since this is the true width of each bin.


```{code-cell} ipython3
smth_decoded_position, smth_decoded_prob = nap.decode_1d(place_fields, smth_counts, ex_run_ep, bin_size=0.2)
```


Let's plot the results.


```{code-cell} ipython3
:tags: [render-all]

fig,ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
c = ax.pcolormesh(smth_decoded_position.index,place_fields.index,np.transpose(smth_decoded_prob))
ax.plot(smth_decoded_position, "--", color="red", label="decoded position")
ax.plot(ex_position, color="red", label="true position")
ax.legend()
fig.colorbar(c, label="decoded probability")
ax.set(xlabel="Time (s)", ylabel="Position (cm)", );
```

- improved approximation

### Bonus: theta sequences


Units phase precessing together creates fast, spatial sequences around the animal's true position. We can reveal this by decoding at an even shorter time scale, which will appear as smooth errors in the decoder.



1. Get smoothed counts for a sliding window of $50 ms$ shifted by $10 ms$.


```{code-cell} ipython3
counts = spikes.restrict(ex_run_ep).count(0.01)
smth_counts = counts.convolve(np.ones(5))
```


2. Use `nap.decode_1d` to get the smoothed decoded position.


```{code-cell} ipython3
smth_decoded_position, smth_decoded_prob = nap.decode_1d(place_fields, smth_counts, ex_run_ep, bin_size=0.05)
```

```{code-cell} ipython3
counts = spikes.restrict(ex_run_ep).count(0.01)
smth_counts = counts.convolve(np.ones(5))
smth_decoded_position, smth_decoded_prob = nap.decode_1d(place_fields, smth_counts, ex_run_ep, bin_size=0.05)
```

    
We'll make the same plot as before to visualize the results, but plot it alongside the raw and filtered LFP.


```{code-cell} ipython3
:tags: [render-all]

fig, axs = plt.subplots(2, 1, figsize=(10, 4), constrained_layout=True, height_ratios=[3,1], sharex=True)
c = axs[0].pcolormesh(smth_decoded_prob.index, smth_decoded_prob.columns, np.transpose(smth_decoded_prob))
p1 = axs[0].plot(smth_decoded_position, "--", color="r")
p2 = axs[0].plot(ex_position, color="r")
axs[0].set_ylabel("Position (cm)")
axs[0].legend([p1[0],p2[0]],["decoded position","true position"])
fig.colorbar(c, label = "predicted probability")

axs[1].plot(ex_lfp_run)
axs[1].plot(theta_band.restrict(ex_run_ep))
axs[1].set_ylabel("LFP (a.u.)")

fig.supxlabel("Time (s)");
```

- position oscillates within cycles of theta 
- "sweep" is a "theta sequence"
- active topic of research

```{code-cell} ipython3

```