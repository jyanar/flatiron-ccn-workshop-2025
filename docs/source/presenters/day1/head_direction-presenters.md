---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.6
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
```{code-cell} ipython3
:tags: [render-all]
%matplotlib inline
```
# Data analysis with pynapple
This notebook has had all its explanatory text removed and has not been run.
 It is intended to be downloaded and run locally (or on the provided binder)
 while listening to the presenter's explanation. In order to see the fully
 rendered of this notebook, go [here](../../full/day1/head_direction.md)
## Learning objectives


- Loading a NWB file
- Compute tuning curves
- Decode neural activity
- Compute correlograms
- Compute perievent


The pynapple documentation can be found [here](https://pynapple.org).

The API documentation is [here](https://pynapple.org/api.html).


Let's start by importing the pynapple package, matplotlib, numpy to see if everything is correctly installed. 
If an import fails, you can do `!pip install pynapple matplotlib` in a cell to fix it.


```{code-cell} ipython3
:tags: [render-all]
import pynapple as nap
import matplotlib.pyplot as plt
import workshop_utils
import numpy as np
```
## Loading a NWB file


Pynapple commit to support NWB for data loading. 
If you have installed the repository, you can run the following cell:


```{code-cell} ipython3
:tags: [render-all]
path = workshop_utils.fetch_data("Mouse32-140822.nwb")

print(path)
```


Pynapple provides the convenience function `nap.load_file` for loading a NWB file.

**Question:** Can you open the NWB file giving the variable `path` to the function `load_file` and call the output `data`?


```{code-cell} ipython3
data = nap.load_file(path)

print(data)
```


The content of the NWB file is not loaded yet. The object `data` behaves like a dictionnary.

**Question:** Can you load the spike times from the NWB and call the variables `spikes`?


```{code-cell} ipython3
spikes = data["units"]  # Get spike timings
```


**Question:** And print it?

```{code-cell} ipython3
print(spikes)
```


There are a lot of neurons. The neurons that interest us are the neurons labeled `adn`. 

**Question:** Using the [slicing method](https://pynapple.org/user_guide/03_metadata.html#using-metadata-to-slice-objects) of your choice, can you select only the neurons in `adn` that are above 1 Hz firing rate?


```{code-cell} ipython3
spikes = spikes[(spikes.location=='adn') & (spikes.rate>1.0)]

print(spikes)
```


The NWB file contains other informations about the recording. `ry` contains the value of the head-direction of the animal over time. 

**Question:** Can you extract the angle of the animal in a variable called `angle` and print it?


```{code-cell} ipython3
angle = data["ry"]
print(angle)
```


But are the data actually loaded ... or not?

**Question:** Can you print the underlying data array of `angle`?


```{code-cell} ipython3
print(angle.d)
```


The animal was recorded during wakefulness and sleep. 

**Question:** Can you extract the behavioral intervals in a variable called `epochs`?


```{code-cell} ipython3
epochs = data["epochs"]

print(epochs)
```


NWB file can save intervals with multiple labels. The object `IntervalSet` includes the labels as a metadata object.

**Question:** Using the column `tags`, can you create one `IntervalSet` object for intervals labeled `wake` and one `IntervalSet` object for intervals labeled `sleep`?


```{code-cell} ipython3
wake_ep = epochs[epochs.tags=="wake"]
sleep_ep = epochs[epochs.tags=="sleep"]
```
## Compute tuning curves


Now that we have spikes and a behavioral feature (i.e. head-direction), we would like to compute the firing rate of neurons as a function of the variable `angle` during `wake_ep`.
To do this in pynapple, all you need is a single line of code!

**Question:** can you compute the firing rate of ADn units as a function of heading direction, i.e. a head-direction tuning curve and call the variable `tuning_curves`?


```{code-cell} ipython3
tuning_curves = nap.compute_1d_tuning_curves(
    group=spikes, 
    feature=angle, 
    nb_bins=61, 
    ep = angle.time_support,
    minmax=(0, 2 * np.pi)
    )
```


**Question:** Can you plot some tuning curves?


```{code-cell} ipython3
plt.figure()
plt.subplot(221)
plt.plot(tuning_curves.iloc[:,0])
plt.subplot(222,projection='polar')
plt.plot(tuning_curves.iloc[:,0])
plt.subplot(223)
plt.plot(tuning_curves.iloc[:,1])
plt.subplot(224,projection='polar')
plt.plot(tuning_curves.iloc[:,1])
```


Most of those neurons are head-directions neurons.

The next cell allows us to get a quick estimate of the neurons's preferred direction.


```{code-cell} ipython3
:tags: [render-all]

pref_ang = tuning_curves.idxmax()
```


**Question:** Can you add it to the metainformation of `spikes`?


```{code-cell} ipython3
spikes['pref_ang'] = pref_ang
```


This index maps a neuron to a preferred direction between 0 and 360 degrees.

**Question:** Can you plot the spiking activity of the neurons based on their preferred direction as well as the head-direction of the animal?
For the sake of visibility, you should restrict the data to the following epoch : `ex_ep = nap.IntervalSet(start=8910, end=8960)`.


```{code-cell} ipython3
ex_ep = nap.IntervalSet(start=8910, end=8960)


plt.figure()
plt.subplot(211)
plt.plot(angle.restrict(ex_ep))
plt.ylim(0, 2*np.pi)

plt.subplot(212)
plt.plot(spikes.restrict(ex_ep).to_tsd("pref_ang"), '|')
```
## Decode neural activity


Population activity clearly codes for head-direction. Can we use the spiking activity of the neurons to infer the current heading of the animal? The process is called bayesian decoding.

**Question:** Using the right pynapple function, can you compute the decoded angle from the spiking activity during wakefulness?


```{code-cell} ipython3
decoded, proba_feature = nap.decode_1d(
    tuning_curves=tuning_curves,
    group=spikes,
    ep=wake_ep,
    bin_size=0.3,  # second
)
```


**Question:** ... and display the decoded angle next to the true angle?


```{code-cell} ipython3
plt.figure()
plt.subplot(211)
plt.plot(angle.restrict(ex_ep))
plt.plot(decoded.restrict(ex_ep), label="decoded")
plt.ylim(0, 2*np.pi)

plt.subplot(212)
plt.plot(spikes.restrict(ex_ep).to_tsd("pref_ang"), '|')
```


Since the tuning curves were computed during wakefulness, it is a circular action to decode spiking activity during wakefulness.
We can try something more interesting by trying to decode the angle during sleep. 

**Question:** Can you instantiate an `IntervalSet` object called `rem_ep` that contains the epochs of REM sleep? You can check the contents of the NWB file by doing first `print(data)`


```{code-cell} ipython3
rem_ep = data['rem'][1]
```


**Question:** Can you compute the decoded angle from the spiking activity during REM sleep?


```{code-cell} ipython3
decoded, proba_feature = nap.decode_1d(
    tuning_curves=tuning_curves,
    group=spikes,
    ep=rem_ep,
    bin_size=0.3,  # second
)
```


**Question:** ... and display the decoded angle next to the spiking activity?


```{code-cell} ipython3
plt.figure()
plt.subplot(211)
plt.plot(decoded.restrict(rem_ep), label="decoded")
plt.ylim(0, 2*np.pi)

plt.subplot(212)
plt.plot(spikes.restrict(rem_ep).to_tsd("pref_ang"), '|')
```
## Compute correlograms


We see that some neurons have a correlated activity. Can we measure it?

**Question:** Can you compute cross-correlograms during wake for all pairs of neurons and call it `cc_wake`?


```{code-cell} ipython3
cc_wake = nap.compute_crosscorrelogram(spikes, binsize=0.2, windowsize=20.0, ep=wake_ep)
```



**Question:** can you plot the cross-correlogram during wake of 2 neurons firing for the same direction?


```{code-cell} ipython3
index = spikes.keys()


plt.figure()
plt.subplot(121)
plt.plot(tuning_curves[7])
plt.plot(tuning_curves[20])
plt.subplot(122)
plt.plot(cc_wake[(7, 20)])
```


**Question:** can you plot the cross-correlogram during wake of 2 neurons firing for opposite directions?


```{code-cell} ipython3
index = spikes.keys()


plt.figure()
plt.subplot(121)
plt.plot(tuning_curves[7])
plt.plot(tuning_curves[26])
plt.subplot(122)
plt.plot(cc_wake[(7, 26)])
```


Pairwise correlation were computed during wakefulness. The activity of the neurons was also recorded during sleep.

**Question:** can you compute the cross-correlograms during sleep?


```{code-cell} ipython3
cc_sleep = nap.compute_crosscorrelogram(spikes, 0.02, 1.0, ep=sleep_ep)
```


**Question:** can you display the cross-correlogram for wakefulness and sleep of the same pairs of neurons?

```{code-cell} ipython3
plt.figure()
plt.subplot(131, projection='polar')
plt.plot(tuning_curves[7])
plt.plot(tuning_curves[20])
plt.subplot(132)
plt.plot(cc_wake[(7, 20)])
plt.subplot(133)
plt.plot(cc_sleep[(7, 20)])
```
```{code-cell} ipython3
plt.figure()
plt.subplot(131, projection='polar')
plt.plot(tuning_curves[7])
plt.plot(tuning_curves[26])
plt.subplot(132)
plt.plot(cc_wake[(7, 26)])
plt.subplot(133)
plt.plot(cc_sleep[(7, 26)])
```
## Compute perievent


Sometimes, some events occurs during recording such as rewards. There was no particular events during this recording but we can look for when the head-direction is close to a particular direction as an event.


```{code-cell} ipython3
:tags: [render-all]

plt.figure()
plt.plot(tuning_curves[9])
plt.axvline(1.5)

crossing_times = np.cos(angle).threshold(np.cos(1.5), "below").time_support.start
crossing_times = nap.Ts(t=crossing_times)
```


**Question:** Can you compute a perievent time histogram around the timestamps defined in `crossing_times` for neuron 9?


```{code-cell} ipython3
peth = nap.compute_perievent(spikes[9], crossing_times, minmax=(-2, 2))
```


**Question:** ...and plot the spikes?


```{code-cell} ipython3
plt.figure()
plt.plot(peth.to_tsd(), '|')
```


**Question:** Can you compute the mean firing rate of the PETH around `crossing_times` in bins of 100 ms?


```{code-cell} ipython3
mean_fr = np.mean(peth.count(0.1)/0.1, 1)
```


**Question:** ... and plot it?


```{code-cell} ipython3
plt.figure()
plt.subplot(211)
plt.plot(peth.to_tsd(), '|')
plt.subplot(212)
plt.plot(mean_fr)
```


Is this a strong effect? We would like to compare this to surrogate dataset.

**Question:** Shuffling the spike trains, can you generate a mean random PETH to compare to the true mean PETH?

```{code-cell} ipython3
rand_ts = nap.shuffle_ts_intervals(spikes[9])
rand_peth = nap.compute_perievent(rand_ts, crossing_times, minmax=(-2, 2))

plt.figure()
plt.subplot(211)
plt.plot(peth.to_tsd(), '|')
plt.subplot(212)
plt.plot(mean_fr)
plt.plot(np.mean(rand_peth.count(0.1)/0.1, 1))
```