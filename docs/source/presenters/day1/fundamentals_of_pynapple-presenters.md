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
:::{admonition} Download
:class: important render-all

This notebook can be downloaded as **{nb-download}`fundamentals_of_pynapple-presenters.ipynb`**. See the button at the top right to download as markdown or pdf.
:::


# Learning the fundamentals of pynapple
This notebook has had all its explanatory text removed and has not been run.
 It is intended to be downloaded and run locally (or on the provided binder)
 while listening to the presenter's explanation. In order to see the fully
 rendered of this notebook, go [here](../../full/day1/fundamentals_of_pynapple.md)

## Learning objectives


- Instantiate the pynapple objects
- Make the pynapple objects interact
- Use numpy with pynapple
- Slicing pynapple objects
- Adding metadata to pynapple objects
- Learn the core functions of pynapple

The pynapple documentation can be found [here](https://pynapple.org).

The documentation for objects and method of the core of pynapple is [here](https://pynapple.org/api.html).


Let's start by importing the pynapple package and matplotlib to see if everything is correctly installed.
If an import fails, you can do `!pip install pynapple matplotlib` in a cell to fix it.


```{code-cell} ipython3
:tags: [render-all]
import pynapple as nap
import matplotlib.pyplot as plt
import numpy as np
import workshop_utils
```
```{code-cell} ipython3
:tags: [render-all]
var1 = np.random.randn(100) # Variable 1
tsp1 = np.arange(100) # The timesteps of variable 1

var2 = np.random.randn(100, 3) # Variable 2
tsp2 = np.arange(0, 100, 1) # The timesteps of variable 2
col2 = ['pineapple', 'banana', 'tomato'] # The name of each columns of var2

var3 = np.random.randn(1000, 4, 5) # Variable 3
tsp3 = np.arange(0, 100, 0.1) # The timesteps of variable 3

random_times_1 = np.array([3.14, 37.0, 42.0])
random_times_2 = np.array([10, 25, 50, 70])
random_times_3 = np.sort(np.random.uniform(10, 80, 100))

starts_1 = np.array([10000, 60000, 90000]) # starts of an epoch in `ms`
ends_1 = np.array([20000, 80000, 95000]) # ends in `ms`
```
## Instantiate pynapple objects 


**Question:** Can you instantiate the right pynapple objects for `var1`, `var2` and `var3`? Objects should be named respectively `tsd1`, `tsd2` and `tsd3`. Don't forget the column name for `var2`.


```{code-cell} ipython3
tsd1 = nap.Tsd(t=tsp1, d=var1)
tsd2 = nap.TsdFrame(t=tsp2, d=var2, columns = col2)
tsd3 = nap.TsdTensor(t=tsp3, d=var3)
```


**Question:** Can you print `tsd1`?


```{code-cell} ipython3
print(tsd1)
```


**Question:** Can you print `tsd2`?


```{code-cell} ipython3
print(tsd2)
```


**Question:** Can you print `tsd3`?


```{code-cell} ipython3
print(tsd3)
```


**Question:** Can you create an `IntervalSet` called `ep` out of `starts_1` and `ends_1` and print it? Be careful, times given above are in `ms`.


```{code-cell} ipython3
ep = nap.IntervalSet(start=starts_1, end=ends_1, time_units='ms')
print(ep)
```


The experiment generated a set of timestamps from 3 different channels.

**Question:** Can you instantiate the corresponding pynapple object (`ts1`, `ts2`, `ts3`) for each one of them?


```{code-cell} ipython3
ts1 = nap.Ts(t=random_times_1)
ts2 = nap.Ts(t=random_times_2)
ts3 = nap.Ts(t=random_times_3)
```


This is a lot of timestamps to carry around as well.

**Question:** Can you instantiate the right pynapple object (call it `tsgroup`) to group them together?


```{code-cell} ipython3
tsgroup = nap.TsGroup({0:ts1, 1:ts2, 2:ts3})
```


**Question:** ... and print it?


```{code-cell} ipython3
print(tsgroup)
```
## Interaction between pynapple objects 


**Question:** Can you print the `time_support` of `tsgroup`?


```{code-cell} ipython3
print(tsgroup.time_support)
```


**Question:** can you recreate the `tsgroup` object passing the right `time_support` during initialisation?


```{code-cell} ipython3
tsgroup = nap.TsGroup({0:ts1, 1:tsd2, 2:ts3}, time_support = nap.IntervalSet(0, 100))
```


**Question:** Can you print the `time_support` and `rate` to see how they changed?


```{code-cell} ipython3
print(tsgroup.time_support)
print(tsgroup.rate)
```


Now you realized the variable `tsd1` has some noise. The good signal is between 10 and 30 seconds and  50 and 100.

**Question:** Can you create an `IntervalSet` object called `ep_signal` and use it to restrict the variable `tsd1`?


```{code-cell} ipython3
ep_signal = nap.IntervalSet(start=[10, 50], end=[30, 100])

tsd1 = tsd1.restrict(ep_signal)
```

You can print `tsd1` to check that the timestamps are in fact within `ep`.
You can also check the `time_support` of `tsd1` to see that it has been updated.

```{code-cell} ipython3
print(tsd1)
print(tsd1.time_support)
```
```{code-cell} ipython3
ep_tmp = nap.IntervalSet(np.sort(np.random.uniform(0, 100, 20)))
print(ep_tmp)
```


**Question:** Can you do the intersection of `ep_signal` and `ep_tmp`?

```{code-cell} ipython3
print(ep_signal.intersect(ep_tmp))
```

You can visualize IntervalSet using the function `workshop_utils.visualize_intervals` we provide.

```{code-cell} ipython3
:tags: [render-all]
workshop_utils.visualize_intervals([ep_signal, ep_tmp, ep_signal.intersect(ep_tmp)])
```


**Question:** Can you do the union of `ep_signal` and `ep_tmp`?

```{code-cell} ipython3
print(ep_signal.union(ep_tmp))
```


**Question:** ... and visualize it?

```{code-cell} ipython3
workshop_utils.visualize_intervals([ep_signal, ep_tmp, ep_signal.union(ep_tmp)])
```


**Question:** Can you do the difference of `ep_signal` and `ep_tmp`?

```{code-cell} ipython3
print(ep_signal.set_diff(ep_tmp))
```


**Question:** ... and visualize it?

```{code-cell} ipython3
workshop_utils.visualize_intervals([ep_signal, ep_tmp, ep_signal.set_diff(ep_tmp)])
```
## Numpy & pynapple


Pynapple objects behaves very similarly like numpy array. They can be sliced with the following syntax :

  `tsd[0:10] # First 10 elements`

Arithmetical operations are available as well :

  `tsd = tsd + 1`

Finally numpy functions works directly. Let's imagine `tsd3` is a movie with frame size (4,5).

**Question:** Can you compute the average frame along the time axis using `np.mean` and print the result?


```{code-cell} ipython3
print(np.mean(tsd3, 0))
```


**Question:**: can you compute the average of `tsd2` for each timestamps and print it?

```{code-cell} ipython3
print(np.mean(tsd2, 1))
```
## Slicing pynapple objects 


**Question:** Can you extract the first and last epoch of `ep` in a new `IntervalSet`?

```{code-cell} ipython3
print(ep[[0,2]])
```


Sometimes you want to get a data point as close as possible in time to another timestamps.

**Question:** Using the `get` method, can you get the data point from `tsd3` as close as possible to the time 50.1 seconds?


```{code-cell} ipython3
print(tsd3.get(50.1))
```
## Metadata


Metadata are ubiquitous in neuroscience. They can be added to 3 pynapple objects :

- `TsGroup` : to label neurons in electrophysiology
- `IntervalSet` : to label intervals
- `TsdFrame` : to label neurons in calcium imaging


**Question:** Can you run the following command `tsgroup['planet'] = ['mars', 'venus', 'saturn']`

```{code-cell} ipython3
tsgroup['planet'] = ['mars', 'venus', 'saturn']
```


**Question:** ... and print it?

```{code-cell} ipython3
print(tsgroup)
```


The object `ep` has 3 epochs labelled `['left', 'right', 'left']`. 

**Question:** Can you add them as a metadata column called `direction`?


```{code-cell} ipython3
ep['direction'] = ['left', 'right', 'left']
print(ep)
```


The object `tsd2` has 3 columns. Each column correspond to the rgb colors `[(0,0,1), (0.5, 0.5, 1), (0.1, 0.2, 0.3)]`. 

**Question:** Can you add them as metadata of `tsd2`?

```{code-cell} ipython3
tsd2['colors'] = [(0,0,1), (0.5, 0.5, 1), (0.1, 0.2, 0.3)]
print(tsd2)
```


You can also add metadata at initialization as a dictionnary using the keyword argument `metadata` : 

```{code-cell} ipython3
:tags: [render-all]

tsgroup = nap.TsGroup({0:ts1, 1:ts2, 2:ts3}, metadata={'planet':['mars','venus', 'saturn']})

print(tsgroup)
```


Metadata are accessible either as attributes (i.e. `tsgroup.planet`) or as dictionnary-like keys (i.e. `ep['direction']`).

They can be used to slice objects. 

**Question:** Can you select only the elements of `tsgroup` with rate below 1Hz?

```{code-cell} ipython
print(tsgroup[tsgroup.rate<1.0])

print(tsgroup[tsgroup['rate']<1.0])

print(tsgroup.getby_threshold("rate", 1, "<"))

```


**Question:** Can you select the intervals in `ep` labelled as `'left'`?

```{code-cell} ipython
print(ep[ep.direction=='left'])
```
### Special case of slicing : `TsdFrame`
```{code-cell} ipython
:tags: [render-all]

tsdframe = nap.TsdFrame(t=np.arange(4), d=np.random.randn(4,3),
  columns = [12, 0, 1], metadata={'alpha':[2,1,0]})

print(tsdframe)
```


**Question:** What happen when you do `tsdframe[0]` vs `tsdframe[:,0]` vs `tsdframe[[12,1]]`

```{code-cell} ipython
print(tsdframe[0])
```


**Question:** What happen when you do `tsdframe.loc[0]` and `tsdframe.loc[[0,1]]`

```{code-cell} ipython
print(tsdframe.loc[0])
print(tsdframe.loc[[0,1]])
```


**Question:** What happen when you do `tsdframe[tsdframe.alpha==2]`


```{code-cell} ipython
print(tsdframe[tsdframe.alpha==2])
```
## Core functions of pynapple 


This part focuses on the most important core functions of pynapple.

**Question:** Using the `count` function, can you count the number of events within 1 second bins for `tsgroup` over the `ep_signal` intervals?

```{code-cell} ipython3
count = tsgroup.count(1, ep_signal)
print(count)
```
```{code-cell} ipython3
plt.figure()
ax = plt.subplot(211)
plt.plot(count, 'o-')
plt.subplot(212, sharex=ax)
plt.plot(tsgroup.restrict(ep_signal).to_tsd(), '|')
```


From a set of timestamps, you want to assign them a set of values with the closest point in time of another time series.

**Question:** Using the function `value_from`, can you assign values to `ts2` from the `tsd1` time series and call the output `new_tsd`?


```{code-cell} ipython3
new_tsd = ts2.value_from(tsd1)
```


**Question:** Can you plot together `tsd1`, `ts2` and `new_tsd`?


```{code-cell} ipython3
plt.figure()
plt.plot(tsd1)
plt.plot(new_tsd, 'o-')
plt.plot(ts2.fillna(0), 'o')
```


One important aspect of data analysis is to bring data to the same size. Pynapple provides the `bin_average` function to downsample data.

**Question:** Can you downsample `tsd2` to one time point every 5 seconds?


```{code-cell} ipython3
new_tsd2 = tsd2.bin_average(5.0)
```


**Question:** Can you plot the `tomato` column from `tsd2` as well as the downsampled version?


```{code-cell} ipython3
plt.figure()
plt.plot(tsd2['tomato'])
plt.plot(new_tsd2['tomato'], 'o-')
```


For `tsd1`, you want to find all the epochs for which the value is above 0.0. Pynapple provides the function `threshold` to get 1 dimensional time series above or below a certain value.

**Question**: Can you print the epochs for which `tsd1` is above 0.0?

```{code-cell} ipython3
ep_above = tsd1.threshold(0.0).time_support

print(ep_above)
```


**Question**: can you plot `tsd1` as well as the epochs for which `tsd1` is above 0.0?

```{code-cell} ipython3
plt.figure()
plt.plot(tsd1)
plt.plot(tsd1.threshold(0.0), 'o-')
[plt.axvspan(s, e, alpha=0.2) for s,e in ep_above.values]
```
## Important


**Question:** Does this work?
If not, please ask a TA.


```{code-cell} ipython3
:tags: [render-all]
import workshop_utils
path = workshop_utils.fetch_data("Mouse32-140822.nwb")
print(path)
```