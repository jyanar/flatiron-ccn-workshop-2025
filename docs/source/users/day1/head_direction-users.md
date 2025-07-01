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
```{code-cell}
# enter code here
```



The content of the NWB file is not loaded yet. The object `data` behaves like a dictionnary.

**Question:** Can you load the spike times from the NWB and call the variables `spikes`?
```{code-cell}
# enter code here
```



**Question:** And print it?
```{code-cell}
# enter code here
```



There are a lot of neurons. The neurons that interest us are the neurons labeled `adn`. 

**Question:** Using the [slicing method](https://pynapple.org/user_guide/03_metadata.html#using-metadata-to-slice-objects) of your choice, can you select only the neurons in `adn` that are above 1 Hz firing rate?
```{code-cell}
# enter code here
```



The NWB file contains other informations about the recording. `ry` contains the value of the head-direction of the animal over time. 

**Question:** Can you extract the angle of the animal in a variable called `angle` and print it?
```{code-cell}
# enter code here
```



But are the data actually loaded ... or not?

**Question:** Can you print the underlying data array of `angle`?
```{code-cell}
# enter code here
```



The animal was recorded during wakefulness and sleep. 

**Question:** Can you extract the behavioral intervals in a variable called `epochs`?
```{code-cell}
# enter code here
```



NWB file can save intervals with multiple labels. The object `IntervalSet` includes the labels as a metadata object.

**Question:** Using the column `tags`, can you create one `IntervalSet` object for intervals labeled `wake` and one `IntervalSet` object for intervals labeled `sleep`?
```{code-cell}
# enter code here
```

## Compute tuning curves


Now that we have spikes and a behavioral feature (i.e. head-direction), we would like to compute the firing rate of neurons as a function of the variable `angle` during `wake_ep`.
To do this in pynapple, all you need is a single line of code!

**Question:** can you compute the firing rate of ADn units as a function of heading direction, i.e. a head-direction tuning curve and call the variable `tuning_curves`?
```{code-cell}
# enter code here
```



**Question:** Can you plot some tuning curves?
```{code-cell}
# enter code here
```



Most of those neurons are head-directions neurons.

The next cell allows us to get a quick estimate of the neurons's preferred direction.
```{code-cell} ipython3
:tags: [render-all]

pref_ang = tuning_curves.idxmax()
```


**Question:** Can you add it to the metainformation of `spikes`?
```{code-cell}
# enter code here
```



This index maps a neuron to a preferred direction between 0 and 360 degrees.

**Question:** Can you plot the spiking activity of the neurons based on their preferred direction as well as the head-direction of the animal?
For the sake of visibility, you should restrict the data to the following epoch : `ex_ep = nap.IntervalSet(start=8910, end=8960)`.
```{code-cell}
# enter code here
```

## Decode neural activity


Population activity clearly codes for head-direction. Can we use the spiking activity of the neurons to infer the current heading of the animal? The process is called bayesian decoding.

**Question:** Using the right pynapple function, can you compute the decoded angle from the spiking activity during wakefulness?
```{code-cell}
# enter code here
```



**Question:** ... and display the decoded angle next to the true angle?
```{code-cell}
# enter code here
```



Since the tuning curves were computed during wakefulness, it is a circular action to decode spiking activity during wakefulness.
We can try something more interesting by trying to decode the angle during sleep. 

**Question:** Can you instantiate an `IntervalSet` object called `rem_ep` that contains the epochs of REM sleep? You can check the contents of the NWB file by doing first `print(data)`
```{code-cell}
# enter code here
```



**Question:** Can you compute the decoded angle from the spiking activity during REM sleep?
```{code-cell}
# enter code here
```



**Question:** ... and display the decoded angle next to the spiking activity?
```{code-cell}
# enter code here
```

## Compute correlograms


We see that some neurons have a correlated activity. Can we measure it?

**Question:** Can you compute cross-correlograms during wake for all pairs of neurons and call it `cc_wake`?
```{code-cell}
# enter code here
```




**Question:** can you plot the cross-correlogram during wake of 2 neurons firing for the same direction?
```{code-cell}
# enter code here
```



**Question:** can you plot the cross-correlogram during wake of 2 neurons firing for opposite directions?
```{code-cell}
# enter code here
```



Pairwise correlation were computed during wakefulness. The activity of the neurons was also recorded during sleep.

**Question:** can you compute the cross-correlograms during sleep?
```{code-cell}
# enter code here
```



**Question:** can you display the cross-correlogram for wakefulness and sleep of the same pairs of neurons?
```{code-cell}
# enter code here
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
```{code-cell}
# enter code here
```



**Question:** ...and plot the spikes?
```{code-cell}
# enter code here
```



**Question:** Can you compute the mean firing rate of the PETH around `crossing_times` in bins of 100 ms?
```{code-cell}
# enter code here
```



**Question:** ... and plot it?
```{code-cell}
# enter code here
```



Is this a strong effect? We would like to compare this to surrogate dataset.

**Question:** Shuffling the spike trains, can you generate a mean random PETH to compare to the true mean PETH?
```{code-cell}
# enter code here
```
