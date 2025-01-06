# Data

This directory will be populated by NWB files once you run the setup script. See the [setup instructions](https://flatironinstitute.github.io/ccn-software-jan-2025/#setup) for more details.

You can access these files either directly or by using `nemos`, and you can load them using pynapple:

``` python
import pynapple as nap
import nemos as nmo

data = nap.load_file(nmo.fetch.fetch_data("allen_478498617.nwb"))
```

We'll explore these datasets during the workshop!
