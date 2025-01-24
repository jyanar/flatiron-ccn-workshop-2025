#!/usr/bin/env python3

import click
import nemos as nmo
import pooch

NEMOS_FILES = [
    "allen_478498617.nwb",
    "Mouse32-140822.nwb",
]

DATA_REGISTRY = {
    "Achilles_10252013_EEG.nwb": "a97a69d231e7e91c07e24890225f8fe4636bac054de50345551f32fc46b9efdd",
}

DATA_URLS = {
    "Achilles_10252013_EEG.nwb": "https://osf.io/2dfvp/download",
}

DATA_ENV = "NEMOS_DATA_DIR"

DOWNLOADABLE_FILES = NEMOS_FILES + list(DATA_REGISTRY.keys())


def fetch_data(dataset_name, path=None):
    """
    Fetch a data set for the neuroRSE workshop, including datasets not included in the NeMoS registry.
    This essentially adds a second registry for the workshop, while still using the default download location as NeMoS.

    Parameters
    ----------
    dataset_name : str
        Name of the data set to fetch.
    path : str, optional
        Path to the directory where the data set should be stored. If not provided, the default NeMoS cache directory is used.

    Returns
    -------
    str
        Path to the downloaded data set.
    """

    if dataset_name in NEMOS_FILES:
        return nmo.fetch.fetch_data(dataset_name, path=path)

    else:
        if path is None:
            path = pooch.os_cache("nemos")

        manager = pooch.create(
            path=path,
            base_url="",
            urls=DATA_URLS,
            registry=DATA_REGISTRY,
            allow_updates="POOCH_ALLOW_UPDATES",
            env=DATA_ENV,
        )

        return manager.fetch(dataset_name)


def fetch_all():
    """Download data.

    By default, this will be in data directory in this repo. To overwrite, set
    NEMOS_DATA_DIR environment variable, e.g.,

    NEMOS_DATA_DIR=path/to/data_dir python fetch.py

    """
    for f in DOWNLOADABLE_FILES:
        fetch_data(f)


# need this seeming duplicate because a function wrapped in the click decorator will
# kill the process when it finishes, and sometimes we want to call fetch_all from within
# an existing python process
@click.command()
def main():
    """fetch all data

    By default, this will be in data directory in this repo. To overwrite, set
    NEMOS_DATA_DIR environment variable, e.g.,

    NEMOS_DATA_DIR=path/to/data_dir python fetch.py

    """
    fetch_all()


if __name__ == "__main__":
    main()
