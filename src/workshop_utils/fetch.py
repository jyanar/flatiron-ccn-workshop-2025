#!/usr/bin/env python3

import click
import nemos as nmo

DOWNLOADABLE_FILES = ["allen_478498617.nwb", "Mouse32-140822.nwb", "Achilles_10252013.nwb"]


@click.command()
def main():
    """Download data.

    By default, this will be in data directory in this repo. To overwrite, set
    NEMOS_DATA_DIR environment variable, e.g.,

    NEMOS_DATA_DIR=path/to/data_dir python fetch.py

    """
    for f in DOWNLOADABLE_FILES:
        nmo.fetch.fetch_data(f)

if __name__ == '__main__':
    main()
