---
sd_hide_title: true
---

# Welcome!

# CCN Software Workshop, Jan 2025

## Setup

Before the workshop, please try to complete the following steps. If you are unable to do so, we have an installation help session at the Flatiron Institute, 160 5th Ave, Room 418, from 2 to 5pm on Wednesday, January 29. Please come by!

0. Make sure you have `git` installed. It is installed by default on most Mac and Linux machines, but you may need to install it if you are on Windows. [These instructions](https://github.com/git-guides/install-git) should help.
1. Clone the github repo for this workshop:
   ```shell
   git clone https://github.com/flatironinstitute/ccn-software-jan-2025.git
   ```
2. Create a new python 3.11 virtual environment. If you do not have a preferred way of managing your python virtual environments, we recommend [miniconda](https://docs.anaconda.com/free/miniconda/). After installing it (if you have not done so already), run 
    ```shell
    conda create --name ccn-jan25 pip python=3.11
    ```
3. Activate your new environment: `
    ```shell
    conda activate ccn-jan25
    ```
4. Navigate to the cloned github repo and install the required dependencies. This will install pynapple and nemos, as well as jupyter and several other packages.
    ```shell
    cd ccn-software-jan-2025
    pip install -e .
    ```
5. Run our setup script to download data and prepare the notebooks:
    ```shell
    python scripts/setup.py
    ```
6. Confirm the installation and setup completed correctly by running:
    ```shell
    python scripts/check_setup.py
    ```

If `check_setup.py` tells you setup was successful, then you're good to go. Otherwise, please come to the installation help session on Wednesday, so everyone is ready to get started Thursday morning.

After doing the above, the `data/` and `notebooks/` directories within your local copy of the `ccn-software-jan-2025` repository will contain the NWB files and jupyter notebooks for the workshop.

We will work through the notebooks in the order they're listed on this website.

## Binder

Some usage notes:

- You are only allowed to have a single binder instance running at a time, so if you get the "already have an instance running error", go to the [binderhub page](https://binder.flatironinstitute.org/hub/hub/home) (or click on "check your currently running servers" on the right of the page) to join your running instance.
- If you lose connection halfway through the workshop, go to the [binderhub page](https://binder.flatironinstitute.org/hub/hub/home) to join your running instance rather than restarting the image.
- This is important because if you restart the image, **you will lose all data and progress**.
- The binder will be shutdown automatically after 1 day of inactivity or 7 days of total usage. Data will not persist after the binder instance shuts down, so **please download any notebooks** you want to keep.
- I will destroy this instance in 2 weeks. You can download your notebooks to keep them after the fact.

```{toctree}
:glob:
:caption: Day 1
:maxdepth: 1
full/day1/*
```

```{toctree}
:glob:
:caption: Day 2
:maxdepth: 1
full/day2/*
```

```{toctree}
:glob:
:caption: Day 1, for users
:maxdepth: 1
users/day1/*
```

```{toctree}
:glob:
:caption: Day 2, for users
:maxdepth: 1
users/day2/*
```
