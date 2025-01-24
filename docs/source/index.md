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

### Create a virtual environment with python 3.11

There are many ways to set up a python virtual environment. You can use your favorite way of doing so. If you don't have a preference or don't know what to do, choose one of the following:

:::::::{tab-set}
:sync-group: category

::::::{tab-item} uv
:sync: uv

:::::{tab-set}
:sync-group: os

::::{tab-item} Mac/Linux
:sync: posix

1. Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) by running:
   ```shell
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. Install python 3.11:
   ```shell
   uv python install 3.11
   ```
   
3. Navigate to your cloned repo and create a new virtual environment:
   ```shell
   cd ccn-software-jan-2025
   uv venv -p 3.11
   ```
   
4. Activate your new virtual environment by running:
   ```shell
   source .venv/bin/activate
   ```
::::

::::{tab-item} Windows
:sync: windows

1. Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) by opening up powershell and running:
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
2. Install python 3.11:
   ```powershell
   uv python install 3.11
   ```
   
3. Navigate to your cloned repo and create a new virtual environment:
   ```powershell
   cd ccn-software-jan-2025
   uv venv -p 3.11
   ```
   
4. Activate your new virtual environment by running:
   ```powershell
   .venv\Scripts\activate
   ```

   :::{warning}
   You may receive an error saying "running scripts is disabled on this system". If so, run `Set-ExecutionPolicy -Scope CurrentUser` and enter `Unrestricted`, then press `Y`.
   
   You may have to do this every time you open powershell.
   
   :::

::::
:::::
::::::

::::::{tab-item} conda / miniforge
:sync: conda

:::{warning}

Due to [recent changes](https://www.anaconda.com/blog/update-on-anacondas-terms-of-service-for-academia-and-research) to Anaconda's Terms of Service, the Simons Foundation cannot use the `defaults` conda channel and it is blocked on all Flatiron Institute wireless networks. You need to specify `conda-forge` instead (which is community-managed and open to all). The following instructions do so, but if you follow your normal workflow, you may hit issues.

:::

1. Install [miniforge](https://github.com/conda-forge/miniforge) if you do not have some version of `conda` or `mamba` installed already.
2. Create the new virtual environment by running:
    ```shell
    conda create --name ccn-jan25 pip python=3.11 -c conda-forge
    ```
    Note the `-c conda-forge`!

3. Activate your new environment and navigate to the cloned repo: 
    ```shell
    conda activate ccn-jan25
    cd ccn-software-jan-2025
    ```
::::::

:::::::

#### Install dependencies and setup notebooks
    
1. Install the required dependencies. This will install pynapple and nemos, as well as jupyter and several other packages.
    ::::{tab-set}
    :sync-group: category
    
    :::{tab-item} uv
    :sync: uv

    ```shell
    uv pip install -e .
    ```
    :::

    :::{tab-item} conda
    :sync: conda

    ```shell
    pip install -e .
    ```
    :::
    ::::

2. Run our setup script to download data and prepare the notebooks:
    ```shell
    python scripts/setup.py
    ```
3. Confirm the installation and setup completed correctly by running:
    ```shell
    python scripts/check_setup.py
    ```

If `check_setup.py` tells you setup was successful, check that you can run `jupyter lab notebooks/day2/current_injection-users.ipynb` and run the first few cells (up until the one containing `path = workshop_utils.fetch_data("allen_478498617.nwb")`). If that all works, then you're good to go. Otherwise, please come to the installation help session on Wednesday, so everyone is ready to get started Thursday morning.

After doing the above, the `data/` and `notebooks/` directories within your local copy of the `ccn-software-jan-2025` repository will contain the NWB files and jupyter notebooks for the workshop.

On the day of the workshop, we will run through the notebooks in the order they're listed on this website. To open them, navigate to the `notebooks/` directory, activate your virtual environment and start `jupyter lab`:

::::::{tab-set}
:sync-group: category

:::::{tab-item} uv
:sync: uv

::::{tab-set}
:sync-group: os

:::{tab-item} Mac/Linux
:sync: posix

```shell
cd path/to/ccn-software-jan-2025/notebooks
source ../.venv/bin/activate
jupyter lab
```
:::

:::{tab-item} Windows
:sync: windows

```powershell
cd path\to\ccn-software-jan-2025\notebooks
..\.venv\Scripts\activate
jupyter lab
```
:::

:::::

:::::{tab-item} conda / miniforge
:sync: conda

```shell
cd path/to/ccn-software-jan-2025/notebooks
conda activate ccn-jan25
jupyter lab
```

:::::

::::::

## Troubleshooting

- If you are on Mac and get an error related to `ruamel.yaml` (or `clang`) when running `pip install -e .`, we think this can be fixed by updating your Xcode Command Line Tools.
- On Windows, you may receive an error saying "running scripts is disabled on this system" when trying to activate the virtual environment. If so, run `Set-ExecutionPolicy -Scope CurrentUser` and enter `Unrestricted`, then press `Y`. (You may have to do this every time you open powershell.)
- When creating the conda environment, you may get errors related to accessing the `defaults` channel. Due to [recent changes](https://www.anaconda.com/blog/update-on-anacondas-terms-of-service-for-academia-and-research) to Anaconda's Terms of Service, the Simons Foundation cannot use the `defaults` conda channel and it is blocked on all Flatiron Institute wireless networks. You need to specify `conda-forge` instead (which is community-managed and open to all). (The instructions above should do so.)
- If you have multiple jupyter installs on your path (because e.g., because you have an existing jupyter installation in a conda environment and you then used `uv` to setup the virtual environment for this workshop), jupyter can get confused. (You can check if this is the case by running `which -a jupyter` on Mac / Linux.)
  To avoid this problem, either make sure you only have one virtual environment active (e.g., by running `conda deactivate`) or prepend `JUPYTER_DATA_DIR=$(realpath ..)/.venv/share/jupyter/` to your jupyter command above:

  ```shell
  JUPYTER_DATA_DIR=$(realpath ..)/.venv/share/jupyter/ jupyter lab
  ```

  (On Windows, replace `$(realpath ..)` with the path to the `ccn-software-jan-2025` directory.)
- We have noticed jupyter notebooks behaving a bit odd in Safari --- if you are running/editing jupyter in Safari and the behavior seems off (scrolling not smooth, lag between creation and display of cells), try a different browser. We've had better luck with Firefox or using the arrow keys to navigate between cells.

## Binder

A binder instance (a virtual environment running on Flatiron's cluster) is provided in case we cannot get your installation working. To access it, click the "launch binder" button in the top left of this site or click [here](https://binder.flatironinstitute.org/v2/user/wbroderick/jan2025?labpath=notebooks).

Some usage notes:

- You must login with the email address you provided when registering for the workshop. If you need to use a different email, find Billy and give him the new email address.
- You are only allowed to have a single binder instance running at a time, so if you get the "already have an instance running error", go to the [binderhub page](https://binder.flatironinstitute.org/hub/hub/home) (or click on "check your currently running servers" on the right of the page) to join your running instance.
- If you lose connection halfway through the workshop, go to the [binderhub page](https://binder.flatironinstitute.org/hub/hub/home) to join your running instance rather than restarting the image.
- This is important because if you restart the image, **you will lose all data and progress**.
- The binder will be shutdown automatically after 1 day of inactivity or 7 days of total usage. Data will not persist after the binder instance shuts down, so **please download any notebooks** you want to keep.
- I will destroy this instance in 2 weeks. You can download your notebooks to keep them after the fact.

```{toctree}
:glob:
:caption: Full notebooks
:titlesonly:
full/*/*
```

```{toctree}
:glob:
:caption: For users (some code, some text)
:titlesonly:
users/*/*
```

```{toctree}
:glob:
:caption: For presenter reference (all code, no text)
:titlesonly:
:maxdepth: 1
presenters/*/*
```
