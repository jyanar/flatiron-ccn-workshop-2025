#!/usr/bin/env python3

import pynapple as nap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Union
from numpy.typing import NDArray

__all__ = ["plot_features"]

def plot_features(
    input_feature: Union[nap.Tsd, nap.TsdFrame, nap.TsdTensor, NDArray],
    sampling_rate: float,
    suptitle: str,
    n_rows: int = 20,
):
    """
    Plot feature matrix.

    Parameters
    ----------
    input_feature:
        The (num_samples, n_neurons, num_feature) feature array.
    sampling_rate:
        Sampling rate in hz.
    n_rows:
        Number of rows to plot.
    suptitle:
        Suptitle of the plot.

    Returns
    -------

    """
    input_feature = np.squeeze(input_feature).dropna()
    window_size = input_feature.shape[1]
    fig = plt.figure(figsize=(8, 8))
    plt.suptitle(suptitle)
    time = np.arange(0, window_size) / sampling_rate
    for k in range(n_rows):
        ax = plt.subplot(n_rows, 1, k + 1)
        plt.step(time, input_feature[k].squeeze(), where="post")

        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.axvspan(0, time[-1], alpha=0.4, color="orange")
        ax.set_yticks([])
        if k != n_rows - 1:
            ax.set_xticks([])
        else:
            ax.set_xlabel("lag (sec)")
        if k in [0, n_rows - 1]:
            ax.set_ylabel("$t_{%d}$" % (window_size + k), rotation=0)

    plt.tight_layout()
    return fig
