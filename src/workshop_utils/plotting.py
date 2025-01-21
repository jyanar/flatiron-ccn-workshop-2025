#!/usr/bin/env python3

import pynapple as nap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Optional, List
from numpy.typing import NDArray
import seaborn as sns
from nemos import _documentation_utils as doc_plots

__all__ = ["plot_features", "plot_head_direction_tuning_model", "plot_feature_mask",
           "plot_heatmap_cv_results", "plot_position_speed",
           "plot_position_speed_tuning", "plot_place_fields", "plot_pos_speed_bases"]

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


def plot_head_direction_tuning_model(
    tuning_curves: pd.DataFrame,
    spikes: nap.TsGroup,
    angle: nap.Tsd,
    predicted_firing_rate: Optional[nap.TsdFrame] = None,
    pref_ang: Optional[pd.Series] = None,
    model_tuning_curves: Optional[pd.DataFrame] = None,
    threshold_hz: int = 1,
    start: float = 8910,
    end: float = 8960,
    cmap_label="hsv",
    figsize=None,
):
    """
    Plot head direction tuning.

    Parameters
    ----------
    tuning_curves:
        The tuning curve dataframe.
    spikes:
        The spike times.
    angle:
        The heading angles.
    predicted_firing_rate:
        The time series of the predicted rate.
    threshold_hz:
        Minimum firing rate for neuron to be plotted.,
    start:
        Start time
    end:
        End time
    cmap_label:
        cmap label ("hsv", "rainbow", "Reds", ...)
    figsize:
        Figure size in inches.

    Returns
    -------
    fig:
        The figure.
    """
    plot_ep = nap.IntervalSet(start, end)
    index_keep = spikes.restrict(plot_ep).getby_threshold("rate", threshold_hz).index

    # filter neurons
    tuning_curves = tuning_curves.loc[:, index_keep]
    if pref_ang is None:
        pref_ang = tuning_curves.idxmax()
    pref_ang = pref_ang.loc[index_keep]
    spike_tsd = (
        spikes.restrict(plot_ep).getby_threshold("rate", threshold_hz).to_tsd(pref_ang)
    )

    # plot raster and heading
    cmap = plt.get_cmap(cmap_label)
    unq_angles = np.unique(pref_ang.values)
    n_subplots = len(unq_angles)
    relative_color_levs = (unq_angles - unq_angles[0]) / (
        unq_angles[-1] - unq_angles[0]
    )

    n_rows = 3
    if predicted_firing_rate is not None:
        n_rows += 1
    if model_tuning_curves is not None:
        model_tuning_curves = model_tuning_curves.loc[:, index_keep]
        n_rows += 1
    if figsize is None:
        figsize = [12, 6]
        if n_rows == 5:
            figsize[1] += 2
    fig = plt.figure(figsize=figsize)
    # plot head direction angle
    ax = plt.subplot2grid(
        (n_rows, n_subplots), loc=(0, 0), rowspan=1, colspan=n_subplots, fig=fig
    )
    ax.plot(angle.restrict(plot_ep), color="k", lw=2)
    ax.set_ylabel("Angle (rad)")
    ax.set_title("Animal's Head Direction")
    ax.set_xlim(8910, 8960)

    ax = plt.subplot2grid(
        (n_rows, n_subplots), loc=(1, 0), rowspan=1, colspan=n_subplots, fig=fig
    )
    ax.set_title("Neural Activity")
    for i, ang in enumerate(unq_angles):
        sel = spike_tsd.d == ang
        ax.plot(
            spike_tsd[sel].t,
            np.ones(sel.sum()) * i,
            "|",
            color=cmap(relative_color_levs[i]),
            alpha=0.5,
        )
    ax.set_ylabel("Sorted Neurons")
    ax.set_xlabel("Time (s)")
    ax.set_xlim(8910, 8960)
    curr_row = 2

    if predicted_firing_rate is not None:
        ax = plt.subplot2grid(
            (n_rows, n_subplots), loc=(curr_row, 0), rowspan=1, colspan=n_subplots, fig=fig
        )
        curr_row += 1
        ax.set_title("Neural Firing Rate")
        fr = predicted_firing_rate.restrict(plot_ep).d
        fr = fr.T / np.max(fr, axis=1)
        ax.imshow(fr[::-1], cmap="Blues", aspect="auto")
        ax.set_ylabel("Sorted Neurons")
        ax.set_xlabel("Time (s)")
        ax.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
        ax.set_xticklabels([8910, 8920, 8930, 8940, 8950, 8960])
        ax.set_xlim(0, 5000)

    for i, ang in enumerate(unq_angles):
        neu_idx = np.argsort(pref_ang.values)[i]
        ax = plt.subplot2grid(
            (n_rows, n_subplots),
            loc=(curr_row + i // n_subplots, i % n_subplots),
            rowspan=1,
            colspan=1,
            fig=fig,
            projection="polar",
        )
        ax.fill_between(
            tuning_curves.iloc[:, neu_idx].index,
            np.zeros(len(tuning_curves)),
            tuning_curves.iloc[:, neu_idx].values,
            color=cmap(relative_color_levs[i]),
            alpha=0.5,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    curr_row += 1

    if model_tuning_curves is not None:
        for i, ang in enumerate(unq_angles):
            neu_idx = np.argsort(pref_ang.values)[i]

            ax = plt.subplot2grid(
                (n_rows, n_subplots),
                loc=(curr_row + i // n_subplots, i % n_subplots),
                rowspan=1,
                colspan=1,
                fig=fig,
                projection="polar",
            )
            ax.fill_between(
                model_tuning_curves.iloc[:, neu_idx].index,
                np.zeros(len(model_tuning_curves)),
                model_tuning_curves.iloc[:, neu_idx].values,
                color=cmap(relative_color_levs[i]),
                alpha=0.5,
            )
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    return fig


def plot_feature_mask(masks, titles=None):
    height = 5
    if not isinstance(masks, list):
        masks = [masks]
    if titles is None:
        titles = [f"Mask {i}" for i in range(len(masks))]
    width = 5 * masks[0].shape[1] / masks[0].shape[0]
    fig, axes = plt.subplots(1, 1+len(masks), sharey=True,
                             figsize=((width+.5)*(.5+len(masks)), height))
    cmap = 'RdBu'
    for i, (ax, m, t) in enumerate(zip(axes, masks, titles)):
        if i == 0:
            ax.set_ylabel("feature")
        if i == (len(axes)-1)//2:
            ax.set_xlabel("neuron")
        ax.imshow(m, vmin=-1, vmax=1, cmap=cmap)
        if m.shape[1] < 5:
            ax.set(xticks=range(m.shape[1]))
        ax.set_title(t)
    norm = plt.Normalize(vmin=-1, vmax=1)
    cmap = plt.get_cmap(cmap)
    handles = [plt.Rectangle((0, 0), 0, 1, color=cmap(norm(1)), label=1),
               plt.Rectangle((0, 0), 0, 0, color=cmap(norm(0)), label=0),]
    fig.legend(handles=handles, loc=5)
    axes[-1].set_visible(False)
    return fig


def facet_heatmap(index, columns, values, data, **heatmap_kwargs):
    data = data.pivot(index=index, columns=columns, values=values)
    ax = sns.heatmap(data, **heatmap_kwargs)
    doc_plots.highlight_max_cell(data, ax)


def format_regularizer(x):
    reg = str(x["param_glm__regularizer"]).split('(')[0]
    reg_str =  x["param_glm__regularizer_strength"]
    if not np.isnan(reg_str):
        reg += f"({reg_str})"
    return reg

def plot_heatmap_cv_results(cv_df, facet_col=None,
                            rows="param_basis__basis1__n_basis_funcs",
                            columns="param_basis__basis2",
                            values="mean_test_score",
                            fmt=".3f"):
    if "param_glm" not in cv_df.columns:
        try:
            cv_df["param_glm"] = cv_df.apply(format_regularizer, 1)
        except KeyError:
            pass
    fig = sns.FacetGrid(cv_df, col=facet_col, height=5)
    vmin = cv_df[values].min()
    vmax = cv_df[values].max()
    fig.map_dataframe(facet_heatmap, index=rows, columns=columns, values=values,
                      annot=True, square=True, vmin=vmin, vmax=vmax,
                      linecolor="white", linewidth=0.5, fmt=fmt)


def _analyze_speed(speed: pd.DataFrame, position: nap.Tsd):
    bins = np.linspace(np.min(position), np.max(position), 20)
    idx = np.digitize(position.values, bins)
    mean_speed = np.array([np.mean(speed[idx==i]) for i in np.unique(idx)])
    std_speed = np.array([np.std(speed[idx==i]) for i in np.unique(idx)])
    return bins, mean_speed, std_speed

def plot_position_speed(position: nap.Tsd, speed: nap.Tsd,
                        position_tuning: pd.DataFrame, speed_tuning: pd.DataFrame,
                        neuron_id: Union[int, List[int]]):
    if not hasattr(neuron_id, "__iter__"):
        neuron_id = [neuron_id]
    fig = plt.figure(figsize=(6*len(neuron_id), 6))
    gs = plt.GridSpec(2, 2*len(neuron_id), wspace=.3, hspace=.35)
    for i, n in enumerate(neuron_id):
        ax = fig.add_subplot(gs[0, 2*i])
        ax.fill_between(position_tuning[n].index.values, np.zeros(len(position_tuning)),
                        position_tuning[n].values)
        ax.set(xlabel="Position (cm)", ylabel="Firing rate (Hz)", title="Position tuning")
        ax.text(1, 1.2, f"Neuron {n}", transform=ax.transAxes, size="x-large")

        ax = fig.add_subplot(gs[1, 2*i])
        bins, mean_speed, std_speed = _analyze_speed(speed, position)

        ax.plot(bins, mean_speed)
        ax.fill_between(
            bins,
            mean_speed - std_speed,
            mean_speed + std_speed,
            alpha=0.1,
        )
        ax.set(xlabel="Position (cm)", ylabel="Speed (cm/s)", title="Animal speed")
        ax = fig.add_subplot(gs[1, 2*i+1])
        ax.fill_between(
            speed_tuning.index.values, np.zeros(len(speed_tuning)),
            speed_tuning[n].values
        )
        ax.set(ylabel="Firing rate (Hz)", xlabel="Speed (cm/s)", title="Speed tuning")
    return fig


def plot_position_speed_tuning(position_tuning: pd.DataFrame, speed_tuning: pd.DataFrame,
                               neuron_id: Union[int, List[int]],
                               model_position_tuning: Optional[pd.DataFrame] = None,
                               model_speed_tuning: Optional[pd.DataFrame] = None):
    if not hasattr(neuron_id, "__iter__"):
        neuron_id = [neuron_id]
    fig = plt.figure(figsize=(6*len(neuron_id), 3))
    gs = plt.GridSpec(1, 2*len(neuron_id), wspace=.3, hspace=.35)
    pos_ax = None
    speed_ax = None
    for i, n in enumerate(neuron_id):
        ax = fig.add_subplot(gs[0, 2*i], sharey=pos_ax, sharex=pos_ax)
        ax.plot(position_tuning[n], '--')
        if model_position_tuning is not None:
            ax.plot(model_position_tuning[n])
        ax.set(xlabel="Position (cm)", ylabel="Firing rate (Hz)", title="Position tuning")
        ax.text(1, 1.2, f"Neuron {n}", transform=ax.transAxes, size="x-large")
        if pos_ax is None:
            pos_ax = ax

        ax = fig.add_subplot(gs[0, 2*i+1], sharex=speed_ax, sharey=pos_ax)
        ax.plot(speed_tuning[n], '--')
        if model_speed_tuning is not None:
            ax.plot(model_speed_tuning[n])
        ax.set(ylabel="Firing rate (Hz)", xlabel="Speed (cm/s)", title="Speed tuning")
        if speed_ax is None:
            speed_ax = ax
    # fig.tight_layout()
    return fig


def plot_place_fields(place_fields, highlight_neurons=[92, 82, 220]):
    # for display purposes, sort place fields by location
    order = place_fields.idxmax().sort_values().index.values
    fig = plt.figure(figsize=(12, 10))
    gs = plt.GridSpec(place_fields.shape[1], 1)
    for i, n in enumerate(order):
        plt.subplot(gs[i, 0])
        if n in highlight_neurons:
            c = 'r'
        else:
            c = 'C0'
        plt.fill_between(place_fields.index.values, np.zeros(len(place_fields)),
                         place_fields[n].values, facecolor=c)
        if i < place_fields.shape[1] - 1:
            plt.xticks([])
        else:
            plt.xlabel("Position (cm)")
        plt.yticks([])

def plot_pos_speed_bases(position_basis, speed_basis):
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    for ax, bas, t in zip(axes, [position_basis, speed_basis], ["Position", "Speed"]):
        time, kernel = bas.evaluate_on_grid(100)
        ax.plot(time, kernel)
        ax.set_title(t)
