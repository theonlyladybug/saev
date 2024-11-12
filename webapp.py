import marimo

__generated_with = "0.9.14"
app = marimo.App(width="full")


@app.cell
def __():
    import os
    import pickle
    import random

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    return mo, np, os, pickle, plt, random, torch


@app.cell
def __():
    webapp_dir = "/local/scratch/stevens.994/cache/saev/webapp/h52suuax/sort_by_patch"
    return (webapp_dir,)


@app.cell
def __(os, webapp_dir):
    neuron_indices = [
        int(name) for name in os.listdir(f"{webapp_dir}/neurons") if name.isdigit()
    ]
    neuron_indices = sorted(neuron_indices)
    return (neuron_indices,)


@app.cell
def __(mo):
    get_neuron_i, set_neuron_i = mo.state(0)
    return get_neuron_i, set_neuron_i


@app.cell
def __(mo, neuron_indices, set_neuron_i):
    next_button = mo.ui.button(
        label="Next",
        on_change=lambda _: set_neuron_i(lambda v: (v + 1) % len(neuron_indices)),
    )

    prev_button = mo.ui.button(
        label="Previous",
        on_change=lambda _: set_neuron_i(lambda v: (v - 1) % len(neuron_indices)),
    )
    return next_button, prev_button


@app.cell
def __(mo, pickle, webapp_dir):
    def get_metadata(neuron: int):
        with open(f"{webapp_dir}/neurons/{neuron}/metadata.pkl", "rb") as fd:
            return pickle.load(fd)

    def format_metadata(metadata: dict[str, float | int]):
        return mo.table([metadata])

    return format_metadata, get_metadata


@app.cell
def __(mo, next_button, prev_button):
    mo.hstack([prev_button, next_button])
    return


@app.cell
def __(get_neuron_i, mo, neuron_indices):
    mo.md(f"""
    Neuron {neuron_indices[get_neuron_i()]} ({get_neuron_i()}/{len(neuron_indices)}; {get_neuron_i() / len(neuron_indices) * 100:.2f}%)
    """)
    return


@app.cell
def __(get_metadata, get_neuron_i, mo, neuron_indices):
    mo.ui.table([get_metadata(neuron_indices[get_neuron_i()])], selection=None)
    return


@app.cell
def __(get_neuron_i, mo, neuron_indices, webapp_dir):
    mo.image(f"{webapp_dir}/neurons/{neuron_indices[get_neuron_i()]}/top_images.png")
    return


@app.cell
def __(os, torch, webapp_dir):
    sparsity_fpath = os.path.join(webapp_dir, "sparsity.pt")
    sparsity = torch.load(sparsity_fpath, weights_only=True, map_location="cpu")

    values_fpath = os.path.join(webapp_dir, "mean_values.pt")
    values = torch.load(values_fpath, weights_only=True, map_location="cpu")
    return sparsity, sparsity_fpath, values, values_fpath


@app.cell
def __(np, plt, sparsity):
    def plot_hist(counts):
        fig, ax = plt.subplots()
        ax.hist(np.log10(counts.numpy() + 1e-9), bins=100)
        return fig

    plot_hist(sparsity)
    return (plot_hist,)


@app.cell
def __(plot_hist, values):
    plot_hist(values)
    return


@app.cell
def __(np, plt, sparsity, values):
    def plot_dist(
        min_log_sparsity: float,
        max_log_sparsity: float,
        min_log_value: float,
        max_log_value: float,
    ):
        fig, ax = plt.subplots()

        log_sparsity = np.log10(sparsity.numpy() + 1e-9)
        log_values = np.log10(values.numpy() + 1e-9)

        mask = np.ones(len(log_sparsity)).astype(bool)
        mask[log_sparsity < min_log_sparsity] = False
        mask[log_sparsity > max_log_sparsity] = False
        mask[log_values < min_log_value] = False
        mask[log_values > max_log_value] = False

        n_shown = mask.sum()
        ax.scatter(
            log_sparsity[mask],
            log_values[mask],
            marker=".",
            alpha=0.1,
            color="tab:blue",
            label=f"Shown ({n_shown})",
        )
        n_filtered = (~mask).sum()
        ax.scatter(
            log_sparsity[~mask],
            log_values[~mask],
            marker=".",
            alpha=0.1,
            color="tab:red",
            label=f"Filtered ({n_filtered})",
        )

        ax.axvline(min_log_sparsity, linewidth=0.5, color="tab:red")
        ax.axvline(max_log_sparsity, linewidth=0.5, color="tab:red")
        ax.axhline(min_log_value, linewidth=0.5, color="tab:red")
        ax.axhline(max_log_value, linewidth=0.5, color="tab:red")

        ax.set_xlabel("Feature Frequency (log10)")
        ax.set_ylabel("Mean Activation Value (log10)")
        ax.legend(loc="upper right")

        return fig

    return (plot_dist,)


@app.cell
def __(mo, plot_dist, sparsity_slider, value_slider):
    mo.md(f"""
    Log Sparsity Range: {sparsity_slider}
    {sparsity_slider.value}

    Log Value Range: {value_slider}
    {value_slider.value}

    {mo.as_html(plot_dist(sparsity_slider.value[0], sparsity_slider.value[1], value_slider.value[0], value_slider.value[1]))}
    """)
    return


@app.cell
def __(mo):
    sparsity_slider = mo.ui.range_slider(start=-8, stop=0, step=0.01, value=[-6, -1])
    return (sparsity_slider,)


@app.cell
def __(mo):
    value_slider = mo.ui.range_slider(start=-3, stop=1, step=0.01, value=[-0.75, 1.0])
    return (value_slider,)


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
