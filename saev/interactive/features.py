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
def __(mo, os):
    def make_ckpt_dropdown():
        try:
            choices = os.listdir(
                "/research/nfs_su_809/workspace/stevens.994/saev/webapp"
            )

        except FileNotFoundError:
            choices = []

        return mo.ui.dropdown(choices, label="Checkpoint:")

    ckpt_dropdown = make_ckpt_dropdown()
    return ckpt_dropdown, make_ckpt_dropdown


@app.cell
def __(ckpt_dropdown, mo):
    mo.hstack([ckpt_dropdown], justify="start")
    return


@app.cell
def __(ckpt_dropdown, mo):
    mo.stop(
        ckpt_dropdown.value is None,
        mo.md(
            "Run `uv run main.py webapp --help` to fill out at least one checkpoint."
        ),
    )

    webapp_dir = f"/research/nfs_su_809/workspace/stevens.994/saev/webapp/{ckpt_dropdown.value}/sort_by_patch"

    get_neuron_i, set_neuron_i = mo.state(0)
    return get_neuron_i, set_neuron_i, webapp_dir


@app.cell
def __(mo, os, webapp_dir):
    neuron_indices = [
        int(name) for name in os.listdir(f"{webapp_dir}/neurons") if name.isdigit()
    ]
    neuron_indices = sorted(neuron_indices)
    mo.md(f"Found {len(neuron_indices)} saved neurons.")
    return (neuron_indices,)


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
def __(get_neuron_i, mo, neuron_indices, set_neuron_i):
    neuron_slider = mo.ui.slider(
        0,
        len(neuron_indices),
        value=get_neuron_i(),
        on_change=lambda i: set_neuron_i(i),
        full_width=True,
    )
    return (neuron_slider,)


@app.cell
def __(
    get_neuron_i,
    mo,
    neuron_indices,
    neuron_slider,
    next_button,
    prev_button,
):
    label = f"Neuron {neuron_indices[get_neuron_i()]} ({get_neuron_i()}/{len(neuron_indices)}; {get_neuron_i() / len(neuron_indices) * 100:.2f}%)"

    mo.md(f"""
    {mo.hstack([prev_button, next_button, label], justify="start")}
    {neuron_slider}
    """)
    return (label,)


@app.cell
def __():
    # mo.image(f"{webapp_dir}/neurons/{neuron_indices[get_neuron_i()]}/top_images.png")
    return


@app.cell
def __(mo, webapp_dir):
    def show_img(n: int, i: int):
        label = "No label found."
        try:
            label = open(f"{webapp_dir}/neurons/{n}/{i}.txt").read().strip()
        except FileNotFoundError:
            return mo.md(f"*Missing image {i + 1}*")

        return mo.vstack([mo.image(f"{webapp_dir}/neurons/{n}/{i}.png"), mo.md(label)])

    return (show_img,)


@app.cell
def __(get_neuron_i, mo, neuron_indices, show_img):
    n = neuron_indices[get_neuron_i()]

    mo.vstack([
        mo.hstack(
            [
                show_img(n, 0),
                show_img(n, 1),
                show_img(n, 2),
                show_img(n, 3),
                show_img(n, 4),
            ],
            widths="equal",
        ),
        mo.hstack(
            [
                show_img(n, 5),
                show_img(n, 6),
                show_img(n, 7),
                show_img(n, 8),
                show_img(n, 9),
            ],
            widths="equal",
        ),
        mo.hstack(
            [
                show_img(n, 10),
                show_img(n, 11),
                show_img(n, 12),
                show_img(n, 13),
                show_img(n, 14),
            ],
            widths="equal",
        ),
        mo.hstack(
            [
                show_img(n, 15),
                show_img(n, 16),
                show_img(n, 17),
                show_img(n, 18),
                show_img(n, 19),
            ],
            widths="equal",
        ),
        mo.hstack(
            [
                show_img(n, 20),
                show_img(n, 21),
                show_img(n, 22),
                show_img(n, 23),
                show_img(n, 24),
            ],
            widths="equal",
        ),
        # mo.hstack(
        #     [show_img(n, 12), show_img(n, 13), show_img(n, 14), show_img(n, 15)],
        #     widths="equal",
        # ),
    ])
    return (n,)


@app.cell
def __(os, torch, webapp_dir):
    sparsity_fpath = os.path.join(webapp_dir, "sparsity.pt")
    sparsity = torch.load(sparsity_fpath, weights_only=True, map_location="cpu")

    values_fpath = os.path.join(webapp_dir, "mean_values.pt")
    values = torch.load(values_fpath, weights_only=True, map_location="cpu")
    return sparsity, sparsity_fpath, values, values_fpath


@app.cell
def __(mo, np, plt, sparsity):
    def plot_hist(counts):
        fig, ax = plt.subplots()
        ax.hist(np.log10(counts.numpy() + 1e-9), bins=100)
        return fig

    mo.md(f"""
    Sparsity Log10

    {mo.as_html(plot_hist(sparsity))}
    """)
    return (plot_hist,)


@app.cell
def __(mo, plot_hist, values):
    mo.md(f"""
    Mean Value Log10

    {mo.as_html(plot_hist(values))}
    """)
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


if __name__ == "__main__":
    app.run()
