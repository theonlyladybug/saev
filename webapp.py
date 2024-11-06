import marimo

__generated_with = "0.9.14"
app = marimo.App(width="full")


@app.cell
def __():
    import os
    import pickle
    import random

    import marimo as mo
    return mo, os, pickle, random


@app.cell
def __():
    webapp_dir = "/local/scratch/stevens.994/cache/saev/webapp/u44m18q1/sort_by_patch"
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
    get_neuron_i()
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
    mo.md(f"""Neuron {neuron_indices[get_neuron_i()]}""")
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
def __():
    return


if __name__ == "__main__":
    app.run()
