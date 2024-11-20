import marimo

__generated_with = "0.9.14"
app = marimo.App(width="full")


@app.cell
def __():
    import itertools
    import os
    import pickle
    import random

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    return itertools, mo, np, os, pickle, plt, random, torch


@app.cell
def __(mo, os):
    ckpts = os.listdir("/research/nfs_su_809/workspace/stevens.994/saev/probing")

    mo.stop(
        not ckpts,
        mo.md("Run `uv run main.py probe --help` to fill out at least one checkpoint."),
    )

    ckpt_dropdown = mo.ui.dropdown(ckpts, label="Checkpoint:", value=ckpts[0])
    return ckpt_dropdown, ckpts


@app.cell
def __(ckpt_dropdown):
    ckpt_dropdown
    return


@app.cell
def __(ckpt_dropdown, mo, os):
    mo.stop(
        ckpt_dropdown.value is None,
        mo.md("Run `uv run main.py probe --help` to fill out at least one checkpoint."),
    )

    tasks = os.listdir(
        f"/research/nfs_su_809/workspace/stevens.994/saev/probing/{ckpt_dropdown.value}"
    )

    mo.stop(
        not tasks,
        mo.md("Run `uv run main.py probe --help` to fill out at least one checkpoint."),
    )

    task_dropdown = mo.ui.dropdown(tasks, label="Task:", value=tasks[0])
    return task_dropdown, tasks


@app.cell
def __(task_dropdown):
    task_dropdown
    return


@app.cell
def __(ckpt_dropdown, mo, os, task_dropdown):
    root = os.path.join(
        "/research/nfs_su_809/workspace/stevens.994/saev/probing",
        ckpt_dropdown.value,
        task_dropdown.value,
    )

    get_neuron_i, set_neuron_i = mo.state(0)
    return get_neuron_i, root, set_neuron_i


@app.cell
def __(mo, os, root):
    neuron_indices = [
        int(name) for name in os.listdir(f"{root}/neurons") if name.isdigit()
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
        len(neuron_indices) - 1,
        value=get_neuron_i(),
        on_change=lambda i: set_neuron_i(i),
        full_width=True,
    )
    return (neuron_slider,)


@app.cell
def __(mo):
    width_slider = mo.ui.slider(start=1, stop=20, label="Images per row", value=8)
    return (width_slider,)


@app.cell
def __(
    get_neuron_i,
    mo,
    n,
    neuron_indices,
    neuron_notes,
    neuron_slider,
    next_button,
    prev_button,
    width_slider,
):
    label = f"Neuron {n} ({get_neuron_i()}/{len(neuron_indices)}; {get_neuron_i() / len(neuron_indices) * 100:.2f}%)"

    mo.md(f"""
    {mo.hstack([prev_button, next_button, label, width_slider], justify="start", gap=1.0)}
    {neuron_slider}

    Notes on Neuron {n}: {neuron_notes}
    """)
    return (label,)


@app.cell
def __(mo, root):
    def show_img(n: int, i: int):
        label = "No label found."
        try:
            label = open(f"{root}/neurons/{n}/{i}.txt").read().strip()
        except FileNotFoundError:
            return mo.md(f"*Missing image {i + 1}*")

        return mo.vstack([
            mo.image(f"{root}/neurons/{n}/{i}-original.png"),
            mo.image(f"{root}/neurons/{n}/{i}.png"),
            mo.md(label),
        ])

    return (show_img,)


@app.cell
def __(
    batched,
    get_neuron_i,
    mo,
    neuron_indices,
    os,
    root,
    show_img,
    width_slider,
):
    n = neuron_indices[get_neuron_i()]
    i_im = [
        int(filename.removesuffix(".txt"))
        for filename in os.listdir(f"{root}/neurons/{n}")
        if filename.endswith(".txt") and filename != "notes.txt"
    ][:200]

    imgs = [show_img(n, i) for i in i_im]

    rows = [
        mo.hstack(batch, widths="equal")
        for batch in batched(imgs, n=width_slider.value)
    ]

    mo.vstack(rows)
    return i_im, imgs, n, rows


@app.cell
def __(n, root):
    try:
        with open(f"{root}/neurons/{n}/notes.txt") as fd:
            neuron_notes = fd.read().strip()
    except FileNotFoundError:
        neuron_notes = "*no notes.*"
    return fd, neuron_notes


@app.cell
def __(itertools):
    def batched(iterable, n, *, strict=False):
        # batched('ABCDEFG', 3) → ABC DEF G
        if n < 1:
            raise ValueError("n must be at least one")
        iterator = iter(iterable)
        while batch := tuple(itertools.islice(iterator, n)):
            if strict and len(batch) != n:
                raise ValueError("batched(): incomplete batch")
            yield batch

    return (batched,)


@app.cell
def __():
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
