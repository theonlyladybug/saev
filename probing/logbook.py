import marimo

__generated_with = "0.9.14"
app = marimo.App(width="full")


@app.cell
def __():
    import marimo as mo

    return (mo,)


@app.cell
def __(mo):
    mo.md(
        r"""
        How can we find examples of interesting features in different checkpoints? How can we predict hypothetical trends in the way different models learn different features?

        In this paper, we assume that manual effort is necessary. This notebook is a method for trying to make it as easy as possible to identify interesting SAEs and predict trends between models.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # An Observational Study of Vision Model Interpretability

        We have multiple ways to compare trained SAEs.

        1. Heuristic measures, like number of dead features, number of dense features, mean L0 norm, etc.
        2. Qualitative plots of feature frequency, mean activation value, L0-MSE tradeoff curves, etc.
        3. Manual inspection of the top K images for each feature.

        After proposing trends, we can construct individual probing datasets (see below).
        """
    )
    return


@app.cell
def __(mo):
    mo.image("docs/assets/experiment1.png")
    return


@app.cell
def __(mo):
    mo.md(r"""`probing/notebooks/lo_mse_tradeoff.py` is a notebook to explore""")
    return


if __name__ == "__main__":
    app.run()
