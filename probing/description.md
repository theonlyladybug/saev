How can we find examples of interesting features in different checkpoints? How can we predict hypothetical trends in the way different models learn different features?

In this work, we assume that manual effort is necessary. This package contains methods for trying to make it as easy as possible to identify interesting SAEs and predict trends between models.

# An Observational Study of Vision Model Interpretability

We have multiple ways to compare trained SAEs.

1. Heuristic measures, like number of dead features, number of dense features, mean L0 norm, etc.
2. Qualitative plots of feature frequency, mean activation value, L0-MSE tradeoff curves, etc.
3. Manual inspection of the top K images for each feature.

After proposing trends, we can construct individual probing datasets (see below).

![experimental-design](docs/assets/experiment1.png)


`probing.notebooks.l0_mse_tradeoff` is a notebook to explore the L0-MSE tradeoff as well as feature frequency and mean activation value distributions.
