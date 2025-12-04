"""
Plot experiment design
References:
    - https://graphviz.readthedocs.io/en/stable/examples.html
"""

from graphviz import Digraph


def plot_basic():
    """
    Plot basic experiment design
    """
    d = Digraph(filename="basic", format="png")
    d.attr(
        "node",
        shape="rect",
        style="filled",
        fillcolor="lightblue",
        fontname="Helvetica",
    )

    d.node("data preprocessing")
    d.node("model training")
    d.node("hyperparameter search")
    d.node("statistical analysis")

    d.edges(
        [
            ("data preprocessing", "model training"),
            ("model training", "hyperparameter search"),
            ("hyperparameter search", "statistical analysis"),
        ]
    )

    d.view()


def plot_detail():
    """
    Plot detailed experiment design
    """
    d = Digraph(filename="detail", format="png")
    d.attr(
        "node",
        shape="rect",
        style="filled",
        fillcolor="lightblue",
        fontname="Helvetica",
    )

    d.node(
        "data preprocessing",
        "8-32 Hz bandpass,\n4th-order Butterworth IIR filter,\nforward-backward pass,\nMNE parameterization",  # noqa: E501
    )
    d.node(
        "model training",
        "within-session,\nshuffled,\nstratified 5-fold splits,\naveraged cross-validation",  # noqa: E501
    )
    d.node("hyperparameter search", "nested 3-fold cross-validation")
    d.node(
        "statistical analysis",
        "within-dataset pairwise comparisons,\neffect sizes and p-values,\none-tailed permutation t-tests, or\nWilcoxon signed-rank test,\n\nmeta-analysis pairwise comparisons,\ncombined effect sizes and p-values,\nStouffer's Z-score method",  # noqa: E501
    )

    d.edges(
        [
            ("data preprocessing", "model training"),
            ("model training", "hyperparameter search"),
            ("hyperparameter search", "statistical analysis"),
        ]
    )

    d.view()


plot_basic()
plot_detail()
