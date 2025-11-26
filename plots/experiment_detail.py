"""
Description: Basic experiment design plot
References:
    - https://graphviz.readthedocs.io/en/stable/examples.html
"""

from graphviz import Digraph

d = Digraph(filename="experiment_detail", format="png")
d.attr("node", shape="rect", style="filled", fillcolor="lightblue", fontname="Helvetica")

d.node("data preprocessing", "8-32 Hz bandpass,\n4th-order Butterworth IIR filter,\nforward-backward pass,\nMNE parameterization")
d.node("model training", "within-session,\nshuffled,\nstratified 5-fold splits,\naveraged cross-validation")
d.node("hyperparameter search", "nested 3-fold cross-validation")
d.node("statistical analysis", "within-dataset pairwise comparisons,\neffect sizes and p-values,\none-tailed permutation t-tests, or\nWilcoxon signed-rank test,\n\nmeta-analysis pairwise comparisons,\ncombined effect sizes and p-values,\nStouffer's Z-score method")

d.edges([
    ("data preprocessing", "model training"),
    ("model training", "hyperparameter search"),
    ("hyperparameter search", "statistical analysis")
])

d.view()
