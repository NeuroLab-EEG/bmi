"""
Description: Basic experiment design plot
References:
    - https://graphviz.readthedocs.io/en/stable/examples.html
"""

from graphviz import Digraph

d = Digraph(filename="experiment_basic", format="png")
d.attr("node", shape="rect", style="filled", fillcolor="lightblue", fontname="Helvetica")

d.node("data preprocessing")
d.node("model training")
d.node("hyperparameter search")
d.node("statistical analysis")

d.edges([
    ("data preprocessing", "model training"),
    ("model training", "hyperparameter search"),
    ("hyperparameter search", "statistical analysis")
])

d.view()
