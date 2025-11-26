"""
Description: Basic end-to-end plot of experiment
References:
    - https://graphviz.readthedocs.io/en/stable/examples.html
"""

from graphviz import Digraph

d = Digraph(filename="e2e_basic", format="png")
d.attr("node", shape="rect", style="filled", fillcolor="lightblue", fontname="Helvetica")

d.node("datasets")

with d.subgraph() as s:
    s.attr(rank="same")
    s.node("frequentist models")
    s.node("bayesian models")

d.node("priors")
d.node("inferences")

with d.subgraph() as s:
    s.attr(rank="same")
    s.node("freq", "<p(y &#124; x, &#952;)>")
    s.node("bayes", "<p(y &#124; x, D)>")

d.edges([
    ("datasets", "frequentist models"),
    ("datasets", "bayesian models"),
    ("bayesian models", "priors"),
    ("priors", "inferences"),
    ("inferences", "bayes"),
    ("frequentist models", "freq")
])

d.view()
