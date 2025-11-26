"""
Description: Detailed end-to-end plot of experiment
References:
    - https://graphviz.readthedocs.io/en/stable/examples.html
"""

from graphviz import Digraph

d = Digraph(filename="e2e_detail", format="png")
d.attr("node", shape="plain", style="filled",
       fillcolor="lightblue", fontname="Helvetica")

d.node("datasets", """<
<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
    <TR>
        <TD>AlexMI</TD>
        <TD>BNCI2014_001</TD>
        <TD>PhysionetMI</TD>
    </TR>
    <TR>
        <TD>Schirrmeister2017</TD>
        <TD>Weibo2014</TD>
        <TD>Zhou2016</TD>
    </TR>
</TABLE>>""")

with d.subgraph() as s:
    s.attr(rank="same")
    s.node("frequentist models", """<
    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
        <TR>
            <TD>CSP+LDA</TD>
            <TD>CSP+SVM</TD>
            <TD>TS+SVM</TD>
        </TR>
        <TR>
            <TD>TS+LR</TD>
            <TD>SCNN</TD>
            <TD>DCNN</TD>
        </TR>
    </TABLE>>""")
    s.node("bayesian models", """<
    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
        <TR>
            <TD>CSP+BLDA</TD>
            <TD>CSP+GP</TD>
            <TD>TS+GP</TD>
        </TR>
        <TR>
            <TD>TS+BLR</TD>
            <TD>BSCNN</TD>
            <TD>BDCNN</TD>
        </TR>
    </TABLE>>""")

d.node("priors", """<
<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
    <TR>
        <TD>Gaussian</TD>
        <TD>Laplace</TD>
        <TD>Cauchy</TD>
    </TR>
</TABLE>>""")
d.node("inferences", """<
<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
    <TR>
        <TD>Laplace</TD>
        <TD>VI</TD>
        <TD>HMC</TD>
    </TR>
</TABLE>>""")

with d.subgraph() as s:
    s.attr(rank="same")
    s.node("freq", """<
    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
        <TR><TD>p(y &#124; x, &#952;)</TD></TR>
    </TABLE>>""")
    s.node("bayes", """<
    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
        <TR><TD>p(y &#124; x, D)</TD></TR>
    </TABLE>>""")

d.edges([
    ("datasets", "frequentist models"),
    ("datasets", "bayesian models"),
    ("bayesian models", "priors"),
    ("priors", "inferences"),
    ("inferences", "bayes"),
    ("frequentist models", "freq")
])

d.view()
