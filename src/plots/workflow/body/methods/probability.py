"""
Plot probability from experiment design
References:
    - https://graphviz.readthedocs.io/en/stable/examples.html
"""

from graphviz import Digraph


def plot_basic():
    """
    Plot basic probability from experiment design
    """
    d = Digraph(filename="basic", format="png")
    d.attr(
        "node",
        shape="rect",
        style="filled",
        fillcolor="lightblue",
        fontname="Helvetica",
    )

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

    d.edges(
        [
            ("datasets", "frequentist models"),
            ("datasets", "bayesian models"),
            ("bayesian models", "priors"),
            ("priors", "inferences"),
            ("inferences", "bayes"),
            ("frequentist models", "freq"),
        ]
    )

    d.view()


def plot_detail():
    """
    Plot detailed probability from experiment design
    """
    d = Digraph(filename="detail", format="png")
    d.attr(
        "node",
        shape="plain",
        style="filled",
        fillcolor="lightblue",
        fontname="Helvetica",
    )

    d.node(
        "datasets",
        """<
    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
        <TR>
            <TD>PhysionetMI</TD>
            <TD>Lee2019_MI</TD>
            <TD>Cho2017</TD>
        </TR>
        <TR>
            <TD>Schirrmeister2017</TD>
            <TD>Shin2017A</TD>
            <TD>BNCI2014_001</TD>
        </TR>
    </TABLE>>""",
    )

    with d.subgraph() as s:
        s.attr(rank="same")
        s.node(
            "frequentist models",
            """<
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
        </TABLE>>""",
        )
        s.node(
            "bayesian models",
            """<
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
        </TABLE>>""",
        )

    d.node(
        "priors",
        """<
    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
        <TR>
            <TD>Gaussian</TD>
            <TD>Laplace</TD>
            <TD>Cauchy</TD>
        </TR>
    </TABLE>>""",
    )
    d.node(
        "inferences",
        """<
    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
        <TR>
            <TD>LA</TD>
            <TD>VI</TD>
            <TD>HMC</TD>
        </TR>
    </TABLE>>""",
    )

    with d.subgraph() as s:
        s.attr(rank="same")
        s.node(
            "freq",
            """<
        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
            <TR><TD>p(y &#124; x, &#952;)</TD></TR>
        </TABLE>>""",
        )
        s.node(
            "bayes",
            """<
        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
            <TR><TD>p(y &#124; x, D)</TD></TR>
        </TABLE>>""",
        )

    d.edges(
        [
            ("datasets", "frequentist models"),
            ("datasets", "bayesian models"),
            ("bayesian models", "priors"),
            ("priors", "inferences"),
            ("inferences", "bayes"),
            ("frequentist models", "freq"),
        ]
    )

    d.view()


plot_basic()
plot_detail()
