"""
Plot of terminology from experiment.

References
----------
.. [1] https://matplotlib.org/stable/gallery/lines_bars_and_markers/broken_barh.html
"""

import matplotlib.pyplot as plt


class Terminology:
    def __init__(self):
        # Define segments
        self.sessions = [(0, 32)]
        self.runs = [(0, 10), (11, 10), (22, 10)]
        self.trials = [
            ((0, 2), (2, 2), (4, 2), (6, 2), (8, 2)),
            ((11, 2), (13, 2), (15, 2), (17, 2), (19, 2)),
            ((22, 2), (24, 2), (26, 2), (28, 2), (30, 2)),
        ]

        # Define sizing
        self.row_height = 0.5
        self.row_gap = 0.7
        self.y_session = 1.0
        self.y_runs = self.y_session + self.row_gap
        self.y_trials = self.y_runs + self.row_gap

        # Create plot
        fig, ax = plt.subplots(figsize=(9, 3), constrained_layout=True)
        self.plot_sessions(ax)
        self.plot_runs(ax)
        self.plot_trials(ax)

        # Configure plot
        ax.set_xlim(0, 32)
        ax.tick_params(axis="x", labelsize=14)
        ax.set_yticks(
            [
                self.y_session + self.row_height / 2,
                self.y_runs + self.row_height / 2,
                self.y_trials + self.row_height / 2,
            ],
            labels=["Session", "Runs", "Trials"],
            fontsize=14,
        )
        ax.invert_yaxis()
        ax.set_title("One Session from One Subject", fontsize=14)
        ax.set_xlabel("Time", fontsize=14)

        fig.savefig("terminology")

    def plot_sessions(self, ax):
        ax.broken_barh(
            self.sessions, (self.y_session, self.row_height), color="tab:red"
        )
        self._label_bars(ax, self.sessions, self.y_session, self.row_height, ["S1"])

    def plot_runs(self, ax):
        ax.broken_barh(
            self.runs,
            (self.y_runs, self.row_height),
            color=["tab:purple", "tab:purple", "tab:purple"],
        )
        self._label_bars(
            ax, self.runs, self.y_runs, self.row_height, ["R1", "R2", "R3"]
        )

    def plot_trials(self, ax):
        for idx in range(3):
            ax.broken_barh(
                self.trials[idx],
                (self.y_trials, self.row_height),
                color=["tab:brown", "tab:brown", "tab:brown", "tab:brown", "tab:brown"],
            )
            self._label_bars(
                ax,
                self.trials[idx],
                self.y_trials,
                self.row_height,
                ["T1", "T2", "T3", "T4", "T5"],
            )
            self._draw_trial_dividers(
                ax, self.trials[idx], self.y_trials, self.row_height
            )

    def _label_bars(self, ax, segments, y, height, labels):
        for (start, width), label in zip(segments, labels):
            ax.text(
                start + width / 2,
                y + height / 2,
                label,
                ha="center",
                va="center",
                fontsize=12,
                color="black",
            )

    def _draw_trial_dividers(self, ax, segments, y, height):
        for start, width in segments[:-1]:
            x = start + width
            ax.vlines(
                x, y, y + height, colors="black", linewidth=1.5, linestyles="solid"
            )


Terminology()
