"""
Plot power spectral density for data validation.

References
----------
.. [1] https://mne.tools/stable/auto_tutorials/time-freq/20_sensors_time_frequency.html
.. [2] https://mne.tools/stable/generated/mne.Epochs.html#mne.Epochs.compute_psd
.. [3] https://mne.tools/stable/generated/mne.time_frequency.EpochsSpectrum.html#mne.time_frequency.EpochsSpectrum
"""

import numpy as np
import matplotlib.pyplot as plt
from moabb.datasets import (
    PhysionetMI,
    Lee2019_MI,
    Cho2017,
    Schirrmeister2017,
    Shin2017A,
    BNCI2014_001,
    BNCI2014_004,
    Dreyer2023,
    Weibo2014,
    GrosseWentrup2009,
    Stieger2021,
)
from src.datasets import Liu2024
from src.paradigm import MultiScoreLeftRightImagery


class PSD:
    def run(self):
        fig, axes = plt.subplots(4, 3, figsize=(12, 12), squeeze=False)

        paradigm = MultiScoreLeftRightImagery(resample=128, fmin=8, fmax=32)

        for DatasetCls, row, col in self._datasets():
            ax = axes[row][col]
            dataset = DatasetCls()
            epochs, labels, _ = paradigm.get_data(dataset, subjects=[1], return_epochs=True)

            epochs_left = epochs[labels == "left_hand"]
            epochs_right = epochs[labels == "right_hand"]
            psd_left = epochs_left.compute_psd(picks="data", fmin=8, fmax=32)
            psd_right = epochs_right.compute_psd(picks="data", fmin=8, fmax=32)
            mean_psd_left = psd_left.average()
            mean_psd_right = psd_right.average()

            for mean_psd, color in [(mean_psd_left, "blue"), (mean_psd_right, "red")]:
                psds, freqs = mean_psd.get_data(return_freqs=True)
                psds = 10 * np.log10(psds)
                psds_mean = psds.mean(axis=0)
                ax.plot(freqs, psds_mean, color=color, label="Left" if color == "blue" else "Right")

            ax.set_title(DatasetCls.__name__, fontsize=14)
            ax.set_xlabel("Frequency (Hz)", fontsize=14)
            ax.set_ylabel("PSD (dB)", fontsize=14)
            ax.legend(loc="lower left", fontsize=14)

        fig.suptitle("Multitaper Power Spectral Density", fontweight="bold", fontsize=16)
        fig.tight_layout()
        fig.savefig("psd")

    def _datasets(self):
        yield (BNCI2014_001, 0, 0)
        yield (BNCI2014_004, 0, 1)
        yield (PhysionetMI, 0, 2)
        yield (Lee2019_MI, 1, 0)
        yield (Cho2017, 1, 1)
        yield (Schirrmeister2017, 1, 2)
        yield (Shin2017A, 2, 0)
        yield (Dreyer2023, 2, 1)
        yield (Weibo2014, 2, 2)
        yield (GrosseWentrup2009, 3, 0)
        yield (Stieger2021, 3, 1)
        yield (Liu2024, 3, 2)
