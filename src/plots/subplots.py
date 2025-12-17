"""
Subplots of images generated from other scripts
References:
    - https://matplotlib.org/stable/gallery/subplots_axes_and_figures/align_labels_demo.html  # noqa: E501
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img_paths = [
    "PhysionetMI.png",
    "Lee2019_MI.png",
    "Cho2017.png",
    "Schirrmeister2017.png",
    "Shin2017A.png",
    "BNCI2014_001.png",
]
captions = [
    "PhysionetMI",
    "Lee2019_MI",
    "Cho2017",
    "Schirrmeister2017",
    "Shin2017A",
    "BNCI2014_001",
]

fig, axs = plt.subplots(2, 3, layout="constrained", figsize=(8, 4))
fig.suptitle("Train/Test Class Balances per Fold", fontsize=12)

for ax, img_path, caption in zip(axs.flat, img_paths, captions):
    img = mpimg.imread(img_path)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(caption, fontsize=10)

fig.align_titles()

plt.savefig("splits.png", dpi=300)
