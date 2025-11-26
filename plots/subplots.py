"""
Description: Subplots of images generated from other scripts
References:
    - https://matplotlib.org/stable/gallery/subplots_axes_and_figures/align_labels_demo.html
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img_paths = ["e2e_basic.png", "e2e_detail.png"]
captions = ["Basic End-to-End Pipeline", "Detailed End-to-End Pipeline"]

fig, axs = plt.subplots(1, 2, layout="constrained", figsize=(8, 4))

for ax, img_path, caption in zip(axs, img_paths, captions):
    img = mpimg.imread(img_path)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(caption, fontsize=10, pad=10)

fig.align_titles()

plt.savefig("e2e.png", dpi=300)
