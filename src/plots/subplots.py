"""
Subplots of images generated from other scripts
References:
    - https://matplotlib.org/stable/gallery/subplots_axes_and_figures/align_labels_demo.html  # noqa: E501
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img_paths = ["picture1.png", "picture2.png", "picture3.png",]
captions = ["caption1", "caption2", "caption3"]

fig, axs = plt.subplots(1, 3, layout="constrained", figsize=(8, 4))
fig.suptitle("Figure Title", fontsize=12)

for ax, img_path, caption in zip(axs.flat, img_paths, captions):
    img = mpimg.imread(img_path)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(caption, fontsize=10)

fig.align_titles()

plt.savefig("figure", dpi=300)
