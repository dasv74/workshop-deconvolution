"""Workshop - 3D Deconvolution Microscopy (EPFL Center for Imaging)
Course material: lecture companion for in-class demonstrations and practical sessions.

Author : Daniel Sage, Biomedical Imaging Group, EPFL, Lausanne, Switzerland
Date   : 21 August 2026
License: BSD 3-Clause + citationware - if this material is useful to your
         research or teaching, please cite:
         D. Sage et al., "DeconvolutionLab2: An open-source software for
         deconvolution microscopy," Methods, vol. 115, pp. 28-41, 2017.

Display helpers for the deconvolution notebooks.

show_images(images, ...) takes a dict {title: image} (numpy 2-D arrays) and shows
the images side by side (4 images are arranged 2 x 2). Axes are sized from the TRUE
pixel dimensions - a 256-px image appears smaller than a 300-px one - and the
display never resizes the data.

Two independent options:

rescale - what happens to the pixel values before display (3 states), and the
colorbar always shows the range that is actually displayed:
- "clip"    : values clipped to [0, 1], the clipped image is displayed;
              colorbar reads 0..1
- "raw"     : values kept intact and displayed over their own full range;
              the true [min, max] is visible on the colorbar
- "rescale" : values stretched to [0, 1] (min -> 0, max -> 1);
              colorbar reads 0..1 - the original levels are lost

display - the colormap:
- "gray"    : grayscale (default)
- "hilo"    : robust grayscale: the gray window covers the [1%, 99%] quantiles
              of the displayed values, the lowest 1% of the pixels show blue
              and the highest 1% red (the colorbar shows the quantile window)
- "viridis" : false color

The actual [min, max] of every image is printed under its title.
"""
import numpy as np
import matplotlib.pyplot as plt

cmap_hilo = plt.get_cmap("gray").copy()
cmap_hilo.set_under("#2060ff")   # below the color range -> blue
cmap_hilo.set_over("#ff3020")    # above the color range -> red

_cmaps = {"gray": "gray", "hilo": cmap_hilo, "viridis": "viridis"}


def show_images(images, rescale="clip", display="gray", px_per_inch=64, fontsize=16):
    """images: dict {title: 2-D array}. See the module docstring for the options.

    rescale ("raw" | "clip" | "rescale") and display ("gray" | "hilo" | "viridis")
    may each be one string applied to every image, or a list/tuple with one value
    per image, so the same image can be compared under different settings."""
    items = list(images.items())
    n = len(items)
    rescale = list(rescale) if isinstance(rescale, (list, tuple)) else [rescale] * n
    display = list(display) if isinstance(display, (list, tuple)) else [display] * n
    ncols = 1 if n <= 1 else 2
    nrows = (n + ncols - 1) // ncols
    grid = [items[r * ncols:(r + 1) * ncols] for r in range(nrows)]
    col_w = [max(row[j][1].shape[1] for row in grid if j < len(row)) for j in range(ncols)]
    row_h = [max(img.shape[0] for _, img in row) for row in grid]
    fig, axes = plt.subplots(nrows, ncols, gridspec_kw={"width_ratios": col_w},
                             figsize=(1.3 * sum(col_w) / px_per_inch, sum(row_h) / px_per_inch + nrows),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for k, (ax, (title, img)) in enumerate(zip(axes, items)):
        if rescale[k] == "clip":
            data = np.clip(img, 0.0, 1.0)
            vmin, vmax = 0.0, 1.0
        elif rescale[k] == "rescale":
            lo, hi = float(img.min()), float(img.max())
            data = (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)
            vmin, vmax = 0.0, 1.0
        else:  # "raw": values untouched, the colorbar shows the true range
            data = img
            vmin, vmax = float(img.min()), float(img.max())
        if display[k] == "hilo":
            # robust window: gray spans the [1%, 99%] quantiles of the displayed
            # values, the 1% extreme tails show blue (low) / red (high)
            qlo, qhi = np.quantile(data, [0.01, 0.99])
            if qhi > qlo:
                vmin, vmax = float(qlo), float(qhi)
        im = ax.imshow(data, cmap=_cmaps[display[k]], vmin=vmin, vmax=vmax)
        ax.set_title(f"{title}\n[{img.min():.2f}, {img.max():.2f}]", fontsize=fontsize)
        fig.colorbar(im, ax=ax, shrink=0.8, extend="both" if display[k] == "hilo" else "neither")
    plt.show()
