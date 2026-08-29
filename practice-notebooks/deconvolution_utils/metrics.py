"""Workshop - 3D Deconvolution Microscopy (EPFL Center for Imaging)
Course material: lecture companion for in-class demonstrations and practical sessions.

Author : Daniel Sage, Biomedical Imaging Group, EPFL, Lausanne, Switzerland
Date   : 27 August 2026
License: BSD 3-Clause + citationware - if this material is useful to your
         research or teaching, please cite:
         D. Sage et al., "DeconvolutionLab2: An open-source software for
         deconvolution microscopy," Methods, vol. 115, pp. 28-41, 2017.

Image quality metrics, one line each. The inputs are numpy arrays or torch tensors
(any shape, converted to flat numpy arrays); the reference r is the ground truth.
- si_snr : scale-invariant SNR (torchmetrics convention), the default metric of the course
- snr    : signal-to-noise ratio in dB
- psnr   : peak SNR in dB, for images in [0, 1]
- ssim   : structural similarity (scikit-image), for images in [0, 1]
"""
import numpy as np
from skimage.metrics import structural_similarity


def to_numpy(a):
    return a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)


def si_snr(e, r): e, r = to_numpy(e).ravel(), to_numpy(r).ravel(); s = (e @ r) / (r @ r) * r; return float(10 * np.log10(np.sum(s ** 2) / (np.sum((e - s) ** 2) + 1e-15)))
def snr(e, r): e, r = to_numpy(e).ravel(), to_numpy(r).ravel(); return float(10 * np.log10(np.sum(r ** 2) / (np.sum((r - e) ** 2) + 1e-15)))
def psnr(e, r): return float(-10 * np.log10(max(np.mean((to_numpy(e) - to_numpy(r)) ** 2), 1e-15)))
def ssim(e, r): return float(structural_similarity(to_numpy(r).squeeze(), to_numpy(e).squeeze(), data_range=1.0))
