"""Workshop - 3D Deconvolution Microscopy (EPFL Center for Imaging)
Course material: lecture companion for in-class demonstrations and practical sessions.

Author : Daniel Sage, Biomedical Imaging Group, EPFL, Lausanne, Switzerland
Date   : 21 August 2026
License: BSD 3-Clause + citationware - if this material is useful to your
         research or teaching, please cite:
         D. Sage et al., "DeconvolutionLab2: An open-source software for
         deconvolution microscopy," Methods, vol. 115, pp. 28-41, 2017.

Simulation toolbox: PSF models, blurring, noise generators, and computed
control targets drawn inside a test image.

PSFs (all normalized to sum 1):
- gaussian_psf(sigma)          : generic smooth blur, h(r) ~ exp(-r^2 / 2 sigma^2)
- airy_psf(radius)             : diffraction-limited lens (circular aperture),
                                 h(r) ~ (2 J1(v)/v)^2, v = 3.83 r/R, R = first dark ring = 0.61 lambda/NA
- disk_psf(radius)             : ideal geometric defocus, a uniform circle
- motion_psf(length, angle)    : linear motion blur, a straight streak
- double_helix_psf(separation) : two Gaussian lobes (engineered PSF; in 3-D
                                 localization microscopy their angle encodes depth)

Blurring is a circular convolution computed in the Fourier domain:
y = h * x  <=>  Y = H X, with H = psf_to_otf(h, image shape).

Noise:
- add_gaussian_noise(image, sigma) : additive, same variance everywhere (readout noise)
- add_poisson_noise(image, k)      : photon counting, y = N/k with N ~ Poisson(k x);
                                     k = expected photons at intensity 1, so
                                     mean(y) = x and var(y) = x/k (signal-dependent)

Control targets (torch tensors (1, 1, H, W) in [0, 1], size unchanged):
- add_ladder(image)      : LEFT strip, resolution ladder (lines of decreasing
                           thickness/spacing) + intensity ladder (3 px lines of
                           increasing brightness)
- add_stair_steps(image) : RIGHT strip, flat intensity steps dark -> bright
"""
import numpy as np
from scipy.special import j1


# ---------------------------------------------------------------- PSF models

def gaussian_psf(sigma, size=None):
    size = size or 2 * int(3 * sigma) + 1
    r = np.arange(size) - size // 2
    gx, gy = np.meshgrid(r, r, indexing="ij")
    psf = np.exp(-(gx ** 2 + gy ** 2) / (2 * sigma ** 2))
    return psf / psf.sum()


def airy_psf(radius, size=None):
    size = size or 2 * int(3 * radius) + 1
    r = np.arange(size) - size // 2
    gx, gy = np.meshgrid(r, r, indexing="ij")
    v = 3.8317 * np.hypot(gx, gy) / radius
    v = np.where(v == 0, 1e-9, v)
    psf = (2 * j1(v) / v) ** 2
    psf[size // 2, size // 2] = 1.0
    return psf / psf.sum()


def disk_psf(radius, size=None):
    size = size or 2 * int(radius + 1) + 1
    r = np.arange(size) - size // 2
    gx, gy = np.meshgrid(r, r, indexing="ij")
    psf = (np.hypot(gx, gy) <= radius).astype(float)
    return psf / psf.sum()


def motion_psf(length, angle=30.0, size=None):
    """Linear motion blur: a straight streak of the given length (px) at `angle` (deg)."""
    size = size or 2 * int(length / 2 + 2) + 1
    r = np.arange(size) - size // 2
    gx, gy = np.meshgrid(r, r, indexing="ij")
    t = np.deg2rad(angle)
    along = gx * np.sin(t) + gy * np.cos(t)     # position along the motion direction
    across = gx * np.cos(t) - gy * np.sin(t)    # distance to the motion line
    psf = np.exp(-across ** 2 / (2 * 0.5 ** 2)) * (np.abs(along) <= length / 2)
    return psf / psf.sum()


def double_helix_psf(separation, angle=45.0, sigma=1.0, size=None):
    """Double-helix PSF: two Gaussian lobes `separation` px apart at `angle` (deg).
    An engineered PSF used in 3-D localization microscopy, where the angle of the
    lobe pair rotates with the emitter's depth z (here a single 2-D slice)."""
    size = size or 2 * int(separation / 2 + 3 * sigma) + 1
    r = np.arange(size) - size // 2
    gx, gy = np.meshgrid(r, r, indexing="ij")
    t = np.deg2rad(angle)
    cx, cy = separation / 2 * np.sin(t), separation / 2 * np.cos(t)
    psf = (np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * sigma ** 2)) +
           np.exp(-((gx + cx) ** 2 + (gy + cy) ** 2) / (2 * sigma ** 2)))
    return psf / psf.sum()


# ------------------------------------------------------------------ blurring

def psf_to_otf(psf, shape):
    otf = np.zeros(shape)
    kh, kw = psf.shape
    otf[:kh, :kw] = psf
    otf = np.roll(otf, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    return np.fft.fft2(otf)


def blur(image, psf):
    return np.real(np.fft.ifft2(np.fft.fft2(image) * psf_to_otf(psf, image.shape)))


# --------------------------------------------------------------------- noise

def add_gaussian_noise(image, sigma, rng=None):
    rng = rng or np.random.default_rng(0)
    return image + rng.normal(0.0, sigma, image.shape)


def add_poisson_noise(image, k, rng=None):
    """Photon-counting noise: y = N/k, N ~ Poisson(k x), Pr(N=n) = (kx)^n e^{-kx} / n!

    k is the expected number of photons at intensity 1 (lower k = noisier)."""
    rng = rng or np.random.default_rng(0)
    return rng.poisson(np.clip(image, 0, None) * k) / k


# ----------------------------------------------------------- control targets

def add_ladder(image, width=30, thicknesses=(8, 6, 5, 4, 3, 2, 1), pairs=3):
    import torch
    out = image.clone()
    h = out.shape[-2]
    strip = torch.zeros(h, width, device=out.device)
    row = 2
    for t in thicknesses:                # line thickness, in pixels
        for _ in range(pairs):           # line/gap pairs per thickness
            if row + 2 * t >= h:
                break
            strip[row:row + t, 2:width - 2] = 1.0
            row += 2 * t
        row += 4
    n_levels = max((h - row - 2) // 6, 1)
    for k in range(n_levels):            # intensity ladder: 3 px lines
        strip[row:row + 3, 2:width - 2] = 0.15 + (1.0 - 0.15) * k / max(n_levels - 1, 1)
        row += 6
    out[..., :, :width] = strip
    return out


def add_stair_steps(image, width=30, steps=20):
    import torch
    out = image.clone()
    h = out.shape[-2]
    strip = torch.zeros(h, width, device=out.device)
    edges = torch.linspace(0, h, steps + 1).round().long()
    for k in range(steps):               # flat steps, dark (top) to bright (bottom)
        strip[edges[k]:edges[k + 1], :] = k / (steps - 1)
    out[..., :, -width:] = strip
    return out
