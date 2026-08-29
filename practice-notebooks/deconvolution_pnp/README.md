# 2-D deconvolution with DeepInverse, Plug-and-Play — `pnp.ipynb`

Two degraded test images (**HeLa nuclei** `nucspot-hela-crop`, 300 × 300, and the **Siemens star** `synthetic_siemens_star`, 256 × 256): Gaussian PSF σ = 2 px (deepinv `Blur`) + Poisson noise (gain 0.005, 200 photons at intensity 1). Kernel `Python (deepinverse)`; runs in about 2 min on an Apple GPU (the executed notebook weighs ≈ 27 MB because of the image panels).

| file | content |
|---|---|
| `pnp.ipynb` | everything, algorithms included (the core of the subproject, a few lines each on top of the deepinv `Blur` operator): `rif`, `landweber` (± positivity, ± regularizer), `richardson_lucy` (± regularizer), and the regularizers `Quadratic` (Tikhonov, Hessian), `TV`, `Denoiser` — objects with one method `prox(x, λ, step)`. Every method runs on both images with a **set of λ**, the images are always shown two per row (undershoots in blue, overshoots in red), the **loss** and the SI-SNR of the iterative methods are plotted per iteration, and the notebook ends with the best setting of every method |

The pretrained denoisers DnCNN and DRUNet come from `models/load_models.py` (`load_denoisers(device, names=["DnCNN", "DRUNet"])` — SwinIR is not loaded, too slow).

## Methods

- **RIF** — regularized inverse filter $\hat X = H^* Y / (|H|^2 + \lambda)$, closed form in Fourier.
- **Landweber** — $x \leftarrow x + \gamma A^T(y - Ax)$, γ = 1, with and without positivity; the number of iterations regularizes (semiconvergence, visible on the SI-SNR curve while the loss keeps decreasing).
- **Richardson–Lucy** — $x \leftarrow x \cdot A^T(y / Ax)$, the Poisson maximum-likelihood estimator; loss = Poisson negative log-likelihood.
- **Regularizers** plugged after each iteration (proximal step): Tikhonov $\tfrac12\|\nabla x\|^2$ and Hessian $\tfrac12\|\Delta x\|^2$ (closed form in Fourier), TV $\|\nabla x\|_1$ (deepinv `TVPrior`), **DRUNet** (Plug-and-Play, λ = denoising strength σ) and **DnCNN** (fixed strength, λ = mixing weight). Landweber + regularizer = proximal gradient (100 iterations, positivity); RL + regularizer = regularized RL (30 iterations).

## Results (best λ of every method, SI-SNR in dB)

| method | nucspot (input 8.65) | siemens (input 4.71) |
|---|---|---|
| RIF | 9.91 (λ = 0.1; set 0.0001 … 0.1) | 8.19 (λ = 0.01) |
| Landweber | 10.21 (30 it) | 8.24 (100 it) |
| Landweber + positivity | 10.21 (30 it) | 9.21 (100 it) |
| Richardson–Lucy | 10.46 (15 it) | 9.29 (50 it) |
| Landweber + Tikhonov | 10.25 (λ = 0.03) | 9.18 (λ = 0.003) |
| Landweber + TV | 10.26 (λ = 0.001) | 8.56 (λ = 0.001) |
| Landweber + Hessian | 10.24 (λ = 0.03) | 9.29 (λ = 0.001) |
| Landweber + DRUNet | 9.94 (σ = 0.02) | **10.18** (σ = 0.05) |
| Landweber + DnCNN | 10.53 (mix 1.0) | 9.53 (mix 0.75) |
| RL + Tikhonov | 10.55 (λ = 0.003) | 8.94 (λ = 0.003) |
| RL + TV | 10.50 (λ = 0.001) | 8.23 (λ = 0.001) |
| RL + Hessian | 10.54 (λ = 0.0003) | 8.99 (λ = 0.0003) |
| RL + DRUNet | 10.49 (σ = 0.003) | 9.53 (σ = 0.02) |
| RL + DnCNN | **10.70** (mix 0.75) | 9.19 (mix 1.0) |

Reading the results:
- On the textured **HeLa image** every method lands within 0.5 dB of the others (10.2–10.7 dB): the Poisson noise at this level limits what any regularizer can recover; RL and the learned priors are slightly ahead.
- On the **Siemens star** (sharp edges, large flat areas, a background at exactly 0) the **positivity constraint alone brings +1 dB** to Landweber, RL is positive by construction, and the **PnP DRUNet prior gains another +1 dB** (10.2 dB) — the learned prior knows what piecewise-flat images look like, where TV (8.6 dB) over-smooths the fine spokes.
- The **loss curves** decrease monotonically for every method while the **SI-SNR curves** peak and then fall (semiconvergence) — the number of iterations is a regularization parameter like λ; the plain solvers peak at 23–25 iterations (Landweber) and 25 iterations (RL) on the HeLa image, later on the star.
- The λ sets are deliberately wide (two decades); the optimum is inside the set for every method, which is what students should check first.
