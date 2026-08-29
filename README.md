>#### Course Material & Practical Sessions

# 3D Deconvolution Microscopy
*Daniel Sage — Biomedical Imaging Group, École Polytechnique Fédérale de Lausanne (EPFL)*

<img src="practice-notebooks/assets/epfl-center-for-imaging.svg" alt="EPFL Center for Imaging" height="50" align="right"/>

**Important note.** This repository contains the material of the workshop **3D Deconvolution Microscopy** given by Daniel Sage (EPFL): the slides of the lecture and two practical sessions. It is a direct complement to the in-class course; it is not self-teaching material.

The practical sessions have one goal: practice deconvolution on simulated and real microscopy images, and build a visual intuition of what it does.

**References**
- DeconvolutionLab2: http://bigwww.epfl.ch/deconvolution/deconvolutionlab2/
- D. Sage, L. Donati, F. Soulez, D. Fortun, G. Schmit, A. Seitz, R. Guiet, C. Vonesch, M. Unser, "DeconvolutionLab2: An open-source software for deconvolution microscopy," *Methods*, vol. 115, pp. 28–41, 2017.

## Practical session — DeconvolutionLab2 (Fiji)

- Download Fiji: [`Fiji`](https://fiji.sc)
- Instructions: [`Practice-Deconvolution.pdf`](practice-deconvolutionlab2/Practice-Deconvolution.pdf)
- Reference: [`DeconvolutionLab2.pdf`](practice-deconvolutionlab2/DeconvolutionLab2.pdf), [website](http://bigwww.epfl.ch/deconvolution/deconvolutionlab2/)
- Installation: copy `DeconvolutionLab_2-2.0.0.jar` and `PSF_Generator.jar` into the `plugins/` folder of Fiji, restart Fiji. `FFTW.zip` is optional (faster FFT).
- Images: [`practice-images/`](practice-deconvolutionlab2/practice-images/) — 2-D and 3-D images, PSFs

## Practical session 2 — Python notebooks

- `deconvolution_assessment/assessment.ipynb` — metrics, display, noise, PSF, regularized inverse filter
- `deconvolution_pnp/pnp.ipynb` — Plug-and-Play deconvolution with deepinv (local only, needs `torch`)
- `deconvolution_utils/` — shared modules: simulation, display, metrics

### In the browser (JupyterLite)
[![jupyter-lite-badge](https://jupyterlite.rtfd.io/en/latest/_static/badge.svg)](https://dasv74.github.io/workshop-deconvolution/lab/index.html)

[Open JupyterLite](https://dasv74.github.io/workshop-deconvolution/lab/index.html) → `deconvolution_assessment/assessment.ipynb`

Nothing to install. The first cell takes about a minute (loading numpy, scipy, scikit-image); if an import fails right after the page opened, run the cell again.

### Locally

```bash
conda create -n deconvolution-practice python=3.11
conda activate deconvolution-practice
pip install numpy scipy matplotlib scikit-image ipywidgets jupyterlab
pip install torch deepinv
cd practice-notebooks && jupyter lab
```

## Deploy

Every push to `main` runs [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml): it builds the JupyterLite site from `practice-notebooks/` with the packages of [`requirements.txt`](requirements.txt) and publishes it on GitHub Pages (once: *Settings → Pages → Source: GitHub Actions*).

## License

BSD 3-Clause + *citationware*: if this material is useful to your research or teaching, please cite the *Methods* paper above.
