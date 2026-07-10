# Neural Network Breath Modeling

Generative models for synthesizing respiratory sensor signals (humidity + temperature), including a physics-informed variant built on an advection–diffusion channel model of the sensor, benchmarked against non-physics generators (CVAE, VAE, GAN, diffusion). Downstream usefulness of the synthetic data is measured with a *train-on-synthetic, test-on-real* (TSTR) protocol against a *train-on-real, test-on-real* (TRTR) ceiling.

Master's thesis project, TU Berlin, Telecommunication Networks Group (TKN).

## Overview

The goal is a generative physics-informed neural network (PINN) that produces synthetic breath signals statistically matching a small real dataset, so that a downstream classifier can be trained on synthetic data. The physics-informed model is benchmarked against purely data-driven generators to quantify what, if anything, the physics prior adds.

Two evaluation axes are used throughout:

- **Fidelity:** signal-space Maximum Mean Discrepancy (MMD) and per-feature morphology distributions (onset, rise slope, plateau) of real vs. synthetic signals.
- **Utility:** TSTR classification accuracy / weighted F1 / ROC-AUC, reported against the TRTR ceiling, under both *k*-fold (in-distribution) and leave-one-subject-out (LOSO, cross-subject) cross-validation.

## Dataset

Based on https://doi.org/10.5281/zenodo.15697086. Three breath-pattern classes: `bradypnea` (slow), `eupnea` (normal), `tachypnea` (fast).

- **Channels:** humidity (%) and temperature (°C), baseline-corrected by subtracting the per-signal minimum.
- **Shape:** 36 samples ≈ 70 s per trial, sampling rate ≈ 0.5 Hz (all trials uniform at 36 samples).
- **Metadata per trial:** participant, region (`mouth` / `nose`), trial number.
- **Regions are kept separate** — mouth signals are ~1.5–2× stronger than nose; the two are never pooled.
- **CIR data** (`dataset/cir/`): channel-impulse-response trials used to fit the advection–diffusion channel model that the physics-informed generator is built on.

Layout under `dataset/`: `bradypnea/`, `eupnea/`, `tachypnea/` (raw `.dat` trials), plus `cir/` and
`sir/` for the channel-model fit.

## Installation

Developed with Python 3.13.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

All entry points are Python modules run from the repository root, e.g. `python -m core.train …`. Two sensor regions (`mouth`, `nose`) are trained and evaluated independently. Complete implementation with extensive explanation are provided in the thesis.

## Citation

```bibtex
@masterthesis{neymeyer2026neural,
  author = {Neymeyer, Ramin Leon},
  title = {{Neural Network Framework for Modeling and Classification of Exhaled Breath Patterns}},
  advisor = {Bhattacharjee, Sunasheer},
  institution = {School of Electrical Engineering and Computer Science (EECS)},
  location = {Berlin, Germany},
  month = {7},
  referee = {Dressler, Falko and Kao, Odej},
  school = {TU Berlin (TUB)},
  type = {Master's Thesis},
  year = {2026},
}
```