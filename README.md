# Neural Network Breath Modeling

A physics-informed conditional variational autoencoder (PI-CVAE) for respiratory sensor signals (humidity + temperature): the decoder emits a calibrated advection–diffusion transport and sensor response, with the conditional decoder of a CVAE as residual. It is compared with an unconstrained CVAE in fidelity (signal-space MMD, morphology, train-on-synthetic-test-on-real) and as augmentation for breathing-pattern classification, all under nested leave-one-participant-out (LOSO) evaluation.

The code of the Master's thesis (TU Berlin, Telecommunication Networks Group), including the VAE, GAN and diffusion baselines and *k*-fold evaluation, is at the git tag [`thesis`](../../tree/thesis). This branch contains the corrected dataset and the evaluation of the paper.

## Dataset

Based on https://doi.org/10.5281/zenodo.15697086. Three breath-pattern classes: `bradypnea` (slow), `eupnea` (normal), `tachypnea` (fast), recorded at the mouth and the nose of eight participants.

- **Channels:** humidity (%) and temperature (°C), 36 samples ≈ 70 s per trial (≈ 0.5 Hz).
- **Metadata per trial:** participant, region (`mouth` / `nose`), trial number.
- **Regions are kept separate**; mouth and nose are trained and evaluated independently.
- **Calibration** (`dataset/sir/`): single-exhalation recordings per region, used to fit the transport and sensor response of the PI-CVAE.

Layout under `dataset/`: `bradypnea/`, `eupnea/`, `tachypnea/` (raw `.dat` trials) and `sir/`.

## Evaluation

Nested LOSO over the eight participants (`core/nested.py`, `core/report.py`). In each outer fold one participant is held out for testing, the next one is the validation participant (CNN epoch, generator checkpoint), and the remaining six are training participants. An inner LOSO loop over the seven non-test participants selects the configuration; the held-out participant influences neither feature selection, tuning, generator training nor selection. Five seeds (0, 1, 7, 42, 123).

- **Classifiers:** the thesis stacker (tsfresh + SHAP top-20 + SMOTE + XGBoost / CatBoost, random-forest meta-learner; `core/stacker.py`), LightGBM on combo features tuned by Optuna inside the training split (`core/lgbm.py`), a raw-signal 1D-CNN (`core/cnn.py`) and the LightGBM + CNN ensemble.
- **Augmentation of the CNN (selected in the inner loop):** physics gain (random sensor gain ×/÷1.8 per channel + DHT22 observation) and PI-CVAE posterior samples + physics gain (`core/augment.py`).
- **Controls (outer folds):** Gaussian jitter, same-class mixup, CVAE posterior samples + physics gain, and the CVAE / PI-CVAE posterior samples without the gain step.
- **Fidelity (outer folds):** MMD and morphology (onset, rise slope, plateau) of CVAE / PI-CVAE prior samples and of every augmentation set against the held-out participant (`core/fidelity.py`), and train-on-synthetic-test-on-real accuracy.
- **k-fold fidelity:** the same MMD and morphology under the thesis protocol (class-stratified 5-fold split of the trials, so training and test trials come from the same participants; `core/kfold.py`), to contrast within-participant with cross-participant fidelity.
- **Generators:** CVAE and PI-CVAE (`models/`, trained by `core/train.py`), calibration by `core/fit_sir.py`.

## Installation

Developed with Python 3.13.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

From the repository root of a run directory (on a Slurm cluster for steps 2 and 3, which can run at the same time):

```bash
experiments/00_calibration.sh                      # results/pinn/params_<region>.npy
mkdir -p logs && sbatch experiments/01_nested.sh   # 640 tasks -> results/nested/<task>.json
sbatch experiments/02_kfold.sh                     # 50 tasks -> results/kfold/<task>.json
experiments/03_report.sh                           # results/summary.csv, fidelity.csv, fidelity_kfold.csv, selection.csv, nested.csv, report.txt
```

Single tasks: `python -m core.nested --region mouth --outer 0 --inner -1 --seed 0` (`--inner -1` = outer task), `python -m core.kfold --region mouth --fold 0 --seed 0`.

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
