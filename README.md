# Neural Network Breath Modeling

A physics-informed conditional variational autoencoder (PI-CVAE) for respiratory sensor signals: the decoder emits a calibrated advection–diffusion transport and sensor response, with the conditional decoder of a CVAE as residual. It is compared with an unconstrained CVAE in fidelity and as augmentation for breathing-pattern classification, all under nested leave-one-participant-out (LOSO) evaluation.

The code is based on my Master's thesis (TU Berlin, Telecommunication Networks Group).

## Dataset

Based on https://doi.org/10.5281/zenodo.15697086. Three breath-pattern classes: `bradypnea` (slow), `eupnea` (normal), `tachypnea` (fast), recorded at the mouth and the nose of eight participants.

- **Channels:** humidity (%) and temperature (°C), 36 samples ≈ 70 s per trial (≈ 0.5 Hz).
- **Metadata per trial:** participant, region (`mouth` / `nose`), trial number.
- **Regions are kept separate:** mouth and nose are trained and evaluated independently.
- **Calibration:** (`dataset/sir/`): single-exhalation recordings per region, used to fit the transport and sensor response of the PI-CVAE.

## Evaluation

Nested LOSO over the eight participants (`core/nested.py`, `core/report.py`). In each outer fold one participant is held out for testing, the next one is the validation participant (CNN epoch, generator checkpoint), and the remaining six are training participants. An inner LOSO loop over the seven non-test participants selects the configuration; the held-out participant influences neither feature selection, tuning, generator training nor selection. Five seeds (0, 1, 7, 42, 123).

- **Classifiers:** downstream classifier is used to check if synthetic augmentation helps training
- **Controls:** Gaussian jitter and same-class mixup are tested to understand if any enlargement of the dataset might help
- **Fidelity:** maximum mean discrepancy (MMD) tests if the created samples look realistic

## Installation

Developed with Python 3.13.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

From the repository root of a run directory (step 2 on a Slurm cluster):

```bash
experiments/00_calibration.sh                         # results/pinn/params_<region>.npy
mkdir -p logs && sbatch experiments/01_experiment.sh  # results/nested/, results/kfold/
experiments/02_report.sh                              # results/summary.csv, fidelity.csv, fidelity_kfold.csv, selection.csv, nested.csv, report.txt
```

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
