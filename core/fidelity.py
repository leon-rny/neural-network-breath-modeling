"""Fidelity of synthetic trials under LOSO: signal-space MMD and morphology, both against the held-out participant.

All signals are baseline-corrected (mean of the first 5 samples subtracted per channel) and, for the MMD, divided per
channel by the SD of the baseline-corrected real training trials, so humidity and temperature weigh equally.
  MMD         squared MMD with a Gaussian kernel on the flattened two-channel trials (72-dim signal space),
              bandwidth by the median heuristic on the pooled sets (biased V-statistic, >= 0)
                mmd           synthetic vs. the held-out participant's trials
                mmd_class     the same per class, mean over the 3 classes
                mmd_train     synthetic vs. the real training trials
                mmd_real_ref  real training trials vs. the held-out participant (reference level: participant shift)
  morphology  per trial and channel: onset (s, first sample >= 10 % of the trial maximum), rise slope (largest
              increase between consecutive samples, per s) and plateau (mean of the last 6 samples, ~12 s); per
              feature the Wasserstein-1 distance between the synthetic and the held-out participant's values within
              each class, in units of the feature's SD over the real training trials, mean over the 3 classes
              (reference: the same between the real training trials and the held-out participant)
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

DT = 2.004  # sampling interval (s)
PLATEAU = 6  # last samples averaged for the plateau level
ONSET_FRACTION = 0.1


def baseline_corrected(signals: np.ndarray) -> np.ndarray:
    S = np.asarray(signals, dtype=float)
    return S - S[:, :, :5].mean(2, keepdims=True)


def signal_mmd(X: np.ndarray, Y: np.ndarray) -> float:
    """Squared MMD between two sets of (n, 2, T) signals: Gaussian kernel on the flattened trials, bandwidth sigma =
    median pairwise distance of the pooled sets, empirical averages over all within and cross pairs."""
    A, B = np.asarray(X, dtype=float).reshape(len(X), -1), np.asarray(Y, dtype=float).reshape(len(Y), -1)
    d_aa, d_bb, d_ab = cdist(A, A, 'sqeuclidean'), cdist(B, B, 'sqeuclidean'), cdist(A, B, 'sqeuclidean')
    sigma2 = np.median(np.concatenate([d_aa[np.triu_indices(len(A), 1)], d_bb[np.triu_indices(len(B), 1)], d_ab.ravel()]))
    k = lambda d: np.exp(-d / (2 * sigma2))  # noqa: E731
    return float(k(d_aa).mean() + k(d_bb).mean() - 2 * k(d_ab).mean())


def mmd_report(synth: np.ndarray, y_synth: np.ndarray, train: np.ndarray, test: np.ndarray, y_test: np.ndarray) -> dict:
    """MMD of a synthetic set (see module doc); all inputs absolute or baseline-corrected (n, 2, T) signals."""
    tr = baseline_corrected(train)
    scale = tr.std(axis=(0, 2)).reshape(1, 2, 1) + 1e-6
    S, R, Q = baseline_corrected(synth) / scale, tr / scale, baseline_corrected(test) / scale
    return {'mmd': signal_mmd(S, Q),
            'mmd_class': float(np.mean([signal_mmd(S[y_synth == c], Q[y_test == c]) for c in range(3)])),
            'mmd_train': signal_mmd(S, R),
            'mmd_real_ref': signal_mmd(R, Q)}


def morphology(signals: np.ndarray) -> pd.DataFrame:
    """Onset (s), rise slope (per s) and plateau of each channel (H = humidity, T = temperature) per trial."""
    S = baseline_corrected(signals)
    rows = []
    for s in S:
        f = {}
        for c, x in zip('HT', s):
            above = np.nonzero(x >= ONSET_FRACTION * x.max())[0] if x.max() > 0 else np.array([len(x) - 1])
            f[f'onset_{c}'] = above[0] * DT
            f[f'slope_{c}'] = np.diff(x).max() / DT
            f[f'plateau_{c}'] = x[-PLATEAU:].mean()
        rows.append(f)
    return pd.DataFrame(rows)


def morphology_report(synth: np.ndarray, y_synth: np.ndarray, train: np.ndarray, y_train: np.ndarray, test: np.ndarray, y_test: np.ndarray) -> dict:
    """Class-wise Wasserstein-1 distances of the morphology features to the held-out participant (see module doc):
    {<feature>: synthetic, <feature>_real_ref: real training trials}."""
    F_s, F_tr, F_te = morphology(synth), morphology(train), morphology(test)
    sd = F_tr.std() + 1e-9
    w = lambda F, y: {k: float(np.mean([wasserstein_distance(F.loc[y == c, k], F_te.loc[y_test == c, k]) / sd[k] for c in range(3)]))  # noqa: E731
                      for k in F_te.columns}
    return {**w(F_s, np.asarray(y_synth)), **{f'{k}_real_ref': v for k, v in w(F_tr, np.asarray(y_train)).items()}}
