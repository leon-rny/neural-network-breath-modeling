'''
Per-trial quality control for the breathing dataset.

Flags CLEAR artifacts only — the design bias is a low false-positive rate, so the
data-derived thresholds are deliberately loose (large MAD multipliers, sane absolute
floors). Reuses `core.data.load_dataset` and stays numpy/pandas only.

CLI:
    python -m core.qc [--dataset_dir dataset] [--out results/qc_breathing.csv]

Programmatic:
    from core.qc import load_flagged
    flagged = load_flagged()  # -> set[str] of flagged filenames
'''
import argparse
import os

import numpy as np
import pandas as pd

from core.data import load_dataset

# --- thresholds (all named, conservative; bias = low false-positive rate) ---

# (1) Head-movement / non-physiological jump.
# A trial's max sample-to-sample |Δ| is compared to a robust population bound
# median(maxdiff) + JUMP_MAD_K * MAD(maxdiff). A per-channel absolute FLOOR guards
# against a tiny MAD turning normal variation into a flag (DHT22 res is 0.1 %RH / 0.1 °C,
# so a real head-bump discontinuity is several units between two ~2 s samples).
JUMP_MAD_K = 6.0           # MAD multiplier for the robust upper bound
JUMP_FLOOR_HUM = 8.0       # %RH; never flag a jump below this absolute |Δ|
JUMP_FLOOR_TEMP = 3.0      # °C;  never flag a jump below this absolute |Δ|

# (2) Unstable pre-onset baseline: std of the first BASELINE_N samples too high.
# A settled pre-onset baseline barely moves; high std means the breath started early
# or the sensor had not stabilised. Floors set well above DHT22 quantisation noise.
BASELINE_N = 5             # number of leading samples treated as the baseline window
BASELINE_STD_HUM = 2.0     # %RH; flag if std(first N humidity) exceeds this
BASELINE_STD_TEMP = 0.8    # °C;  flag if std(first N temperature) exceeds this

# (3) Dropouts / flat segments: a long run of identical consecutive values, or any NaN/inf.
# DHT22 holds the last value on a read failure, producing a stuck flat run. The breath
# transient also ends in a genuine PLATEAU that flattens — and in this dataset every long
# humidity flat run sits right at that plateau (near the signal's own max). So a flat run is
# only an artifact when it is BOTH long AND well below the trial's peak (i.e. the sensor
# froze mid-rise, not at the physiological plateau). NaN/inf always flags.
FLAT_RUN_LEN = 8           # consecutive identical samples to call it a stuck/flat run
FLAT_PLATEAU_MARGIN = 5.0  # only flag if the stuck value is >this far below the trial peak

# (4) Saturation: humidity pinned at the DHT22 HARD cap for a sustained span.
# The sensor ceiling is 100.0 %RH; readings of ~99.9 are the normal humidity plateau, not a
# clip, so the level must be the true hardware cap to avoid flagging real plateaus.
SAT_HUM_LEVEL = 100.0      # %RH; samples >= this are pinned at the sensor cap
SAT_RUN_LEN = 6            # consecutive capped samples to flag as saturated

# (5) Amplitude outlier within its (participant, class, region) cell.
# Robust MAD z-score on the baseline-corrected peak amplitude; only meaningful with
# enough siblings, so require a minimally populated cell.
AMP_Z_THRESH = 3.5         # |robust z| above this is an amplitude outlier
AMP_MIN_CELL = 4           # minimum trials in the cell to even run this check

# scale factor making MAD a consistent estimator of std for normal data (1/Phi^-1(0.75))
_MAD_TO_STD = 1.4826


def _mad(x: np.ndarray) -> float:
    '''Median absolute deviation, scaled to be comparable to a standard deviation.'''
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    return _MAD_TO_STD * np.median(np.abs(x - med))


def _max_abs_diff(sig: np.ndarray) -> float:
    '''Largest sample-to-sample |Δ| (0.0 for a degenerate single-sample signal).'''
    sig = np.asarray(sig, dtype=float)
    finite = sig[np.isfinite(sig)]
    if finite.size < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(finite))))


def _longest_equal_run(sig: np.ndarray) -> tuple:
    '''Longest run of identical consecutive values: returns (length, value_at_run).'''
    sig = np.asarray(sig)
    if sig.size == 0:
        return 0, np.nan
    changes = np.diff(sig) != 0
    # run boundaries are where the value changes; diffs of the boundaries are run lengths
    bounds = np.flatnonzero(np.concatenate(([True], changes, [True])))
    run_lengths = np.diff(bounds)
    i = int(np.argmax(run_lengths))
    return int(run_lengths[i]), float(sig[bounds[i]])


def _longest_true_run(mask: np.ndarray) -> int:
    '''Length of the longest run of True in a boolean mask.'''
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not mask.any():
        return 0
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def _peak_amplitude(sig: np.ndarray) -> float:
    '''Baseline-corrected peak amplitude: max(signal) - mean(first BASELINE_N samples).'''
    sig = np.asarray(sig, dtype=float)
    finite = sig[np.isfinite(sig)]
    if finite.size == 0:
        return np.nan
    baseline = np.mean(sig[:BASELINE_N])
    if not np.isfinite(baseline):
        baseline = np.median(finite)
    return float(np.nanmax(sig) - baseline)


def compute_qc(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Run all per-trial QC checks and return a results DataFrame (one row per trial).

    :param df: dataset DataFrame as returned by core.data.load_dataset.
    :return: DataFrame with identity columns, metric values, per-check booleans,
             an overall `flagged` boolean, and a `reasons` string.
    '''
    rows = []
    for r in df.to_dict('records'):
        hum = np.asarray(r['humidity'], dtype=float)
        temp = np.asarray(r['temperature'], dtype=float)

        # longest stuck run per channel, with the value it is stuck at and the channel peak,
        # so a genuine plateau (stuck at the peak) can be told apart from a frozen sensor.
        flat_h_len, flat_h_val = _longest_equal_run(hum)
        flat_t_len, flat_t_val = _longest_equal_run(temp)
        peak_h = float(np.nanmax(hum)) if np.isfinite(hum).any() else np.nan
        peak_t = float(np.nanmax(temp)) if np.isfinite(temp).any() else np.nan
        # a stuck run that is long AND sits well below the trial peak == dropout artifact
        stuck_h = flat_h_len >= FLAT_RUN_LEN and (peak_h - flat_h_val) > FLAT_PLATEAU_MARGIN
        stuck_t = flat_t_len >= FLAT_RUN_LEN and (peak_t - flat_t_val) > FLAT_PLATEAU_MARGIN

        rows.append({
            # NOTE: filenames are NOT unique across classes (e.g. b_mouth_trial_13.dat exists in
            # bradypnea/eupnea/tachypnea), so `path` = class/filename is the per-trial unique key.
            'path': f"{r['class']}/{r['filename']}",
            'filename': r['filename'],
            'participant': r['participant'],
            'region': r['region'],
            'class': r['class'],
            'trial_num': r['trial_num'],
            # (1) jump metrics
            'max_diff_hum': _max_abs_diff(hum),
            'max_diff_temp': _max_abs_diff(temp),
            # (2) baseline-stability metrics
            'baseline_std_hum': float(np.std(hum[:BASELINE_N])),
            'baseline_std_temp': float(np.std(temp[:BASELINE_N])),
            # (3) dropout/flat metrics
            'has_nan_inf': bool(not np.all(np.isfinite(hum)) or not np.all(np.isfinite(temp))),
            'flat_run': max(flat_h_len, flat_t_len),
            'stuck_offpeak': bool(stuck_h or stuck_t),
            # (4) saturation metric
            'sat_run': _longest_true_run(hum >= SAT_HUM_LEVEL),
            # (5) amplitude metric (cell z-score added after the population pass)
            'peak_amp_hum': _peak_amplitude(hum),
        })

    out = pd.DataFrame(rows)

    # --- (1) population-derived jump bounds (robust, with absolute floor) ---
    hum_bound = max(np.median(out['max_diff_hum']) + JUMP_MAD_K * _mad(out['max_diff_hum']),
                    JUMP_FLOOR_HUM)
    temp_bound = max(np.median(out['max_diff_temp']) + JUMP_MAD_K * _mad(out['max_diff_temp']),
                     JUMP_FLOOR_TEMP)
    out['jump_bound_hum'] = hum_bound
    out['jump_bound_temp'] = temp_bound
    chk_jump = (out['max_diff_hum'] > hum_bound) | (out['max_diff_temp'] > temp_bound)

    # --- (2) unstable baseline ---
    chk_baseline = (out['baseline_std_hum'] > BASELINE_STD_HUM) | \
                   (out['baseline_std_temp'] > BASELINE_STD_TEMP)

    # --- (3) dropouts / flat / NaN-inf (flat run flags only when stuck off the plateau) ---
    chk_dropout = out['has_nan_inf'] | out['stuck_offpeak']

    # --- (4) saturation ---
    chk_saturation = out['sat_run'] >= SAT_RUN_LEN

    # --- (5) amplitude outlier within (participant, class, region) cell ---
    amp_z = pd.Series(0.0, index=out.index)
    cell_keys = ['participant', 'class', 'region']
    for _, idx in out.groupby(cell_keys).groups.items():
        if len(idx) < AMP_MIN_CELL:
            continue
        vals = out.loc[idx, 'peak_amp_hum'].to_numpy(dtype=float)
        med = np.median(vals)
        mad = _mad(vals)
        if mad <= 0:
            continue  # no spread -> cannot define an outlier
        amp_z.loc[idx] = (vals - med) / mad
    out['amp_zscore'] = amp_z
    chk_amplitude = out['amp_zscore'].abs() > AMP_Z_THRESH

    # --- assemble per-check booleans, overall flag, and reasons ---
    out['chk_jump'] = chk_jump
    out['chk_baseline'] = chk_baseline
    out['chk_dropout'] = chk_dropout
    out['chk_saturation'] = chk_saturation
    out['chk_amplitude'] = chk_amplitude
    out['flagged'] = chk_jump | chk_baseline | chk_dropout | chk_saturation | chk_amplitude

    reason_map = [
        ('chk_jump', 'jump'),
        ('chk_baseline', 'unstable_baseline'),
        ('chk_dropout', 'dropout'),
        ('chk_saturation', 'saturation'),
        ('chk_amplitude', 'amplitude_outlier'),
    ]
    out['reasons'] = [
        ';'.join(label for col, label in reason_map if row[col])
        for _, row in out.iterrows()
    ]

    return out


def load_flagged(csv_path: str = 'results/qc_breathing.csv') -> set:
    '''
    Return the set of flagged trial identifiers from a written QC CSV.

    Identifiers are `class/filename` (the `path` column), because plain filenames are NOT
    unique across the three class folders. A loader hook should therefore match each trial on
    `f"{class}/{filename}"`, not on `filename` alone, to avoid over-excluding sibling trials.

    :param csv_path: path to the QC CSV produced by this module.
    :return: set of `class/filename` paths whose `flagged` column is True (empty if file missing).
    '''
    if not os.path.exists(csv_path):
        return set()
    qc = pd.read_csv(csv_path)
    return set(qc.loc[qc['flagged'].astype(bool), 'path'])


def _print_summary(out: pd.DataFrame) -> None:
    '''Print a per-(participant,class,region) flagged-count summary and the flagged list.'''
    total = len(out)
    n_flag = int(out['flagged'].sum())
    print(f'\nQC summary: {n_flag}/{total} trials flagged '
          f'({100.0 * n_flag / total:.1f}%)\n')

    print('By reason:')
    for col, label in [('chk_jump', 'jump'), ('chk_baseline', 'unstable_baseline'),
                       ('chk_dropout', 'dropout'), ('chk_saturation', 'saturation'),
                       ('chk_amplitude', 'amplitude_outlier')]:
        print(f'  {label:18s}: {int(out[col].sum())}')

    print('\nBy (participant, class, region) cell [flagged/total | dominant reasons]:')
    for keys, cell in out.groupby(['participant', 'class', 'region']):
        cf = int(cell['flagged'].sum())
        if cf == 0:
            continue
        reasons = ';'.join(cell.loc[cell['flagged'], 'reasons']).split(';')
        reasons = [r for r in reasons if r]
        dom = pd.Series(reasons).value_counts()
        dom_str = ', '.join(f'{r}({n})' for r, n in dom.items())
        p, c, reg = keys
        print(f'  {p} / {c:10s} / {reg:5s}: {cf}/{len(cell)}  [{dom_str}]')

    flagged = out.loc[out['flagged'], ['path', 'reasons']].sort_values('path')
    print(f'\nFlagged trials ({len(flagged)}):')
    for path, reason in zip(flagged['path'], flagged['reasons']):
        print(f'  {path}  ->  {reason}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Per-trial QC for the breathing dataset.')
    parser.add_argument('--dataset_dir', default='dataset',
                        help='dataset directory passed to core.data.load_dataset')
    parser.add_argument('--out', default='results/qc_breathing.csv',
                        help='output CSV path')
    args = parser.parse_args()

    df = load_dataset(dataset_dir=args.dataset_dir)
    out = compute_qc(df)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f'Wrote {len(out)} rows to {args.out}')

    _print_summary(out)


if __name__ == '__main__':
    main()
