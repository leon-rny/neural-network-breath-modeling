"""Nested LOSO selection and report over the task results of core.nested (and the k-fold fidelity of core.kfold).

Writes (to --results, default results/):
  summary.csv    one row per task x candidate: region, outer / inner test participant (inner '' for outer tasks), seed,
                 candidate, accuracy and macro F1 (%), n_test
  fidelity.csv   one row per outer task x synthetic set: MMD and morphology distances (core.fidelity)
  fidelity_kfold.csv  one row per k-fold task x synthetic set (region, fold, seed, set, ...), if --kfold has results
  selection.csv  per menu x region x outer participant: inner score of every menu candidate and the selected one
  nested.csv     per menu x region x outer participant x seed: outer test scores of the selected candidate and of the
                 menu's reference
Selection: inner score = mean over the inner test participants of the seed-mean inner test score (--criterion); the
menu candidate with the highest inner score (ties: earlier in the menu) is evaluated once on the outer participant.
  A  ens, ens_physgain, ens_postgain                  vs. ens       LightGBM + CNN: augmentation selected
  B  stacker, lgbm, ens, ens_physgain, ens_postgain   vs. stacker   pipeline selected vs. the thesis stacker
  C  cnn, cnn_physgain, cnn_postgain                  vs. cnn       CNN: augmentation selected
Paired tests: two-sided Wilcoxon signed-rank on seed means per participant, over the 8 participants of a region, the
16 participant x region pairs and the 8 participants' means over both regions. Besides the menus: fixed comparisons on
the outer tasks (augmentation vs. controls, PI-CVAE vs. CVAE in TSTR accuracy, MMD and morphology).
k-fold fidelity: MMD of every synthetic set and of the real training trials against the test fold (same participants)
next to the LOSO values (unseen participant), and the PI-CVAE vs. CVAE tests over the seed means of the 5 folds per
region / 10 region x fold units (the folds share participants and most training trials, so these tests are optimistic).

Usage: python -m core.report [results/nested] [--results results] [--kfold results/kfold] [--criterion accuracy|macro_f1]
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from core.nested import MENU

MENUS = {'A': (['ens', 'ens_physgain', 'ens_postgain'], 'ens', 'LightGBM + CNN: augmentation selected on the training participants vs. real-only'),
         'B': (['stacker', 'lgbm', 'ens', 'ens_physgain', 'ens_postgain'], 'stacker', 'Pipeline selected on the training participants vs. stacker real-only'),
         'C': (['cnn', 'cnn_physgain', 'cnn_postgain'], 'cnn', 'CNN: augmentation selected on the training participants vs. real-only')}
COMPARISONS = [(f'{m}_{a}', f'{m}_{b}' if b else m) for m in ('ens', 'cnn')
               for a, b in (('jitter', ''), ('mixup', ''), ('cvaegain', ''), ('post', ''), ('cvaepost', ''), ('physgain', 'jitter'), ('physgain', 'mixup'),
                            ('postgain', 'cvaegain'), ('post', 'cvaepost'), ('postgain', 'post'), ('cvaegain', 'cvaepost'))]
COMPARISONS += [(f'tstr_picvae_{c}', f'tstr_cvae_{c}') for c in ('lgbm', 'cnn', 'ens')]
# fidelity PI-CVAE vs. CVAE: (set, set, label, metrics; None = all)
FIDELITY_PAIRS = [('picvae_prior', 'cvae_prior', 'prior samples', None), ('post', 'cvaepost', 'posterior samples', ['mmd', 'mmd_class']),
                  ('postgain', 'cvaegain', 'posterior samples + gain', ['mmd', 'mmd_class'])]
SEEDS = (0, 1, 7, 42, 123)


def load(out_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, fid = [], []
    for path in sorted(glob.glob(f'{out_dir}/*.json')):
        r = json.load(open(path))
        c = r['config']
        base = {'region': c['region'], 'outer': c['outer'], 'inner': c['inner'], 'seed': c['seed'], 'participant': r['test']}
        for cand, v in r['results'].items():
            rows.append({**base, 'candidate': cand, 'accuracy': 100 * v['accuracy'], 'macro_f1': 100 * v['macro_f1'], 'n_test': len(r['y_test'])})
        fid += [{**base, 'set': name, **v} for name, v in r['fidelity'].items()]
    return pd.DataFrame(rows), pd.DataFrame(fid)


def load_kfold(kfold_dir: str) -> pd.DataFrame:
    """One row per k-fold task (core.kfold) x synthetic set."""
    rows = []
    for path in sorted(glob.glob(f'{kfold_dir}/*.json')):
        r = json.load(open(path))
        c = r['config']
        rows += [{'region': c['region'], 'fold': c['fold'], 'seed': c['seed'], 'set': name, **v} for name, v in r['fidelity'].items()]
    return pd.DataFrame(rows)


def complete(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the (region, outer) cells with all N tasks (1 outer + N-1 inner) x 5 seeds of every menu candidate."""
    menu = df[df['candidate'].isin(MENU)]
    n = menu.groupby(['region', 'outer']).size()
    want = df.groupby('region')['participant'].nunique() * len(SEEDS) * len(MENU)
    full = [cell for cell in n.index if n[cell] == want[cell[0]]]
    for cell in n.index.difference(full):
        print(f'[REPORT] incomplete {cell}: {n[cell]} of {want[cell[0]]} menu rows -> skipped')
    return df.set_index(['region', 'outer']).loc[list(full)].reset_index()


def test(d: pd.Series) -> str:
    p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
    return f'Δ {d.mean():+.2f}, {(d > 0).sum()}/{len(d)} higher, p = {p:.3f}'


def paired(wide: pd.DataFrame, a: str, b: str, label: str, unit: str = 'pp', pool: str | None = 'participant') -> None:
    """Paired comparison a - b of a (region, unit) x column table of seed means; pool: index level whose means over both
    regions are tested as well (None: skip)."""
    d = wide[a] - wide[b]
    print(f'\n== {label} ({unit})')
    for region in d.index.unique('region'):
        print(f'  {region:5s} {wide.loc[region, b].mean():.4g} -> {wide.loc[region, a].mean():.4g} | {test(d.loc[region])}')
    print(f'  both regions ({len(d)} units): {test(d)}' + (f' | {pool} means over both regions: {test(d.groupby(pool).mean())}' if pool else ''))


def fidelity_tests(fw: pd.DataFrame, protocol: str = '', pool: str | None = 'participant') -> None:
    """PI-CVAE vs. CVAE (FIDELITY_PAIRS) on a (region, unit, set) x metric table of seed means."""
    metrics = [c for c in fw.columns if not c.endswith('real_ref')]
    for a, b, what, only in FIDELITY_PAIRS:
        for metric in only or metrics:
            paired(fw[metric].unstack('set'), a, b, f'{protocol}{metric}: PI-CVAE vs. CVAE {what}', unit='lower is closer', pool=pool)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('out_dir', nargs='?', default='results/nested')
    ap.add_argument('--results', default='results')
    ap.add_argument('--kfold', default='results/kfold', help='core.kfold results (skipped when empty)')
    ap.add_argument('--criterion', default='accuracy', choices=['accuracy', 'macro_f1'])
    a = ap.parse_args()
    os.makedirs(a.results, exist_ok=True)
    tag = '' if a.criterion == 'accuracy' else f'_{a.criterion}'
    df, fid = load(a.out_dir)
    df = complete(df)
    names = df[df['inner'] < 0].groupby(['region', 'outer'])['participant'].first()
    summary = df.assign(outer_participant=[names[(r, o)] for r, o in zip(df['region'], df['outer'])])
    summary = summary.assign(inner_participant=summary['participant'].where(summary['inner'] >= 0, ''))
    summary[['region', 'outer_participant', 'inner_participant', 'seed', 'candidate', 'accuracy', 'macro_f1', 'n_test']].to_csv(f'{a.results}/summary.csv', index=False)
    fid.drop(columns=['outer', 'inner']).to_csv(f'{a.results}/fidelity.csv', index=False)

    inner = df[(df['inner'] >= 0) & df['candidate'].isin(MENU)]
    outer = df[df['inner'] < 0].drop(columns=['inner'])
    scores = (inner.groupby(['region', 'outer', 'candidate', 'participant'])[['accuracy', 'macro_f1']].mean()
              .groupby(['region', 'outer', 'candidate']).mean().reset_index())
    selection, nested = [], []
    for menu, (cands, ref, _) in MENUS.items():
        for (region, o), s in scores.groupby(['region', 'outer']):
            s = s.set_index('candidate').loc[cands, a.criterion]
            chosen = cands[int(np.argmax(s.values))]  # first maximum = earlier in the menu
            selection.append({'menu': menu, 'region': region, 'participant': names[(region, o)], 'selected': chosen, **{f'inner_{c}': s[c] for c in cands}})
            o_rows = outer[(outer['region'] == region) & (outer['outer'] == o)].set_index(['seed', 'candidate'])
            for seed in SEEDS:
                sel, base = o_rows.loc[(seed, chosen)], o_rows.loc[(seed, ref)]
                nested.append({'menu': menu, 'region': region, 'participant': names[(region, o)], 'seed': seed, 'selected': chosen, 'reference': ref,
                               'accuracy': sel['accuracy'], 'macro_f1': sel['macro_f1'], 'reference_accuracy': base['accuracy'],
                               'reference_macro_f1': base['macro_f1'], 'delta': sel[a.criterion] - base[a.criterion], 'n_test': sel['n_test']})
    selection, nested = pd.DataFrame(selection), pd.DataFrame(nested)
    selection.to_csv(f'{a.results}/selection{tag}.csv', index=False)
    nested.to_csv(f'{a.results}/nested{tag}.csv', index=False)

    print(f'[REPORT] criterion {a.criterion}; {df.groupby(["region", "outer"]).ngroups} complete (region, outer) cells')
    part = outer.groupby(['region', 'participant', 'candidate'])[[a.criterion, 'n_test']].mean()
    print(f'\nEvery candidate on the outer participants without selection ({a.criterion}, mean over participants):')
    print(part[a.criterion].groupby(['region', 'candidate']).mean().unstack(0).round(1).to_string())
    for menu, (cands, ref, title) in MENUS.items():
        print(f'\n== Menu {menu}: {title}')
        d = nested[nested['menu'] == menu]
        units = d.groupby(['region', 'participant'])['delta'].mean()
        ref_col, sel_col = ('reference_accuracy', 'accuracy') if a.criterion == 'accuracy' else ('reference_macro_f1', 'macro_f1')
        for region in units.index.unique('region'):
            pr = d[d['region'] == region].groupby('participant').agg(real=(ref_col, 'mean'), aug=(sel_col, 'mean'), n=('n_test', 'first'))
            pooled = lambda c: (pr[c] * pr['n']).sum() / pr['n'].sum()  # noqa: E731
            sel = selection[(selection['menu'] == menu) & (selection['region'] == region)].set_index('participant')['selected']
            print(f'  {region:5s} {ref} {pr["real"].mean():.1f} -> nested {pr["aug"].mean():.1f} % (trials pooled {pooled("real"):.1f} -> {pooled("aug"):.1f}) | '
                  f'{test(units.loc[region])} | selected: ' + ', '.join(f'{p}={c}' for p, c in sel.items()))
        print(f'  both regions ({len(units)} units): {test(units)} | participant means over both regions: {test(units.groupby("participant").mean())}')

    wide = part[a.criterion].unstack('candidate')
    for x, y in COMPARISONS:
        if x in wide and y in wide:
            paired(wide, x, y, f'{x} vs. {y} (outer tasks, fixed)')
    mmd = {}  # protocol -> (set x region) MMD against the test trials, mean over units
    if len(fid):
        fw = fid.groupby(['region', 'participant', 'set']).mean(numeric_only=True).drop(columns=['outer', 'inner', 'seed', 'n'])
        print('\nFidelity against the held-out participant (mean over participants; lower = closer):')
        print(fw.groupby(['region', 'set']).mean().T.round(4).to_string())
        fidelity_tests(fw)
        mmd['LOSO'] = fw
    kf = load_kfold(a.kfold)
    if len(kf):
        kf.to_csv(f'{a.results}/fidelity_kfold.csv', index=False)
        n = kf.groupby(['region', 'fold', 'seed']).ngroups
        print(f'\n[REPORT] k-fold fidelity: {n} tasks' + ('' if n == 2 * 5 * len(SEEDS) else f' (incomplete, expected {2 * 5 * len(SEEDS)})'))
        kw = kf.groupby(['region', 'fold', 'set']).mean(numeric_only=True).drop(columns=['seed', 'n'])
        print('\nk-fold fidelity against the test fold of the training participants (mean over folds; lower = closer):')
        print(kw.groupby(['region', 'set']).mean().T.round(4).to_string())
        mmd['k-fold'] = kw
        tab = {}
        for protocol, w in mmd.items():
            m = w.groupby(['region', 'set']).mean()
            tab[protocol] = pd.concat([m['mmd'].unstack('region'), m['mmd_real_ref'].groupby('region').mean().to_frame('real training').T])
        print('\nMMD against the test trials, k-fold (same participants) vs. LOSO (unseen participant); real training = '
              'the real training trials themselves:')
        print(pd.concat(tab, axis=1).round(4).to_string())
        fidelity_tests(kw, 'k-fold ', pool=None)


if __name__ == '__main__':
    main()
