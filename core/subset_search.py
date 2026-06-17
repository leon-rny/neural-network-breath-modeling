'''Participant-subset search driver (Phase 2).

Enumerates every participant subset of size >= MIN_SIZE and, via the TRTR proxy (train-real-test-real,
no generator -> cheap, isolates *subset learnability*), measures k-fold accuracy per subset. The actual
TRTR runs are launched by experiments/28_subset_search.sh (one array task per subset x region x fold),
which call `core.tstr --model trtr --include_subjects <letters>`; this module only (a) enumerates the
subsets (so the array can index them) and (b) aggregates the resulting per-run trtr json into
results/subset_search.csv, annotated with subset SIZE and SEASON composition for the data-dredging
controls (a "best" subset that is just smaller or season-homogeneous is not a real finding).

Usage:
  python -m core.subset_search enumerate --min_size 5            # line1=count, then one subset/line
  python -m core.subset_search enumerate --min_size 5 --index N  # just the Nth subset (comma-joined)
  python -m core.subset_search aggregate --min_size 5 --init_seed 0 --folds 1,2,3,4,5
'''
import argparse
import csv
import itertools
import json
import os

from core.data import PARTICIPANTS, subset_tag

SUMMER = set(['b', 'c', 'd'])  # summer cohort (higher baseline, weaker temp rise); see loop_journal

def all_subsets(min_size: int, parts=tuple(PARTICIPANTS)) -> list[tuple]:
    out = []
    for k in range(min_size, len(parts) + 1):
        out.extend(itertools.combinations(parts, k))
    return out

def season_comp(sub: tuple) -> str:
    s = sum(1 for p in sub if p in SUMMER)
    return f'{len(sub) - s}w{s}s'  # e.g. 4w1s = 4 winter + 1 summer

def _trtr_json_path(region: str, init_seed: int, fold: int, sub: tuple) -> str:
    # mirrors build_run_id for model='trtr': '{region}_s{seed}' + '_f{fold}' + subset_tag (kfold, no markers)
    return f'results/trtr/{region}_s{init_seed}_f{fold}{subset_tag(sub)}_tstr.json'

def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    e = sp.add_parser('enumerate'); e.add_argument('--min_size', type=int, default=5)
    e.add_argument('--index', type=int, default=None)
    a = sp.add_parser('aggregate'); a.add_argument('--min_size', type=int, default=5)
    a.add_argument('--regions', default='mouth,nose'); a.add_argument('--init_seed', type=int, default=0)
    a.add_argument('--folds', default='1,2,3,4,5'); a.add_argument('--out', default='results/subset_search.csv')
    args = ap.parse_args()

    subs = all_subsets(args.min_size)
    if args.cmd == 'enumerate':
        if args.index is not None:
            print(','.join(subs[args.index]))
        else:
            print(len(subs))
            for c in subs:
                print(','.join(c))
        return

    # aggregate: read each subset's per-(region,fold) trtr json, annotate with size + season
    regions = args.regions.split(','); folds = [int(x) for x in args.folds.split(',')]
    rows, missing = [], 0
    for c in subs:
        for region in regions:
            for fold in folds:
                p = _trtr_json_path(region, args.init_seed, fold, c)
                if not os.path.exists(p):
                    missing += 1
                    continue
                with open(p) as f:
                    d = json.load(f)
                rows.append({'subset': '+'.join(c), 'size': len(c), 'season': season_comp(c),
                             'region': region, 'init_seed': args.init_seed, 'fold': fold,
                             'accuracy': d['metrics']['accuracy'], 'f1_weighted': d['metrics']['f1_weighted']})
    os.makedirs('results', exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['subset', 'size', 'season', 'region', 'init_seed', 'fold', 'accuracy', 'f1_weighted'])
        w.writeheader(); w.writerows(rows)
    print(f'[subset_search] wrote {len(rows)} rows to {args.out} ({missing} missing json)')

if __name__ == '__main__':
    main()
