"""TRTR-level ablation for the SSL idea (research #4): does appending the pretrained contrastive
embeddings to the top-20 tsfresh features improve the downstream classifier? Runs trtr with and
without --ssl_ckpt over seeds x folds for one region+cv and reports the paired delta.

    python -m core.ssl_eval --ssl_ckpt results/ssl/ssl_encoder.pt --region nose --cv_mode kfold
"""
import argparse
import json
import numpy as np
from scipy.stats import wilcoxon
from core.data import load_dataset, n_loso_folds
from core.tstr import trtr


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ssl_ckpt', required=True)
    p.add_argument('--region', required=True, choices=['mouth', 'nose'])
    p.add_argument('--cv_mode', default='kfold', choices=['kfold', 'loso'])
    p.add_argument('--seeds', default='0,1,7,42,123')
    p.add_argument('--split_seed', type=int, default=42)
    p.add_argument('--n_jobs', type=int, default=4)
    p.add_argument('--out', default='')
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    nf = n_loso_folds(load_dataset('dataset')) if args.cv_mode == 'loso' else 5
    base, ssl = [], []
    for seed in seeds:
        for fold in range(1, nf + 1):
            kw = dict(dataset_dir='dataset', region=args.region, n_jobs=args.n_jobs, init_seed=seed,
                      split_seed=args.split_seed, fold=fold, n_folds=nf, cv_mode=args.cv_mode, preprocessing='baseline')
            b = trtr(**kw)['trtr_metrics']['accuracy']
            s = trtr(**kw, ssl_ckpt=args.ssl_ckpt)['trtr_metrics']['accuracy']
            base.append(b); ssl.append(s)
            print(f'{args.region} {args.cv_mode} seed{seed} fold{fold}: base={b:.3f} ssl={s:.3f} d{s - b:+.3f}', flush=True)
    base, ssl = np.array(base), np.array(ssl)
    pv = wilcoxon(ssl, base).pvalue if np.any(ssl != base) else float('nan')
    print(f'\n=== {args.region} {args.cv_mode}: TRTR base={base.mean():.3f} ssl={ssl.mean():.3f} '
          f'd{ssl.mean() - base.mean():+.3f} p={pv:.4f} (n={len(base)}) ===', flush=True)
    if args.out:
        json.dump({'region': args.region, 'cv_mode': args.cv_mode, 'base': base.tolist(), 'ssl': ssl.tolist(),
                   'base_mean': float(base.mean()), 'ssl_mean': float(ssl.mean()), 'p': float(pv)}, open(args.out, 'w'))


if __name__ == '__main__':
    main()
