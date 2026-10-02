"""Fine-tune a trained TPINN with the real-only classifier's selected features as an informed prior.

For one partition: (1) build a differentiable emulator of the partition's top-20 tsfresh features (the
features selected for the real-only classifier, computed on the sensor-observed signal), trained on real
training trials and on generated posterior / prior samples (input = the generator's normalized output);
(2) fine-tune the generator with its ELBO (+ shape loss) plus lambda * class-conditional MMD between the
emulated features of its posterior and prior samples and the real training features of the same class.
The emulator is frozen during fine-tuning; evaluation always uses the exact tsfresh features.

Needs the real-only cache (core.tstr --model trtr) and the TPINN checkpoint (core.train) of the partition.
Usage (from the run directory): python -m core.featprior --region R --cv_mode CV --fold F --seed S
Writes results/tpinn/<run_id>_fp<lam>_checkpoint.pt (core.train format, evaluate with core.tstr
--model tpinn --hp_tag <tpinn hp_tag>_fp<lam>) and a JSON diagnostic.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from core import tstr as T
from core.data import PARTICIPANT_TO_IDX, PhysicsInformedDataset, get_split, load_dataset
from core.utils import seed_everything
from core.augment import denormalize, observe_with, parent_baselines, top_features, training_baselines
from models.vae import elbo_loss


class Emulator(nn.Module):
    def __init__(self, n_out: int, T_len: int = 36):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(2, 32, 3, padding=1), nn.ReLU(), nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
                                  nn.Conv1d(64, 64, 3, padding=1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(64 * T_len + 2 * T_len, 256), nn.ReLU(), nn.Linear(256, n_out))

    def forward(self, x):
        d = torch.cat([x[:, :, 1:] - x[:, :, :-1], torch.zeros_like(x[:, :, :1])], dim=2)  # explicit differences help texture features
        return self.fc(torch.cat([self.conv(x).flatten(1), d.flatten(1)], dim=1))


def mmd2(a, b, scales=(10.0, 40.0, 160.0)):
    def k(x, y):
        d = torch.cdist(x, y) ** 2
        return sum(torch.exp(-d / s) for s in scales)
    return k(a, a).mean() + k(b, b).mean() - 2 * k(a, b).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_run', default='.')  # directory with dataset/ and results/trtr (real-only caches)
    ap.add_argument('--gen_root', default='.')  # directory with results/tpinn (TPINN checkpoints)
    ap.add_argument('--out_root', default='.')
    ap.add_argument('--template', default='{region}_s{seed}_ld16_ed8_tphys_res_f{fold}{loso}_ppstdscale_a0n10_shape3.0')
    ap.add_argument('--region', required=True)
    ap.add_argument('--cv_mode', required=True)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--lam', type=float, default=0.01)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--emu_epochs', type=int, default=150)
    a = ap.parse_args()
    torch.set_num_threads(1)
    seed_everything(a.seed)
    torch.use_deterministic_algorithms(False)
    n_folds = 8 if a.cv_mode == 'loso' else 5
    loso = '_loso' if a.cv_mode == 'loso' else ''
    cache = pd.read_pickle(f'{a.base_run}/results/trtr/{a.region}_is{a.seed}_ss42_fold{a.fold}of{n_folds}{loso}_prepbaseline_checkpoint.pkl')
    df = load_dataset(f'{a.base_run}/dataset')
    df = df[df.region == a.region].reset_index(drop=True)
    tr, va, _ = get_split(df, a.cv_mode, a.fold - 1, n_folds, 42)
    run_id = a.template.format(region=a.region, seed=a.seed, fold=a.fold, loso=('_loso_drop0.1' if a.cv_mode == 'loso' else ''))
    cwd = os.getcwd()
    os.chdir(a.gen_root)
    model, stats = T.load_model('tpinn', run_id, torch.device('cpu'))
    ckpt = torch.load(f'results/tpinn/{run_id}_checkpoint.pt', map_location='cpu', weights_only=False)
    os.chdir(cwd)
    X_real, y_real = cache['X_train_top'], cache['y_train']
    f_mu, f_sd = X_real.mean(0), X_real.std(0) + 1e-9
    R = torch.tensor((X_real - f_mu) / f_sd, dtype=torch.float32)
    yr = torch.tensor(y_real)
    ds = PhysicsInformedDataset(tr, stats=stats)
    x_real = torch.stack([ds[i][0] for i in range(len(ds))])
    y_t = torch.tensor([ds[i][2] for i in range(len(ds))])
    p_t = torch.tensor([PARTICIPANT_TO_IDX[q] for q in tr.participant])
    null = model.null_part_idx if a.cv_mode == 'loso' else None
    pool = torch.tensor(model.trained_participants)
    bl = training_baselines(tr)
    allb = np.concatenate(list(bl.values()))
    rng = np.random.RandomState(a.seed)

    def decode_observe(xn, base):
        sig = observe_with(denormalize(xn.numpy(), stats), base)
        return torch.tensor((top_features(sig, cache) - f_mu) / f_sd, dtype=torch.float32)

    def gen_set(m):
        """Normalized posterior + prior decodes of the current generator with exact observed features."""
        m.eval()
        with torch.no_grad():
            mu, lv = m.encoder(x_real, y_t, p_t)
            zp = mu + (0.5 * lv).exp() * torch.randn_like(mu)
            xp = m._physics(zp, y_t, p_t)[0]
            yq = torch.tensor(rng.randint(0, 3, len(y_t)))
            pq = torch.full_like(yq, null) if null is not None else pool[torch.randint(len(pool), (len(yq),))]
            xq = m._physics(torch.randn_like(mu), yq, pq)[0]
        fp = decode_observe(xp, parent_baselines(tr))
        fq = decode_observe(xq, allb[rng.randint(len(allb), size=len(yq))])
        return xp, fp, y_t, xq, fq, yq

    # 1) emulator of the selected features on observed signals
    xp, fp, yp, xq, fq, yq = gen_set(model)
    Xe = torch.cat([x_real, xp, xq])
    Fe = torch.cat([R, fp, fq])
    perm = torch.randperm(len(Xe))
    n_val = len(Xe) // 5
    iv, it = perm[:n_val], perm[n_val:]
    emu = Emulator(R.shape[1])
    opt = torch.optim.Adam(emu.parameters(), lr=1e-3, weight_decay=1e-5)
    for ep in range(a.emu_epochs):
        emu.train()
        for b in torch.split(it[torch.randperm(len(it))], 64):
            loss = F.mse_loss(emu(Xe[b]), Fe[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    emu.eval()
    with torch.no_grad():
        pred = emu(Xe[iv])
    r2 = (1 - ((pred - Fe[iv]) ** 2).mean(0) / Fe[iv].var(0)).numpy()
    for prm in emu.parameters():
        prm.requires_grad_(False)

    def diag(m):
        """Exact-feature diagnostics of posterior samples: real-classifier label agreement and per-class MMD to real."""
        _, f_post, y_post, _, f_prior, y_prior = gen_set(m)
        out = {}
        for name, Fg, yg in (('posterior', f_post, y_post), ('prior', f_prior, y_prior)):
            out[name + '_mmd'] = float(np.mean([mmd2(Fg[yg == c], R[yr == c]).item() for c in range(3)]))
        return out

    before = diag(model)
    # 2) fine-tune with ELBO + lambda * class-conditional MMD on emulated features
    loader = DataLoader(PhysicsInformedDataset(tr, stats=stats), batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(a.seed))
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    hist = []
    for ep in range(a.epochs):
        model.train()
        tot = {'recon': 0.0, 'feat': 0.0, 'n': 0}
        for signal, _t, label, part, _o in loader:
            x_hat, mu, logvar, *_ = model(signal, label.long(), part.long())
            elbo, recon, kl = elbo_loss(signal, x_hat, mu, logvar, 0.01, 0.0)
            shape = F.mse_loss(x_hat[:, :, 1:] - x_hat[:, :, :-1], signal[:, :, 1:] - signal[:, :, :-1])
            yq = torch.randint(0, 3, (len(label),))
            pq = torch.full_like(yq, null) if null is not None else pool[torch.randint(len(pool), (len(yq),))]
            xq = model._physics(torch.randn(len(yq), model.latent_dim), yq, pq)[0]
            fg, fq = emu(x_hat), emu(xq)
            feat = 0.0
            for c in range(3):
                rc = R[yr == c][torch.randint(int((yr == c).sum()), (64,))]
                if (label == c).sum() > 1:
                    feat = feat + mmd2(fg[label == c], rc)
                if (yq == c).sum() > 1:
                    feat = feat + 0.5 * mmd2(fq[yq == c], rc)
            loss = elbo + 3.0 * shape + a.lam * feat
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot['recon'] += recon.item() * len(label)
            tot['feat'] += float(feat.detach() if torch.is_tensor(feat) else feat) * len(label)
            tot['n'] += len(label)
        sched.step()
        hist.append({'epoch': ep + 1, 'recon': tot['recon'] / tot['n'], 'feat_mmd': tot['feat'] / tot['n']})
    after = diag(model)
    tag = f'_fp{a.lam}'
    os.makedirs(f'{a.out_root}/results/tpinn', exist_ok=True)
    ckpt['model_state'] = model.state_dict()
    ckpt['featprior'] = {'lam': a.lam, 'epochs': a.epochs, 'lr': a.lr, 'emulator_r2': r2.tolist(), 'parent': run_id}
    torch.save(ckpt, f'{a.out_root}/results/tpinn/{run_id}{tag}_checkpoint.pt')
    json.dump({'emulator_r2_mean': float(r2.mean()), 'emulator_r2': r2.tolist(), 'before': before, 'after': after, 'history': hist},
              open(f'{a.out_root}/results/tpinn/{run_id}{tag}_featprior.json', 'w'), indent=1)
    print(f'[FP] {run_id}{tag}: emulator R2 mean {r2.mean():.3f}; posterior MMD {before["posterior_mmd"]:.4f} -> {after["posterior_mmd"]:.4f}; '
          f'prior MMD {before["prior_mmd"]:.4f} -> {after["prior_mmd"]:.4f}', flush=True)


if __name__ == '__main__':
    main()
