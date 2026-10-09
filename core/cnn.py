"""Raw-signal 1D-CNN classifier.

Input: baseline-corrected (first 5 samples) humidity / temperature scaled per channel by the real training SD;
Conv(2-32, k5)-BN-ReLU, Conv(32-64, k5)-BN-ReLU, Conv(64-64, k3)-BN-ReLU, global average + max pooling, dropout 0.3,
linear(128-3); Adam 1e-3, weight decay 1e-4, batch 32, 150 epochs, the epoch with the best accuracy on the validation
participant is kept.
"""

import copy

import numpy as np
import torch
from torch import nn

EPOCHS = 150


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Conv1d(2, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
                               nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
                               nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 3))

    def forward(self, x):
        h = self.f(x)
        return self.head(torch.cat([h.mean(2), h.amax(2)], 1))


def prep(signals: np.ndarray, scale: np.ndarray) -> torch.Tensor:
    """Absolute (n, 2, T) trials -> baseline-corrected channels scaled by the real training SD."""
    s = np.asarray(signals, dtype=float)
    s = s - s[:, :, :5].mean(2, keepdims=True)
    return torch.tensor(s / scale[None, :, None], dtype=torch.float32)


class CNNClassifier:
    """predict / predict_proba on absolute signals."""

    def __init__(self, net: Net, scale: np.ndarray):
        self.net, self.scale = net.eval(), scale

    def predict_proba(self, signals: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return torch.softmax(self.net(prep(signals, self.scale)), 1).numpy()

    def predict(self, signals: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.net(prep(signals, self.scale)).argmax(1).numpy()


def fit(X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, seed: int, scale: np.ndarray, epochs: int = EPOCHS) -> CNNClassifier:
    """Train on (X_tr, y_tr) signals and keep the epoch with the best validation accuracy."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    xt, yt = prep(X_tr, scale), torch.tensor(np.asarray(y_tr), dtype=torch.long)
    xv = prep(X_val, scale)
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    best, state = -1.0, None
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(yt), generator=g)
        for i in range(0, len(yt), 32):
            b = perm[i:i + 32]
            if len(b) < 2:
                continue
            loss = nn.functional.cross_entropy(net(xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            va = float((net(xv).argmax(1).numpy() == y_val).mean())
        if va > best:
            best, state = va, copy.deepcopy(net.state_dict())
    net.load_state_dict(state)
    return CNNClassifier(net, scale)
