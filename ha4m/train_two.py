"""
HA4M: two classifiers بـ 3 طبقات LSTM — واحد بيشوف أقل من 5 ثواني وواحد بيشوف أكتر من 5 ثواني حوالين نفس الفريم.
(بيتنادى من run_all2.py)

  S    نافذة 90 فريم (3 ث) → 30 خطوة. اللابل = أغلب لابل في النافذة (زي باقي الموديلات).
  L    نافذة 240 فريم (8 ث) → 60 خطوة. اللابل = أغلب لابل في الـ 90 فريم اللي في النص،
       يعني نفس هدف S بالظبط بس بسياق أطول (الأفعال الطويلة زي screws وallen key متوسطها 10–13 ث).
  S+L  متوسط موزون لاحتمالات الاتنين لكل فريم. الوزن بيتختار على الـ validation بس (مش الـ test).
الاتنين: 3 طبقات LSTM (hidden 64، dropout 0.3)، مدخل كل مفصل = موضع + سرعة، وباقي التدريب زي train_att.py.
"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_att import BATCH, DROPOUT, HIDDEN, LR, PATIENCE, STEPS, WD
from train_stream import STEP, WIN

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
LAYERS = 3
LONG_WIN, LONG_N, SHORT_N = 240, 60, 30
NOISE_STEP = {SHORT_N: 1.5, LONG_N: 2.0}      # خطوة النافذة المتطبّعة بوحدة خطوة YOLO (فريم كل 2)


def normn(w, n):
    """زي norm17 بس بعدد خطوات n: (T، 17، 2) → (n، 34)."""
    hip = (w[:, 11] + w[:, 12]) / 2
    sho = (w[:, 5] + w[:, 6]) / 2
    t = np.median(np.linalg.norm(sho - hip, axis=-1)) + 1e-6
    w = (w - hip[:, None]) / t
    return w[np.linspace(0, len(w) - 1, n).astype(int)].reshape(n, 34).astype(np.float32)


def windows_long(x, y, step=STEP, win=LONG_WIN, n=LONG_N):
    """win، n: طول النافذة بالفريم وعدد الخطوات (الافتراضي 8 ث → 60؛ run_mp بيستخدم كمان 12 ث = 360 → 90)."""
    starts = np.arange(0, len(x) - win + 1, step)
    X = np.stack([normn(x[s:s + win], n) for s in starts])
    c0 = (win - WIN) // 2
    lab, pur = [], []
    for s in starts:
        c = np.bincount(y[s + c0:s + c0 + WIN], minlength=13)
        lab.append(c.argmax()); pur.append(c.max() / WIN)
    return X, np.array(lab), np.array(pur), starts + win // 2


def feats(X):
    """X: (B، T، 34) → (B، T، 17، 4) = موضع + سرعة."""
    p = X.reshape(len(X), X.shape[1], 17, 2)
    return torch.cat([p, torch.diff(p, dim=1, prepend=p[:, :1])], -1)


class LSTM3(nn.Module):
    def __init__(self, n_cls=13):
        super().__init__()
        self.lstm = nn.LSTM(17 * 4, HIDDEN, num_layers=LAYERS, dropout=DROPOUT, batch_first=True)
        self.drop = nn.Dropout(DROPOUT)
        self.cls = nn.Linear(HIDDEN, n_cls)

    def forward(self, x):
        B, T = x.shape[:2]
        return self.cls(self.drop(self.lstm(x.reshape(B, T, -1))[0][:, -1]))


def train(Xtr, ytr, Xva, yva, seed, aug):
    torch.manual_seed(seed); np.random.seed(seed)
    Xt = torch.as_tensor(Xtr, dtype=torch.float32, device=DEV)
    vel = feats(Xt)[..., 2:].std().item()
    scale = torch.tensor([1, 1, 1 / vel, 1 / vel], device=DEV)
    prep = lambda X: feats(X) * scale
    Xv = prep(torch.as_tensor(Xva, dtype=torch.float32, device=DEV))
    yt, yv = torch.as_tensor(ytr, device=DEV), torch.as_tensor(yva, device=DEV)
    cnt = np.bincount(ytr, minlength=13).astype(float)
    wts = torch.as_tensor(1.0 / cnt[ytr], dtype=torch.float32)
    m = LSTM3().to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
    best, state, bad = 1e9, None, 0
    for ep in range(1, 61):
        m.train()
        idx = torch.multinomial(wts, STEPS * BATCH, replacement=True).to(DEV)
        for i in range(0, len(idx), BATCH):
            b = idx[i:i + BATCH]
            loss = F.cross_entropy(m(prep(aug(Xt[b]))), yt[b])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        m.eval()
        with torch.no_grad():
            vl = sum(F.cross_entropy(m(Xv[i:i + 4096]), yv[i:i + 4096], reduction='sum').item()
                     for i in range(0, len(Xv), 4096)) / len(Xv)
        if vl < best - 1e-4:
            best, bad, state = vl, 0, copy.deepcopy(m.state_dict())
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    m.load_state_dict(state); m.eval()
    return m, prep


def predict(m, prep, X):
    with torch.no_grad():
        return np.concatenate([F.softmax(m(prep(torch.as_tensor(X[i:i + 4096], dtype=torch.float32, device=DEV))), 1).cpu().numpy()
                               for i in range(0, len(X), 4096)]) if len(X) else np.zeros((0, 13), np.float32)


def best_weight(Ps, Pl, ys):
    """Ps، Pl: لستة احتمالات لكل فريم لفيديوهات الـ validation → أحسن وزن لـ S (والباقي لـ L)."""
    grid = np.round(np.arange(0, 1.01, 0.1), 1)
    acc = [np.mean(np.concatenate([(w * a + (1 - w) * b).argmax(1) == y for a, b, y in zip(Ps, Pl, ys)])) for w in grid]
    return float(grid[int(np.argmax(acc))])
