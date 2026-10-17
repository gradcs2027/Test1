"""
HA4M: موديلات نوتبوك lstm_attention على نفس تقسيم train_models.py (results/split.json) — 17 مفصل 2D.

    python -I ha4m/train_att.py

  A  LSTM بس (آخر خرج)
  B  LSTM + temporal attention (أهمية كل فريم)
  C  LSTM + spatial attention (أهمية كل مفصل قبل الـ LSTM)
  D  الاتنين مع بعض
زي النوتبوك: LSTM واحدة hidden=64، مدخل كل مفصل = (x, y, سرعة x, سرعة y)، balanced sampling،
weight_decay 1e-4، clip 1.0، اختيار أحسن epoch بأقل val loss. الفرق: مدخل HA4M (17 مفصل 2D من Kinect، 13 فئة).
"""
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_stream import NAMES, N, aug, frame_pred, load_raw, smooth, windows  # noqa: E402

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
HIDDEN, DROPOUT, LR, WD, BATCH, STEPS, PATIENCE = 64, 0.3, 1e-3, 1e-4, 128, 150, 12
CFG = {'A: LSTM': (0, 0), 'B: temporal att': (1, 0), 'C: spatial att': (0, 1), 'D: both': (1, 1)}


def feats(X):
    """X: (B، 30، 34) → (B، 30، 17، 4) = موضع + سرعة."""
    p = X.reshape(len(X), N, 17, 2)
    return torch.cat([p, torch.diff(p, dim=1, prepend=p[:, :1])], -1)


class AttLSTM(nn.Module):
    def __init__(self, temporal, spatial, n_cls=13):
        super().__init__()
        self.t, self.s = temporal, spatial
        if spatial:
            self.joint = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 1))
        self.lstm = nn.LSTM(17 * 4, HIDDEN, batch_first=True)
        if temporal:
            self.time = nn.Sequential(nn.Linear(HIDDEN, 32), nn.Tanh(), nn.Linear(32, 1))
        self.drop = nn.Dropout(DROPOUT)
        self.cls = nn.Linear(HIDDEN, n_cls)

    def forward(self, x):
        B, T, J, _ = x.shape
        if self.s:
            x = x * torch.softmax(self.joint(x).squeeze(-1), -1).unsqueeze(-1) * J
        o = self.lstm(x.reshape(B, T, J * 4))[0]
        h = (o * torch.softmax(self.time(o).squeeze(-1), 1).unsqueeze(-1)).sum(1) if self.t else o[:, -1]
        return self.cls(self.drop(h))


def train(cfg, Xtr, ytr, Xva, yva, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xt = torch.as_tensor(Xtr, dtype=torch.float32, device=DEV)
    vel = feats(Xt)[..., 2:].std().item()
    scale = torch.tensor([1, 1, 1 / vel, 1 / vel], device=DEV)
    prep = lambda X: feats(X) * scale
    Xv = prep(torch.as_tensor(Xva, dtype=torch.float32, device=DEV))
    yt, yv = torch.as_tensor(ytr, device=DEV), torch.as_tensor(yva, device=DEV)
    cnt = np.bincount(ytr, minlength=13).astype(float)
    wts = torch.as_tensor(1.0 / cnt[ytr], dtype=torch.float32)
    m = AttLSTM(*cfg).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
    best, state, bad, bep = 1e9, None, 0, 0
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
            vl = F.cross_entropy(m(Xv), yv).item()
        if vl < best - 1e-4:
            best, bad, bep, state = vl, 0, ep, copy.deepcopy(m.state_dict())
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    m.load_state_dict(state); m.eval()
    return m, prep, bep


def predict(m, prep, X):
    with torch.no_grad():
        return F.softmax(m(prep(torch.as_tensor(X, dtype=torch.float32, device=DEV))), 1).cpu().numpy()


def main():
    raw = load_raw()
    subj = {v: int(v[3:6]) for v in raw}
    split = json.loads((HERE / 'results' / 'split.json').read_text())
    W = {v: windows(*raw[v]) for v in raw}

    def pick(S, pur=0.8):
        L = [(W[v][0][W[v][2] >= pur], W[v][1][W[v][2] >= pur]) for v in raw if subj[v] in S]
        return np.concatenate([a for a, _ in L]), np.concatenate([b for _, b in L])

    Xtr, ytr = pick(split['train']); Xva, yva = pick(split['val'])
    print(f'train {len(Xtr)} / val {len(Xva)} نوافذ | test أشخاص {split["test"]}', flush=True)
    final = {}
    for name, cfg in CFG.items():
        n_par = sum(p.numel() for p in AttLSTM(*cfg).parameters())
        accs, wins, cm = [], [], np.zeros((13, 13), int)
        for seed in range(3):
            t0 = time.time()
            m, prep, bep = train(cfg, Xtr, ytr, Xva, yva, seed)
            fa, wa = [], []
            for v in raw:
                if subj[v] not in split['test']:
                    continue
                Xw, yw, _, cen = W[v]
                P = predict(m, prep, Xw)
                y = raw[v][1]
                p = frame_pred(smooth(P, 5), cen, len(y)).argmax(1)
                fa.append(p == y); wa.append(P.argmax(1) == yw)
                np.add.at(cm, (y, p), 1)
            accs.append(np.concatenate(fa).mean() * 100); wins.append(np.concatenate(wa).mean() * 100)
            print(f'{name} seed {seed}: test فريم (smooth5) {accs[-1]:.1f}% | نوافذ {wins[-1]:.1f}% | best epoch {bep} ({time.time()-t0:.0f}s)', flush=True)
        mf1 = np.mean([2 * cm[c, c] / max(1, cm[c].sum() + cm[:, c].sum()) for c in range(13)]) * 100
        final[name] = (accs, wins, mf1, [100 * cm[c, c] / max(1, cm[c].sum()) for c in range(13)], n_par)
    out = [f'{"الموديل":<18}{"عدد الباراميترات":>16}{"test فريم (smooth5)":>26}{"macro-F1":>10}{"نوافذ":>8}']
    for k, (a, w, f1, _, n) in final.items():
        out.append(f'{k:<18}{n:>16,}{np.mean(a):>9.1f}%  ({", ".join(f"{x:.1f}" for x in a)}){f1:>8.1f}%{np.mean(w):>8.1f}%')
    out += ['', 'recall لكل فئة % (test):', f'{"الفئة":<18}' + ''.join(f'{k[:1]:>8}' for k in final)]
    for c in range(13):
        out.append(f'{NAMES[c]:<18}' + ''.join(f'{final[k][3][c]:>8.0f}' for k in final))
    print('\n'.join(out))
    (HERE / 'results' / 'train_att.txt').write_text('\n'.join(out), encoding='utf-8')


if __name__ == '__main__':
    main()
