"""
HA4M: هل تقليل الهيكل من 32 لـ 17 مفصل بيضر بالتفرقة بين الأفعال الـ 12؟

    python ha4m/compare.py

4 أشكال مدخلات لنفس الموديل (BiLSTM زي SkeletonEncoder بس Dense(12) بدل كود 384) ونفس الإعدادات:
  A  COCO17 2D     (34)   = اللي موديلنا/YOLO بيفهمه
  D  Kinect32 2D   (64)   = نفس 2D + مفاصل اليد/العمود الفقري (بيعزل أثر المفاصل الزيادة)
  C  COCO17 3D     (51)   = نفس مفاصل A + العمق Z (بيعزل أثر العمق)
  B  Kinect32 2D+3D(160)  = كل حاجة
التقسيم بالشخص (train/val/test = أشخاص مختلفين). 3 seeds. اختيار epoch بأقل val loss.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import K2C, OUT  # noqa: E402

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
NAMES = ['1 carrier', '2 gear bearings', '3 planet gears', '4 carrier shaft', '5 sun shaft', '6 sun gear',
         '7 sun gear bearing', '8 ring gear', '9 block2→block1', '10 cover', '11 screws', '12 allen key']


class Clf(nn.Module):
    def __init__(self, in_dim, n=12):
        super().__init__()
        self.l1 = nn.LSTM(in_dim, 256, batch_first=True, bidirectional=True)
        self.l2 = nn.LSTM(512, 128, batch_first=True, bidirectional=True)
        self.l3 = nn.LSTM(256, 128, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.4), nn.Linear(256, n))

    def forward(self, x):
        x = self.drop(self.l1(x)[0]); x = self.drop(self.l2(x)[0])
        _, (h, _) = self.l3(x)
        return self.head(self.drop(torch.cat([h[-2], h[-1]], 1)))


def feats(seg):
    N = len(seg)
    return {
        'A  COCO17 2D': seg[:, :, K2C, :2].reshape(N, 30, -1),
        'D  Kinect32 2D': seg[:, :, :, :2].reshape(N, 30, -1),
        'C  COCO17 3D': seg[:, :, K2C, 2:].reshape(N, 30, -1),
        'B  Kinect32 2D+3D': seg.reshape(N, 30, -1),
    }


def run(Xtr, ytr, Xva, yva, Xte, yte, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    mu, sd = Xtr.reshape(-1, Xtr.shape[-1]).mean(0), Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6
    T = lambda X: torch.as_tensor((X - mu) / sd, dtype=torch.float32, device=DEV)
    Xtr_, Xva_, Xte_ = T(Xtr), T(Xva), T(Xte)
    ytr_, yva_ = torch.as_tensor(ytr, device=DEV), torch.as_tensor(yva, device=DEV)
    m = Clf(Xtr.shape[-1]).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    best, state, bad = 1e9, None, 0
    for ep in range(120):
        m.train()
        perm = torch.randperm(len(Xtr_), device=DEV)
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            if len(b) < 2: continue
            loss = F.cross_entropy(m(Xtr_[b]), ytr_[b])
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl = F.cross_entropy(m(Xva_), yva_).item()
        sched.step(vl)
        if vl < best - 1e-4:
            best, bad, state = vl, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 12: break
    m.load_state_dict(state); m.eval()
    with torch.no_grad():
        return m(Xte_).argmax(1).cpu().numpy()


def macro_f1(y, p, n=12):
    f = []
    for c in range(n):
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum()
        f.append(2 * tp / max(1, 2 * tp + fp + fn))
    return float(np.mean(f))


def main():
    d = np.load(OUT / 'segments.npz')
    seg, y, subj = d['seg_full'], d['y'] - 1, d['subject']
    subs = np.array(sorted(set(subj)))
    rng = np.random.RandomState(0); rng.shuffle(subs)
    nte, nva = max(1, round(len(subs) * 0.2)), max(1, round(len(subs) * 0.1))
    te, va, tr = subs[:nte], subs[nte:nte + nva], subs[nte + nva:]
    mk = lambda S: np.isin(subj, S)
    out = [f'HA4M | segments {len(y)} | أشخاص train {len(tr)} / val {len(va)} / test {len(te)} | عينات train {mk(tr).sum()} / val {mk(va).sum()} / test {mk(te).sum()}',
           f'صدفة {100/12:.1f}% | أكبر فئة (test) {100*np.bincount(y[mk(te)], minlength=12).max()/mk(te).sum():.1f}%', '']
    print(out[0]); print(out[1])
    conf_all = {}
    for name, X in feats(seg).items():
        accs, f1s, recs = [], [], []
        cm = np.zeros((12, 12), int)
        t0 = time.time()
        for seed in range(3):
            p = run(X[mk(tr)], y[mk(tr)], X[mk(va)], y[mk(va)], X[mk(te)], y[mk(te)], seed)
            yt = y[mk(te)]
            accs.append((p == yt).mean() * 100); f1s.append(macro_f1(yt, p) * 100)
            recs.append([(p[yt == c] == c).mean() * 100 if (yt == c).any() else np.nan for c in range(12)])
            np.add.at(cm, (yt, p), 1)
        line = f'{name:<20} دقة test {np.mean(accs):5.1f}% ({", ".join(f"{a:.1f}" for a in accs)})  macro-F1 {np.mean(f1s):5.1f}%  ({time.time()-t0:.0f}s)'
        print(line); out.append(line)
        conf_all[name] = (np.nanmean(recs, 0), cm)
    out += ['', 'دقة كل فعل (recall %، متوسط 3 seeds):', f'{"الفعل":<22}' + ''.join(f'{k[:1]:>8}' for k in conf_all)]
    for c in range(12):
        out.append(f'{NAMES[c]:<22}' + ''.join(f'{v[0][c]:>8.0f}' for v in conf_all.values()))
    for name, (_, cm) in conf_all.items():
        pairs = sorted(((cm[i, j], i, j) for i in range(12) for j in range(12) if i != j), reverse=True)[:5]
        out.append(f'\nأكتر لخبطة في {name.strip()} (الحقيقي ← المتوقع، عدد): ' +
                   ' | '.join(f'{NAMES[i][:12]}←{NAMES[j][:12]} {n}' for n, i, j in pairs))
    print('\n'.join(out[3:]))
    (Path(__file__).parent / 'results' / 'compare.txt').write_text('\n'.join(out), encoding='utf-8')


if __name__ == '__main__':
    main()
