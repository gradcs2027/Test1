"""
HA4M: تدريب موديلاتنا التلاتة على الـ 217 فيديو بتقسيم ثابت train / val / test (بالشخص) ومقارنتهم على فيديوهات متصلة.

    python -I ha4m/train_models.py

الموديلات (نفس مدخل 17 مفصل 2D من train_stream.py):
  lstm  BiLSTM 256→128→128 + آخر حالة + Dense(13)         (زي compare.py)
  attn  نفس الـ BiLSTM بس بنجمع كل الفريمات بـ attention بدل آخر حالة
  text  SkeletonEncoder بتاع zero-shot (كود 384 + cosine على أكواد MiniLM لوصف كل فعل)
التقسيم: بالشخص (مفيش شخص في أكتر من مجموعة). بيتحفظ في results/split.json. اختيار الـ epoch = أقل val loss.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'zeroshot'))
from train_stream import (NAMES, NONE, N, WIN, aug, frame_pred, load_raw, smooth, windows)  # noqa: E402
from prepare import OUT  # noqa: E402
from zsl_model import SkeletonEncoder  # noqa: E402

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DESC = ['a person picks up the carrier and places it on the table',
        'a person picks up three gear bearings and places them',
        'a person picks up three planet gears and places them on the carrier',
        'a person inserts the carrier shaft',
        'a person inserts the sun shaft',
        'a person places the sun gear',
        'a person places the sun gear bearing',
        'a person places the ring gear',
        'a person picks up block 2 and places it onto block 1',
        'a person places the cover on the assembly',
        'a person picks up two screws and inserts them',
        'a person turns the screws with an allen key',
        'a person is standing idle and not doing any task']


class Attn(nn.Module):
    def __init__(self, in_dim=34, n=13):
        super().__init__()
        self.l1 = nn.LSTM(in_dim, 256, batch_first=True, bidirectional=True)
        self.l2 = nn.LSTM(512, 128, batch_first=True, bidirectional=True)
        self.l3 = nn.LSTM(256, 128, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.3)
        self.att = nn.Linear(256, 1)
        self.head = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.4), nn.Linear(256, n))

    def forward(self, x):
        x = self.drop(self.l1(x)[0]); x = self.drop(self.l2(x)[0])
        o = self.l3(x)[0]
        w = torch.softmax(self.att(o), 1)
        return self.head(self.drop((w * o).sum(1)))


class Last(nn.Module):
    def __init__(self, in_dim=34, n=13):
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


class Text(nn.Module):
    def __init__(self, txt):
        super().__init__()
        self.enc = SkeletonEncoder()
        self.register_buffer('txt', txt)

    def forward(self, x):
        return self.enc.logits(x, self.txt)


def text_codes():
    """أكواد MiniLM لوصف الأفعال — بتتحسب مرة بـ ha4m/text_codes.py (من غير -I) وبتتخزن."""
    return torch.as_tensor(np.load(OUT / 'text_codes.npy'), dtype=torch.float32)


def make(kind, txt):
    return {'lstm': Last, 'attn': Attn}[kind]().to(DEV) if kind != 'text' else Text(txt).to(DEV)


def train(kind, txt, Xtr, ytr, Xva, yva, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    mu, sd = Xtr.reshape(-1, 34).mean(0), Xtr.reshape(-1, 34).std(0) + 1e-6
    T = lambda X: torch.as_tensor(X, dtype=torch.float32, device=DEV)
    Xt, Xv = T(Xtr), T((Xva - mu) / sd)
    yt, yv = torch.as_tensor(ytr, device=DEV), torch.as_tensor(yva, device=DEV)
    mu_t, sd_t = T(mu), T(sd)
    m = make(kind, txt)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    best, state, bad = 1e9, None, 0
    for ep in range(60):
        m.train()
        perm = torch.randperm(len(Xt), device=DEV)
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]
            if len(b) < 2:
                continue
            loss = F.cross_entropy(m((aug(Xt[b]) - mu_t) / sd_t), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl = F.cross_entropy(m(Xv), yv).item()
        sch.step(vl)
        if vl < best - 1e-4:
            best, bad, state = vl, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 8:
                break
    m.load_state_dict(state); m.eval()
    return m, mu, sd, ep + 1


def predict(m, mu, sd, X):
    with torch.no_grad():
        return F.softmax(m(torch.as_tensor((X - mu) / sd, dtype=torch.float32, device=DEV)), 1).cpu().numpy()


def main():
    raw = load_raw()
    subj = {v: int(v[3:6]) for v in raw}
    subs = sorted(set(subj.values()))
    rng = np.random.RandomState(0)
    others = [s for s in subs if s != 1]
    rng.shuffle(others)
    split = {'test': sorted([1] + others[:7]), 'val': sorted(others[7:11]), 'train': sorted(others[11:])}
    (HERE / 'results' / 'split.json').write_text(json.dumps(
        {**split, 'videos': {k: sorted(v for v in raw if subj[v] in s) for k, s in split.items()}}, indent=1))
    W = {v: windows(*raw[v]) for v in raw}

    def pick(S, pur=0.8):
        L = [(W[v][0][W[v][2] >= pur], W[v][1][W[v][2] >= pur]) for v in raw if subj[v] in S]
        return np.concatenate([a for a, _ in L]), np.concatenate([b for _, b in L])

    Xtr, ytr = pick(split['train']); Xva, yva = pick(split['val'])
    nf = {k: sum(len(raw[v][1]) for v in raw if subj[v] in s) for k, s in split.items()}
    out = [f'HA4M | 217 فيديو، 41 شخص | train {len(split["train"])} شخص / val {len(split["val"])} / test {len(split["test"])} | '
           f'فريمات train {nf["train"]} / val {nf["val"]} / test {nf["test"]} | نوافذ train {len(Xtr)} / val {len(Xva)}', '']
    print(out[0], flush=True)
    txt = text_codes().to(DEV)
    final = {}
    for kind in ('lstm', 'attn', 'text'):
        accs, f1s, wins, cms = [], [], [], np.zeros((13, 13), int)
        for seed in range(3):
            t0 = time.time()
            m, mu, sd, ep = train(kind, txt, Xtr, ytr, Xva, yva, seed)
            fa, wa = [], []
            for split_name in ('val', 'test'):
                for v in raw:
                    if subj[v] not in split[split_name]:
                        continue
                    Xw, yw, _, cen = W[v]
                    P = predict(m, mu, sd, Xw)
                    y = raw[v][1]
                    p = frame_pred(smooth(P, 5), cen, len(y)).argmax(1)
                    if split_name == 'test':
                        fa.append(p == y); wa.append(P.argmax(1) == yw)
                        np.add.at(cms, (y, p), 1)
            fa, wa = np.concatenate(fa), np.concatenate(wa)
            accs.append(fa.mean() * 100); wins.append(wa.mean() * 100)
            print(f'{kind} seed {seed}: test فريم (smooth5) {accs[-1]:.1f}% | نوافذ {wins[-1]:.1f}% | {ep} epoch ({time.time()-t0:.0f}s)', flush=True)
        rec = [100 * cms[c, c] / max(1, cms[c].sum()) for c in range(13)]
        mf1 = np.mean([2 * cms[c, c] / max(1, cms[c].sum() + cms[:, c].sum()) for c in range(13)]) * 100
        final[kind] = (accs, wins, mf1, rec, cms)
    out.append(f'{"الموديل":<8}{"test فريم (smooth5)":>24}{"macro-F1":>10}{"نوافذ":>8}')
    for k, (a, w, f1, rec, _) in final.items():
        out.append(f'{k:<8}{np.mean(a):>8.1f}%  ({", ".join(f"{x:.1f}" for x in a)}){f1:>8.1f}%{np.mean(w):>8.1f}%')
    out += ['', 'recall لكل فئة % (test، متوسط 3 seeds مجمّع):', f'{"الفئة":<18}' + ''.join(f'{k:>8}' for k in final)]
    for c in range(13):
        out.append(f'{NAMES[c]:<18}' + ''.join(f'{final[k][3][c]:>8.0f}' for k in final))
    for k, (_, _, _, _, cm) in final.items():
        pairs = sorted(((cm[i, j], i, j) for i in range(13) for j in range(13) if i != j), reverse=True)[:4]
        out.append(f'لخبطة {k} (حقيقي←متوقع): ' + ' | '.join(f'{NAMES[i]}←{NAMES[j]} {n}' for n, i, j in pairs))
    print('\n'.join(out[2:]))
    (HERE / 'results' / 'train_models.txt').write_text('\n'.join(out), encoding='utf-8')
    np.savez(HERE / 'results' / 'confusion.npz', **{k: v[4] for k, v in final.items()})


if __name__ == '__main__':
    main()
