"""
HA4M: تدريب BiLSTM على نوافذ 3 ثواني من فيديو متصل (17 مفصل 2D) + تقييم لكل فريم على فيديوهات أشخاص مخبّيين.

    python -I ha4m/train_stream.py

  - 13 فئة: الأفعال 1..12 + "none" (فعل 0 = مفيش فعل / مش مهم)
  - نافذة 90 فريم (3 ثواني @30fps) خطوة 15، تتعمل resample لـ 30 فريم. اللابل = أغلب لابل في النافذة.
  - تدريب: نوافذ نقاوتها >= 0.8. تقييم: كل النوافذ، وكل فريم بياخد توقّع أقرب نافذة.
  - التقسيم بالشخص. الشخص 1 دايماً في الـ test عشان نقارن Kinect بـ YOLO على V001.
  - التطبيع على COCO17 (منتصف الحوضين / منتصف الكتفين) عشان Kinect و YOLO يتطبّعوا بنفس الطريقة.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare import Clf  # noqa: E402
from prepare import K2C, OUT, SRC, labels, read_video  # noqa: E402

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
WIN, STEP, N, NONE = 90, 15, 30, 12
NAMES = ['carrier', 'gear bearings', 'planet gears', 'carrier shaft', 'sun shaft', 'sun gear',
         'sun gear bearing', 'ring gear', 'block2→block1', 'cover', 'screws', 'allen key', 'none']
YOLO_NPZ = SRC.parent / 'ha4m_kp' / 'IDU001V001_kp2.npz'


def norm17(w):
    """w: (T، 17، 2) خام → (30، 34) متوسّط على منتصف الحوضين ومقسوم على طول الجذع."""
    hip = (w[:, 11] + w[:, 12]) / 2
    sho = (w[:, 5] + w[:, 6]) / 2
    t = np.median(np.linalg.norm(sho - hip, axis=-1)) + 1e-6
    w = (w - hip[:, None]) / t
    return w[np.linspace(0, len(w) - 1, N).astype(int)].reshape(N, 34).astype(np.float32)


def load_raw():
    cache = OUT / 'raw17.npz'
    if cache.exists():
        d = np.load(cache)
        return {k[:-2]: (d[k[:-2] + '_x'], d[k[:-2] + '_y']) for k in d.files if k.endswith('_x')}
    raw = {}
    for vdir in sorted(p for p in SRC.iterdir() if (p / 'Skeletons.zip').exists() and (p / 'Labels.txt').exists()):
        fr, lab = read_video(vdir), labels(vdir)
        ids = sorted(fr)
        if len(ids) < WIN:
            continue
        x = np.stack([fr[i][K2C][:, 4:6] for i in ids])
        y = np.array([lab[i][0] if i in lab else 0 for i in ids])
        raw[vdir.name] = (x, np.where(y == 0, NONE, y - 1))
        print(vdir.name, len(ids), flush=True)
    np.savez_compressed(cache, **{f'{k}_x': v[0] for k, v in raw.items()}, **{f'{k}_y': v[1] for k, v in raw.items()})
    return raw


def windows(x, y, win=WIN, step=STEP):
    starts = np.arange(0, len(x) - win + 1, step)
    X = np.stack([norm17(x[s:s + win]) for s in starts])
    lab, pur = [], []
    for s in starts:
        c = np.bincount(y[s:s + win], minlength=13)
        lab.append(c.argmax()); pur.append(c.max() / win)
    return X, np.array(lab), np.array(pur), starts + win // 2


def frame_pred(P, centers, T):
    """P: (K، 13) احتمالات النوافذ → توقّع لكل فريم (أقرب نافذة)."""
    return P[np.abs(np.arange(T)[:, None] - centers[None]).argmin(1)]


def aug(x):
    B, T = x.shape[:2]          # T = 30 هنا، و60 في النافذة الطويلة (train_two.py)
    s = x.view(B, T, 17, 2)
    th = torch.empty(B, 1, 1, device=x.device).uniform_(-0.15, 0.15)
    c, sn = th.cos(), th.sin()
    s = torch.stack([s[..., 0] * c - s[..., 1] * sn, s[..., 0] * sn + s[..., 1] * c], -1)
    s = s * torch.empty(B, 1, 1, 1, device=x.device).uniform_(0.9, 1.1) + torch.randn_like(s) * 0.02
    return s.reshape(B, T, 34)


def train(Xtr, ytr, Xva, yva, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    mu, sd = Xtr.reshape(-1, 34).mean(0), Xtr.reshape(-1, 34).std(0) + 1e-6
    T = lambda X: torch.as_tensor(X, dtype=torch.float32, device=DEV)
    Xt, Xv = T(Xtr), T((Xva - mu) / sd)
    yt, yv = torch.as_tensor(ytr, device=DEV), torch.as_tensor(yva, device=DEV)
    mu_t, sd_t = T(mu), T(sd)
    m = Clf(34, 13).to(DEV)
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
    return m, mu, sd


def predict(m, mu, sd, X):
    with torch.no_grad():
        return F.softmax(m(torch.as_tensor((X - mu) / sd, dtype=torch.float32, device=DEV)), 1).cpu().numpy()


def smooth(P, k):
    pad = np.pad(P, ((k // 2, k // 2), (0, 0)), mode='edge')
    return np.stack([pad[i:i + k].mean(0) for i in range(len(P))])


def main():
    raw = load_raw()
    subj = {v: int(v[3:6]) for v in raw}
    subs = sorted(set(subj.values()))
    rng = np.random.RandomState(0)
    others = [s for s in subs if s != 1]
    rng.shuffle(others)
    te, va, tr = [1] + others[:7], others[7:11], others[11:]
    W = {v: windows(*raw[v]) for v in raw}

    def pick(S, pur=0.8):
        L = [(W[v][0][W[v][2] >= pur], W[v][1][W[v][2] >= pur]) for v in raw if subj[v] in S]
        return np.concatenate([a for a, _ in L]), np.concatenate([b for _, b in L])

    Xtr, ytr = pick(tr)
    Xva, yva = pick(va)
    out = [f'HA4M متصل | أشخاص train {len(tr)} / val {len(va)} / test {len(te)} (الشخص 1 + 7) | نوافذ train {len(Xtr)} / val {len(Xva)}',
           'فئات train: ' + ' '.join(f'{NAMES[i][:6]}={n}' for i, n in enumerate(np.bincount(ytr, minlength=13))), '']
    print('\n'.join(out), flush=True)

    yolo = None
    if YOLO_NPZ.exists():
        d = np.load(YOLO_NPZ)
        yolo = (d['new'][:, :, :2], np.array([NONE if a <= 0 else a - 1 for a in d['act']]))
    yres = []
    cm = np.zeros((13, 13), int)
    res = {'raw': [], 's5': [], 'win': [], 'act': []}
    for seed in range(3):
        t0 = time.time()
        m, mu, sd = train(Xtr, ytr, Xva, yva, seed)
        fa, fs, wa, ao = [], [], [], []
        for v in raw:
            if subj[v] not in te:
                continue
            Xw, yw, _, cen = W[v]
            P = predict(m, mu, sd, Xw)
            y = raw[v][1]
            p0 = frame_pred(P, cen, len(y)).argmax(1)
            p5 = frame_pred(smooth(P, 5), cen, len(y)).argmax(1)
            fa.append(p0 == y); fs.append(p5 == y); wa.append(P.argmax(1) == yw); ao.append((p0 == y)[y != NONE])
            np.add.at(cm, (y, p0), 1)
            if v == 'IDU001V001' and yolo is not None:
                n = len(yolo[1])
                for name, src in (('Kinect', raw[v][0][:2 * n:2]), ('YOLO', yolo[0])):
                    st = np.arange(0, n - 45 + 1, 8)
                    Ps = predict(m, mu, sd, np.stack([norm17(src[s:s + 45]) for s in st]))
                    pf = frame_pred(Ps, st + 22, n).argmax(1)
                    yres.append((name, (pf == yolo[1]).mean() * 100, (pf == yolo[1])[yolo[1] != NONE].mean() * 100))
        fa, fs, wa, ao = map(np.concatenate, (fa, fs, wa, ao))
        for k, a in zip(('raw', 's5', 'win', 'act'), (fa, fs, wa, ao)):
            res[k].append(a.mean() * 100)
        print(f'seed {seed}: فريم {fa.mean()*100:.1f}% | +smooth5 {fs.mean()*100:.1f}% | نوافذ {wa.mean()*100:.1f}% | '
              f'فريمات الأفعال بس {ao.mean()*100:.1f}% ({time.time()-t0:.0f}s)', flush=True)
    mean = lambda k: f'{np.mean(res[k]):.1f}% ({", ".join(f"{a:.1f}" for a in res[k])})'
    tail = [f'دقة لكل فريم (test): {mean("raw")}', f'  + smoothing 5 نوافذ: {mean("s5")}',
            f'  فريمات فيها فعل بس (من غير none): {mean("act")}', f'دقة النوافذ: {mean("win")}',
            f'نسبة فريمات none في الـ test: {100*cm[NONE].sum()/cm.sum():.1f}% | أكبر فئة: {100*cm.sum(1).max()/cm.sum():.1f}%', '']
    rec = [100 * cm[c, c] / max(1, cm[c].sum()) for c in range(13)]
    tail.append('recall لكل فئة (فريمات): ' + ' | '.join(f'{NAMES[c][:10]} {rec[c]:.0f}' for c in range(13)))
    pairs = sorted(((cm[i, j], i, j) for i in range(13) for j in range(13) if i != j), reverse=True)[:6]
    tail.append('أكتر لخبطة (حقيقي←متوقع): ' + ' | '.join(f'{NAMES[i]}←{NAMES[j]} {n}' for n, i, j in pairs))
    if yres:
        tail += ['', 'V001: نفس الفريمات، Kinect مقابل YOLO (الموديل اتدرّب على Kinect بس)']
        for name in ('Kinect', 'YOLO'):
            r = [x for x in yres if x[0] == name]
            tail.append(f'  {name:<7} كل الفريمات {np.mean([x[1] for x in r]):.1f}% | فريمات الأفعال {np.mean([x[2] for x in r]):.1f}%')
    print('\n'.join(tail))
    (Path(__file__).parent / 'results' / 'train_stream.txt').write_text('\n'.join(out + tail), encoding='utf-8')


if __name__ == '__main__':
    main()
