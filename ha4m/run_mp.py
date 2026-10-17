"""
HA4M بـ MediaPipe بس (الحل 2): فيديو واحد لكل شخص، ColorAligned، 5 فريم/ث (fetch_mp.py).

    python -I ha4m/run_mp.py mp full|upper     # (أ) كل الموديلات متدرّبة ومتختبرة على MediaPipe بس
    python -I ha4m/run_mp.py kin full|upper    # (ب) متدرّب على Kinect + ضجيج MediaPipe، و(ج) نفس الموديل + fine-tune على MediaPipe
    python -I ha4m/run_mp.py dtw full|upper    # FastDTW بقوالب MediaPipe
    python -I ha4m/run_mp.py merge             # → results/run_mp.txt

التقسيم: نفس results/split.json (29 شخص train، 4 val، 8 test)، فيديو واحد لكل شخص.
النوافذ: 3 ث = 15 فريم و8 ث = 40 فريم، بيتعملهم interpolation خطي لـ 30 و60 خطوة (نفس مدخل الموديلات القديمة).
  train: خطوة فريم (0.2 ث)، نقاوة >= 0.8. val/test: خطوة فريمين (0.4 ث)، smooth 5، كل فريم بياخد أقرب نافذة.
الدقة = لكل فريم (5 فريم/ث) على فيديوهات أشخاص الـ test الـ 8. الاختبار دايماً على MediaPipe.
(ب) الضجيج بيتقاس على فيديوهات الـ val (Kinect مقابل MediaPipe لنفس الفريمات، بعد التطبيع)، مش من الـ test.
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
import train_att  # noqa: E402
import train_models as tm  # noqa: E402
import train_two  # noqa: E402
from prepare import OUT  # noqa: E402
from run_all import MODELS as OLD, RES, mask_legs  # noqa: E402
from train_att import BATCH, CFG, PATIENCE, STEPS, WD  # noqa: E402
from train_noise import make_noisy_aug, measure  # noqa: E402
from train_stream import NAMES, NONE, aug as base_aug, frame_pred, load_raw, smooth, windows  # noqa: E402

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

DEV = train_att.DEV
SEEDS = range(3)
KP5 = OUT.parent / 'ha4m_kp' / 'mp5'
SHORT, LONG, CENTER = 15, 40, 15            # فريمات @5fps: 3 ث، 8 ث، والـ 3 ث اللي في نص الطويلة
TR_STEP, EV_STEP = 1, 2
TWO = ['S: short 3s', 'L: long 8s', 'S+L']
MODELS = OLD + TWO
KIN_MODELS = ['A: LSTM'] + TWO
MP_DT = 0.2                                 # ثانية بين فريمين MediaPipe
NOISE_STEP = {30: 3 / 30 / MP_DT, 60: 8 / 60 / MP_DT}     # خطوة النافذة المتطبّعة بوحدة فريم MediaPipe


def gt_of(act):
    return np.array([NONE if a <= 0 else a - 1 for a in act])


def fill(kp):
    """فريم مفيهوش شخص (كل الثقة صفر) → ياخد أقرب فريم فيه شخص."""
    ok = kp[:, :, 2].max(1) > 0
    if ok.all() or not ok.any():
        return kp
    idx = np.where(ok)[0]
    return kp[idx[np.abs(np.arange(len(kp))[:, None] - idx[None]).argmin(1)]]


def norm_i(w, n):
    """w: (T، 17، ≥2) → (n، 34): متوسّط على منتصف الحوضين، مقسوم على طول الجذع، و interpolation خطي لـ n خطوة."""
    w = w[..., :2]
    hip = (w[:, 11] + w[:, 12]) / 2
    sho = (w[:, 5] + w[:, 6]) / 2
    t = np.median(np.linalg.norm(sho - hip, axis=-1)) + 1e-6
    w = ((w - hip[:, None]) / t).reshape(len(w), 34)
    s = np.linspace(0, len(w) - 1, n)
    i0 = np.floor(s).astype(int)
    i1, a = np.minimum(i0 + 1, len(w) - 1), (s - i0)[:, None]
    return ((1 - a) * w[i0] + a * w[i1]).astype(np.float32)


def wins(x, y, win, step, n, center=None):
    """center=None: اللابل = أغلب لابل في النافذة. center=k: أغلب لابل في الـ k فريم اللي في النص."""
    starts = np.arange(0, len(x) - win + 1, step)
    X = np.stack([norm_i(x[s:s + win], n) for s in starts])
    c0, k = (0, win) if center is None else ((win - center) // 2, center)
    lab, pur = [], []
    for s in starts:
        c = np.bincount(y[s + c0:s + c0 + k], minlength=13)
        lab.append(c.argmax()); pur.append(c.max() / k)
    return X, np.array(lab), np.array(pur), starts + win // 2


def setup():
    split = json.loads((RES / 'split.json').read_text())
    mp = {}
    for p in sorted(KP5.glob('*.npz')):
        d = np.load(p)
        mp[p.stem] = dict(kp=fill(d['kp']), y=gt_of(d['act']), fid=d['fid'])
    part = {s: k for k in ('train', 'val', 'test') for s in split[k]}
    vids = {k: [v for v in mp if part.get(int(v[3:6])) == k] for k in ('train', 'val', 'test')}
    W = {}
    for v, d in mp.items():
        W[v] = dict(S_tr=wins(d['kp'], d['y'], SHORT, TR_STEP, 30), S_ev=wins(d['kp'], d['y'], SHORT, EV_STEP, 30),
                    L_tr=wins(d['kp'], d['y'], LONG, TR_STEP, 60, CENTER), L_ev=wins(d['kp'], d['y'], LONG, EV_STEP, 60, CENTER))
    print('فيديوهات MediaPipe:', {k: len(x) for k, x in vids.items()}, '| فريمات test', sum(len(mp[v]['y']) for v in vids['test']), flush=True)
    return split, mp, vids, W


def pick(W, vids, key, fx, pur=0.8):
    X = np.concatenate([W[v][key][0][W[v][key][2] >= pur] for v in vids])
    y = np.concatenate([W[v][key][1][W[v][key][2] >= pur] for v in vids])
    return fx(X), y


def frame_probs(pred, W, key, vids, mp):
    return {v: frame_pred(smooth(pred(W[v][key][0]), 5), W[v][key][3], len(mp[v]['y'])) for v in vids}


def stats(P, vids, mp):
    cm = np.zeros((13, 13), int)
    per = {}
    for v in vids:
        p, y = P[v].argmax(1), mp[v]['y']
        np.add.at(cm, (y, p), 1)
        per[v] = float((p == y).mean() * 100)
    acc = float(np.trace(cm) / cm.sum() * 100)
    f1 = float(np.mean([2 * cm[c, c] / max(1, cm[c].sum() + cm[:, c].sum()) for c in range(13)]) * 100)
    return dict(acc=acc, f1=f1, per=per, cm=cm.tolist())


def add(out, name, r):
    o = out.setdefault(name, {})
    for k in ('acc', 'f1', 'per'):
        o.setdefault(k, [])
    o.setdefault('cm', np.zeros((13, 13), int).tolist())
    o['acc'].append(r['acc']); o['f1'].append(r['f1']); o['per'].append(r['per'])
    o['cm'] = (np.array(o['cm']) + np.array(r['cm'])).tolist()


def log(tag, name, seed, r, t0, extra=''):
    print(f'{tag} | {name} | seed {seed}: test MP {r["acc"]:.1f} | macro-F1 {r["f1"]:.1f}{extra} ({time.time()-t0:.0f}s)', flush=True)


def finetune(m, prep, Xtr, ytr, Xva, yva, seed, aug, lr=3e-4):
    """نفس طريقة train_att / train_two بالظبط، بس بيبدأ من موديل متدرّب وبـ lr أقل."""
    torch.manual_seed(seed); np.random.seed(seed)
    m = copy.deepcopy(m)
    Xt = torch.as_tensor(Xtr, dtype=torch.float32, device=DEV)
    Xv = torch.as_tensor(Xva, dtype=torch.float32, device=DEV)
    yt, yv = torch.as_tensor(ytr, device=DEV), torch.as_tensor(yva, device=DEV)
    cnt = np.bincount(ytr, minlength=13).astype(float)
    wts = torch.as_tensor(1.0 / cnt[ytr], dtype=torch.float32)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=WD)

    def vloss():
        m.eval()
        with torch.no_grad():
            return sum(F.cross_entropy(m(prep(Xv[i:i + 4096])), yv[i:i + 4096], reduction='sum').item()
                       for i in range(0, len(Xv), 4096)) / len(Xv)
    best, state, bad = vloss(), copy.deepcopy(m.state_dict()), 0
    for ep in range(1, 61):
        m.train()
        idx = torch.multinomial(wts, STEPS * BATCH, replacement=True).to(DEV)
        for i in range(0, len(idx), BATCH):
            b = idx[i:i + BATCH]
            loss = F.cross_entropy(m(prep(aug(Xt[b]))), yt[b])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        vl = vloss()
        if vl < best - 1e-4:
            best, bad, state = vl, 0, copy.deepcopy(m.state_dict())
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    m.load_state_dict(state); m.eval()
    return m, prep


def run_mp(body, ctx):
    split, mp, vids, W = ctx
    fx = mask_legs if body == 'upper' else (lambda x: x)
    aug = (lambda x: mask_legs(base_aug(x))) if body == 'upper' else base_aug
    train_att.aug = tm.aug = aug
    Xtr, ytr = pick(W, vids['train'], 'S_tr', fx); Xva, yva = pick(W, vids['val'], 'S_tr', fx)
    XtrL, ytrL = pick(W, vids['train'], 'L_tr', fx); XvaL, yvaL = pick(W, vids['val'], 'L_tr', fx)
    print(f'mp_{body}: نوافذ train {len(Xtr)} / val {len(Xva)} | طويلة {len(XtrL)} / {len(XvaL)}', flush=True)
    txt = tm.text_codes().to(tm.DEV)
    tag, out = f'mp_{body}', {}
    for seed in SEEDS:
        for mname in OLD:
            t0 = time.time()
            if mname in CFG:
                m, prep, _ = train_att.train(CFG[mname], Xtr, ytr, Xva, yva, seed)
                pred = lambda X, m=m, prep=prep: train_att.predict(m, prep, fx(X))
            else:
                m, mu, sd, _ = tm.train(mname, txt, Xtr, ytr, Xva, yva, seed)
                pred = lambda X, m=m, mu=mu, sd=sd: tm.predict(m, mu, sd, fx(X))
            r = stats(frame_probs(pred, W, 'S_ev', vids['test'], mp), vids['test'], mp)
            add(out, mname, r); log(tag, mname, seed, r, t0)
        t0 = time.time()
        mS, pS = train_two.train(Xtr, ytr, Xva, yva, seed, aug)
        mL, pL = train_two.train(XtrL, ytrL, XvaL, yvaL, seed, aug)
        two(out, tag, seed, t0, lambda X: train_two.predict(mS, pS, fx(X)), lambda X: train_two.predict(mL, pL, fx(X)), W, vids, mp)
    (RES / f'run_mp_mp_{body}.json').write_text(json.dumps(out))
    print('DONE', tag, flush=True)


def two(out, tag, seed, t0, predS, predL, W, vids, mp, prefix=''):
    ev = vids['test'] + vids['val']
    PS, PL = frame_probs(predS, W, 'S_ev', ev, mp), frame_probs(predL, W, 'L_ev', ev, mp)
    w = train_two.best_weight([PS[v] for v in vids['val']], [PL[v] for v in vids['val']], [mp[v]['y'] for v in vids['val']])
    out.setdefault(prefix + 'S+L', {}).setdefault('w_short', []).append(w)
    for mname, P in ((TWO[0], PS), (TWO[1], PL), (TWO[2], {k: w * PS[k] + (1 - w) * PL[k] for k in PS})):
        r = stats(P, vids['test'], mp)
        add(out, prefix + mname, r); log(tag, prefix + mname, seed, r, t0, f' w={w}' if mname == 'S+L' else '')


def kin_noise(raw, vids, mp):
    """ضجيج MediaPipe مقابل Kinect على فيديوهات الـ val، بعد ما كل واحد يتطبّع لوحده (زي مدخل الموديل)."""
    def nrm(w):
        hip = (w[:, 11] + w[:, 12]) / 2
        t = np.median(np.linalg.norm((w[:, 5] + w[:, 6]) / 2 - hip, axis=-1)) + 1e-6
        return (w - hip[:, None]) / t
    K, M, C = [], [], []
    for v in vids['val']:
        fid = mp[v]['fid']; ok = fid < len(raw[v][0])
        K.append(nrm(raw[v][0][fid[ok]])); M.append(nrm(mp[v]['kp'][ok, :, :2])); C.append(mp[v]['kp'][ok, :, 2])
    return measure(np.concatenate(K), np.concatenate(M), np.concatenate(C))


def run_kin(body, ctx):
    split, mp, vids, W = ctx
    raw = load_raw()
    subj = {v: int(v[3:6]) for v in raw}
    fx = mask_legs if body == 'upper' else (lambda x: x)
    WS = {v: windows(*raw[v]) for v in raw}
    WL = {v: train_two.windows_long(*raw[v]) for v in raw}

    def kpick(Wk, S):
        L = [(Wk[v][0][Wk[v][2] >= 0.8], Wk[v][1][Wk[v][2] >= 0.8]) for v in raw if subj[v] in S]
        return fx(np.concatenate([a for a, _ in L])), np.concatenate([b for _, b in L])
    sig, rho, bias, drop = kin_noise(raw, vids, mp)
    print(f'kin_{body}: ضجيج MediaPipe (val): sigma متوسط {sig.mean():.3f}، rho متوسط {rho.mean():.2f} (بين فريمين 0.2 ث)', flush=True)
    augs = {}
    for n in (30, 60):
        f = make_noisy_aug(sig, rho, bias, drop, step=NOISE_STEP[n])
        augs[n] = (lambda x, f=f: mask_legs(f(x))) if body == 'upper' else f
    mp_aug = (lambda x: mask_legs(base_aug(x))) if body == 'upper' else base_aug
    train_att.aug = augs[30]
    KX, Ky = kpick(WS, split['train']); KXv, Kyv = kpick(WS, split['val'])
    KXL, KyL = kpick(WL, split['train']); KXLv, KyLv = kpick(WL, split['val'])
    Xtr, ytr = pick(W, vids['train'], 'S_tr', fx); Xva, yva = pick(W, vids['val'], 'S_tr', fx)
    XtrL, ytrL = pick(W, vids['train'], 'L_tr', fx); XvaL, yvaL = pick(W, vids['val'], 'L_tr', fx)
    tag, out = f'kin_{body}', {}
    for seed in SEEDS:
        t0 = time.time()
        m, prep, _ = train_att.train(CFG['A: LSTM'], KX, Ky, KXv, Kyv, seed)
        r = stats(frame_probs(lambda X: train_att.predict(m, prep, fx(X)), W, 'S_ev', vids['test'], mp), vids['test'], mp)
        add(out, '(ب) A: LSTM', r); log(tag, '(ب) A: LSTM', seed, r, t0)
        t0 = time.time()
        m2, _ = finetune(m, prep, Xtr, ytr, Xva, yva, seed, mp_aug)
        r = stats(frame_probs(lambda X: train_att.predict(m2, prep, fx(X)), W, 'S_ev', vids['test'], mp), vids['test'], mp)
        add(out, '(ج) A: LSTM', r); log(tag, '(ج) A: LSTM', seed, r, t0)
        t0 = time.time()
        mS, pS = train_two.train(KX, Ky, KXv, Kyv, seed, augs[30])
        mL, pL = train_two.train(KXL, KyL, KXLv, KyLv, seed, augs[60])
        two(out, tag, seed, t0, lambda X: train_two.predict(mS, pS, fx(X)), lambda X: train_two.predict(mL, pL, fx(X)), W, vids, mp, '(ب) ')
        t0 = time.time()
        fS, _ = finetune(mS, pS, Xtr, ytr, Xva, yva, seed, mp_aug)
        fL, _ = finetune(mL, pL, XtrL, ytrL, XvaL, yvaL, seed, mp_aug)
        two(out, tag, seed, t0, lambda X: train_two.predict(fS, pS, fx(X)), lambda X: train_two.predict(fL, pL, fx(X)), W, vids, mp, '(ج) ')
    (RES / f'run_mp_kin_{body}.json').write_text(json.dumps(out))
    print('DONE', tag, flush=True)


def run_dtw(body, ctx):
    from dtw_baseline import PER_CLASS, classify
    split, mp, vids, W = ctx
    fx = mask_legs if body == 'upper' else (lambda x: x)
    Xtr, ytr = pick(W, vids['train'], 'S_tr', fx)
    out = {}
    for seed in SEEDS:
        t0 = time.time()
        rng = np.random.RandomState(seed)
        idx = np.concatenate([rng.choice(np.where(ytr == c)[0], min(PER_CLASS, (ytr == c).sum()), replace=False)
                              for c in range(13) if (ytr == c).any()])
        r = stats(frame_probs(lambda X: classify(fx(X), Xtr[idx], ytr[idx]), W, 'S_ev', vids['test'], mp), vids['test'], mp)
        add(out, 'fastdtw', r); log(f'mp_{body}', 'fastdtw', seed, r, t0)
    (RES / f'run_mp_dtw_{body}.json').write_text(json.dumps(out))
    print('DONE dtw', body, flush=True)


def merge():
    f = lambda a: f'{np.mean(a):.1f}' if a else '—'
    rows = {}
    for p in sorted(RES.glob('run_mp_*_*.json')):
        _, _, mode, body = p.stem.split('_')
        for name, r in json.loads(p.read_text()).items():
            if 'acc' in r:
                rows[(name, body)] = (mode, r)
    lines = ['HA4M بـ MediaPipe بس: test = 8 أشخاص، فيديو لكل شخص، 5 فريم/ث، دقة لكل فريم (متوسط 3 seeds)',
             '(أ) = متدرّب على MediaPipe بس | (ب) = Kinect + ضجيج MediaPipe | (ج) = (ب) + fine-tune على MediaPipe', '',
             f'{"الموديل":<20}{"الجسم":<8}{"الدقة":>8}{"macro-F1":>10}   الدقة لكل seed']
    order = sorted(rows, key=lambda k: -np.mean(rows[k][1]['acc']))
    for k in order:
        mode, r = rows[k]
        lines.append(f'{k[0]:<20}{k[1]:<8}{f(r["acc"]):>8}{f(r["f1"]):>10}   ' + ', '.join(f'{a:.1f}' for a in r['acc']))
    best = order[0]
    r = rows[best][1]
    cm = np.array(r['cm'])
    lines += ['', f'أحسن موديل: {best[0]} ({best[1]})', 'recall لكل فئة: ' + ' | '.join(
        f'{NAMES[c]} {100 * cm[c, c] / max(1, cm[c].sum()):.0f}' for c in range(13))]
    pairs = sorted(((cm[i, j], i, j) for i in range(13) for j in range(13) if i != j), reverse=True)[:6]
    lines.append('أكتر لخبطة (حقيقي←متوقع، فريمات على 3 seeds): ' + ' | '.join(f'{NAMES[i]}←{NAMES[j]} {n}' for n, i, j in pairs))
    per = {v: np.mean([s[v] for s in r['per']]) for v in r['per'][0]}
    lines.append('الدقة لكل شخص: ' + ' | '.join(f'{v[3:6]} {a:.1f}' for v, a in sorted(per.items())))
    print('\n'.join(lines))
    (RES / 'run_mp.txt').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'merge':
        merge()
    else:
        ctx = setup()
        {'mp': run_mp, 'kin': run_kin, 'dtw': run_dtw}[a[0]](a[1], ctx)
