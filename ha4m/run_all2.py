"""
HA4M: إعادة تشغيل كل الموديلات بعد إضافة MediaPipe وفيديو V002 والـ two classifiers (train_two.py).

    python -I ha4m/run_all2.py full_clean full_noiseyolo ...   # ظرف أو أكتر ورا بعض
    python -I ha4m/run_all2.py dtw full|upper                  # FastDTW على الكاميرات (الـ test من run_all_*_dtw.json)
    python -I ha4m/run_all2.py merge                           # → results/run_all2.txt

الظروف: {full, upper} × {clean, noiseyolo, noisemp}
  noiseyolo / noisemp: ضجيج متقاس من النص الأول من V001 لـ YOLO أو لـ MediaPipe (زي train_noise.py).
الأعمدة:
  ktest              test Kinect (8 أشخاص)
  v1_kin/yolo/mp     V001 النص التاني (النص الأول اتقاس منه الضجيج)، صور Color
  v2_kin/yolo/mp     V002 كله، صور ColorAligned (640×576)، نفس الشخص 1 بس فيديو تاني ماتقاسش منه أي ضجيج
⚠️ الكاميرا لسه شخص واحد (الشخص 1 في الـ test) في فيديوهين.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_att  # noqa: E402
import train_models as tm  # noqa: E402
import train_two  # noqa: E402
from prepare import OUT  # noqa: E402
from run_all import MODELS as OLD, RES, mask_legs  # noqa: E402
from train_att import CFG  # noqa: E402
from train_noise import make_noisy_aug, measure  # noqa: E402
from train_stream import NONE, aug as base_aug, frame_pred, load_raw, smooth, windows  # noqa: E402
from train_two import LONG_N, LONG_WIN, SHORT_N, NOISE_STEP, normn, windows_long  # noqa: E402

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

SEEDS = range(3)
CONDS = [f'{b}_{n}' for b in ('full', 'upper') for n in ('clean', 'noiseyolo', 'noisemp')]
TWO = ['S: short 3s', 'L: long 8s', 'S+L']
MODELS = OLD + TWO
COLS = ['ktest', 'v1_kin', 'v1_yolo', 'v1_mp', 'v2_kin', 'v2_yolo', 'v2_mp']
KP = OUT.parent / 'ha4m_kp'
CAM_WIN = {SHORT_N: 45, LONG_N: LONG_WIN // 2}     # طول النافذة بفريمات الكاميرا (15 فريم/ث)


def gt_of(act):
    return np.array([NONE if a <= 0 else a - 1 for a in act])


def cameras(raw):
    d1, m1 = np.load(KP / 'IDU001V001_kp2.npz'), np.load(KP / 'IDU001V001_mp.npz')
    d2, m2 = np.load(KP / 'IDU001V002_yolo.npz'), np.load(KP / 'IDU001V002_mp.npz')
    assert (d1['fid'] == m1['fid']).all() and (d2['fid'] == m2['fid']).all()
    n1 = len(d1['act'])
    return {
        'v1': dict(gt=gt_of(d1['act']), kin=raw['IDU001V001'][0][d1['fid']], yolo=d1['new'], mp=m1['kp'], lo=n1 // 2),
        'v2': dict(gt=gt_of(d2['act']), kin=raw['IDU001V002'][0][d2['fid']], yolo=d2['kp'], mp=m2['kp'], lo=0),
    }


def setup():
    raw = load_raw()
    split = json.loads((RES / 'split.json').read_text())
    subj = {v: int(v[3:6]) for v in raw}
    WS = {v: windows(*raw[v]) for v in raw}
    WL = {v: windows_long(*raw[v]) for v in raw}
    return raw, split, subj, WS, WL, cameras(raw)


def cam_X(src, n_steps):
    win = CAM_WIN[n_steps]
    st = np.arange(0, len(src) - win + 1, 8)
    return np.stack([normn(src[s:s + win, :, :2], n_steps) for s in st]), st + win // 2


def frame_probs(pred, n_steps, tests, raw, W, cams):
    """احتمالات لكل فريم: لكل فيديو test ولكل (كاميرا، مصدر)."""
    out = {v: frame_pred(smooth(pred(W[v][0]), 5), W[v][3], len(raw[v][1])) for v in tests}
    for c, d in cams.items():
        for src in ('kin', 'yolo', 'mp'):
            X, cen = cam_X(d[src], n_steps)
            out[(c, src)] = frame_pred(smooth(pred(X), 5), cen, len(d['gt']))
    return out


def score(P, tests, raw, cams):
    r = {'ktest': float(np.concatenate([P[v].argmax(1) == raw[v][1] for v in tests]).mean() * 100)}
    for c, d in cams.items():
        for src in ('kin', 'yolo', 'mp'):
            r[f'{c}_{src}'] = float((P[(c, src)].argmax(1)[d['lo']:] == d['gt'][d['lo']:]).mean() * 100)
    return r


def run(cond, ctx):
    raw, split, subj, WS, WL, cams = ctx
    upper, noise = cond.startswith('upper'), cond.split('_')[1]
    fx = mask_legs if upper else (lambda x: x)

    def pick(W, S, pur=0.8):
        L = [(W[v][0][W[v][2] >= pur], W[v][1][W[v][2] >= pur]) for v in raw if subj[v] in S]
        return fx(np.concatenate([a for a, _ in L])), np.concatenate([b for _, b in L])

    v1 = cams['v1']; half = v1['lo']
    augs = {}
    for n_steps in (SHORT_N, LONG_N):
        f = base_aug
        if noise != 'clean':
            src = v1['yolo'] if noise == 'noiseyolo' else v1['mp']
            f = make_noisy_aug(*measure(v1['kin'][:half], src[:half, :, :2], src[:half, :, 2]), step=NOISE_STEP[n_steps])
        augs[n_steps] = (lambda x, f=f: mask_legs(f(x))) if upper else f
    train_att.aug = tm.aug = augs[SHORT_N]

    Xtr, ytr = pick(WS, split['train']); Xva, yva = pick(WS, split['val'])
    XtrL, ytrL = pick(WL, split['train']); XvaL, yvaL = pick(WL, split['val'])
    tests = [v for v in raw if subj[v] in split['test']]
    vals = [v for v in raw if subj[v] in split['val']]
    txt = tm.text_codes().to(tm.DEV)
    out = {m: {c: [] for c in COLS} for m in MODELS}
    out['S+L']['w_short'] = []
    for seed in SEEDS:
        for mname in OLD:
            t0 = time.time()
            if mname in CFG:
                m, prep, _ = train_att.train(CFG[mname], Xtr, ytr, Xva, yva, seed)
                pred = lambda X, m=m, prep=prep: train_att.predict(m, prep, fx(X))
            else:
                m, mu, sd, _ = tm.train(mname, txt, Xtr, ytr, Xva, yva, seed)
                pred = lambda X, m=m, mu=mu, sd=sd: tm.predict(m, mu, sd, fx(X))
            r = score(frame_probs(pred, SHORT_N, tests, raw, WS, cams), tests, raw, cams)
            for c in COLS:
                out[mname][c].append(r[c])
            log(cond, mname, seed, r, t0)
        t0 = time.time()
        mS, pS = train_two.train(Xtr, ytr, Xva, yva, seed, augs[SHORT_N])
        mL, pL = train_two.train(XtrL, ytrL, XvaL, yvaL, seed, augs[LONG_N])
        predS = lambda X: train_two.predict(mS, pS, fx(X))
        predL = lambda X: train_two.predict(mL, pL, fx(X))
        PS = frame_probs(predS, SHORT_N, tests + vals, raw, WS, cams)
        PL = frame_probs(predL, LONG_N, tests + vals, raw, WL, cams)
        w = train_two.best_weight([PS[v] for v in vals], [PL[v] for v in vals], [raw[v][1] for v in vals])
        out['S+L']['w_short'].append(w)
        for mname, P in ((TWO[0], PS), (TWO[1], PL), (TWO[2], {k: w * PS[k] + (1 - w) * PL[k] for k in PS})):
            r = score(P, tests, raw, cams)
            for c in COLS:
                out[mname][c].append(r[c])
            log(cond, mname, seed, r, t0, f' w={w}' if mname == 'S+L' else '')
    (RES / f'run_all2_{cond}.json').write_text(json.dumps(out, indent=1))
    print('DONE', cond, flush=True)


def log(cond, mname, seed, r, t0, extra=''):
    print(f'{cond} | {mname} | seed {seed}: test {r["ktest"]:.1f} | V001 K {r["v1_kin"]:.1f} Y {r["v1_yolo"]:.1f} MP {r["v1_mp"]:.1f}'
          f' | V002 K {r["v2_kin"]:.1f} Y {r["v2_yolo"]:.1f} MP {r["v2_mp"]:.1f}{extra} ({time.time()-t0:.0f}s)', flush=True)


def dtw(body, ctx):
    sys.path.insert(0, str(HERE.parent / 'fastdtw'))
    from dtw_baseline import PER_CLASS, classify
    raw, split, subj, WS, WL, cams = ctx
    fx = mask_legs if body == 'upper' else (lambda x: x)
    Xtr = np.concatenate([WS[v][0][WS[v][2] >= 0.8] for v in raw if subj[v] in split['train']])
    ytr = np.concatenate([WS[v][1][WS[v][2] >= 0.8] for v in raw if subj[v] in split['train']])
    old = json.loads((RES / f'run_all_{body}_dtw.json').read_text())['fastdtw']
    out = {c: [] for c in COLS}
    for seed in SEEDS:
        t0 = time.time()
        rng = np.random.RandomState(seed)
        idx = np.concatenate([rng.choice(np.where(ytr == c)[0], min(PER_CLASS, (ytr == c).sum()), replace=False) for c in range(13)])
        T, ty = fx(Xtr[idx]), ytr[idx]
        r = {'ktest': old['ktest'][seed]}
        for c, d in cams.items():
            for src in ('kin', 'yolo', 'mp'):
                X, cen = cam_X(d[src], SHORT_N)
                p = frame_pred(smooth(classify(fx(X), T, ty), 5), cen, len(d['gt'])).argmax(1)
                r[f'{c}_{src}'] = float((p[d['lo']:] == d['gt'][d['lo']:]).mean() * 100)
        for c in COLS:
            out[c].append(r[c])
        log(f'{body}_clean', 'fastdtw', seed, r, t0)
    (RES / f'run_all2_{body}_dtw.json').write_text(json.dumps({'fastdtw': out}, indent=1))
    print('DONE dtw', body, flush=True)


def merge():
    f = lambda a: f'{np.mean(a):.1f}' if a else '—'
    head = ['test', 'V1 Kin', 'V1 YOLO', 'V1 MP', 'V2 Kin', 'V2 YOLO', 'V2 MP']
    lines = ['كل الموديلات × كل الظروف (متوسط 3 seeds). test = 8 أشخاص Kinect | V1 = النص التاني من V001 | V2 = V002 كله (ColorAligned)',
             f'{"الموديل":<16}{"الظرف":<18}' + ''.join(f'{h:>9}' for h in head)]
    for mname in MODELS + ['fastdtw']:
        for cond in CONDS:
            if mname == 'fastdtw':
                p = RES / f'run_all2_{cond.split("_")[0]}_dtw.json'
                if not cond.endswith('clean') or not p.exists():
                    continue
            else:
                p = RES / f'run_all2_{cond}.json'
                if not p.exists():
                    continue
            r = json.loads(p.read_text())[mname]
            extra = f'   w_short={r["w_short"]}' if 'w_short' in r else ''
            lines.append(f'{mname:<16}{cond:<18}' + ''.join(f'{f(r[c]):>9}' for c in COLS) + extra)
    print('\n'.join(lines))
    (RES / 'run_all2.txt').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'merge':
        merge()
    elif a[0] == 'dtw':
        dtw(a[1], setup())
    else:
        ctx = setup()
        for cond in a:
            assert cond in CONDS, cond
            run(cond, ctx)
