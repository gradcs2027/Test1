"""
HA4M: تجربة كل الموديلات في كل الظروف على نفس التقسيم (results/split.json) و3 seeds.

    python -I ha4m/run_all.py <full_clean|full_noise|upper_clean|upper_noise>
    python -I ha4m/run_all.py merge        # بيجمّع كل النتايج في results/run_all.txt

الموديلات: lstm, attn, text (train_models.py) و A, B, C, D (train_att.py).
الظروف:
  full / upper : 17 مفصل، أو الجسم العلوي بس (الركب والكعبين = صفر، لأن YOLO مش بيشوفهم ورا الترابيزة)
  clean / noise: تدريب عادي، أو بضجيج YOLO (مقاس من النص الأول من V001 زي train_noise.py)
التقييم: test Kinect (8 أشخاص) | V001 النص التاني Kinect | V001 النص التاني YOLO.
⚠️ YOLO متاح لشخص واحد (V001) بس، فعمود YOLO إشارة مش دليل.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_att  # noqa: E402
import train_models as tm  # noqa: E402
from prepare import OUT  # noqa: E402
from train_att import CFG  # noqa: E402
from train_noise import make_noisy_aug, measure  # noqa: E402
from train_stream import NONE, aug as base_aug, frame_pred, load_raw, norm17, smooth, windows  # noqa: E402

RES = HERE / 'results'
MODELS = ['lstm', 'attn', 'text'] + list(CFG)
SEEDS = range(3)
LEGS = slice(13, 17)


def mask_legs(x):
    """x: (..., 30, 34) → الركب والكعبين صفر."""
    s = x.reshape(*x.shape[:-1], 17, 2).clone() if torch.is_tensor(x) else x.reshape(*x.shape[:-1], 17, 2).copy()
    s[..., LEGS, :] = 0
    return s.reshape(x.shape)


def run(cond):
    upper, noise = cond.startswith('upper'), cond.endswith('noise')
    raw = load_raw()
    split = json.loads((RES / 'split.json').read_text())
    subj = {v: int(v[3:6]) for v in raw}
    W = {v: windows(*raw[v]) for v in raw}
    fx = mask_legs if upper else (lambda x: x)

    def pick(S, pur=0.8):
        L = [(W[v][0][W[v][2] >= pur], W[v][1][W[v][2] >= pur]) for v in raw if subj[v] in S]
        return fx(np.concatenate([a for a, _ in L])), np.concatenate([b for _, b in L])

    Xtr, ytr = pick(split['train']); Xva, yva = pick(split['val'])
    d = np.load(OUT.parent / 'ha4m_kp' / 'IDU001V001_kp2.npz')
    yolo, conf, act = d['new'][:, :, :2], d['new'][:, :, 2], d['act']
    n = len(act); half = n // 2
    gt = np.array([NONE if a <= 0 else a - 1 for a in act])
    kin = raw['IDU001V001'][0][:2 * n:2]
    base = noisy = base_aug
    if noise:
        noisy = make_noisy_aug(*measure(kin[:half], yolo[:half], conf[:half]))
    augf = (lambda x: mask_legs(noisy(x))) if upper else noisy
    train_att.aug = augf
    tm.aug = augf
    txt = tm.text_codes().to(tm.DEV)
    tests = [(v, W[v]) for v in raw if subj[v] in split['test']]
    out = {}
    for mname in MODELS:
        r = {'ktest': [], 'v_kin': [], 'v_yolo': []}
        for seed in SEEDS:
            t0 = time.time()
            if mname in CFG:
                m, prep, _ = train_att.train(CFG[mname], Xtr, ytr, Xva, yva, seed)
                pred = lambda X: train_att.predict(m, prep, fx(X))
            else:
                m, mu, sd, _ = tm.train(mname, txt, Xtr, ytr, Xva, yva, seed)
                pred = lambda X: tm.predict(m, mu, sd, fx(X))
            fa = []
            for v, (Xw, yw, _, cen) in tests:
                y = raw[v][1]
                fa.append(frame_pred(smooth(pred(Xw), 5), cen, len(y)).argmax(1) == y)
            r['ktest'].append(float(np.concatenate(fa).mean() * 100))
            st = np.arange(0, n - 45 + 1, 8)
            for key, src in (('v_kin', kin), ('v_yolo', yolo)):
                P = pred(np.stack([norm17(src[s:s + 45]) for s in st]))
                p = frame_pred(smooth(P, 5), st + 22, n).argmax(1)
                r[key].append(float((p[half:] == gt[half:]).mean() * 100))
            print(f'{cond} | {mname} | seed {seed}: test {r["ktest"][-1]:.1f} | V001 Kinect {r["v_kin"][-1]:.1f} YOLO {r["v_yolo"][-1]:.1f} ({time.time()-t0:.0f}s)', flush=True)
        out[mname] = r
    (RES / f'run_all_{cond}.json').write_text(json.dumps(out, indent=1))
    print('DONE', cond, flush=True)


def merge():
    f = lambda a: f'{np.mean(a):.1f}'
    lines = ['كل الموديلات × كل الظروف (متوسط 3 seeds). test = 8 أشخاص Kinect | V001 = النص التاني من فيديو شخص 1',
             f'{"الموديل":<16}{"الظرف":<14}{"test Kinect":>12}{"V001 Kinect":>13}{"V001 YOLO":>11}   (YOLO لكل seed)']
    for mname in MODELS + ['fastdtw']:
        for cond in ('full_clean', 'full_noise', 'upper_clean', 'upper_noise'):
            p = RES / (f'run_all_{cond[:-6]}_dtw.json' if mname == 'fastdtw' else f'run_all_{cond}.json')
            if not p.exists() or (mname == 'fastdtw' and cond.endswith('noise')):
                continue
            r = json.loads(p.read_text())[mname]
            lines.append(f'{mname:<16}{cond:<14}{f(r["ktest"]):>12}{f(r["v_kin"]):>13}{f(r["v_yolo"]):>11}   ({", ".join(f"{x:.0f}" for x in r["v_yolo"])})')
    print('\n'.join(lines))
    (RES / 'run_all.txt').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    merge() if sys.argv[1] == 'merge' else run(sys.argv[1])
