"""
HA4M: خط أساس FastDTW (من غير تدريب) على نفس التقسيم والتقييم بتاع run_all.py.

    python -I ha4m/dtw_baseline.py <full|upper>

- قوالب: 15 نافذة نقية (>=0.8) لكل فئة من أشخاص train، كل seed بيختار قوالب مختلفة.
- التوقع: أقرب قالب (FastDTW radius=1 على 30×34، موضع بس) → فئته. بعدها نفس smooth5 وأقرب نافذة لكل فريم.
- ⚠️ لسرعة الحساب: نوافذ الـ test بنقيّم كل خامس نافذة بس (التوقع لكل فريم بياخد أقرب نافذة متقيّمة).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'fastdtw'))
from fastdtw_algorithm import fastdtw  # noqa: E402
from prepare import OUT  # noqa: E402
from run_all import RES, mask_legs  # noqa: E402
from train_stream import NONE, frame_pred, load_raw, norm17, smooth, windows  # noqa: E402

PER_CLASS, SUB = 15, 5


def classify(Q, T, ty):
    """Q: (k، 30، 34) → احتمالات one-hot لأقرب قالب."""
    P = np.zeros((len(Q), 13), np.float32)
    for i, q in enumerate(Q):
        d = np.array([fastdtw(q, t, radius=1)[0] for t in T])
        P[i, ty[d.argmin()]] = 1
    return P


def main(cond):
    upper = cond == 'upper'
    fx = mask_legs if upper else (lambda x: x)
    raw = load_raw()
    split = json.loads((RES / 'split.json').read_text())
    subj = {v: int(v[3:6]) for v in raw}
    W = {v: windows(*raw[v]) for v in raw}
    Xtr = np.concatenate([W[v][0][W[v][2] >= 0.8] for v in raw if subj[v] in split['train']])
    ytr = np.concatenate([W[v][1][W[v][2] >= 0.8] for v in raw if subj[v] in split['train']])
    d = np.load(OUT.parent / 'ha4m_kp' / 'IDU001V001_kp2.npz')
    yolo, act = d['new'][:, :, :2], d['act']
    n = len(act); half = n // 2
    gt = np.array([NONE if a <= 0 else a - 1 for a in act])
    kin = raw['IDU001V001'][0][:2 * n:2]
    tests = [(v, W[v]) for v in raw if subj[v] in split['test']]
    r = {'ktest': [], 'v_kin': [], 'v_yolo': []}
    for seed in range(3):
        t0 = time.time()
        rng = np.random.RandomState(seed)
        idx = np.concatenate([rng.choice(np.where(ytr == c)[0], min(PER_CLASS, (ytr == c).sum()), replace=False) for c in range(13)])
        T, ty = fx(Xtr[idx]), ytr[idx]
        fa = []
        for v, (Xw, yw, _, cen) in tests:
            y = raw[v][1]
            sel = np.arange(0, len(Xw), SUB)
            P = smooth(classify(fx(Xw[sel]), T, ty), 5)
            fa.append(frame_pred(P, cen[sel], len(y)).argmax(1) == y)
        r['ktest'].append(float(np.concatenate(fa).mean() * 100))
        st = np.arange(0, n - 45 + 1, 8)
        for key, src in (('v_kin', kin), ('v_yolo', yolo)):
            P = classify(fx(np.stack([norm17(src[s:s + 45]) for s in st])), T, ty)
            p = frame_pred(smooth(P, 5), st + 22, n).argmax(1)
            r[key].append(float((p[half:] == gt[half:]).mean() * 100))
        print(f'{cond} | fastdtw | seed {seed}: test {r["ktest"][-1]:.1f} | V001 Kinect {r["v_kin"][-1]:.1f} YOLO {r["v_yolo"][-1]:.1f} ({time.time()-t0:.0f}s)', flush=True)
    (RES / f'run_all_{cond}_dtw.json').write_text(json.dumps({'fastdtw': r}, indent=1))
    print('DONE', cond, flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
