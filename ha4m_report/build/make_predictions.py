"""
تقرير HA4M — الخطوة 1: بيدرّب كل موديل (seed 0) في ظرفين وبيحفظ توقعاته عشان النوتبوك والفيديوهات.

    python -I ha4m_report/build/make_predictions.py

  قبل = full_clean  (17 مفصل، من غير ضجيج)        — نقطة البداية
  بعد = upper_noise (الجسم العلوي + ضجيج YOLO)    — أحسن وصفة على YOLO في run_all.py
  FastDTW: قبل = full، بعد = upper (ملوش تدريب فمفيش ضجيج).

المخرج في ha4m_report/data/:
  predictions.npz  توقع كل فريم على V001 (Kinect و YOLO) + confusion على الـ test لكل موديل/ظرف
  summary.json     الدقة، عدد الباراميترات، إحصائيات الداتا، ضجيج YOLO المقاس
  + نسخة من ملفات ha4m/results (الجداول الكاملة بـ 3 seeds)
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'ha4m'))
sys.path.insert(0, str(ROOT / 'fastdtw'))
import train_att  # noqa: E402
import train_models as tm  # noqa: E402
from dtw_baseline import PER_CLASS, classify  # noqa: E402
from prepare import OUT  # noqa: E402
from run_all import MODELS, mask_legs  # noqa: E402
from train_att import CFG, AttLSTM  # noqa: E402
from train_noise import make_noisy_aug, measure  # noqa: E402
from train_stream import NAMES, NONE, aug as base_aug, frame_pred, load_raw, norm17, smooth, windows  # noqa: E402

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

RES = ROOT / 'ha4m' / 'results'
DATA = ROOT / 'ha4m_report' / 'data'
SEED = 0
CONDS = ('full_clean', 'upper_noise')


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    for f in sorted(RES.glob('*.json')) + sorted(RES.glob('*.txt')) + [RES / 'confusion.npz', RES / 'actions_sheet.jpg', RES / 'V001_frame600.jpg']:
        shutil.copy2(f, DATA / f.name)

    raw = load_raw()
    split = json.loads((RES / 'split.json').read_text())
    subj = {v: int(v[3:6]) for v in raw}
    W = {v: windows(*raw[v]) for v in raw}

    def pick(S, pur=0.8):
        L = [(W[v][0][W[v][2] >= pur], W[v][1][W[v][2] >= pur]) for v in raw if subj[v] in S]
        return np.concatenate([a for a, _ in L]), np.concatenate([b for _, b in L])

    Xtr0, ytr = pick(split['train']); Xva0, yva = pick(split['val'])
    d = np.load(OUT.parent / 'ha4m_kp' / 'IDU001V001_kp2.npz')
    yolo, conf, act, old = d['new'][:, :, :2], d['new'][:, :, 2], d['act'], d['old']
    n = len(act); half = n // 2
    gt = np.array([NONE if a <= 0 else a - 1 for a in act])
    kin = raw['IDU001V001'][0][:2 * n:2]
    sig, rho, bias, drop = measure(kin[:half], yolo[:half], conf[:half])
    noisy = make_noisy_aug(sig, rho, bias, drop)
    txt = tm.text_codes().to(tm.DEV)
    tests = [(v, W[v]) for v in raw if subj[v] in split['test']]
    st = np.arange(0, n - 45 + 1, 8)
    V = {k: np.stack([norm17(s[i:i + 45]) for i in st]) for k, s in (('kin', kin), ('yolo', yolo))}

    pred, acc, params = {}, {}, {}
    for cond in CONDS:
        upper, noise = cond.startswith('upper'), cond.endswith('noise')
        fx = mask_legs if upper else (lambda x: x)
        augf = noisy if noise else base_aug
        augf = (lambda x, f=augf: mask_legs(f(x))) if upper else augf
        train_att.aug = augf; tm.aug = augf
        Xtr, Xva = fx(Xtr0), fx(Xva0)
        for mname in MODELS:
            t0 = time.time()
            if mname in CFG:
                m, prep, _ = train_att.train(CFG[mname], Xtr, ytr, Xva, yva, SEED)
                P = lambda X: train_att.predict(m, prep, fx(X))
            else:
                m, mu, sd, _ = tm.train(mname, txt, Xtr, ytr, Xva, yva, SEED)
                P = lambda X: tm.predict(m, mu, sd, fx(X))
            params[mname] = int(sum(p.numel() for p in m.parameters() if p.requires_grad))
            cm, ok = np.zeros((13, 13), int), []
            for v, (Xw, _, _, cen) in tests:
                y = raw[v][1]
                p = frame_pred(smooth(P(Xw), 5), cen, len(y)).argmax(1)
                np.add.at(cm, (y, p), 1); ok.append(p == y)
            key = f'{mname}|{cond}'
            pred[key + '|cm'] = cm
            a = {'test': float(np.concatenate(ok).mean() * 100)}
            for src in ('kin', 'yolo'):
                pf = frame_pred(smooth(P(V[src]), 5), st + 22, n).argmax(1)
                pred[f'{key}|{src}'] = pf
                a[src] = float((pf[half:] == gt[half:]).mean() * 100)
            acc[key] = a
            print(f'{key}: test {a["test"]:.1f} | V001 Kinect {a["kin"]:.1f} YOLO {a["yolo"]:.1f} ({time.time()-t0:.0f}s)', flush=True)

    # FastDTW على V001 بس (الـ test الكامل بتاعه في run_all_*_dtw.json)
    rng = np.random.RandomState(SEED)
    idx = np.concatenate([rng.choice(np.where(ytr == c)[0], min(PER_CLASS, (ytr == c).sum()), replace=False) for c in range(13)])
    for cond, fx in (('full_clean', lambda x: x), ('upper_noise', mask_legs)):
        t0 = time.time()
        T, ty = fx(Xtr0[idx]), ytr[idx]
        key = f'fastdtw|{cond}'
        dtw = json.loads((RES / f'run_all_{cond.split("_")[0]}_dtw.json').read_text())['fastdtw']
        a = {'test': dtw['ktest'][SEED]}
        for src in ('kin', 'yolo'):
            pf = frame_pred(smooth(classify(fx(V[src]), T, ty), 5), st + 22, n).argmax(1)
            pred[f'{key}|{src}'] = pf
            a[src] = float((pf[half:] == gt[half:]).mean() * 100)
        acc[key] = a
        print(f'{key}: test {a["test"]:.1f} (من run_all) | V001 Kinect {a["kin"]:.1f} YOLO {a["yolo"]:.1f} ({time.time()-t0:.0f}s)', flush=True)
    params['fastdtw'] = 0

    for k in CFG:
        params[k] = int(sum(p.numel() for p in AttLSTM(*CFG[k]).parameters()))
    nf = {s: np.bincount(np.concatenate([raw[v][1] for v in raw if subj[v] in split[s]]), minlength=13).tolist()
          for s in ('train', 'val', 'test')}
    summary = {
        'seed': SEED, 'conds': CONDS, 'models': MODELS + ['fastdtw'], 'names': NAMES, 'acc': acc, 'params': params,
        'frames_per_class': nf, 'videos': {s: len(split['videos'][s]) for s in ('train', 'val', 'test')},
        'subjects': {s: split[s] for s in ('train', 'val', 'test')},
        'n_videos': len(raw), 'n_frames': int(sum(len(raw[v][1]) for v in raw)),
        'video_len_s': [len(raw[v][1]) / 30 for v in raw],
        'windows': {'train': int(len(ytr)), 'val': int(len(yva)),
                    'test_all': int(sum(len(W[v][1]) for v, _ in tests))},
        'purity_all_train': [float(x) for v in raw if subj[v] in split['train'] for x in W[v][2][::5]],
        'noise': {'sigma': sig.tolist(), 'rho': rho.tolist(), 'drop': drop.tolist()},
        'v001': {'n': int(n), 'half': int(half), 'fps': 15},
    }
    (DATA / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding='utf-8')
    np.savez_compressed(DATA / 'predictions.npz', gt=gt, kin=kin, yolo=yolo, conf=conf,
                        old_conf=old[:, :, 2], fid=d['fid'], **pred)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
