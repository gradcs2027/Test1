"""
الخطوة 4: تجهيز بيانات الـ zero shot.

بيطلّع 4 مجموعات في _scratch/zeroshot/data.npz:
  train       NTU xsub_train، الـ 46 حركة المتشافة، + 3 نسخ augmentation لكل عينة
  val_seen    NTU xsub_val، نفس الـ 46 حركة — لإيقاف التدريب بدري (early stopping)
  val_hidden  NTU xsub_val، الـ 3 حركات المخبّية — فحص zero shot على NTU نفسه
  ours        الـ 40 قصاصة GT من الـ 4 فيديوهات (نفس اختبار experiment_frame_scales)

كل عينة (30 فريم، 34 رقم) متطبّعة بـ shared/skeleton_norm (نفس الـ LSTM).
الحركات المخبّية مابتدخلش التدريب خالص، لا فيديو ولا نص.
"""
import sys
import pickle
import time
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from ground_truth import VIDEOS, spans
from paths import load_keypoints
from skeleton_norm import NUM_FRAMES, normalize_skeleton, resample_to

import descriptions as D

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
NTU_PKL = ROOT / '_scratch' / 'ntu60_2d.pkl'
OUT = ROOT / '_scratch' / 'zeroshot' / 'data.npz'
N_AUG = 3
SEED = 0

# نفس ربط experiment_frame_scales: sitting بتتقاس كمان على sit_down
GT_LABEL_ALIASES = {'sitting': ['sitting', 'sit_down']}

# ─── augmentation: منسوخة حرفياً من lstm/lstm_ntu60.ipynb (خلية 3) ───
COCO_FLIP = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15])
AUG_MODES = ('full', 'crop', 'pad')
AUG_P = (0.25, 0.40, 0.35)


def sample_indices(n, rng, mode):
    """قص زمني: الكليب كامل، أو جزء منه، أو الحركة + وقفة قبلها/بعدها."""
    if mode == 'full':
        return np.linspace(0, n - 1, NUM_FRAMES, dtype=int)
    if mode == 'crop':
        span = max(2, int(n * rng.uniform(0.6, 1.0)))
        lo = int(rng.integers(0, n - span + 1))
        return np.linspace(lo, lo + span - 1, NUM_FRAMES, dtype=int)
    k = max(2, int(NUM_FRAMES * rng.uniform(0.5, 0.9)))
    before = int(rng.integers(0, NUM_FRAMES - k + 1))
    return np.concatenate([
        np.zeros(before, dtype=int),
        np.linspace(0, n - 1, k, dtype=int),
        np.full(NUM_FRAMES - k - before, n - 1, dtype=int),
    ])


def augment_geom(seq, rng):
    """قلب + دوران ±12° + مقياس + رعشة + إخفاء مفصل (معايرة على فرق NTU/YOLO)."""
    s = seq.reshape(NUM_FRAMES, 17, 2).copy()
    if rng.random() < 0.5:
        s = s[:, COCO_FLIP, :]
        s[..., 0] *= -1
    th = rng.uniform(-0.21, 0.21)
    c, sn = np.cos(th), np.sin(th)
    s = s @ np.array([[c, sn], [-sn, c]], dtype=np.float32)
    s *= rng.uniform(0.85, 1.15)
    s += rng.normal(0, rng.uniform(0.0, 0.045), s.shape).astype(np.float32)
    dead = rng.random(17) < 0.05
    if dead.any():
        s[:, dead, :] = s[:, dead, :].mean(axis=0, keepdims=True)
    return s.reshape(NUM_FRAMES, -1).astype(np.float32)


def ntu_sets():
    t = time.time()
    data = pickle.load(open(NTU_PKL, 'rb'))
    print(f'📂 NTU60: {len(data["annotations"])} عينة ({time.time() - t:.0f}s)')
    tr = set(data['split']['xsub_train'])
    hidden = set(D.HIDDEN.values())
    rng = np.random.default_rng(SEED)
    sets = {k: ([], []) for k in ('train', 'val_seen', 'val_hidden')}
    for a in data['annotations']:
        lab = a['label']
        if lab >= len(D.NTU49):          # حركات شخصين — برّه
            continue
        kp = np.asarray(a['keypoint'][0], dtype=np.float32)
        n = len(kp)
        seq = normalize_skeleton(kp[np.linspace(0, n - 1, NUM_FRAMES, dtype=int)])
        if a['frame_dir'] in tr:
            if lab in hidden:            # المخبّية مابتدخلش التدريب خالص
                continue
            X, y = sets['train']
            X.append(seq); y.append(lab)
            for _ in range(N_AUG):
                mode = 'full' if n < 4 else rng.choice(AUG_MODES, p=AUG_P)
                s = normalize_skeleton(kp[sample_indices(n, rng, mode)])
                X.append(augment_geom(s, rng)); y.append(lab)
        else:
            X, y = sets['val_hidden' if lab in hidden else 'val_seen']
            X.append(seq); y.append(lab)
    return {k: (np.array(X, dtype=np.float16), np.array(y, dtype=np.int16))
            for k, (X, y) in sets.items()}


def our_clips():
    """نفس نوافذ build_test_windows في experiment_frame_scales بس بتطبيع الـ LSTM."""
    X, y, vid = [], [], []
    names = list(D.OURS)
    for v in VIDEOS:
        kp, fps = load_keypoints(v)
        kp = kp[..., :2]
        for label in names:
            for gt_label in GT_LABEL_ALIASES.get(label, [label]):
                for t0, t1 in spans(v, gt_label):
                    lo, hi = int(round(t0 * fps)), int(round(t1 * fps))
                    clip = kp[max(0, lo):min(len(kp), hi)]
                    if len(clip) < 4:
                        continue
                    X.append(resample_to(normalize_skeleton(clip)))
                    y.append(names.index(label)); vid.append(v)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int16), np.array(vid)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sets = ntu_sets()
    Xo, yo, vo = our_clips()
    np.savez(OUT, **{f'X_{k}': X for k, (X, _) in sets.items()},
             **{f'y_{k}': y for k, (_, y) in sets.items()},
             X_ours=Xo, y_ours=yo, vid_ours=vo)

    for k, (X, y) in sets.items():
        print(f'  {k:<11} {X.shape}  حركات: {len(np.unique(y))}')
    names = list(D.OURS)
    unseen = sum(names[i] in D.UNSEEN for i in yo)
    print(f'  ours        {Xo.shape}  قصاصات على حركات مش متشافة: {unseen} من {len(yo)}')
    leak = set(np.unique(sets['train'][1])) & set(D.HIDDEN.values())
    assert not leak, f'حركات مخبّية دخلت التدريب: {leak}'
    assert np.isfinite(Xo).all() and np.isfinite(sets['train'][0].astype(np.float32)).all()
    print(f'✅ مفيش حركة مخبّية في التدريب — {OUT} ({OUT.stat().st_size / 1e6:.0f}MB)')


if __name__ == '__main__':
    main()
