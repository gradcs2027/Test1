"""
Few-Shot FastDTW على vidtest2

    python run_vidtest2.py

المهمة: بناخد **قصاصة قصيرة واحدة** من كل حركة في الفيديو كـ template،
وبنصنّف باقي الفيديو بأقرب template بمسافة FastDTW.

عدد الحركات بييجي من `ground_truth.py` مش مكتوب هنا بالإيد — لما الجدول
اتصحّح وبقى `sit_down` و `stand_up` منفصلين بدل حركة واحدة، عدد الكلاسات
اتغيّر لوحده والصدفة معاه.

⚠️ المنهجية — تلات نقاط:
  ١. مافيش أي معامل بيتظبط — أقرب template وخلاص
  ٢. الـ template بيتستثنى من التقييم
  ٣. (لو فيه حركات بتتكرر) اختبار نظيف على الظهور التاني
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from oneshot_core import (build_multiscale_windows, cut_templates,
                          distance_cache, majority_smooth, norm_distance,
                          window_edges)
from ground_truth import GROUND_TRUTH, REPEATED as _REP

GT = GROUND_TRUTH['vidtest2']
REPEATED = _REP['vidtest2']

SCALES = (1.5, 2.0, 3.0)
STRIDE = 5
RADIUS = 1
SMOOTH_K = 3

TMPL_MAX, TMPL_MIN, TMPL_FRAC = 2.5, 1.0, 0.5


def load():
    kp = np.load('keypoints/vidtest2_keypoints.npy')
    meta = np.load('keypoints/vidtest2_meta.npy', allow_pickle=True).item()
    return kp, float(meta['effective_fps'])


def template_spans():
    """لكل حركة: أول جزء من الظهور الأول."""
    first = {}
    for s, e, lab in GT:
        if lab != '?' and lab not in first:
            first[lab] = (s, e)

    clips = []
    for lab, (s, e) in first.items():
        dur = min(TMPL_MAX, max(TMPL_MIN, (e - s) * TMPL_FRAC))
        clips.append((s, min(e, s + dur), lab))
    return clips


def evaluate(labels, edges, clips, step=0.1):
    """تقييم على مستوى الزمن. بيستثني '?' والـ template spans."""
    tmpl_spans = [(s, e) for s, e, _ in clips]
    per_class = defaultdict(lambda: {'n': 0, 'hit': 0})
    confusion = Counter()
    n = hit = 0

    for s, e, truth in GT:
        if truth == '?':
            continue
        for t in np.arange(s, e, step):
            if any(ts <= t < te for ts, te in tmpl_spans):
                continue
            w = int(np.searchsorted(edges, t, side='right') - 1)
            if not (0 <= w < len(labels)):
                continue

            pred = labels[w]
            ok = (pred == truth)
            n += 1
            hit += ok
            per_class[truth]['n'] += 1
            per_class[truth]['hit'] += ok
            if not ok:
                confusion[(truth, pred)] += 1

    return {'n': n, 'hit': hit, 'per_class': per_class, 'confusion': confusion}


def hubness_correct(D):
    """تصحيح الـ hubness."""
    med = np.median(D, axis=0, keepdims=True)
    mad = np.median(np.abs(D - med), axis=0, keepdims=True)
    return (D - med) / np.maximum(mad, 1e-6)


def run(mode='vel', shape_norm=True, smooth=SMOOTH_K, hubness=True,
        verbose=True):
    kp, fps = load()
    clips = template_spans()

    templates = cut_templates(kp, fps, clips, source='vidtest2', mode=mode,
                              shape_norm=shape_norm, scales=SCALES)
    tl = [t['label'] for t in templates]

    centers = np.arange(0, len(kp), STRIDE)
    edges = window_edges(centers, fps, len(kp))
    feats, E = build_multiscale_windows(kp, fps, centers, SCALES,
                                        mode=mode, shape_norm=shape_norm)

    key = f'v2_{mode}_{int(shape_norm)}_s{STRIDE}_r{RADIUS}'
    cache = Path('_scratch/dtw_cache')
    cache.mkdir(parents=True, exist_ok=True)
    cf = cache / f'{key}.npy'

    if cf.exists():
        D = np.load(cf)
    else:
        if verbose:
            print(f'  حساب {len(centers) * len(SCALES) * len(templates):,} '
                  f'مسافة DTW...')
        D = distance_cache(feats, templates, SCALES, radius=RADIUS)
        np.save(cf, D)

    if hubness:
        D = hubness_correct(D)

    flat = D.reshape(len(centers), -1)
    best = flat.argmin(axis=1)
    raw = [tl[i % len(templates)] for i in best]
    labels = majority_smooth(raw, smooth) if smooth > 1 else raw

    return evaluate(labels, edges, clips), labels, edges, clips, templates


def main():
    clips = template_spans()

    print('=' * 72)
    print(f'  Few-Shot FastDTW — vidtest2، {len(clips)} حركات')
    print('=' * 72)

    print(f'\n📚 الـ templates ({len(clips)} حركة):')
    for s, e, lab in clips:
        print(f'    {lab:<20} [{s:5.1f} - {e:5.1f}]  ({e - s:.1f}s)')

    r, labels, edges, _, templates = run()

    acc = r['hit'] / max(1, r['n'])
    chance = 1 / len(clips)

    print(f'\n{"=" * 72}')
    print('  النتيجة')
    print('=' * 72)
    print(f'  الدقة        {acc * 100:5.1f}%   على {r["n"]} عيّنة زمنية')
    print(f'  الصدفة       {chance * 100:5.1f}%   (1 من {len(clips)})')
    if acc > 0:
        print(f'  ↑ أحسن من الصدفة بـ {acc / chance:.1f}×')

    print(f'\n  --- الدقة لكل حركة ---')
    pc = r['per_class']
    for lab in sorted(pc, key=lambda k: -pc[k]['hit'] / max(1, pc[k]['n'])):
        v = pc[lab]
        a = v['hit'] / max(1, v['n'])
        bar = '█' * int(a * 24)
        print(f'    {lab:<20} {a * 100:5.1f}%  {bar:<24} '
              f'({v["hit"]}/{v["n"]})')

    print(f'\n  --- أكتر 6 التباسات ---')
    for (t, p), c in r['confusion'].most_common(6):
        print(f'    {t:<20} → {p:<20} {c:4}×')

    print(f'\n{"=" * 72}\n  تجارب الحذف\n{"=" * 72}')
    variants = [
        ('كامل',                    dict()),
        ('من غير تصحيح hubness',    dict(hubness=False)),
        ('مواضع بدل الفروق',        dict(mode='pos')),
        ('مواضع + من غير hubness',  dict(mode='pos', hubness=False)),
        ('من غير تطبيع الشكل',      dict(shape_norm=False)),
        ('من غير تنعيم',            dict(smooth=1)),
    ]
    for name, kw in variants:
        rv, *_ = run(verbose=False, **kw)
        a = rv['hit'] / max(1, rv['n'])
        print(f'  {name:<26} دقة={a * 100:5.1f}%')


if __name__ == '__main__':
    main()
