"""
FastDTW — تجربة RADIUS مختلفة (بدون confidence filtering)

الفكرة: جرّب RADIUS=[1, 2, 3, 4] وشوف أيهم أحسن accuracy مباشرة
"""

import sys
from collections import Counter

import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import _bootstrap  # noqa: F401

from ground_truth import GROUND_TRUTH, VIDEOS, shared_labels, spans
from classifier import (NUM_FRAMES, balance_templates, cut_templates,
                          normalize_window, norm_distance, resample_linear,
                          to_features)
from paths import load_keypoints

SCALES = (1.0, 1.5, 2.0, 3.0)
STRIDE = 5
MAX_PER_LABEL = 8
SHARED = tuple(shared_labels())
RADII_TO_TRY = [1, 2, 3, 4, 5]


def load(video):
    return load_keypoints(video)


def build_templates(sources, mode='vel', shape_norm=True, exclude=()):
    out = []
    for v in sources:
        kp, fps = load(v)
        clips = [(s, e, lab) for lab in SHARED for s, e in spans(v, lab)
                 if (v, s, e) not in exclude]
        out += cut_templates(kp, fps, clips, source=v, mode=mode,
                             shape_norm=shape_norm, scales=SCALES)
    return balance_templates(out, MAX_PER_LABEL)


def featurize(kp, fps, t0, t1, mode, shape_norm):
    lo, hi = int(round(t0 * fps)), int(round(t1 * fps))
    clip = kp[max(0, lo):min(len(kp), hi)]
    if len(clip) < 4:
        return None
    seq = resample_linear(normalize_window(clip), NUM_FRAMES)
    return to_features(seq, mode=mode, shape_norm=shape_norm)


def nearest(feat, templates, radius=1):
    """أقرب template (بدون confidence)."""
    distances = []
    for t in templates:
        try:
            d = norm_distance(feat, t['feat'], radius=radius)
            distances.append(d)
        except:
            distances.append(float('inf'))

    if not distances or all(d == float('inf') for d in distances):
        return None, float('inf')

    best_idx = int(np.argmin(distances))
    return templates[best_idx]['label'], distances[best_idx]


def segment_protocol(radius=1, mode='vel', shape_norm=True, within=False):
    """الـ protocol الأساسي."""
    rows = []
    for v in VIDEOS:
        kp, fps = load(v)
        for lab in SHARED:
            for (s, e) in spans(v, lab):
                if within:
                    srcs, excl = [v], {(v, s, e)}
                    tmpl = build_templates(srcs, mode, shape_norm, exclude=excl)
                else:
                    srcs = [o for o in VIDEOS if o != v]
                    tmpl = build_templates(srcs, mode, shape_norm)

                if not tmpl or not any(t['label'] == lab for t in tmpl):
                    continue

                feat = featurize(kp, fps, s, e, mode, shape_norm)
                if feat is None:
                    continue

                pred, dist = nearest(feat, tmpl, radius=radius)
                rows.append({
                    'video': v, 'span': (s, e), 'truth': lab,
                    'pred': pred, 'dist': dist,
                })
    return rows


def score(rows):
    """احسب الدقة والتفاصيل."""
    if not rows:
        return None

    correct = sum(1 for r in rows if r['pred'] == r['truth'])
    accuracy = correct / len(rows)

    truths = Counter(r['truth'] for r in rows)
    majority_lab, majority_n = truths.most_common(1)[0]

    per_class = {}
    for lab, c in truths.items():
        hits = sum(1 for r in rows if r['truth'] == lab and r['pred'] == lab)
        per_class[lab] = (hits, c)

    return {
        'n': len(rows),
        'hit': correct,
        'acc': accuracy,
        'majority': majority_n / len(rows),
        'majority_lab': majority_lab,
        'per_class': per_class,
    }


def main():
    print('=' * 72)
    print('  تجربة RADIUS مختلفة (بدون confidence filtering)')
    print('=' * 72)
    print(f'\n  RADII to try: {RADII_TO_TRY}')

    # جرّب كل RADIUS
    print(f'\n{"=" * 72}')
    print('  النتائج')
    print('=' * 72)

    results = {}

    for radius in RADII_TO_TRY:
        cross = segment_protocol(radius=radius)
        s = score(cross)
        results[radius] = s

        print(f'\n  RADIUS = {radius}:')
        print(f'    الدقة:   {s["acc"] * 100:5.1f}% ({s["hit"]}/{s["n"]})')
        print(f'    الأغلبية: {s["majority"] * 100:5.1f}% ✓' if s['acc'] > s['majority'] else f'    الأغلبية: {s["majority"] * 100:5.1f}% ✗')

        for lab, (h, c) in sorted(s['per_class'].items()):
            acc = h / c if c > 0 else 0
            print(f'      {lab:<12} {acc * 100:5.1f}%  ({h}/{c})')

    # قارن
    print(f'\n{"=" * 72}')
    print('  مقارنة')
    print('=' * 72)

    print(f'\n  {"RADIUS":>6}  {"Accuracy":>10}  {"vs Baseline":>12}')
    print(f'  {"-" * 40}')

    baseline = results[1]['acc']

    for radius in RADII_TO_TRY:
        acc = results[radius]['acc']
        diff = (acc - baseline) * 100
        symbol = '✅' if diff > 0 else '❌' if diff < 0 else '='
        print(f'  {radius:>6}  {acc * 100:>9.1f}%  {diff:+7.1f}%  {symbol}')

    # أفضل
    best_radius = max(results.keys(), key=lambda r: results[r]['acc'])
    best_score = results[best_radius]

    print(f'\n  ⭐ أفضل RADIUS: {best_radius} → {best_score["acc"] * 100:.1f}%')


if __name__ == '__main__':
    main()
