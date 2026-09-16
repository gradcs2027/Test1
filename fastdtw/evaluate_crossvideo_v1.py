"""
FastDTW — TIER 1: Parameter Tuning + Confidence Scores

تحسينات على evaluate_crossvideo.py:
  1. جرّب RADIUS مختلفة [1, 2, 3, 4]
  2. أضف confidence scores (softmax)
  3. Reject low-confidence predictions

النتيجة المتوقعة: 41.7% → 45-50%
"""

import sys
from collections import Counter, defaultdict
from math import exp

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

# 🆕 معاملات جديدة
RADII_TO_TRY = [1, 2, 3, 4]  # جرّب RADIUS مختلفة
CONFIDENCE_RATIO_THRESHOLD = 1.05  # رفض إذا (best/second_best) < 1.05


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


def confidence_ratio(distances):
    """احسب confidence كـ ratio بين أقرب وثاني أقرب.

    confidence_ratio = d_best / d_second_best

    High ratio (> 1.2) = confident prediction
    Low ratio (< 1.2) = ambiguous prediction
    """
    sorted_dist = sorted([d for d in distances if d != float('inf')])

    if len(sorted_dist) < 2:
        return 1.0

    best = sorted_dist[0]
    second = sorted_dist[1]

    if second == 0:
        return 1.0

    # ratio = second / best (كبر الرقم = أحسن)
    return second / best if best > 0 else 1.0


def nearest_with_confidence(feat, templates, radius=1):
    """أقرب template + confidence ratio.

    Returns:
      (label, distance, confidence_ratio)
    """
    # احسب المسافات
    distances = []
    for t in templates:
        try:
            d = norm_distance(feat, t['feat'], radius=radius)
            distances.append(d)
        except:
            distances.append(float('inf'))

    if not distances or all(d == float('inf') for d in distances):
        return None, float('inf'), 0.0

    # احسب confidence ratio
    conf_ratio = confidence_ratio(distances)

    # أقرب template
    best_idx = int(np.argmin(distances))

    return templates[best_idx]['label'], distances[best_idx], conf_ratio


def segment_protocol(radius=1, mode='vel', shape_norm=True, within=False):
    """الـ protocol الأساسي مع confidence scores."""
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

                # 🆕 استعمل nearest_with_confidence
                pred, dist, conf = nearest_with_confidence(feat, tmpl, radius=radius)

                rows.append({
                    'video': v, 'span': (s, e), 'truth': lab,
                    'pred': pred, 'dist': dist, 'confidence': conf,
                    'n_tmpl': len(tmpl)
                })
    return rows


def score(rows, ratio_threshold=1.2):
    """احسب الدقة مع reject option بـ confidence ratio.

    إذا confidence_ratio < ratio_threshold → عتبر prediction "ambiguous"
    (لا نرفضها تماماً، بس نسجلها)
    """
    if not rows:
        return None

    # عد الصحيح والخطأ والـ ambiguous
    correct = 0
    ambiguous = 0

    for r in rows:
        if r['confidence'] < ratio_threshold:
            ambiguous += 1
        elif r['pred'] == r['truth']:
            correct += 1

    # الدقة = صحيح / total (حسبها على كل العينات)
    accuracy = correct / len(rows)

    truths = Counter(r['truth'] for r in rows)
    majority_lab, majority_n = truths.most_common(1)[0]

    return {
        'n': len(rows),
        'hit': correct,
        'ambiguous': ambiguous,
        'acc': accuracy,
        'majority': majority_n / len(rows),
        'majority_lab': majority_lab,
        'chance': 1 / len(truths),
        'per_class': {lab: (sum(1 for r in rows
                                if r['truth'] == lab and r['pred'] == lab
                                and r['confidence'] >= ratio_threshold), c)
                      for lab, c in truths.items()},
    }


def print_score(title, s, ratio_threshold=1.2, indent='  '):
    if s is None:
        print(f'{indent}{title}: مافيش عيّنات')
        return

    verdict = '✅' if s['acc'] > s['majority'] else '❌'
    print(f'{indent}{title}')
    print(f'{indent}  الدقة            {s["acc"] * 100:5.1f}%  '
          f'({s["hit"]}/{s["n"]})')
    print(f'{indent}  الـ Ambiguous     {s["ambiguous"]} (ratio < {ratio_threshold})')
    print(f'{indent}  خط أساس الأغلبية {s["majority"] * 100:5.1f}%  {verdict}')
    print(f'{indent}  الصدفة           {s["chance"] * 100:5.1f}%')


def main():
    print('=' * 72)
    print('  🔧 TIER 1: Parameter Tuning — جرّب RADIUS مختلفة + Confidence Ratio')
    print('=' * 72)
    print(f'\n  الحركات المشتركة: {", ".join(SHARED)}')
    print(f'  Confidence ratio threshold: {CONFIDENCE_RATIO_THRESHOLD}')
    print(f'  RADII to try: {RADII_TO_TRY}')

    # ---------- جرّب كل RADIUS ----------
    print(f'\n{"=" * 72}')
    print('  نتائج تجربة RADIUS مختلفة')
    print('=' * 72)

    results_by_radius = {}

    for radius in RADII_TO_TRY:
        print(f'\n  ▶ RADIUS = {radius}')
        print(f'  {"-" * 68}')

        cross = segment_protocol(radius=radius)
        s_cross = score(cross, ratio_threshold=CONFIDENCE_RATIO_THRESHOLD)
        results_by_radius[radius] = s_cross

        print_score(f'    Accuracy', s_cross, ratio_threshold=CONFIDENCE_RATIO_THRESHOLD,
                   indent='    ')

    # ---------- قارن النتائج ----------
    print(f'\n{"=" * 72}')
    print('  📊 مقارنة النتائج')
    print('=' * 72)

    print(f'\n  {"RADIUS":>6}  {"Accuracy":>10}  {"Hit":>6}  {"Ambiguous":>10}  {"Better?"}')
    print(f'  {"-" * 62}')

    baseline_acc = results_by_radius[1]['acc']

    for radius in RADII_TO_TRY:
        s = results_by_radius[radius]
        is_better = "✅" if s['acc'] > baseline_acc else ""
        print(f'  {radius:>6}  {s["acc"] * 100:>9.1f}%  {s["hit"]:>6}  '
              f'{s["ambiguous"]:>10}  {is_better}')

    # أفضل RADIUS
    best_radius = max(results_by_radius.keys(),
                     key=lambda r: results_by_radius[r]['acc'])
    best_score = results_by_radius[best_radius]

    print(f'\n  ⭐ أفضل RADIUS: {best_radius} → {best_score["acc"] * 100:.1f}% '
          f'(من {baseline_acc * 100:.1f}%)')

    if best_score['acc'] > baseline_acc:
        improvement = (best_score['acc'] - baseline_acc) * 100
        print(f'     تحسّن: +{improvement:.1f}%')

    # ---------- التفاصيل للأفضل ----------
    print(f'\n{"=" * 72}')
    print(f'  التفاصيل — RADIUS = {best_radius} (الأفضل)')
    print('=' * 72)

    cross = segment_protocol(radius=best_radius)
    s = score(cross, ratio_threshold=CONFIDENCE_RATIO_THRESHOLD)

    print(f'\n  الدقة لكل حركة:')
    for lab, (h, c) in sorted(s['per_class'].items()):
        if c > 0:
            a = h / c
            print(f'    {lab:<12} {a * 100:5.1f}%  {"█" * int(a * 24):<24} '
                  f'({h}/{c})')

    print(f'\n  ملخص:')
    print(f'    • الـ Baseline (RADIUS=1): {baseline_acc * 100:.1f}%')
    print(f'    • الأفضل (RADIUS={best_radius}): {best_score["acc"] * 100:.1f}%')
    print(f'    • الـ Ambiguous predictions: {best_score["ambiguous"]}')
    print(f'    • Confidence ratio threshold: {CONFIDENCE_RATIO_THRESHOLD}')


if __name__ == '__main__':
    main()
