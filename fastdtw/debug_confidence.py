"""Debug: شوف قيم المسافات والـ confidence"""

import sys
from math import exp
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import _bootstrap  # noqa: F401

from ground_truth import VIDEOS, shared_labels, spans
from classifier import (NUM_FRAMES, balance_templates, cut_templates,
                          normalize_window, resample_linear, to_features,
                          norm_distance)
from paths import load_keypoints

SCALES = (1.0, 1.5, 2.0, 3.0)
MAX_PER_LABEL = 8
SHARED = tuple(shared_labels())
RADIUS = 1


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


def softmax(values):
    """تحويل المسافات إلى probabilities."""
    neg_values = [-v for v in values]
    max_val = max(neg_values)
    exp_values = [exp(v - max_val) for v in neg_values]
    sum_exp = sum(exp_values)
    return [e / sum_exp for e in exp_values]


# جرّب عينة واحدة
print("📊 Debug: شوف المسافات والـ confidence")
print("=" * 70)

v = 'vidtest1'
kp, fps = load(v)

# بني templates من فيديوهات تانية
srcs = [o for o in VIDEOS if o != v]
tmpl = build_templates(srcs, mode='vel', shape_norm=True)

print(f"\n  الـ templates: {len(tmpl)}")
for i, t in enumerate(tmpl[:5]):
    print(f"    {i}: {t['label']} (from {t.get('video', '?')})")

# اختبر على عينة واحدة
lab = 'stand_up'
span_list = spans(v, lab)
s, e = span_list[0]

feat = featurize(kp, fps, s, e, mode='vel', shape_norm=True)

print(f"\n  العينة: {v} @ {s:.1f}-{e:.1f} ({lab})")

# احسب المسافات
distances = []
for t in tmpl:
    try:
        d = norm_distance(feat, t['feat'], radius=RADIUS)
        distances.append(d)
    except:
        distances.append(float('inf'))

print(f"\n  المسافات (أول 10):")
for i, d in enumerate(distances[:10]):
    label = tmpl[i]['label']
    print(f"    {i}: {label:<12} d={d:8.4f}")

# احسب softmax
confidences = softmax(distances)

print(f"\n  Confidence scores (softmax(-distances)):")
for i, (d, c) in enumerate(zip(distances[:10], confidences[:10])):
    label = tmpl[i]['label']
    print(f"    {i}: {label:<12} c={c:.4f} (d={d:8.4f})")

print(f"\n  Max confidence: {max(confidences):.4f}")
print(f"  Min confidence: {min(confidences):.4f}")
print(f"  Mean confidence: {np.mean(confidences):.4f}")

best_idx = np.argmin(distances)
print(f"\n  ✅ أقرب template:")
print(f"    Index: {best_idx}")
print(f"    Label: {tmpl[best_idx]['label']}")
print(f"    Distance: {distances[best_idx]:.4f}")
print(f"    Confidence: {confidences[best_idx]:.4f}")

print("\n" + "=" * 70)
print("المشكلة: المسافات كبيرة جداً، الـ confidence كلها قليلة؟")
print("أم المشكلة في softmax implementation؟")
