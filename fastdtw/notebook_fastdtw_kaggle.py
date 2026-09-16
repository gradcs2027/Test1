"""
🎬 FastDTW على Kaggle — بدون أي dataset

الاستخدام:
    شُغّل الملف ده في خلية واحدة في Kaggle notebook

الخطوات:
    1. استخرج الـ keypoints من الفيديوهات (اختياري)
    2. حمّل الـ keypoints المحفوظة
    3. شغّل FastDTW (تجربة النوافذ)
    4. اعرض النتائج
"""

import sys
from pathlib import Path

# ✓ لازم يكون أول import
import _bootstrap  # noqa: F401

import numpy as np
from ground_truth import GROUND_TRUTH, VIDEOS, shared_labels, spans
from classifier import (NUM_FRAMES, balance_templates, cut_templates,
                          normalize_window, norm_distance, resample_linear,
                          to_features)
from paths import load_keypoints

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SCALES = (1.0, 1.5, 2.0, 3.0)
STRIDE = 5
RADIUS = 1
MAX_PER_LABEL = 8

SHARED = tuple(shared_labels())


# ==============================================================================
# التطبيق الكامل
# ==============================================================================

def main():
    """الكود الكامل — شُغّله في خلية واحدة"""

    print("\n" + "="*70)
    print("🎬 FastDTW على Kaggle — الاختبار النظيف")
    print("="*70)

    # 1️⃣ حمّل الـ keypoints
    print("\n1️⃣ تحميل الـ keypoints...")
    videos_data = {}
    for video in VIDEOS:
        try:
            kp, fps = load_keypoints(video)
            videos_data[video] = (kp, fps)
            print(f"   ✓ {video}: {len(kp)} فريم @ {fps} fps")
        except Exception as e:
            print(f"   ✗ {video}: {e}")

    # 2️⃣ بني بنك الـ templates
    print("\n2️⃣ بناء بنك الـ templates...")
    templates = []
    for v in VIDEOS:
        if v not in videos_data:
            continue
        kp, fps = videos_data[v]
        clips = [(s, e, lab) for lab in SHARED for s, e in spans(v, lab)]

        print(f"   {v}: {len(clips)} قصاصة")

        for t0, t1, label in clips:
            lo, hi = int(round(t0 * fps)), int(round(t1 * fps))
            kp_clip = kp[max(0, lo):min(len(kp), hi)]

            if len(kp_clip) < 10:
                continue

            kp_norm = normalize_window(kp_clip)
            feat = to_features(kp_norm, mode='vel', shape_norm=True)

            # متعددة المقاسات
            for scale in SCALES:
                target_len = int(30 * scale)
                if len(feat) > 1:
                    feat_scaled = resample_linear(feat, n=target_len)
                else:
                    feat_scaled = feat

                templates.append({
                    'label': label,
                    'video': v,
                    'span': (t0, t1),
                    'features': feat_scaled,
                })

    # توازن الـ templates
    templates = balance_templates(templates, MAX_PER_LABEL)
    print(f"\n   📚 إجمالي templates: {len(templates)}")

    # 3️⃣ اختبر النوافذ
    print("\n3️⃣ الاختبار النظيف (عبر الفيديوهات)...")

    predictions = {}
    for v in VIDEOS:
        if v not in videos_data:
            continue

        kp, fps = videos_data[v]
        predictions[v] = []

        for label in SHARED:
            for t0, t1 in spans(v, label):
                # النافذة
                lo, hi = int(round(t0 * fps)), int(round(t1 * fps))

                for w_dur in (1.5, 2.0, 3.0, 4.0):
                    for offset in range(0, hi - lo, STRIDE):
                        w_lo = lo + offset
                        w_hi = w_lo + int(round(w_dur * fps))

                        if w_hi > hi:
                            break

                        window_clip = kp[w_lo:w_hi]
                        if len(window_clip) < 5:
                            continue

                        kp_norm = normalize_window(window_clip)
                        feat = to_features(kp_norm, mode='vel', shape_norm=True)

                        # احسب المسافات
                        dists = {}
                        for i, tmpl in enumerate(templates):
                            try:
                                d = norm_distance(feat, tmpl['features'],
                                                 radius=RADIUS)
                                dists[i] = d
                            except:
                                dists[i] = float('inf')

                        # أقرب template
                        if dists:
                            best_idx = min(dists, key=dists.get)
                            pred_label = templates[best_idx]['label']
                            predictions[v].append({
                                'true': label,
                                'pred': pred_label,
                                'dist': dists[best_idx],
                            })

    # 4️⃣ احسب الدقة
    print("\n4️⃣ النتائج...")

    all_preds = []
    for v in predictions:
        all_preds.extend(predictions[v])

    if all_preds:
        correct = sum(1 for p in all_preds if p['true'] == p['pred'])
        total = len(all_preds)
        accuracy = correct / total * 100 if total > 0 else 0

        print(f"\n   ✓ الصح: {correct} من {total}")
        print(f"   📊 الدقة: {accuracy:.1f}%")

        # كسر بكل حركة
        print("\n   بكل حركة:")
        for label in SHARED:
            label_preds = [p for p in all_preds if p['true'] == label]
            if label_preds:
                label_correct = sum(1 for p in label_preds if p['pred'] == label)
                label_acc = label_correct / len(label_preds) * 100
                print(f"     {label}: {label_correct}/{len(label_preds)} = {label_acc:.1f}%")

    print("\n" + "="*70)
    print("✅ خلص!")
    print("="*70)


if __name__ == '__main__':
    main()
