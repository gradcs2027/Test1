"""
مشترك بين الـ DTW و $1 Recognizer baselines.
- استخرج templates
- normalize windows
- sliding window setup
"""

import numpy as np
from scipy.spatial.distance import euclidean
from collections import defaultdict

# ==============================================================================
# Templates Extraction — استخرج متوسط كل حركة من training data
# ==============================================================================

def extract_templates(X_train_sk, y_train_sk, le_skeleton, action_names):
    """
    بياخد X_train_sk (samples, 30, 34) و y_train_sk (samples, num_classes)
    ويطلع dict بفيه متوسط كل حركة.

    لو عندك 100 عينة sit_down، بيجمعهم ويقسم على 100.
    النتيجة: شكل "متوسط" لـ sit_down.
    """
    templates = {}

    # تحويل من one-hot لـ indices
    y_indices = np.argmax(y_train_sk, axis=1)

    for class_idx, class_name in enumerate(le_skeleton.classes_):
        mask = y_indices == class_idx
        if mask.sum() > 0:
            # متوسط كل العينات بتاعة الكلاس ده
            template = X_train_sk[mask].mean(axis=0)
            action_name = action_names.get(int(class_name), f"class_{class_name}")
            templates[action_name] = template
            print(f"✅ Template {action_name}: {template.shape} ({mask.sum()} عينة)")

    return templates


def normalize_for_baseline(keypoints):
    """
    تطبيع سريع لنافذة — نفس اللي في التدريب.
    keypoints: (frames, 34) بعد الاستخراج من الفيديو
    """
    from cell3_normalization import normalize_skeleton

    # تحويل لـ (frames, 17, 2) عشان normalize_skeleton تتوقعها كذا
    kp_reshaped = keypoints.reshape(-1, 17, 2)
    # تطبيع
    normalized = normalize_skeleton(kp_reshaped)
    # ضغط لـ 30 فريم (نفس التدريب بالظبط)
    indices = np.linspace(0, len(normalized) - 1, 30, dtype=int)
    return normalized[indices]  # (30, 34)


# ==============================================================================
# Sliding Window Setup — نوافذ الفيديو
# ==============================================================================

def setup_sliding_windows(all_keypoints, fps, effective_fps=30.0):
    """
    بناء مراكز النوافذ (centers_sk) و حدودها (edges_sk).
    نفس الـ setup بتاع LSTM بالظبط.
    """
    WINDOW_SECONDS = 3.0
    WINDOW_FRAMES = int(round(WINDOW_SECONDS * effective_fps))
    STRIDE = 10

    T_frames = len(all_keypoints)
    half_base = WINDOW_FRAMES // 2
    centers_sk = np.arange(half_base, T_frames - half_base, STRIDE)

    # حدود النوافذ
    edges_sk = np.empty(len(centers_sk) + 1)
    edges_sk[0] = 0
    if len(centers_sk) > 1:
        edges_sk[1:-1] = (np.arange(len(centers_sk) - 1) + np.arange(1, len(centers_sk))) / 2 * (STRIDE / effective_fps)
    edges_sk[-1] = (T_frames - 1) / effective_fps

    frame_indices = np.arange(T_frames)
    window_times_sk = frame_indices[centers_sk] / effective_fps

    print(f"✅ Centers: {len(centers_sk)} نافذة")
    print(f"✅ Edges: {edges_sk[0]:.1f}s لـ {edges_sk[-1]:.1f}s")

    return centers_sk, edges_sk, window_times_sk, WINDOW_FRAMES


def build_windows_for_baseline(all_keypoints, centers_sk, WINDOW_FRAMES, effective_fps=30.0):
    """
    بناء النوافذ من الفيديو.
    كل نافذة = WINDOW_FRAMES فريم مضغوطة لـ 30.
    النتيجة: (len(centers_sk), 30, 34)
    """
    MODEL_FRAMES = 30
    T_frames = len(all_keypoints)
    half = WINDOW_FRAMES // 2

    windows = np.empty((len(centers_sk), MODEL_FRAMES, 34), dtype=np.float32)

    for i, c in enumerate(centers_sk):
        lo = max(0, c - half)
        hi = min(T_frames, c + half + 1)
        # ضغط للـ 30 فريم بالظبط زي التدريب
        sub = np.linspace(lo, hi - 1, MODEL_FRAMES, dtype=int)

        from cell3_normalization import normalize_skeleton
        windows[i] = normalize_skeleton(all_keypoints[sub])

    return windows  # (num_windows, 30, 34)


# ==============================================================================
# Minimum Duration Filtering — شيل الحركات القصيرة
# ==============================================================================

def enforce_min_duration(labels, min_len, protect=None):
    """
    أي حركة ظهرت أقل من min_len نافذة بنشيلها.
    protect: كلاس محمي (مثلاً 'other').
    """
    labels = list(labels)
    i = 0
    while i < len(labels):
        j = i
        while j < len(labels) and labels[j] == labels[i]:
            j += 1
        if j - i < min_len and i > 0 and labels[i] != protect:
            labels[i:j] = [labels[i - 1]] * (j - i)
        i = j
    return labels


# ==============================================================================
# Moving Average Smoothing — تنعيم زمني
# ==============================================================================

def moving_average_predictions(predictions, k=5):
    """
    تنعيم الإجابات باستخدام متوسط متحرك.
    predictions: list من الأكشن (strings)
    k: عدد النوافذ المستخدمة (خمسة = نافذتين قبل وبعد + الحالية)
    """
    if k <= 1:
        return predictions

    # Convert to indices for averaging
    unique_actions = sorted(set(predictions))
    action_to_idx = {a: i for i, a in enumerate(unique_actions)}
    idx_to_action = {i: a for a, i in action_to_idx.items()}

    indices = np.array([action_to_idx[p] for p in predictions])

    # Moving average on indices
    pad = k // 2
    padded = np.pad(indices, pad, mode='edge')
    kernel = np.ones(k) / k

    smoothed_indices = np.array([
        np.round(np.convolve(padded, kernel, mode='valid'))[i]
        for i in range(len(indices))
    ], dtype=int)

    return [idx_to_action[int(i)] for i in smoothed_indices]


# ==============================================================================
# Timeline Visualization — نفس الرسم بتاع LSTM
# ==============================================================================

def score_predictions(predicted_actions, edges_sk, ground_truth_segments, step=0.1):
    """
    نفس الـ scoring بتاع LSTM بالظبط.

    Args:
        predicted_actions: list من الـ actions
        edges_sk: حدود النوافذ (من الوقت)
        ground_truth_segments: list من (start, end, label)
        step: خطوة العينة (0.1s)

    Returns:
        dict بـ metrics
    """
    result = {
        'total': 0, 'hit': 0,
        'other_total': 0, 'other_hit': 0,
        'real_total': 0, 'real_hit': 0,
        'other_false': 0, 'skip': 0
    }

    # Sample times
    t_min = edges_sk[0]
    t_max = edges_sk[-1]
    times = np.arange(t_min, t_max, step)

    for t in times:
        # الـ GT عند الوقت ده
        gt = None
        for start, end, label in ground_truth_segments:
            if start <= t < end:
                gt = label
                break

        # Window index
        w = int(np.searchsorted(edges_sk, t, side='right') - 1)

        # تخطي لو مش في الـ GT أو خارج النطاق
        if gt is None or gt.endswith('?') or not (0 <= w < len(predicted_actions)):
            result['skip'] += 1
            continue

        # تحويل 'name*' لـ 'other'
        gt_label = 'other' if gt.endswith('*') else gt
        pred = predicted_actions[w]

        result['total'] += 1
        result['hit'] += (pred == gt_label)

        if gt_label == 'other':
            result['other_total'] += 1
            result['other_hit'] += (pred == gt_label)
        else:
            result['real_total'] += 1
            result['real_hit'] += (pred == gt_label)
            result['other_false'] += (pred == 'other')

    return result


def print_results(name, predicted_actions, edges_sk, ground_truth_segments, baseline_only=False):
    """
    طبع النتايج بنفس الشكل بتاع LSTM.
    """
    print(f"\n{'='*70}")
    print(f"📊 {name}")
    print(f"{'='*70}")

    # Scoring
    cur = score_predictions(predicted_actions, edges_sk, ground_truth_segments)
    base = score_predictions(['other'] * len(predicted_actions), edges_sk, ground_truth_segments)

    if cur['total'] == 0:
        print("⚠️ لا توقعات في نطاق الـ ground truth")
        return

    print(f"\n📏 مقارنة بالـ ground truth (عينة كل 0.1s):")
    print(f"   دقة إجمالية:     {cur['hit']:3d}/{cur['total']:<3d} = {cur['hit']/cur['total']*100:5.1f}%")
    print(f"   خط الأساس:       {base['hit']:3d}/{base['total']:<3d} = {base['hit']/base['total']*100:5.1f}%")
    print(f"   other recall:    {cur['other_hit']:3d}/{cur['other_total']:<3d} = {cur['other_hit']/max(1,cur['other_total'])*100:5.1f}%")
    print(f"   real recall:     {cur['real_hit']:3d}/{cur['real_total']:<3d} = {cur['real_hit']/max(1,cur['real_total'])*100:5.1f}%")

    if baseline_only:
        print(f"\n✅ النتيجة النهائية: {cur['hit']/cur['total']*100:.1f}% (مقابل LSTM: 71.8%)")



# ==============================================================================
# Rejection Threshold Sweep — بندوّر على "عتبة معرفش" الأنسب
# ==============================================================================

def sweep_rejection_threshold(best_distances, best_actions, edges_sk, ground_truth_segments,
                               thresholds=None):
    """
    بتجرب عدة عتبات مسافة، ولكل واحدة تستبدل أي نافذة مسافتها أكبر من
    العتبة بـ 'other'، وتقيس النتيجة مقابل الـ ground truth.
    """
    best_distances = np.asarray(best_distances, dtype=float)
    finite = best_distances[np.isfinite(best_distances)]

    if thresholds is None:
        lo, hi = np.percentile(finite, [5, 95]) if len(finite) else (0, 1)
        thresholds = np.linspace(lo, hi, 15)

    rows = []
    for th in thresholds:
        labels = [a if d <= th else 'other' for a, d in zip(best_actions, best_distances)]
        s = score_predictions(labels, edges_sk, ground_truth_segments)
        rows.append({
            'threshold': th,
            'overall': s['hit'] / max(1, s['total']) * 100,
            'other_recall': s['other_hit'] / max(1, s['other_total']) * 100,
            'real_recall': s['real_hit'] / max(1, s['real_total']) * 100,
            'other_pct': labels.count('other') / len(labels) * 100,
        })
    return rows


def print_threshold_sweep(rows, baseline_acc):
    print(f"\n🎚️ العتبة مقابل الـ ground truth (خط الأساس {baseline_acc:.1f}%):")
    print(f"   {'عتبة':>8} {'إجمالي':>8} {'other':>8} {'real':>8}  {'other%':>7}")
    for r in rows:
        print(f"   {r['threshold']:8.2f} {r['overall']:7.1f}% "
              f"{r['other_recall']:7.1f}% {r['real_recall']:7.1f}% {r['other_pct']:6.0f}%")