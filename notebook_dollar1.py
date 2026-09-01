"""
$1 Recognizer Baseline
"""

import numpy as np
import matplotlib.pyplot as plt

print("💾 تحميل البيانات...")

from baselines_common import (extract_templates, setup_sliding_windows, build_windows_for_baseline,
                               enforce_min_duration, moving_average_predictions, score_predictions,
                               print_results, sweep_rejection_threshold, print_threshold_sweep)

templates = extract_templates(X_train_sk, y_train_sk, le_skeleton, action_names)
print(f"\n✅ عدد الـ templates: {len(templates)}")


def resample_curve(curve, n_points=64):
    distances = np.sqrt(np.sum(np.diff(curve, axis=0)**2, axis=1))
    total_distance = np.sum(distances)
    if total_distance == 0:
        return np.repeat(curve[[0]], n_points, axis=0)
    cumsum = np.concatenate([[0], np.cumsum(distances)])
    new_indices = np.linspace(0, total_distance, n_points)
    resampled = np.zeros((n_points, curve.shape[1]), dtype=curve.dtype)
    for d in range(curve.shape[1]):
        resampled[:, d] = np.interp(new_indices, cumsum, curve[:, d])
    return resampled


def translate_curve(curve):
    return curve - np.mean(curve, axis=0)


def scale_curve(curve, width=100):
    min_val = np.min(curve, axis=0)
    max_val = np.max(curve, axis=0)
    range_val = max_val - min_val
    range_val[range_val == 0] = 1
    return (curve - min_val) / range_val * width - width / 2


def recognize_gesture_dollar1(curve, templates_dict, resample_n=64):
    resampled = resample_curve(curve, n_points=resample_n)
    wrist_traj = resampled[:, 20:22]
    if np.all(wrist_traj == wrist_traj[0]):
        rotated = resampled.copy()
    else:
        start, end = wrist_traj[0], wrist_traj[-1]
        angle = np.arctan2((end - start)[1], (end - start)[0])
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)
        rotated = np.zeros_like(resampled)
        center = np.mean(resampled, axis=0)
        for i in range(0, resampled.shape[1], 2):
            if i + 1 < resampled.shape[1]:
                point = resampled[:, i:i+2] - center[[i, i+1]]
                rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                rotated[:, i:i+2] = point @ rot_matrix.T
            else:
                rotated[:, i] = resampled[:, i] - center[i]

    translated = translate_curve(rotated)
    scaled = scale_curve(translated, width=100)

    distances = {}
    for action_name, template in templates_dict.items():
        t_resampled = resample_curve(template, n_points=resample_n)
        t_scaled = scale_curve(translate_curve(t_resampled), width=100)
        if len(scaled) != len(t_scaled):
            continue
        distances[action_name] = np.mean(np.sqrt(np.sum((scaled - t_scaled)**2, axis=1)))
    return distances


print("\n🎬 بناء نوافذ الفيديو...")
fps = 60
effective_fps = fps / 2
centers_sk, edges_sk, window_times_sk, WINDOW_FRAMES = setup_sliding_windows(
    all_keypoints, fps, effective_fps
)
windows = build_windows_for_baseline(all_keypoints, centers_sk, WINDOW_FRAMES, effective_fps)
print(f"✅ شكل النوافذ: {windows.shape}")

# ==============================================================================
# تطبيق $1 على كل نافذة + عتبة الرفض
# ==============================================================================

print("\n🔄 تطبيق $1 Recognizer على كل نافذة...")
predictions_raw = []
best_distances = []
for i, window in enumerate(windows):
    dist = recognize_gesture_dollar1(window, templates)
    if not dist:
        predictions_raw.append('other')
        best_distances.append(np.inf)
    else:
        best_action = min(dist, key=dist.get)
        predictions_raw.append(best_action)
        best_distances.append(dist[best_action])

best_distances = np.array(best_distances)
print(f"✅ توقعات لـ {len(predictions_raw)} نافذة")

ground_truth_segments = [
    (0.3,   4.4, 'rub_hands'), (4.4,   4.8, '?'), (4.8,  11.4, 'wave'),
    (11.4, 12.2, 'other*'), (12.2, 14.3, 'sit_down'), (14.3, 19.5, 'other*'),
    (19.5, 20.4, 'stand_up'), (20.4, 21.2, 'other*'), (21.2, 27.1, '?'),
    (27.1, 28.4, 'other*'), (28.4, 29.7, 'sit_down'), (29.7, 32.4, 'other*'),
    (32.4, 33.4, 'stand_up'), (33.4, 33.7, '?'), (33.7, 37.4, 'wave'),
]

base_score = score_predictions(['other'] * len(predictions_raw), edges_sk, ground_truth_segments)
baseline_acc = base_score['hit'] / max(1, base_score['total']) * 100

sweep_rows = sweep_rejection_threshold(best_distances, predictions_raw, edges_sk, ground_truth_segments)
print_threshold_sweep(sweep_rows, baseline_acc)

best_row = max(sweep_rows, key=lambda r: min(r['real_recall'], r['other_recall']))
REJECT_THRESHOLD = best_row['threshold']
print(f"\n✅ العتبة المختارة: {REJECT_THRESHOLD:.2f} "
      f"(real={best_row['real_recall']:.1f}%, other={best_row['other_recall']:.1f}%)")

predictions = ['other' if d > REJECT_THRESHOLD else a
               for a, d in zip(predictions_raw, best_distances)]

print("\n⏱️  تطبيق minimum duration filtering...")
predictions = enforce_min_duration(predictions, 2, protect='other')

print("✨ تنعيم زمني...")
predictions = moving_average_predictions(predictions, k=5)

print("\n📊 المقارنة مع الـ ground truth...")
print_results("$1 Recognizer Baseline", predictions, edges_sk, ground_truth_segments)

print("\n📈 رسم النتائج...")
fig, ax = plt.subplots(figsize=(16, 5))
unique_actions = list(set(predictions))
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_actions)))
action_color_map = dict(zip(unique_actions, colors))
for i in range(len(windows) - 1):
    ax.axvspan(edges_sk[i], edges_sk[i+1], color=action_color_map[predictions[i]], alpha=0.6)
for start, end, label in ground_truth_segments:
    ax.text((start+end)/2, 1.05, label, ha='center', fontsize=9, fontweight='bold',
             rotation=30, transform=ax.get_xaxis_transform())
    ax.axvline(start, color='k', lw=0.6, ls='--', alpha=0.4)
handles = [plt.Rectangle((0,0),1,1, color=action_color_map[a]) for a in unique_actions]
ax.legend(handles, unique_actions, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
ax.set_xlabel('Time (s)')
ax.set_yticks([])
ax.set_title('$1 Recognizer Baseline — vidtest1.mp4', fontsize=13)
plt.tight_layout()
plt.savefig('/kaggle/working/dollar1_timeline.png', dpi=120, bbox_inches='tight')
plt.show()
print("✅ الصورة اتحفظت: /kaggle/working/dollar1_timeline.png")