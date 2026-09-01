"""
DTW (Dynamic Time Warping) Baseline
"""

import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

print("💾 تحميل البيانات...")

from baselines_common import (extract_templates, setup_sliding_windows, build_windows_for_baseline,
                               enforce_min_duration, moving_average_predictions, score_predictions,
                               print_results, sweep_rejection_threshold, print_threshold_sweep)

templates = extract_templates(X_train_sk, y_train_sk, le_skeleton, action_names)
print(f"\n✅ عدد الـ templates: {len(templates)}")
print(f"✅ الأكشنات المتاحة: {list(templates.keys())}")

print("\n🎬 بناء نوافذ الفيديو...")
fps = 60
effective_fps = fps / 2

centers_sk, edges_sk, window_times_sk, WINDOW_FRAMES = setup_sliding_windows(
    all_keypoints, fps, effective_fps
)
windows = build_windows_for_baseline(all_keypoints, centers_sk, WINDOW_FRAMES, effective_fps)
print(f"✅ شكل النوافذ: {windows.shape}")


def dtw_distance(seq1, seq2):
    n, m = len(seq1), len(seq2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = euclidean(seq1[i - 1], seq2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]
            )
    return dtw_matrix[n, m]


print("\n🔄 حساب DTW لكل نافذة...")
distances = {}
for action_name in templates.keys():
    template = templates[action_name]
    distances[action_name] = [dtw_distance(w, template) for w in windows]
    print(f"   {action_name}: تم")

# ==============================================================================
# اختيار الإجابة + عتبة الرفض
# ==============================================================================

print("\n🎯 اختيار الإجابات...")

predictions_raw = []
best_distances = []
for window_idx in range(len(windows)):
    scores = {action: distances[action][window_idx] for action in templates.keys()}
    best_action = min(scores, key=scores.get)
    predictions_raw.append(best_action)
    best_distances.append(scores[best_action])

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
print_results("DTW Baseline", predictions, edges_sk, ground_truth_segments)

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
ax.set_title('DTW Baseline — vidtest1.mp4', fontsize=13)
plt.tight_layout()
plt.savefig('/kaggle/working/dtw_timeline.png', dpi=120, bbox_inches='tight')
plt.show()
print("✅ الصورة اتحفظت: /kaggle/working/dtw_timeline.png")