"""
DTW (Dynamic Time Warping) Baseline — ⚠️ نسخة قديمة، استخدم notebook_fastdtw.py

⚠️⚠️ تحذير منهجي — متعرضش نتيجة الملف ده للدكتور:
   الكود ده بياخد min(distances) على طول من غير أي rejection، يعني
   **مستحيل** يقول 'other'. والفيديوهات معظمها 'other'
   (vidtest3: ~120 ثانية من 126.5 هي سكون).
   يعني الرقم اللي هيطلع منه واطي مش لأن الـ DTW ضعيفة، لأننا مانعناها
   تجاوب صح. لو عرضت الرقم ده كمقارنة عادلة، ده تضليل.

   الملف ده متسيب للمرجعية بس (فيه الـ DTW الكامل O(n²) مشروح خطوة خطوة).
   للنتايج الحقيقية استخدم: notebook_fastdtw.py — فيه rejection معاير من
   بيانات التدريب، يوازي الـ CONF_REJECT بتاع الـ LSTM.

التعريف:
  يقارن بين سلسلتين من الأرقام بأطوال مختلفة بـ "warp" ديناميكي.
  النقاط بتنزلق عشان تطابق الأشكال، مش بتقف عند حد معين.

الهدف:
  استخرج template (متوسط) من كل حركة في NTU dataset.
  على الفيديو، قارن كل نافذة بالـ templates بـ DTW.

النتيجة المتوقعة:
  أقل من LSTM (40-50% تقريباً) لأن template واحد ما بيكفيش للتنوع البشري.
"""

import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

# ==============================================================================
# ١. تحميل البيانات
# ==============================================================================

print("💾 تحميل البيانات...")

# بناء على الكود الموجود بتاعك
# (انت بتشتغل على Kaggle، فالبيانات محملة خلال الحل)
# هنا بنفترض إن:
#   - X_train_sk, y_train_sk متحملين
#   - le_skeleton متحمل
#   - action_names متعرّفة
#   - model_skeleton محمّل (للاستخراج بس للمقارنة لاحقاً)

# أول حاجة: استخرج الـ templates من training data
from baselines_common import extract_templates, setup_sliding_windows, build_windows_for_baseline
from baselines_common import enforce_min_duration, moving_average_predictions, score_predictions, print_results

templates = extract_templates(X_train_sk, y_train_sk, le_skeleton, action_names)

print(f"\n✅ عدد الـ templates: {len(templates)}")
print(f"✅ الأكشنات المتاحة: {list(templates.keys())}")

# ==============================================================================
# ٢. بناء النوافذ (sliding window على الفيديو)
# ==============================================================================

print("\n🎬 بناء نوافذ الفيديو...")

fps = 60  # vidtest1 هو 60fps (تحقق من cell7_video.py)
effective_fps = fps / 2  # FRAME_SKIP = 2

centers_sk, edges_sk, window_times_sk, WINDOW_FRAMES = setup_sliding_windows(
    all_keypoints, fps, effective_fps
)

# بناء النوافذ
windows = build_windows_for_baseline(all_keypoints, centers_sk, WINDOW_FRAMES, effective_fps)
print(f"✅ شكل النوافذ: {windows.shape}")  # (num_windows, 30, 34)

# ==============================================================================
# ٣. تنفيذ DTW
# ==============================================================================

def dtw_distance(seq1, seq2):
    """
    Dynamic Time Warping — حساب المسافة بين سلسلتين.

    الفكرة:
      matrix[i, j] = أقل مسافة من أول seq1[:i] لـ seq2[:j]

    يعني:
      seq1[i] ممكن يطابق seq2[j] أو seq2[j+1] (warp)
      بدل ما يطابق seq1[i] مع seq2[i] بالضبط.
    """
    n, m = len(seq1), len(seq2)

    # initialize matrix
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    # fill the matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # المسافة بين النقاط (euclidean في الـ 34 بعد)
            cost = euclidean(seq1[i - 1], seq2[j - 1])

            # أقل طريق من الثلاث اتجاهات (قطري، أفقي، رأسي)
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],      # من فوق
                dtw_matrix[i, j - 1],      # من الشمال
                dtw_matrix[i - 1, j - 1]   # من القطر
            )

    return dtw_matrix[n, m]


print("\n🔄 حساب DTW لكل نافذة...")

# حساب المسافة من كل نافذة لكل template
distances = {}
for action_name in templates.keys():
    template = templates[action_name]
    distances[action_name] = []

    for i, window in enumerate(windows):
        dist = dtw_distance(window, template)
        distances[action_name].append(dist)

        if (i + 1) % max(1, len(windows) // 4) == 0 or i == len(windows) - 1:
            print(f"   {action_name}: تم {i+1}/{len(windows)}")

# ==============================================================================
# ٤. اختيار الإجابة (أقل مسافة)
# ==============================================================================

print("\n🎯 اختيار الإجابات...")

predictions = []
confidences = []

for window_idx in range(len(windows)):
    # أقل مسافة = أقرب match
    scores = {action: distances[action][window_idx] for action in templates.keys()}
    best_action = min(scores, key=scores.get)
    best_distance = scores[best_action]

    # الثقة = معكوس المسافة (كلما أقل المسافة، كلما أعلى الثقة)
    # normalize بين 0 و 1
    all_distances = list(scores.values())
    min_dist, max_dist = min(all_distances), max(all_distances)
    if max_dist > min_dist:
        confidence = 1 - (best_distance - min_dist) / (max_dist - min_dist)
    else:
        confidence = 1.0

    predictions.append(best_action)
    confidences.append(confidence)

print(f"✅ توقعات لـ {len(predictions)} نافذة")

# ==============================================================================
# ٥. تطبيق Minimum Duration Filtering
# ==============================================================================

print("\n⏱️  تطبيق minimum duration filtering...")

MIN_SEGMENT = 2
predictions = enforce_min_duration(predictions, MIN_SEGMENT, protect='other')

# ==============================================================================
# ٦. التنعيم الزمني (Moving Average)
# ==============================================================================

print("✨ تنعيم زمني...")

predictions = moving_average_predictions(predictions, k=5)

# ==============================================================================
# ٧. المقارنة مع Ground Truth
# ==============================================================================

print("\n📊 المقارنة مع الـ ground truth...")

# Ground truth بتاع vidtest1 (من cell10_timeline.py)
# ⚠️ الجدول ده كان مكتوب هنا بإيد — واحدة من 5 نسخ متطابقة في المشروع.
#    اتشال 2026-08-31 وبقى بييجي من مصدر واحد. السبب: نسخة vidtest3
#    عاشت شهر وهي غلط تماماً من غير ما حد ياخد باله. HANDOFF قسم 6.8
from ground_truth import GROUND_TRUTH

ground_truth_segments = GROUND_TRUTH['vidtest1']

# طبع النتايج
print_results("DTW Baseline", predictions, edges_sk, ground_truth_segments)

# ==============================================================================
# ٨. رسم النتائج (optional)
# ==============================================================================

print("\n📈 رسم النتائج...")

fig, ax = plt.subplots(figsize=(16, 5))

# رسم الإجابات
unique_actions = list(set(predictions))
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_actions)))
action_color_map = dict(zip(unique_actions, colors))

for i in range(len(windows) - 1):
    ax.axvspan(edges_sk[i], edges_sk[i+1],
               color=action_color_map[predictions[i]], alpha=0.6)

# رسم الـ ground truth
for start, end, label in ground_truth_segments:
    ax.text((start+end)/2, 1.05, label, ha='center', fontsize=9,
            fontweight='bold', rotation=30, transform=ax.get_xaxis_transform())
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

# ==============================================================================
# ٩. الخلاصة
# ==============================================================================

print("\n" + "="*70)
print("🔍 الملاحظات:")
print("="*70)
print("""
DTW Baseline بتقارن كل نافذة من الفيديو بـ template واحد من كل حركة.

المشاكل:
  1. Template واحد ما بيكفيش للتنوع البشري
  2. لا يأخذ في الاعتبار السياق (الحركة قبل وبعد)
  3. القطع الناعمة (soft parts) مثل وضع الذراع معقدة

الفائدة:
  ✅ سريع (بدون تدريب)
  ✅ سهل الفهم
  ❌ دقة أقل من LSTM

المتوقع: 40-50% دقة
الفعلي: تحقق من النتيجة أعلاه!
""")

print("="*70)
