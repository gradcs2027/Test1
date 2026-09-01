"""
$1 Recognizer Baseline

التعريف:
  خوارزمية للتعرف على الأشكال المرسومة بالإيد.
  بتقلل الحركات لنقاط كم كم محددة وتقارن الأشكال مباشرة.

التطبيق على مشروعنا:
  بدل رسمة واحدة (stroke واحد)، عندنا 17 مفصل (17 "strokes").
  الفكرة: خذ مسار كل مفصل، طبّق $1 على كل واحد، واجمع النتايج.

النتيجة المتوقعة:
  أقل من DTW (20-30% تقريباً) لأن معلومة واحد مفصل ناقصة جداً.
"""

import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

# ==============================================================================
# ١. تحميل البيانات
# ==============================================================================

print("💾 تحميل البيانات...")

from baselines_common import extract_templates, setup_sliding_windows, build_windows_for_baseline
from baselines_common import enforce_min_duration, moving_average_predictions, score_predictions, print_results

# استخرج templates
templates = extract_templates(X_train_sk, y_train_sk, le_skeleton, action_names)

print(f"\n✅ عدد الـ templates: {len(templates)}")

# ==============================================================================
# ٢. تنفيذ $1 Recognizer
# ==============================================================================

def resample_curve(curve, n_points=64):
    """
    Resample — توحيد عدد النقاط.

    curve: (T, D) حيث T = عدد النقاط الخام، D = أبعاد (34 في حالتنا)
    n_points: عدد النقاط المستهدفة (64 في $1 الأصلي)

    النتيجة: (n_points, D) مع توزيع متساوي على طول المسار
    """
    # احسب المسافات التراكمية
    distances = np.sqrt(np.sum(np.diff(curve, axis=0)**2, axis=1))
    total_distance = np.sum(distances)

    if total_distance == 0:
        # الحركة ثابتة (كل النقاط في نفس المكان)
        return np.repeat(curve[[0]], n_points, axis=0)

    # النقاط المراكمة (cumulative)
    cumsum = np.concatenate([[0], np.cumsum(distances)])

    # النقاط الجديدة موزعة بالتساوي
    new_indices = np.linspace(0, total_distance, n_points)

    # استيفاء خطي
    resampled = np.zeros((n_points, curve.shape[1]), dtype=curve.dtype)
    for d in range(curve.shape[1]):
        resampled[:, d] = np.interp(new_indices, cumsum, curve[:, d])

    return resampled


def rotate_curve(curve):
    """
    Rotate — استخرج الميلان الأساسي وأعد تعيين الأصل.

    الفكرة: تدوير الحركة لوضع قياسي (الخط الرئيسي أفقي).
    """
    # النقاط الأول والأخير
    start = curve[0]
    end = curve[-1]

    # الزاوية من البداية للنهاية
    delta = end - start
    angle = np.arctan2(delta[1], delta[0])

    # مصفوفة التدوير
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    rotation_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])

    # لكن انتظر — عندنا 34 بعد (17 مفصل × 2)
    # بدل ما نشتغل على كل مفصل لوحده (معقد)، نشتغل على متوسط المسار

    # احسب متوسط كل فريم
    center = np.mean(curve, axis=0)  # (34,)
    curve_centered = curve - center  # بعد ما ننقل

    # دوّر الإحداثيات
    # نحتاج نطبق التدوير على كل pair (x, y)
    rotated = np.zeros_like(curve)
    for i in range(0, curve.shape[1], 2):
        if i + 1 < curve.shape[1]:
            point = curve_centered[:, i:i+2]  # (T, 2)
            rotated_point = point @ rotation_matrix.T
            rotated[:, i:i+2] = rotated_point
        else:
            rotated[:, i] = curve_centered[:, i]

    return rotated


def translate_curve(curve):
    """
    Translate — انقل لنقطة الصفر.
    """
    center = np.mean(curve, axis=0)
    return curve - center


def scale_curve(curve, width=100):
    """
    Scale — عدّل الحجم.

    كل الحركات بتدخل في مربع بحجم width × width (على كل بعد).
    """
    # احسب أكبر وأصغر قيمة لكل بعد
    min_val = np.min(curve, axis=0)
    max_val = np.max(curve, axis=0)
    range_val = max_val - min_val

    # تجنب القسمة على صفر
    range_val[range_val == 0] = 1

    # تطبيع
    scaled = (curve - min_val) / range_val * width - width / 2

    return scaled


def recognize_gesture_dollar1(curve, templates_dict, resample_n=64):
    """
    الخطوات الكاملة للـ $1 Recognizer.

    curve: (T, 34) — الحركة الخام من النافذة
    templates_dict: dict من الأكشنات وقوالبهم
    """
    # 1. Resample
    resampled = resample_curve(curve, n_points=resample_n)

    # 2. Rotate (نقطة مهمة: هنا بنركز على أول مسار)
    # بما أن عندنا 17 مفصل، بناخد مسار الرسغ (مفصل 10)
    wrist_traj = resampled[:, 20:22]  # مفصل 10 = 20-21 (x, y)

    if np.all(wrist_traj == wrist_traj[0]):
        # الرسغ ما تحرك
        rotated = resampled.copy()
    else:
        start = wrist_traj[0]
        end = wrist_traj[-1]
        delta = end - start
        angle = np.arctan2(delta[1], delta[0])

        cos_a = np.cos(-angle)
        sin_a = np.sin(-angle)

        rotated = np.zeros_like(resampled)
        center = np.mean(resampled, axis=0)

        for i in range(0, resampled.shape[1], 2):
            if i + 1 < resampled.shape[1]:
                point = resampled[:, i:i+2] - center[[i, i+1]]
                rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                rotated[:, i:i+2] = point @ rot_matrix.T
            else:
                rotated[:, i] = resampled[:, i] - center[i]

    # 3. Translate
    translated = translate_curve(rotated)

    # 4. Scale
    scaled = scale_curve(translated, width=100)

    # 5. Compare بـ كل template
    distances = {}
    for action_name, template in templates_dict.items():
        # template لازم يكون نفس الشكل
        template_resampled = resample_curve(template, n_points=resample_n)
        template_centered = translate_curve(template_resampled)
        template_scaled = scale_curve(template_centered, width=100)

        # حساب أقرب نقطة لنقطة (point-to-point distance)
        if len(scaled) != len(template_scaled):
            # في حالة نادرة، اعادة عينة
            continue

        dist = np.mean(np.sqrt(np.sum((scaled - template_scaled)**2, axis=1)))
        distances[action_name] = dist

    return distances


# ==============================================================================
# ٣. بناء النوافذ
# ==============================================================================

print("\n🎬 بناء نوافذ الفيديو...")

fps = 60
effective_fps = fps / 2

centers_sk, edges_sk, window_times_sk, WINDOW_FRAMES = setup_sliding_windows(
    all_keypoints, fps, effective_fps
)

windows = build_windows_for_baseline(all_keypoints, centers_sk, WINDOW_FRAMES, effective_fps)
print(f"✅ شكل النوافذ: {windows.shape}")

# ==============================================================================
# ٤. تطبيق $1 على كل نافذة
# ==============================================================================

print("\n🔄 تطبيق $1 Recognizer على كل نافذة...")

predictions = []
confidences = []

for i, window in enumerate(windows):
    # Recognition
    distances = recognize_gesture_dollar1(window, templates)

    if not distances:
        # فشل الـ recognition (نادر)
        best_action = 'other'
        confidence = 0.0
    else:
        # أقل مسافة = أفضل match
        best_action = min(distances, key=distances.get)
        best_distance = distances[best_action]

        # الثقة = معكوس المسافة
        all_distances = list(distances.values())
        min_dist = min(all_distances)
        max_dist = max(all_distances)

        if max_dist > min_dist:
            # normalize بين 0 و 1
            confidence = 1 - (best_distance - min_dist) / (max_dist - min_dist)
        else:
            confidence = 1.0

    predictions.append(best_action)
    confidences.append(confidence)

    if (i + 1) % max(1, len(windows) // 4) == 0 or i == len(windows) - 1:
        print(f"   تم {i+1}/{len(windows)}")

print(f"✅ توقعات لـ {len(predictions)} نافذة")

# ==============================================================================
# ٥. تطبيق Minimum Duration Filtering
# ==============================================================================

print("\n⏱️  تطبيق minimum duration filtering...")

MIN_SEGMENT = 2
predictions = enforce_min_duration(predictions, MIN_SEGMENT, protect='other')

# ==============================================================================
# ٦. التنعيم الزمني
# ==============================================================================

print("✨ تنعيم زمني...")

predictions = moving_average_predictions(predictions, k=5)

# ==============================================================================
# ٧. المقارنة مع Ground Truth
# ==============================================================================

print("\n📊 المقارنة مع الـ ground truth...")

# ⚠️ الجدول ده كان مكتوب هنا بإيد — واحدة من 5 نسخ متطابقة في المشروع.
#    اتشال 2026-08-31 وبقى بييجي من مصدر واحد. السبب: نسخة vidtest3
#    عاشت شهر وهي غلط تماماً من غير ما حد ياخد باله. HANDOFF قسم 6.8
from ground_truth import GROUND_TRUTH

ground_truth_segments = GROUND_TRUTH['vidtest1']

print_results("$1 Recognizer Baseline", predictions, edges_sk, ground_truth_segments)

# ==============================================================================
# ٨. رسم النتائج
# ==============================================================================

print("\n📈 رسم النتائج...")

fig, ax = plt.subplots(figsize=(16, 5))

unique_actions = list(set(predictions))
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_actions)))
action_color_map = dict(zip(unique_actions, colors))

for i in range(len(windows) - 1):
    ax.axvspan(edges_sk[i], edges_sk[i+1],
               color=action_color_map[predictions[i]], alpha=0.6)

for start, end, label in ground_truth_segments:
    ax.text((start+end)/2, 1.05, label, ha='center', fontsize=9,
            fontweight='bold', rotation=30, transform=ax.get_xaxis_transform())
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

# ==============================================================================
# ٩. الخلاصة
# ==============================================================================

print("\n" + "="*70)
print("🔍 الملاحظات:")
print("="*70)
print("""
$1 Recognizer بتطبّق خطوات توحيد صارمة (resample, rotate, scale, translate)
ثم تقارن الأشكال مباشرة.

المشاكل:
  1. Resample و Rotate قد تفقد معلومات مهمة
  2. نقطة واحد مفصل (الرسغ) = معلومة ناقصة جداً
  3. في حالتنا، عندنا 17 مفصل متزامنة، لكن $1 بتركز على مسار واحد

الفائدة:
  ✅ سريع جداً
  ✅ بسيط
  ❌ دقة منخفضة جداً

المتوقع: 20-30% دقة
الفعلي: تحقق من النتيجة أعلاه!

ملاحظة: في النسخة الكاملة من $1، تقدر تستخدم kd-tree لتسريع البحث.
ولكن هنا التركيز على الفهم بدل الأداء.
""")

print("="*70)
