"""
FastDTW Baseline — التشغيل على فيديوهات الاختبار

بيشتغل على أي فيديو من التلاتة. غيّر VIDEO_NAME بس.

الفرق الجوهري عن notebook_dtw.py القديم:
  الكود القديم كان بياخد min(distances) على طول، يعني **مستحيل** يقول
  'other'. وده مشكلة قاتلة لأن الفيديوهات معظمها 'other'
  (vidtest3: ~120 ثانية من 126.5 هي سكون!). النتيجة إن الـ baseline كان
  هيطلع رقم واطي مش لأنه ضعيف، لأننا مانعناه يجاوب صح.

  هنا فيه **rejection معاير من بيانات التدريب** — يوازي الـ CONF_REJECT
  بتاع الـ LSTM. لو النافذة بعيدة عن كل الـ templates، يبقى 'other'.
  كده المقارنة عادلة والدكتور مايقدرش يقول إننا كسّرنا الـ baseline.

المتطلبات (لازم تكون متحمّلة من خلايا الـ notebook الأصلي):
    X_train_sk, y_train_sk, le_skeleton, action_names, all_keypoints, fps
"""

import numpy as np
import time
import matplotlib.pyplot as plt

from fastdtw_core import fastdtw, dtw_full, count_cells_evaluated
from baselines_common import (
    extract_templates, setup_sliding_windows, build_windows_for_baseline,
    enforce_min_duration, moving_average_predictions, print_results,
)

# ==============================================================================
# الإعدادات
# ==============================================================================

VIDEO_NAME = 'vidtest1'      # غيّرها لـ 'vidtest2' أو 'vidtest3'
RADIUS = 1                   # نصف قطر البحث في FastDTW
REJECT_PERCENTILE = 80       # عتبة الرفض (مئوية من توزيع مسافات التدريب)
MIN_SEGMENT = 2
SMOOTH_K = 5

VIDEO_FPS = {'vidtest1': 60, 'vidtest2': 60, 'vidtest3': 30}

print("=" * 70)
print(f"  FastDTW Baseline — {VIDEO_NAME}")
print("=" * 70)

# ==============================================================================
# ١. استخراج الـ templates
# ==============================================================================

print("\n📦 استخراج الـ templates من بيانات التدريب...")
templates = extract_templates(X_train_sk, y_train_sk, le_skeleton, action_names)
template_names = list(templates.keys())
print(f"\n✅ {len(templates)} template: {template_names}")

# ==============================================================================
# ٢. معايرة عتبة الرفض  ← ده الجزء المهم
# ==============================================================================

print("\n" + "=" * 70)
print("🎚️  معايرة عتبة الرفض (rejection threshold)")
print("=" * 70)
print("""
الفكرة:
  ناخد عينات من بيانات التدريب، ونحسب مسافة FastDTW من كل عينة لـ template
  بتاع كلاسها هي. التوزيع اللي هيطلع بيقولنا: "الحركة الحقيقية بتبعد قد ايه
  عن الـ template بتاعها؟"

  أي نافذة في الفيديو مسافتها أكبر من العتبة دي => مش شبه أي حركة نعرفها
  => 'other'.

  ده الموازي للـ CONF_REJECT = 0.60 بتاع الـ LSTM، بس معاير من البيانات
  مش رقم مختار بالمزاج.
""")


def calibrate_rejection(X_train_sk, y_train_sk, templates, le_skeleton,
                        action_names, n_per_class=30, percentile=80,
                        radius=1, seed=0):
    """
    بيرجّع (العتبة, توزيع المسافات) — المسافات من عينات تدريب لـ template كلاسها.
    """
    rng = np.random.default_rng(seed)
    y_indices = np.argmax(y_train_sk, axis=1)
    in_class_distances = []

    for class_idx, class_name in enumerate(le_skeleton.classes_):
        action_name = action_names.get(int(class_name), f"class_{class_name}")
        if action_name not in templates:
            continue

        mask = np.where(y_indices == class_idx)[0]
        if len(mask) == 0:
            continue

        # عينة عشوائية عشان ما ناخدش وقت طويل
        sample_idx = rng.choice(mask, size=min(n_per_class, len(mask)), replace=False)
        tmpl = templates[action_name]

        for idx in sample_idx:
            d, _ = fastdtw(X_train_sk[idx], tmpl, radius=radius)
            in_class_distances.append(d)

    in_class_distances = np.array(in_class_distances)
    threshold = np.percentile(in_class_distances, percentile)
    return threshold, in_class_distances


t0 = time.perf_counter()
REJECT_THRESHOLD, calib_distances = calibrate_rejection(
    X_train_sk, y_train_sk, templates, le_skeleton, action_names,
    n_per_class=30, percentile=REJECT_PERCENTILE, radius=RADIUS
)
print(f"⏱️  المعايرة خدت {time.perf_counter() - t0:.1f} ثانية "
      f"({len(calib_distances)} عينة)")
print(f"\n📊 توزيع مسافات الحركات الحقيقية عن الـ templates بتاعتها:")
print(f"   أقل مسافة:   {calib_distances.min():8.2f}")
print(f"   الوسيط:      {np.median(calib_distances):8.2f}")
print(f"   المئوية 80:  {np.percentile(calib_distances, 80):8.2f}")
print(f"   المئوية 95:  {np.percentile(calib_distances, 95):8.2f}")
print(f"   أكبر مسافة:  {calib_distances.max():8.2f}")
print(f"\n🎚️  العتبة المختارة (مئوية {REJECT_PERCENTILE}): {REJECT_THRESHOLD:.2f}")
print(f"   أي نافذة مسافتها > {REJECT_THRESHOLD:.2f} => 'other'")

# ==============================================================================
# ٣. بناء نوافذ الفيديو
# ==============================================================================

print("\n" + "=" * 70)
print("🎬 بناء نوافذ الفيديو")
print("=" * 70)

fps = VIDEO_FPS[VIDEO_NAME]
effective_fps = fps / 2          # FRAME_SKIP = 2

centers_sk, edges_sk, window_times_sk, WINDOW_FRAMES = setup_sliding_windows(
    all_keypoints, fps, effective_fps
)
windows = build_windows_for_baseline(
    all_keypoints, centers_sk, WINDOW_FRAMES, effective_fps
)
print(f"✅ شكل النوافذ: {windows.shape}")

# ==============================================================================
# ٤. التنبؤ بـ FastDTW
# ==============================================================================

print("\n" + "=" * 70)
print("🔄 حساب FastDTW لكل نافذة")
print("=" * 70)

n_windows = len(windows)
dist_matrix = np.zeros((n_windows, len(template_names)))

t0 = time.perf_counter()
for wi in range(n_windows):
    for ti, name in enumerate(template_names):
        dist_matrix[wi, ti], _ = fastdtw(windows[wi], templates[name], radius=RADIUS)

    if (wi + 1) % max(1, n_windows // 5) == 0:
        elapsed = time.perf_counter() - t0
        print(f"   {wi + 1}/{n_windows} نافذة — {elapsed:.1f}s")

total_time = time.perf_counter() - t0
per_window_ms = total_time / n_windows * 1000
print(f"\n⏱️  إجمالي: {total_time:.1f}s | لكل نافذة: {per_window_ms:.1f}ms")

# اختيار الإجابة + تطبيق الرفض
predictions = []
confidences = []

for wi in range(n_windows):
    best_ti = int(np.argmin(dist_matrix[wi]))
    best_dist = dist_matrix[wi, best_ti]

    if best_dist > REJECT_THRESHOLD:
        # بعيدة عن كل الـ templates => مش حركة معروفة
        predictions.append('other')
    else:
        predictions.append(template_names[best_ti])

    # ثقة بين 0 و 1 (1 = قريبة جداً من الـ template)
    confidences.append(float(np.clip(1.0 - best_dist / REJECT_THRESHOLD, 0.0, 1.0)))

n_rejected = sum(1 for p in predictions if p == 'other')
print(f"\n🚫 اترفض: {n_rejected}/{n_windows} نافذة ({n_rejected / n_windows * 100:.1f}%) => 'other'")

# ==============================================================================
# ٥. التنعيم الزمني
# ==============================================================================

print("\n✨ تنعيم زمني...")
predictions = enforce_min_duration(predictions, MIN_SEGMENT, protect='other')
predictions = moving_average_predictions(predictions, k=SMOOTH_K)

# ==============================================================================
# ٦. المقارنة مع الـ ground truth
# ==============================================================================

# ⚠️ الجدول كان مكتوب هنا بإيد، ونسخ منه في 4 ملفات تانية. اتشال
#    2026-08-31 وبقى بييجي من مصدر واحد. السبب: نسخة vidtest3 كانت
#    غلط تماماً وعاشت شهر من غير ما حد ياخد باله. HANDOFF قسم 6.8
from ground_truth import GROUND_TRUTH

gt = GROUND_TRUTH[VIDEO_NAME]

print_results(f"FastDTW Baseline — {VIDEO_NAME}", predictions, edges_sk, gt,
              baseline_only=True)

# ==============================================================================
# ٧. Benchmark — إثبات الـ O(n) بالأرقام
# ==============================================================================

print("\n" + "=" * 70)
print("⚡ Benchmark: FastDTW مقابل DTW الكامل")
print("=" * 70)

rng = np.random.default_rng(0)
print(f"\n{'n':>6} | {'خلايا كامل':>12} | {'خلايا Fast':>12} | {'النسبة':>7} | "
      f"{'كامل ms':>9} | {'Fast ms':>9} | {'كسب':>6}")
print("-" * 78)

for n in [30, 60, 120, 240]:
    c = count_cells_evaluated(n, n, radius=RADIUS)
    x = rng.normal(size=(n, 34))
    y = rng.normal(size=(n, 34))

    t0 = time.perf_counter(); dtw_full(x, y);  t_full = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter(); fastdtw(x, y, radius=RADIUS); t_fast = (time.perf_counter() - t0) * 1000

    print(f"{n:>6} | {c['full_dtw_cells']:>12,} | {c['fastdtw_cells']:>12,} | "
          f"{c['ratio']:>7.2f} | {t_full:>9.1f} | {t_fast:>9.1f} | {t_full / t_fast:>5.2f}x")

# دقة التقريب على النوافذ الحقيقية
print("\n🎯 دقة التقريب على نوافذ الفيديو الحقيقية (عينة 20 نافذة):")
sample = rng.choice(n_windows, size=min(20, n_windows), replace=False)
errors = []
for wi in sample:
    tmpl = templates[template_names[0]]
    d_full, _ = dtw_full(windows[wi], tmpl)
    d_fast, _ = fastdtw(windows[wi], tmpl, radius=RADIUS)
    if d_full > 0:
        errors.append((d_fast - d_full) / d_full * 100)

errors = np.array(errors)
print(f"   متوسط الخطأ: {errors.mean():+.4f}%  |  أكبر خطأ: {errors.max():+.4f}%")
if abs(errors).max() < 0.01:
    print("   ✅ عند n=30 الـ FastDTW بيدّي نفس إجابة الـ DTW الكامل بالظبط")

# ==============================================================================
# ٨. الرسم
# ==============================================================================

print("\n📈 رسم الـ timeline...")

fig, ax = plt.subplots(figsize=(16, 5))
unique_actions = sorted(set(predictions))
colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_actions), 2)))
action_color_map = dict(zip(unique_actions, colors))

for i in range(len(predictions)):
    ax.axvspan(edges_sk[i], edges_sk[i + 1],
               color=action_color_map[predictions[i]], alpha=0.6)

for start, end, label in gt:
    ax.text((start + end) / 2, 1.05, label, ha='center', fontsize=8,
            fontweight='bold', rotation=30, transform=ax.get_xaxis_transform())
    ax.axvline(start, color='k', lw=0.6, ls='--', alpha=0.4)

handles = [plt.Rectangle((0, 0), 1, 1, color=action_color_map[a]) for a in unique_actions]
ax.legend(handles, unique_actions, loc='upper center',
          bbox_to_anchor=(0.5, -0.15), ncol=5)
ax.set_xlabel('Time (s)')
ax.set_yticks([])
ax.set_title(f'FastDTW Baseline — {VIDEO_NAME}.mp4', fontsize=13)
plt.tight_layout()
plt.savefig(f'/kaggle/working/fastdtw_timeline_{VIDEO_NAME}.png', dpi=120, bbox_inches='tight')
plt.show()

print(f"✅ اتحفظت: /kaggle/working/fastdtw_timeline_{VIDEO_NAME}.png")

# ==============================================================================
# ٩. الخلاصة
# ==============================================================================

print("\n" + "=" * 70)
print("📋 الخلاصة")
print("=" * 70)
print(f"""
الفيديو:            {VIDEO_NAME}
عدد النوافذ:        {n_windows}
الزمن لكل نافذة:    {per_window_ms:.1f}ms  ({len(templates)} template)
نسبة الرفض:         {n_rejected / n_windows * 100:.1f}%
عتبة الرفض:         {REJECT_THRESHOLD:.2f} (مئوية {REJECT_PERCENTILE} من التدريب)

⚠️ للأمانة عند عرض النتيجة:
   الرقم ده لـ template matching بـ template **واحد متوسط** لكل حركة.
   المتوسط بيمسح التنوع البشري — الناس بتعمل نفس الحركة بسرعات وأشكال
   مختلفة، والمتوسط بيطلع شكل مامحدش بيعمله بالظبط.
   ده السبب الأساسي إن الـ LSTM بتكسب، مش إن الـ DTW خوارزمية وحشة.
""")
