"""
One-Shot FastDTW — تعرّف على الحركات بـ **مثال واحد** لكل حركة، من غير أي داتاسِت.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
الفكرة
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
الـ LSTM محتاجة 56,000 عينة من NTU-60 وساعات تدريب.
الـ FastDTW محتاجة **مثال واحد** لكل حركة وصفر تدريب.

هنا بناخد قصاصة واحدة لكل حركة من vidtest1 (بالاعتماد على الـ ground truth)،
ونستخدمها كـ template للتعرّف على vidtest3. مفيش NTU خالص.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
تجنّب التسريب (leakage) — مهم جداً
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
الـ templates بتتقص من **vidtest1**، والقياس على **vidtest3**.
فيديو مختلف تماماً، جلسة تصوير مختلفة، دقة مختلفة (852x480 مقابل 576x1024)،
و fps مختلف (30 مقابل 60).

لو قصّينا الـ templates من نفس الفيديو اللي بنقيس عليه، الرقم يبقى مزيّف.

⚠️ **حاجة لازم تتقال بصراحة في التقرير:**
   الشخص في الفيديوهات هو **نفس الشخص**. يعني الـ one-shot عندها ميزة
   "نفس الجسم ونفس أسلوب الحركة" — والـ LSTM (المدرّبة على أشخاص تانيين
   خالص من NTU) مالهاش الميزة دي.

   ده **مش غش** — ده بالظبط الـ use case بتاع الـ one-shot: "سجّل نفسك
   مرة واحدة، والنظام يعرفك بعد كده". بس لازم يتكتب صريح، مايتخبّاش.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
تغطية الحركات
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  vidtest1 (مصدر الـ templates): rub_hands, wave, sit_down, stand_up
  vidtest3 (الاختبار):           sit_down, stand_up  ← الاتنين متغطيين ✅
  vidtest2:                      فيه phone_call مالهوش template من vidtest1

  عشان كده vidtest3 هو أنضف حالة اختبار — وهو كمان أهم فيديو.

المتطلبات: شغّل `python pose_extract.py` الأول.
"""

import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fastdtw_core import fastdtw
from skeleton_norm import normalize_skeleton, resample_to, NUM_FRAMES
from baselines_common import (
    enforce_min_duration, moving_average_predictions, score_predictions,
)

KP_DIR = Path(__file__).resolve().parent / 'keypoints'
OUT_DIR = Path(__file__).resolve().parent / '_scratch'

# ==============================================================================
# الإعدادات
# ==============================================================================

TEMPLATE_VIDEO = 'vidtest1'
TEST_VIDEO = 'vidtest3'

RADIUS = 1
WINDOW_SECONDS = 3.0
STRIDE_FRAMES = 10
MIN_SEGMENT = 2
SMOOTH_K = 5

# 'vel' = الفروق بين الفريمات (السرعة) | 'pos' = المواضع الخام
#
# ⚠️ 'pos' **بيفشل تماماً** — اتقاس: real recall = 0.0% على vidtest3.
#    السبب: الـ DTW على المواضع بتقيس تشابه *الوضعية* مش *الحركة*، وبتجمع
#    المسافة على الـ 34 بعد بالتساوي. إشارة القعود موجودة في بُعدين تلاتة
#    بس (الورك->الركبة)، فبتغرق وسط ضوضاء الدراعات والراس. النتيجة إن
#    template الوقفة الساكنة (rub_hands) بقى أقرب حاجة لأي نافذة.
#
#    'vel' بيشيل المكوّن الثابت ويخلي المقارنة على التغيّر الفعلي.
FEATURE_MODE = 'vel'

# قصاصات الـ templates: (بداية, نهاية, اسم الحركة) بالثواني من vidtest1
# مأخوذة من الـ ground truth بتاع vidtest1 — مثال واحد بس لكل حركة.
TEMPLATE_CLIPS = [
    (0.5,   4.2, 'rub_hands'),
    (5.0,  11.0, 'wave'),
    (12.2, 14.3, 'sit_down'),
    (19.5, 20.4, 'stand_up'),
]

# الـ ground truth بتاع فيديو الاختبار — من المصدر الواحد
# ⚠️ كان بيستورد من vidtest3_ground_truth.py اللي طلع **غلط تماماً**
#    (4 حركات بدل 13). أي نتيجة قديمة من الملف ده لاغية. HANDOFF قسم 6.8
from ground_truth import GROUND_TRUTH  # noqa: F401  (vidtest3 جوّاه)


def to_features(seq_norm, mode=None):
    """
    (frames, 34) -> تمثيل المقارنة.

    'pos' = زي ما هي (مواضع متطبّعة)
    'vel' = الفروق بين الفريمات المتتالية -> (frames-1, 34)
    """
    mode = mode or FEATURE_MODE
    if mode == 'pos':
        return seq_norm
    if mode == 'vel':
        return np.diff(seq_norm, axis=0)
    raise ValueError(f'FEATURE_MODE غير معروف: {mode}')


def load_keypoints(name):
    kp_path = KP_DIR / f'{name}_keypoints.npy'
    meta_path = KP_DIR / f'{name}_meta.npy'
    if not kp_path.exists():
        raise FileNotFoundError(
            f'{kp_path} مش موجود — شغّل `python pose_extract.py {name}` الأول'
        )
    kp = np.load(kp_path)
    meta = np.load(meta_path, allow_pickle=True).item()
    return kp, meta


# ==============================================================================
# ١. قص الـ templates — مثال واحد لكل حركة
# ==============================================================================

print("=" * 74)
print("  One-Shot FastDTW — من غير داتاسِت")
print("=" * 74)

tpl_kp, tpl_meta = load_keypoints(TEMPLATE_VIDEO)
eff_fps_tpl = tpl_meta['effective_fps']

print(f"\n📼 مصدر الـ templates: {TEMPLATE_VIDEO}")
print(f"   {tpl_kp.shape[0]} فريم @ {eff_fps_tpl:.1f}fps فعلي | "
      f"اكتشاف {tpl_meta['detect_rate'] * 100:.1f}%")

print(f"\n✂️  قص الـ templates (مثال واحد لكل حركة):")
templates = {}
for t0, t1, name in TEMPLATE_CLIPS:
    lo, hi = int(t0 * eff_fps_tpl), int(t1 * eff_fps_tpl)
    clip = tpl_kp[lo:hi]
    if len(clip) < 2:
        print(f"   ⚠️ {name}: القصاصة قصيرة جداً، اتشالت")
        continue
    templates[name] = to_features(resample_to(normalize_skeleton(clip), NUM_FRAMES))
    print(f"   ✅ {name:12} [{t0:5.1f}s - {t1:5.1f}s]  "
          f"{len(clip):3d} فريم -> {NUM_FRAMES}")

template_names = list(templates)
print(f"\n📦 إجمالي: {len(templates)} template — كل واحد من **مثال واحد**")
print(f"   (الـ LSTM للمقارنة: 56,000 عينة)")

# ==============================================================================
# ٢. بناء نوافذ فيديو الاختبار
# ==============================================================================

test_kp, test_meta = load_keypoints(TEST_VIDEO)
eff_fps = test_meta['effective_fps']

print(f"\n🎬 فيديو الاختبار: {TEST_VIDEO}")
print(f"   {test_kp.shape[0]} فريم @ {eff_fps:.1f}fps فعلي | "
      f"{test_meta['duration_s']:.1f}s | اكتشاف {test_meta['detect_rate'] * 100:.1f}%")

WINDOW_FRAMES = int(round(WINDOW_SECONDS * eff_fps))
half = WINDOW_FRAMES // 2
T = len(test_kp)
centers = np.arange(half, T - half, STRIDE_FRAMES)

_n_feat = NUM_FRAMES if FEATURE_MODE == 'pos' else NUM_FRAMES - 1
windows = np.empty((len(centers), _n_feat, 34), dtype=np.float32)
for i, c in enumerate(centers):
    sub = np.linspace(max(0, c - half), min(T, c + half + 1) - 1,
                      NUM_FRAMES, dtype=int)
    windows[i] = to_features(normalize_skeleton(test_kp[sub]))

# حدود النوافذ بالثواني
edges = np.empty(len(centers) + 1)
edges[0] = 0.0
edges[1:-1] = (centers[:-1] + centers[1:]) / 2.0 / eff_fps
edges[-1] = (T - 1) / eff_fps

print(f"   نافذة = {WINDOW_SECONDS}s ({WINDOW_FRAMES} فريم) | "
      f"خطوة = {STRIDE_FRAMES / eff_fps:.2f}s")
print(f"   إجمالي النوافذ: {len(windows)}")

# ==============================================================================
# ٣. معايرة عتبة الرفض — من الـ templates نفسها، من غير داتاسِت
# ==============================================================================

print("\n" + "=" * 74)
print("🎚️  معايرة عتبة الرفض — من غير داتاسِت")
print("=" * 74)
print("""
مفيش داتا تدريب نعاير عليها، فبنعاير من الـ templates نفسها:

  نحسب المسافة بين كل template والتاني. المسافة دي بتقولنا "حركتين
  مختلفتين بيبعدوا عن بعض قد ايه؟". أي نافذة أبعد من أقرب template
  بأكتر من المدى ده => مش شبه أي حاجة نعرفها => 'other'.

ده معناه إن العتبة مستنتجة من الـ 4 أمثلة اللي عندنا بس — مفيش أي
معلومة خارجية.
""")

cross = []
for i, a in enumerate(template_names):
    for b in template_names[i + 1:]:
        d, _ = fastdtw(templates[a], templates[b], radius=RADIUS)
        cross.append(d)
cross = np.array(cross)

REJECT_THRESHOLD = float(np.percentile(cross, 50))

print(f"📊 المسافات بين الـ templates وبعضها ({len(cross)} زوج):")
print(f"   أقل:     {cross.min():8.2f}")
print(f"   الوسيط:  {np.median(cross):8.2f}")
print(f"   أكبر:    {cross.max():8.2f}")
print(f"\n🎚️  العتبة (المئوية 25): {REJECT_THRESHOLD:.2f}")

# ==============================================================================
# ٤. التصنيف
# ==============================================================================

print("\n" + "=" * 74)
print("🔄 التصنيف بـ FastDTW")
print("=" * 74)

dist = np.zeros((len(windows), len(template_names)))
t0 = time.perf_counter()

for wi in range(len(windows)):
    for ti, name in enumerate(template_names):
        dist[wi, ti], _ = fastdtw(windows[wi], templates[name], radius=RADIUS)
    if (wi + 1) % max(1, len(windows) // 4) == 0:
        print(f"   {wi + 1}/{len(windows)} — {time.perf_counter() - t0:.1f}s")

elapsed = time.perf_counter() - t0
per_window_ms = elapsed / len(windows) * 1000

predictions = []
for wi in range(len(windows)):
    best = int(np.argmin(dist[wi]))
    predictions.append('other' if dist[wi, best] > REJECT_THRESHOLD
                       else template_names[best])

n_rej = sum(p == 'other' for p in predictions)
print(f"\n⏱️  {elapsed:.1f}s إجمالي | {per_window_ms:.1f}ms لكل نافذة")
print(f"🚫 اترفض: {n_rej}/{len(windows)} ({n_rej / len(windows) * 100:.1f}%) => 'other'")

raw_predictions = list(predictions)
predictions = enforce_min_duration(predictions, MIN_SEGMENT, protect='other')
predictions = moving_average_predictions(predictions, k=SMOOTH_K)

# ==============================================================================
# ٥. التقييم
# ==============================================================================

print("\n" + "=" * 74)
print(f"📊 النتيجة — One-Shot FastDTW على {TEST_VIDEO}")
print("=" * 74)

gt = GROUND_TRUTH[TEST_VIDEO]
cur = score_predictions(predictions, edges, gt)
base = score_predictions(['other'] * len(predictions), edges, gt)

if cur['total'] == 0:
    print("⚠️ مفيش توقعات في نطاق الـ ground truth")
else:
    acc = cur['hit'] / cur['total'] * 100
    print(f"\n   دقة إجمالية:  {cur['hit']:4d}/{cur['total']:<4d} = {acc:5.1f}%")
    print(f"   خط الأساس:    {base['hit']:4d}/{base['total']:<4d} = "
          f"{base['hit'] / base['total'] * 100:5.1f}%   (لو قال 'other' على طول)")
    print(f"   other recall: {cur['other_hit']:4d}/{cur['other_total']:<4d} = "
          f"{cur['other_hit'] / max(1, cur['other_total']) * 100:5.1f}%")
    print(f"   real recall:  {cur['real_hit']:4d}/{cur['real_total']:<4d} = "
          f"{cur['real_hit'] / max(1, cur['real_total']) * 100:5.1f}%")

    print(f"\n⚠️ خط الأساس مهم: {TEST_VIDEO} معظمه سكون، فلو النظام قال")
    print(f"   'other' على طول هياخد {base['hit'] / base['total'] * 100:.1f}%.")
    print(f"   القيمة الحقيقية في الـ real recall — قدرته يمسك الحركات الفعلية.")

# توزيع التوقعات
print(f"\n📈 توزيع التوقعات:")
for name in sorted(set(predictions)):
    n = sum(p == name for p in predictions)
    print(f"   {name:12} {n:4d} نافذة ({n / len(predictions) * 100:5.1f}%)")

# ==============================================================================
# ٦. الرسم
# ==============================================================================

OUT_DIR.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(16, 4))

uniq = sorted(set(predictions))
cmap = dict(zip(uniq, plt.cm.tab10(np.linspace(0, 1, max(len(uniq), 2)))))

for i in range(len(predictions)):
    ax.axvspan(edges[i], edges[i + 1], color=cmap[predictions[i]], alpha=0.65)

for s, e, label in gt:
    if not label.endswith('*') and label != '?':
        ax.axvspan(s, e, ymin=0.88, ymax=1.0, color='k', alpha=0.8)
        ax.text((s + e) / 2, 1.03, label, ha='center', fontsize=8,
                fontweight='bold', transform=ax.get_xaxis_transform())

handles = [plt.Rectangle((0, 0), 1, 1, color=cmap[a]) for a in uniq]
ax.legend(handles, uniq, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=6)
ax.set_xlabel('Time (s)')
ax.set_yticks([])
ax.set_title(f'One-Shot FastDTW — templates from {TEMPLATE_VIDEO}, '
             f'tested on {TEST_VIDEO} — {FEATURE_MODE} features, no dataset', fontsize=12)
plt.tight_layout()
out = OUT_DIR / f'oneshot_fastdtw_{TEST_VIDEO}_{FEATURE_MODE}.png'
plt.savefig(out, dpi=120, bbox_inches='tight')
print(f"\n💾 الرسم: {out}")

# ==============================================================================
# ٧. الخلاصة
# ==============================================================================

print("\n" + "=" * 74)
print("📋 الخلاصة")
print("=" * 74)
print(f"""
  عدد الأمثلة المستخدمة:  {len(templates)} (واحد لكل حركة)
  حجم الداتاسِت:          صفر — مفيش NTU خالص
  وقت التدريب:            صفر
  الزمن لكل نافذة:        {per_window_ms:.1f}ms
  التمثيل:                {FEATURE_MODE}

  مصدر الـ templates:     {TEMPLATE_VIDEO}
  فيديو الاختبار:         {TEST_VIDEO}  (فيديو مختلف — مفيش تسريب)

  ⚠️ نفس الشخص في الفيديوهين — الميزة دي لازم تتقال في التقرير.
""")
