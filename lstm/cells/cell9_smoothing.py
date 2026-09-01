import numpy as np

effective_fps = fps / FRAME_SKIP  # ~30 fps

# ----------------------------------------------------------------------
# المقياس الزمني — أهم إعداد في الخلية دي
# ----------------------------------------------------------------------
# التدريب بياخد الكليب *كامل* (في NTU متوسطه ~3 ثواني) وبيعمله linspace
# لـ 30 فريم. يعني الموديل اتعلم إن الـ 30 صف = حركة من أولها لآخرها،
# والخطوة بينهم ~0.1 ثانية. لازم الاستنتاج يعمل نفس الحاجة بالظبط:
# ناخد نافذة بطول كليب التدريب وبعدين نضغطها لـ MODEL_FRAMES بنفس الـ
# linspace بتاع process_skeleton_sample.
WINDOW_SECONDS = 3.0
WINDOW_FRAMES = int(round(WINDOW_SECONDS * effective_fps))  # ~90 فريم
MODEL_FRAMES = NUM_FRAMES   # 30 — اللي الموديل مستنيه، جاي من خلية 3

# نافذة واحدة بطول ثابت بتفترض إن كل الحركات ليها نفس المدة، وده مش صحيح:
# falling ثانية ونص، phone_call ممكن تاخد 5. بنجرب أربع أطوال حوالين نفس
# المركز وبناخد متوسط الاحتمالات — الطول اللي بيظبط الحركة بيدي احتمال
# عالي والباقي بيبقى مشتّت، فالمتوسط بيميل للصح من غير ما نختار طول واحد.
# 1.5s مهمة تحديداً للحركات السريعة (sit_down 1.1s و stand_up 0.8s في
# vidtest3): جوه نافذة 3 ثواني بتبقى ربع المحتوى والباقي "قاعد/واقف".
WINDOW_SCALES = (1.5, 2.0, 3.0, 4.0)

STRIDE = 10          # 0.33 ثانية هوب (~89% overlap) — الدقة الزمنية محدودة
                     # بطول النافذة نفسها أصلاً، فمفيش داعي لـ stride أصغر
SMOOTH_K = 5         # مدى التنعيم مربوط بالـ stride: لو غيّرت واحد غيّر التاني
MIN_SEGMENT = 2      # نافذتين = ~0.7s. 3 كانت بتمسح stand_up (0.8s) قبل
                     # ما نشوفه أصلاً

# ----------------------------------------------------------------------
# عتبة الرفض -> 'other'
# ----------------------------------------------------------------------
# الفيديو فيه حركات مش ضمن الكلاسات المختارة (مشي مثلاً — مش في NTU
# أصلاً). softmax مجبور يوزّع الاحتمال على الكلاسات الموجودة، فبيطلع
# أكشن غلط بثقة معقولة. العتبة دي هي آلية الرفض الوحيدة.
#
# ⚠️ ماتزوّدهاش فوق كده على أساس sweep رن واحد. الموديل بيتدرب من أول
# وجديد كل رن فدرجة الحرارة المعايرة بتتغيّر، والعتبة العالية بتنقل عشرات
# النوافذ معاها. 0.60 أثبتت إنها أثبت قيمة بين الرنات (real recall
# 84.6% -> 70.3%) مقابل 0.80 اللي نطّت من 62.6% لـ 39.6%.
# التفاصيل في .wolf/buglog.json :: threshold-overfit-to-single-run-weights
#
# ⚠️ رن 25: رفعناها لـ 0.70 بحساب نظري (التوزيع المتساوي بقى 25% بدل
# 8.3% مع تقليل الكلاسات) — وده كان غلط تماماً. الـ sweep الفعلي طلّع
# real recall 28.4% عند 0.30 و 0.0% عند 0.70. الحساب النظري مالوش لازمة
# لأن الموديل مش معاير أصلاً على مدخل خارج النطاق (درجة الحرارة اتحسبت
# 1.05 يعني صفر تعايير). القاعدة: العتبة تتحدد من الـ sweep مش من العدد.
#
# رجعناها 0.60 مع رجوع الـ 10 كلاسات — دي القيمة اللي اشتغلت أيام الـ 12
# كلاس (رن 23: قمة الـ sweep عند 0.60 بالظبط). الـ 0.50 كانت مضبوطة على
# رن 26 اللي كان 3 كلاسات ودرجة حرارته 0.50. الـ sweep تحت هيقول الصح.
CONF_REJECT = 0.60    # أعلى احتمال لازم يعدّي كده

T_frames = len(all_keypoints)
half_base = WINDOW_FRAMES // 2
# كل النوافذ بتتبني حوالين نفس المراكز مهما كان طولها — كده كل المقاييس
# بتدّي نفس عدد التوقعات وبنقدر نجمعهم بالمتوسط مباشرة
centers_sk = np.arange(half_base, T_frames - half_base, STRIDE)
assert len(centers_sk) > 0, "الفيديو أقصر من نافذة واحدة"


def build_windows(win_frames):
    """نوافذ بطول win_frames حوالين centers_sk، كل واحدة مضغوطة لـ MODEL_FRAMES"""
    half = win_frames // 2
    out = np.empty((len(centers_sk), MODEL_FRAMES, 34), dtype=np.float32)
    for i, c in enumerate(centers_sk):
        lo = max(0, c - half)
        hi = min(T_frames, c + half + 1)
        # نفس الـ subsampling بتاع process_skeleton_sample بالحرف: نضغط
        # النافذة كلها لـ MODEL_FRAMES بدل ما ناخد أول 30 فريم منها
        sub = np.linspace(lo, hi - 1, MODEL_FRAMES, dtype=int)
        # نفس دالة التدريب بالظبط — مفيش نسخة تانية تروح تختلف عنها
        out[i] = normalize_skeleton(all_keypoints[sub])
    return out


def mirror_windows(W):
    """قلب أفقي — نفس اللي بيتعمل في الـ augmentation، بس هنا كـ TTA"""
    s = W.reshape(len(W), MODEL_FRAMES, 17, 2)[:, :, COCO_FLIP, :].copy()
    s[..., 0] *= -1
    return s.reshape(len(W), MODEL_FRAMES, -1)


# النافذة الأساسية (3s) هي المرجع للتوقيت
window_starts_sk = np.array([frame_indices[max(0, c - half_base)] / fps for c in centers_sk])
window_ends_sk = np.array([frame_indices[min(T_frames - 1, c + half_base)] / fps
                           for c in centers_sk])
window_times_sk = frame_indices[centers_sk] / fps

print(f"✅ النافذة الأساسية: {WINDOW_FRAMES} فريم = {WINDOW_FRAMES/effective_fps:.1f}s "
      f"-> مضغوطة لـ {MODEL_FRAMES} صف (زي التدريب)")
print(f"✅ عدد المراكز: {len(centers_sk)} | مقاييس النوافذ: {WINDOW_SCALES}")

# ----------------------------------------------------------------------
# حدود زمنية للسيجمنتس
# ----------------------------------------------------------------------
# window_times_sk هي مراكز النوافذ، والنافذة نفسها طولها WINDOW_SECONDS. لو حسبنا
# المدة بمركز البداية ناقص مركز النهاية هنقلّل كل سيجمنت بحوالي ثانية.
# فبناخد الحد بين كل نافذتين متجاورتين عند نص المسافة بين مركزيهم — كده
# المدد بتبقى متلاصقة ومجموعها = المدى المغطى كله من غير عدّ مزدوج.
edges_sk = np.empty(len(window_times_sk) + 1)
edges_sk[0] = window_starts_sk[0]
edges_sk[1:-1] = (window_times_sk[:-1] + window_times_sk[1:]) / 2
edges_sk[-1] = window_ends_sk[-1]
covered_span = edges_sk[-1] - edges_sk[0]

# ----------------------------------------------------------------------
# التوقع: 4 مقاييس × (أصلي + مقلوب) = 8 تمريرات، بالمتوسط
# ----------------------------------------------------------------------
scale_probs = []
for sec in WINDOW_SCALES:
    W = build_windows(int(round(sec * effective_fps)))
    assert W.shape[1:] == (MODEL_FRAMES, 34), "شكل النافذة مش زي اللي الموديل اتدرب عليه"
    p = model_skeleton.predict(W, verbose=0)
    p_m = model_skeleton.predict(mirror_windows(W), verbose=0)
    scale_probs.append((p + p_m) / 2.0)
    print(f"   مقياس {sec}s: تم")
raw_probs = np.mean(scale_probs, axis=0)

# ----------------------------------------------------------------------
# معايرة الثقة (temperature scaling) على الـ validation
# ----------------------------------------------------------------------
# CONF_REJECT رقم بلا معنى لو الموديل مش معاير. شبكات التصنيف بتطلع
# واثقة أكتر من اللازم بشكل منهجي، يعني 0.90 الحقيقية ممكن تبقى 0.65.
# بندوّر على درجة حرارة واحدة بتقلّل الـ NLL على الـ validation، وبنطبقها
# على توقعات الفيديو — بعد كده العتبة بتبقى احتمال حقيقي.
def temp_scale(p, temp):
    lg = np.log(np.clip(p, 1e-12, None)) / temp
    lg -= lg.max(axis=1, keepdims=True)
    e = np.exp(lg)
    return e / e.sum(axis=1, keepdims=True)


val_probs = model_skeleton.predict(X_val_sk, verbose=0)
val_idx = np.argmax(y_val_sk, axis=1)
grid_T = np.arange(0.5, 3.01, 0.05)
nlls = [-np.log(np.clip(temp_scale(val_probs, t)[np.arange(len(val_idx)), val_idx],
                        1e-12, None)).mean() for t in grid_T]
i_best = int(np.argmin(nlls))
i_one = int(np.argmin(np.abs(grid_T - 1.0)))   # مش .index() — 1.0 مش بالظبط في arange
BEST_T = float(grid_T[i_best])
print(f"\n🌡️ درجة الحرارة المعايرة: {BEST_T:.2f} "
      f"(NLL على الـ val: {nlls[i_best]:.4f} بدل {nlls[i_one]:.4f} عند 1.0)")
raw_probs = temp_scale(raw_probs, BEST_T)


def smooth_probs(probs, k):
    """متوسط متحرك على محور الزمن — بيشيل الرفرفة بين النوافذ المتجاورة"""
    if k <= 1:
        return probs
    pad = k // 2
    padded = np.pad(probs, ((pad, pad), (0, 0)), mode='edge')
    kernel = np.ones(k) / k
    return np.stack([np.convolve(padded[:, c], kernel, mode='valid')
                     for c in range(probs.shape[1])], axis=1)


def enforce_min_duration(labels, min_len, protect=None):
    """
    أي أكشن ظهر لفترة أقصر من min_len نافذة بنعتبره ضوضاء وبنمدد اللي قبله.

    protect: لابل مستثنى من الشيل. بنمرر فيه other — لأن other هنا قرار
    مقصود ("مش عارفين")، فلو شلناه بنستبدل إجابة أمينة بأكشن غلط. العكس
    مسموح: other يقدر يبلع أكشن قصير قبله.
    """
    labels = labels.copy()
    i = 0
    while i < len(labels):
        j = i
        while j < len(labels) and labels[j] == labels[i]:
            j += 1
        if j - i < min_len and i > 0 and labels[i] != protect:
            labels[i:j] = labels[i - 1]
        i = j
    return labels


probs_sk = smooth_probs(raw_probs, SMOOTH_K)
argmax_sk = np.argmax(probs_sk, axis=1)
max_conf = probs_sk.max(axis=1)

# ----------------------------------------------------------------------
# الرفض
# ----------------------------------------------------------------------
# 'other' فهرس صناعي بعد آخر كلاس حقيقي — مالوش عمود في probs_sk، لأن
# الموديل ماتدربش عليه. النافذة اللي الموديل مش واثق في أي كلاس فيها
# بتتحط other.
class_labels_list = [action_names[le_skeleton.classes_[i]] for i in range(num_classes_skeleton)]
OTHER_IDX = num_classes_skeleton
class_labels_list.append('other')

# سويب على العتبة عشان نظبطها من غير ما نعيد الـ run كله
print("\n🎚️ نسبة الرفض عند عتبات ثقة مختلفة:")
for th in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
    r = (max_conf < th).mean()
    mark = "  <-- المستخدمة" if abs(th - CONF_REJECT) < 1e-9 else ""
    print(f"   {th:.2f} -> {r*100:5.1f}% من النوافذ تبقى other{mark}")

predicted_classes_sk = argmax_sk.copy()
rejected = max_conf < CONF_REJECT
predicted_classes_sk[rejected] = OTHER_IDX
predicted_classes_sk = enforce_min_duration(predicted_classes_sk, MIN_SEGMENT,
                                            protect=OTHER_IDX)

# الثقة لازم تتحسب على الكلاس النهائي بعد التعديل مش على الـ argmax الأصلي.
# الـ other مالوش عمود في probs_sk، فبنستبدل فهرسه بـ 0 عشان الفهرسة ما
# تطلعش برّه الحدود، وبعدين بنكتب فوقها أعلى احتمال (اللي اترفض) — ده
# بيوريك الموديل كان قريب قد إيه من إنه يقرر.
is_other = predicted_classes_sk == OTHER_IDX
safe_idx = np.where(is_other, 0, predicted_classes_sk)
predicted_confidence_sk = probs_sk[np.arange(len(safe_idx)), safe_idx].copy()
predicted_confidence_sk[is_other] = max_conf[is_other]

predicted_actions_sk = [class_labels_list[c] for c in predicted_classes_sk]

n = len(max_conf)
print(f"\n🔎 توزيع الثقة على {n} نافذة:")
for p in (10, 25, 50, 75, 90):
    print(f"   p{p:<2d}: {np.percentile(max_conf, p)*100:5.1f}%")
print(f"   اترفضت بالعتبة {CONF_REJECT:.2f}: {int(rejected.sum())} نافذة ({rejected.sum()/n*100:.0f}%)")
print(f"   الإجمالي النهائي other: {int(is_other.sum())} نافذة ({is_other.sum()/n*100:.0f}%)")

# بنطبع سيجمنتس متصلة بدل كل نافذة لوحدها عشان النتيجة تبقى مقروءة،
# ومعاها التوقع التاني عشان نشوف الموديل كان متردد بين إيه وإيه
second = np.argsort(probs_sk, axis=1)[:, -2]

segments_sk = []   # (بداية, نهاية, مدة, أكشن, ثقة, التوقع التاني)
i = 0
while i < len(predicted_actions_sk):
    j = i
    while j < len(predicted_actions_sk) and predicted_actions_sk[j] == predicted_actions_sk[i]:
        j += 1
    t0, t1 = edges_sk[i], edges_sk[j]
    segments_sk.append((t0, t1, t1 - t0, predicted_actions_sk[i],
                        predicted_confidence_sk[i:j].mean(),
                        class_labels_list[np.bincount(second[i:j]).argmax()]))
    i = j

print(f"\n📊 السيجمنتس ({len(segments_sk)}) — المدى المغطى "
      f"{edges_sk[0]:.1f}s إلى {edges_sk[-1]:.1f}s = {covered_span:.1f}s "
      f"من {total_frames/fps:.1f}s فيديو:\n")
print(f"   {'من':>7} {'لـ':>7} {'المدة':>7}  {'الأكشن':<16} {'ثقة':>4}  التاني")
for t0, t1, dur, act, conf, alt in segments_sk:
    print(f"   {t0:6.1f}s {t1:6.1f}s {dur:6.1f}s  {act:<16} {conf*100:3.0f}%  {alt}")

# ----------------------------------------------------------------------
# إجمالي الوقت لكل أكشن
# ----------------------------------------------------------------------
totals = {}
for _, _, dur, act, _, _ in segments_sk:
    t, c = totals.get(act, (0.0, 0))
    totals[act] = (t + dur, c + 1)

print(f"\n⏳ إجمالي الوقت لكل أكشن:\n")
print(f"   {'الأكشن':<16} {'الإجمالي':>9} {'من الفيديو':>10} {'مرات':>6} {'أطول مرة':>9}")
for act, (tot, cnt) in sorted(totals.items(), key=lambda kv: -kv[1][0]):
    longest = max(d for _, _, d, a, _, _ in segments_sk if a == act)
    print(f"   {act:<16} {tot:8.1f}s {tot/covered_span*100:9.1f}% {cnt:6d} {longest:8.1f}s")
print(f"   {'—':<16} {sum(t for t, _ in totals.values()):8.1f}s (مجموع التحقق)")
