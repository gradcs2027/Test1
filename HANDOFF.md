# HANDOFF — تسليم الشغل

> الملف ده بيتحدّث بعد كل خطوة جديدة. اقرأه من فوق لتحت عشان تفهم إحنا فين.
>
> **آخر تحديث:** 2026-08-30

---

## 1. المشروع ايه؟

التعرّف على أفعال الإنسان (Human Action Recognition) من الـ skeleton.

```
فيديو ──> YOLOv8n-pose ──> 17 keypoint لكل frame ──> BiLSTM ──> اسم الحركة
```

- **الـ training data:** NTU-60 (2D skeletons)، الـ split هو Cross-Subject (X-Sub)، حوالي 56,000 عينة
- **الـ model:** Bidirectional LSTM (256 → 128 → 128)
- **الـ inference:** sliding window على الفيديو، كل window 3 ثواني، الـ stride 0.33 ثانية

## 2. الهدف من الشغل الحالي

نثبت إن الـ **LSTM أحسن من الـ baselines الأبسط**، وده بمقارنتها بـ:

| Algorithm | الفكرة |
|---|---|
| **LSTM** | شبكة عصبية اتدرّبت على 56k عينة |
| **DTW** (Dynamic Time Warping) | template matching — بيلوي الزمن عشان يطابق حركتين بأطوال مختلفة |
| **$1 Recognizer** | normalize (resample → rotate → scale → translate) وبعدين مقارنة مباشرة |

المقارنة على **3 فيديوهات**: `vidtest1` (37.5s)، `vidtest2` (28.5s)، `vidtest3` (126.5s).
**الأهم فيهم `vidtest3`** لأنه الأطول والأصعب.

---

## 3. النتايج — الحالة الحقيقية دلوقتي

⚠️ **مهم:** لسه **نتيجة واحدة بس** هي اللي اتقاست فعلاً. الباقي لسه ما اتشغّلش.

| Algorithm | vidtest1 | vidtest2 | vidtest3 |
|---|---|---|---|
| **LSTM** | **71.8%** ✅ مقاسة | ⬜ لسه | ⬜ لسه |
| **DTW** | ⬜ لسه | ⬜ لسه | ⬜ لسه |
| **$1** | ⬜ لسه | ⬜ لسه | ⬜ لسه |

**تفصيل نتيجة LSTM على vidtest1 (الوحيدة المقاسة):**
- Overall accuracy: **71.8%**
- Real recall (الحركات الحقيقية): **72.6%**
- Other recall (السكون / مش حركة): **70.4%**

**الأرقام المتوقعة (تخمين — مش نتايج!):** DTW حوالي 40-50%، $1 حوالي 20-30%.
الأرقام دي مكتوبة كـ placeholders في `FULL_PLAN.md` و `run_all_baselines.py` — **متتقالش للدكتور كأنها نتايج** لحد ما نشغّلها فعلاً.

---

## 4. أهم اكتشاف تقني (ده اللي ضيّع وقت كتير)

لما قلّلنا عدد الكلاسات (من 12 → 4 → 3):
- الـ **accuracy على NTU طلعت** (97.4% → 100%) — منطقي، المسألة بقت أسهل
- بس الأداء على **الفيديو بقى أوحش** ❌

**السبب:** لما الكلاسات تقلّ، الـ softmax بيتشبّع (saturation) — الموديل بيدّي confidence عالي على طول حتى لما يكون غلطان. وده بيكسّر الـ **rejection mechanism** اللي شغّال بـ `CONF_REJECT = 0.60`، فالموديل ما بيبقاش قادر يقول "الحركة دي مش من اللي أنا عارفها".

**الخلاصة:** رجّعنا الـ setup الأصلي بـ 10 كلاسات.

```python
action_names = {
    0: 'drink_water',  7: 'sit_down',    8: 'stand_up',   9: 'clap',
   22: 'wave',        27: 'phone_call', 33: 'rub_hands', 34: 'nod_head',
   39: 'cross_hands', 43: 'touch_head',
}
```

---

## 5. الملفات — ايه بيعمل ايه

### الـ pipeline الأساسي (الـ LSTM)
| الملف | الدور |
|---|---|
| `notebook520b8d324a.ipynb` | النوتبوك الكامل (ده الأصل) |
| `cell2_other_class.py` | تعريف الكلاسات + كلاس `other` |
| `cell3_normalization.py` | تطبيع الـ skeleton (توسيط على الـ mid-hip، قسمة على طول الجذع، resample لـ 30 frame) |
| `cell4_split.py` | تقسيم X-Sub |
| `cell5_model.py` | معمارية الـ BiLSTM |
| `cell6_eval.py` | التقييم على NTU |
| `cell7_video.py` / `cell8_extract.py` | تحميل الفيديو + استخراج الـ keypoints |
| `cell9_smoothing.py` | temporal smoothing + temperature calibration |
| `cell10_timeline.py` | رسم الـ timeline + الـ scoring |

### الـ baselines (الشغل الجديد)
| الملف | الدور |
|---|---|
| `baselines_common.py` | دوال مشتركة: `extract_templates`, `setup_sliding_windows`, `build_windows_for_baseline`, `moving_average_predictions`, `enforce_min_duration`, `score_predictions` |
| `fastdtw_core.py` | **تنفيذ FastDTW** (Salvador & Chan 2007) — coarsening / projection / refinement |
| `notebook_fastdtw.py` | **تشغيل FastDTW على الفيديوهات** ← ده اللي تستخدمه |
| `notebook_dtw.py` | ⚠️ نسخة قديمة، **متعرضش نتيجتها** (شوف تحت) |
| `notebook_dollar1.py` | الـ $1 Recognizer كامل |
| `run_all_baselines.py` | runner بيجمع النتايج ويرسم المقارنة |
| `comparison_results.py` | جدول المقارنة |

### الـ Ground Truth
| الملف | الدور |
|---|---|
| `vidtest2_ground_truth.py` | اتصحّح: `phone_call` بقت 18.2-23.6 متواصلة، الـ closeups اتشالت (`?`) |
| `vidtest3_ground_truth.py` | اتستخرج يدوي من contact sheets — 4 حركات: `sit_down` ×2، `stand_up` ×2 |

### التوثيق
| الملف | الدور |
|---|---|
| `HANDOFF.md` | الملف ده |
| `README.md` | نظرة عامة |
| `BASELINES_README.md` | شرح الـ baselines + خطوات التشغيل |
| `FULL_PLAN.md` | خطة التشغيل على Kaggle |
| `QA.md` | أسئلة الدكتور المتوقعة + إجاباتها |

---

## 6. تفاصيل مهمة للي هيكمّل

**تطبيع الـ skeleton لازم يكون واحد في التدريب والفيديو** — أي اختلاف بيكسّر النتيجة:
1. توسيط على الـ mid-hip
2. قسمة على median طول الجذع
3. resample لـ 30 frame (بـ `linspace` subsampling)

**إعدادات الـ inference:**
- `CONF_REJECT = 0.60` — أقل من كده يترمي كـ `other`
- moving average بـ `k=5`
- `MIN_SEGMENT = 2` — أي segment أقصر من كده يتشال (ما عدا `other`)
- multi-scale windows: 1.5s / 2s / 3s / 4s + mirror TTA
- temperature scaling: grid search من 0.5 لـ 3.0

**فرق مهم:** `vidtest3` بـ **30fps**، بينما `vidtest1` و `vidtest2` بـ **60fps**. خلي بالك في أي حساب بالثواني.

**اصطلاح اللابلز في الـ ground truth:**
- `'other*'` = فترة سكون (واقف/قاعد ثابت) — بتتحسب
- `'?'` = frames مش صالحة (closeup مثلاً) — **مبتتحسبش خالص**

---

## 6.5 مشكلة الـ rejection في الـ baselines ⚠️ (مهمة جداً)

**المشكلة:** مفيش كلاس `other` **متدرّب** في المشروع ده. الرفض في الـ LSTM بيتم
بعتبة ثقة (`CONF_REJECT = 0.60`) — لو الموديل مش واثق، الإجابة تبقى `other`.
(اتجرّب كلاس `other` متدرّب في رن 18/19 وفشل: قال `other` صفر مرة من 77 نافذة.)

**الخطأ اللي كان في `notebook_dtw.py`:** بياخد `min(distances)` على طول، فمستحيل
يقول `other` خالص. والفيديوهات معظمها سكون — `vidtest3` فيه ~120 ثانية من 126.5
هي `other`. يعني الـ DTW كانت هتطلع برقم واطي **مش لأنها ضعيفة، لأننا مانعناها
تجاوب صح**. عرض الرقم ده كمقارنة عادلة = تضليل، والدكتور هيمسكها.

**الحل في `notebook_fastdtw.py`:** عتبة رفض **معايرة من بيانات التدريب**:
1. ناخد 30 عينة من كل كلاس
2. نحسب مسافة FastDTW من كل عينة لـ template بتاع كلاسها
3. العتبة = المئوية 80 من التوزيع ده
4. أي نافذة في الفيديو مسافتها أكبر من العتبة => `other`

كده الرفض مبني على البيانات مش على رقم مختار بالمزاج، وموازي لمنطق الـ LSTM.

---

## 6.6 نتايج FastDTW — القياسات التقنية

الـ implementation **اتتحقق منه محلياً** قبل التشغيل:

| الاختبار | النتيجة |
|---|---|
| نفس السلسلة مع نفسها | مسافة = 0 ✅ |
| إشارة ممطوطة ×2 مقابل الأصلية | مسافة = 0 ✅ (الـ warp شغال) |
| مقابل إشارة عشوائية | مسافة = 65.2 ✅ |
| الفرق عن الـ DTW الكامل عند n=30 | **0.000%** ✅ |

**إثبات الـ O(n) مقابل O(n²) بالأرقام** (34 بعد، radius=1):

| n | خلايا DTW كامل | خلايا FastDTW | النسبة | كسب السرعة |
|---|---|---|---|---|
| 30 | 900 | 391 | 0.43 | 1.41x |
| 60 | 3,600 | 855 | 0.24 | 2.55x |
| 120 | 14,400 | 1,799 | 0.12 | 6.02x |
| 240 | 57,600 | 3,703 | 0.06 | 10.94x |
| 480 | 230,400 | 7,527 | 0.03 | **24.50x** |

**اقرا العمود التاني:** لما `n` تتضاعف، خلايا الـ DTW الكامل بتتربّع (×4)،
بينما خلايا الـ FastDTW بتتضاعف بس (×2). ده الفرق بين O(n²) و O(n) بالأرقام.

**نقطة مهمة للدكتور:** عند n=30 (اللي إحنا شغالين عليه) الـ FastDTW بيدّي
**نفس إجابة الـ DTW الكامل بالظبط** (خطأ 0.000%) وأسرع 1.41x. يعني مفيش أي
تضحية بالدقة مقابل السرعة عند الطول ده.

⬜ **لسه:** الـ accuracy على الفيديوهات — محتاج تشغيل على Kaggle.

---

## 7. الخطوات الجاية

```
☐ 1. شغّل FastDTW على vidtest1  ← الخطوة الجاية دلوقتي
☐ 2. شغّل FastDTW على vidtest2 و vidtest3
☐ 3. شغّل LSTM على vidtest2 و vidtest3
☐ 4. شغّل $1 على الـ 3 فيديوهات
☐ 5. حط النتايج الحقيقية في run_all_baselines.py وشيل الـ placeholders
☐ 6. اعمل الجدول النهائي للدكتور
```

**تشغيل الـ FastDTW على Kaggle:**
```python
# بعد ما تشغّل خلايا الـ notebook الأصلي (عشان X_train_sk و all_keypoints يبقوا متحملين)
exec(open('notebook_fastdtw.py').read())
```

عشان تغيّر الفيديو، عدّل `VIDEO_NAME` في أول `notebook_fastdtw.py`
(`'vidtest1'` / `'vidtest2'` / `'vidtest3'`). الـ fps بيتظبط لوحده.

---

## 8. لوج الشغل (بيتزوّد مع كل خطوة)

### 2026-08-30
- ✅ رجّعنا الـ setup الأصلي بـ 10 كلاسات بعد ما اكتشفنا مشكلة الـ softmax saturation
- ✅ اتكتبت الـ baselines: `baselines_common.py`, `notebook_dtw.py`, `notebook_dollar1.py`, `run_all_baselines.py`
- ✅ اتصحّح `vidtest2_ground_truth.py` (الـ `phone_call` كانت مقسومة غلط)
- ✅ اتعمل `vidtest3_ground_truth.py` — اتستخرج يدوي من contact sheets (overview كل 3 ثواني + zoom كل 0.5 ثانية)
- ✅ اتعمل push للنوتبوك على Kaggle (version 28)
- ✅ اتعمل push للفولدر كله على GitHub → `gradcs2027/Test1`
- ✅ اتعمل `HANDOFF.md` (الملف ده)
- ✅ **اتكتب الـ FastDTW** — `fastdtw_core.py` + `notebook_fastdtw.py`
  - اتتحقق من صحته محلياً: خطأ 0.000% مقابل الـ DTW الكامل عند n=30
  - اتقاس الـ speedup: من 1.41x عند n=30 لـ 24.5x عند n=480
  - اتضافت عتبة رفض معايرة من البيانات (شوف قسم 6.5)
- 🐛 اتصلّح بج في `notebook_dtw.py`: الـ progress print كان بره اللوب فبيطبع رقم غلط
- ⚠️ اتحطّ تحذير في `notebook_dtw.py` إن نتيجته مضللة (مفيش rejection)
