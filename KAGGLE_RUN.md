# تشغيل FastDTW على Kaggle

انسخ ده في خلية واحدة في أي نوتبوك على Kaggle وشغّلها.

```python
!git clone -q -b crossvideo-and-gt-unification https://github.com/gradcs2027/Test1.git
%cd Test1/kaggle_nb_520b8d324a
!python shared/paths.py
```

⚠️ **الـ `-b` مش اختياري.** الشغل ده على فرع `crossvideo-and-gt-unification`،
و `main` لسه 6 commits وراه ومافيهاش `paths.py` ولا `run_vidtest4.py`.
من غير الـ `-b` هتاخد `main` وهتقع بـ `ModuleNotFoundError`.

(لما الفرع يتدمج في `main`، شيل الـ `-b` من السطر ده.)

لازم تشوف `الفيديوهات المتاحة (4)`. لو شفت `(0)` اقرا `KP_DIR` اللي طبعه —
هو ده المكان اللي دوّر فيه.

بعدين:

```python
!python fastdtw/run_crossvideo.py   # ⭐ الرقم النضيف — 6 ثواني
!python fastdtw/run_vidtest4.py     # تنبؤ أعمى على vidtest4 — 29 ثانية
!python fastdtw/run_vidtest3.py     # 13 حركة — 43 ثانية
!python fastdtw/run_vidtest2.py     # 5 حركات
```

كل واحد بيكتب نتايجه في `fastdtw/results/`. مافيش `results/` واحد للمشروع —
كل طريقة ليها فولدرها عشان المخرجات ماتختلطش.

⚠️ **مش لازم تعمل `cd fastdtw`.** كل سكريبت بيبدأ بـ `import _bootstrap` اللي
بيحط `shared/` في الـ `sys.path`، فبيشتغل من أي مكان. بس لو عملت `cd`، الأمر
يبقى `!python run_crossvideo.py` من غير البادئة.

---

## مش محتاج: pip install، ولا داتاسِت، ولا GPU

**مافيش حاجة تتثبّت.** `fastdtw_core.py` تطبيق محلي في الريبو، مش حزمة pip.
مسار الـ few-shot كله معتمد على **numpy بس**.

**مافيش داتاسِت.** التصنيف بيشتغل على الـ keypoints (828 KB، متسجّلة في git
فبتيجي مع الـ clone) مش على ملفات الـ mp4.

**مافيش GPU.** كله numpy على CPU.

الفيديوهات (`/kaggle/input/testvid/`) محتاجينها في حالتين بس:

```python
!python shared/pose_extract.py vidtest4   # إعادة استخراج الـ pose بـ YOLO
!python fastdtw/render_pred.py            # رسم التوقّعات على الفيديو
```

دول محتاجين كمان `ultralytics` و `opencv`، وموجودين على Kaggle أصلاً.

---

## اقرا الأرقام صح

`run_crossvideo.py` هو الرقم اللي يتقال. الباقي بياخد الـ template من **نفس
الفيديو** اللي بيختبر عليه، فبيطلّع أرقام أعلى (92.3% على vidtest2) بس فيها
ضعف معروف.

⚠️ الـ 92.3% دي **تلات حركات بس هي اللي اتقاست فعلاً** — `sit_down` و
`stand_up` في vidtest2 طول كل واحدة ثانية واحدة بالظبط فالقصاصة بتاكل
الحركة كلها. متقولش الرقم من غير التحفّظ ده.

⚠️ **أهم سطر في الخرج كله:**

```
🔬 القيد الحقيقي: عندنا 12 ظهور حقيقي للحركة في التلات فيديوهات كلهم.
```

أي رقم من هنا لازم يتقال ومعاه العدد ده. بـ 12 عيّنة، 58.3% مقابل 41.7% خط
أساس الأغلبية بيدّي `p=0.19` — **مش دال إحصائياً**. مينفعش نقول إن الطريقة
"نجحت".

⚠️ `run_vidtest4.py` **مالوش رقم دقة** لأن vidtest4 لسه من غير ground truth.
بيطلّع خط زمني للمراجعة بالعين بس.

---

## لو حاجة وقعت

| الرسالة | السبب |
|---|---|
| `مافيش keypoints لـ vidtestN في <مسار>` | الملفات مش في المسار المطبوع. شغّل `python shared/paths.py` وقارن |
| `الفيديوهات المتاحة (0)` | نفس الحاجة — الـ clone ناقص أو `KEYPOINTS_DIR` متظبّط غلط |
| `ModuleNotFoundError: fastdtw_core` | مش في الفولدر الصح. لازم تكون جوّه `Test1/kaggle_nb_520b8d324a` |
| `ModuleNotFoundError: paths` | سطر `import _bootstrap` اتشال أو اتنقل تحت الاستيرادات — لازم يفضل أول واحد |
| `مش قادر أفتح <مسار>.mp4` | ده `render_*.py` بس — محتاج الداتاسِت متوصّل بالنوتبوك |

التفاصيل الكاملة في `HANDOFF.md` قسم 5.5، والتفاصيل بتاعة الـ FastDTW
نفسها في [`fastdtw/HANDOFF.md`](fastdtw/HANDOFF.md).
