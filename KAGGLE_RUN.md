# تشغيل FastDTW على Kaggle

انسخ ده في خلية واحدة في أي نوتبوك على Kaggle وشغّلها.

```python
!git clone -q -b hmdb51-frame-scale-experiment https://github.com/gradcs2027/Test1.git /kaggle/working/Test1
%cd /kaggle/working/Test1
!git log --oneline -1
!python shared/paths.py
```

⚠️ **الـ `-b` مش اختياري.** الشغل الحالي على فرع
`hmdb51-frame-scale-experiment`، و `main` وراه بكتير. من غير الـ `-b` هتاخد
`main` وهتقع بـ `ModuleNotFoundError` أو هتلاقي ملفات ناقصة.

⚠️ **اسم الفرع ده بيتغيّر مع الشغل.** لو الشغل اتنقل لفرع جديد، غيّر الاسم
في السطر اللي فوق. سطر `git log --oneline -1` موجود عشان كده بالظبط: لو
آخر commit مش اللي انت مستنيه، يبقى انت على فرع قديم. (استنساخ فرع ملغي
بيقع بصوت عالي `Remote branch not found`، لكن استنساخ فرع **قديم موجود**
بينجح بالسكوت — ده اللي بيضيّع الوقت.)

⚠️ **الجلسة لما تعمل ريستارت، `/kaggle/working` بيتمسح بالكامل** — الريبو
وكل المخرجات والكاش بيروحوا. لازم تعيد الخلية دي من الأول.

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

## مسار التصنيف: مش محتاج pip install، ولا داتاسِت، ولا GPU

**مافيش حاجة تتثبّت** للتصنيف. `fastdtw_core.py` تطبيق محلي في الريبو، مش
حزمة pip. مسار الـ few-shot كله معتمد على **numpy بس**.

(ده بيخصّ التصنيف بس. استخراج الـ pose محتاج تثبيت — تحت.)

**مافيش داتاسِت.** التصنيف بيشتغل على الـ keypoints (828 KB، متسجّلة في git
فبتيجي مع الـ clone) مش على ملفات الـ mp4.

**مافيش GPU.** كله numpy على CPU.

الفيديوهات (`/kaggle/input/testvid/`) محتاجينها في حالتين بس:

```python
!python shared/pose_extract.py vidtest4   # إعادة استخراج الـ pose بـ YOLO
!python fastdtw/render_pred.py            # رسم التوقّعات على الفيديو
```

دول محتاجين كمان `ultralytics` (اللي جوّاه YOLO) و `opencv`.

⚠️ **`ultralytics` مش متثبّت على Kaggle.** الملف ده كان بيقول إنه موجود
أصلاً، وده كان صح زمان وبقى غلط — الصورة بتاعة Kaggle اتغيّرت. أي سكريبت
بيستخرج pose بيقع بـ `ModuleNotFoundError: No module named 'ultralytics'`
(اتأكّد 2026-09-19). قبله بخلية:

```python
!pip install -q ultralytics
```

`opencv` موجود أصلاً. والسكريبتات اللي بتستخرج pose هي:
`shared/pose_extract.py` · `fastdtw/build_local_templates.py` ·
`fastdtw/build_external_templates.py` · `fastdtw/experiment_ten_actions.py`.

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
| `ModuleNotFoundError: fastdtw_core` | مش في الفولدر الصح. لازم تكون جوّه `/kaggle/working/Test1` |
| `ModuleNotFoundError: ultralytics` | `!pip install -q ultralytics` — مش متثبّت على Kaggle |
| `can't open file '/kaggle/working/fastdtw/...'` | الجلسة عملت ريستارت ومسحت `/kaggle/working`. أعد خلية الاستنساخ |
| `cannot change to '/kaggle/working/Test1'` | نفس الحاجة — الريبو اتمسح |
| `ModuleNotFoundError: paths` | سطر `import _bootstrap` اتشال أو اتنقل تحت الاستيرادات — لازم يفضل أول واحد |
| `مش قادر أفتح <مسار>.mp4` | ده `render_*.py` بس — محتاج الداتاسِت متوصّل بالنوتبوك |

التفاصيل الكاملة في `HANDOFF.md` قسم 5.5، والتفاصيل بتاعة الـ FastDTW
نفسها في [`fastdtw/HANDOFF.md`](fastdtw/HANDOFF.md).
