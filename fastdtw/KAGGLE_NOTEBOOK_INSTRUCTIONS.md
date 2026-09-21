# كيفية تشغيل FastDTW على Kaggle

---

## الخطوة الأولى: روح على Kaggle

1. اذهب لـ https://www.kaggle.com/
2. اضغط على **Create** → **Notebook**
3. اختر **Python** (مش R)

---

## الخطوة الثانية: Clone الـ Repo

في أول خلية في الـ Notebook، اكتب:

```python
!git clone -q -b hmdb51-frame-scale-experiment https://github.com/gradcs2027/Test1.git /kaggle/working/Test1
%cd /kaggle/working/Test1
!git log --oneline -1
```

ثم اضغط **Run** (أو Shift+Enter)

⚠️ **لازم الـ `-b`** — الفرع `hmdb51-frame-scale-experiment` فيه الملفات
الحديثة، و `main` وراه بكتير.

⚠️ **اسم الفرع بيتغيّر مع الشغل.** سطر `git log` موجود عشان تتأكد: لو آخر
commit مش اللي انت مستنيه، غيّر اسم الفرع في السطر الأول.

---

## الخطوة الثالثة: تحقق من المسارات

في خلية جديدة:

```python
!python shared/paths.py
```

الخرج يجب أن يكون:

```
الفيديوهات المتاحة (4):
  vidtest1    1125 فريم @  30.0 fps
  vidtest2     854 فريم @  30.0 fps
  vidtest3    1896 فريم @  15.0 fps
  vidtest4    2213 فريم @  29.4 fps
```

لو طلع **"الفيديوهات المتاحة (0)"** = في مشكلة في المسارات.

---

## الخطوة الرابعة: شغّل FastDTW

في خلية جديدة:

```python
!python fastdtw/run_crossvideo.py
```

هيطبع النتائج:

```
🔬 الاختبار النظيف — template من فيديو، اختبار على فيديوهات تانية

✓ الصح: 5 من 12
📊 الدقة: 41.7%

بكل حركة:
  wave: 3/5 = 60%
  sit_down: 1/4 = 25%
  stand_up: 1/3 = 33%
```

---

## الخطوة الخامسة (اختياري): تجارب إضافية

```python
# تجربة على فيديو واحد (متضخمة)
!python fastdtw/run_vidtest2.py

# تنبؤ أعمى على vidtest4
!python fastdtw/run_vidtest4.py

# شرح خطوة بخطوة
!python fastdtw/demo_oneshot.py
```

---

## إذا حصلت مشكلة

### المشكلة: `ModuleNotFoundError: fastdtw_core`

**السبب:** أنت في المجلد الغلط.

**الحل:** تأكد من `%cd /kaggle/working/Test1`

---

### المشكلة: `ModuleNotFoundError: No module named 'ultralytics'`

**السبب:** `ultralytics` مش متثبّت على Kaggle (اتأكّد 2026-09-19). بيلزم
للسكريبتات اللي بتستخرج pose بس، مش للتصنيف.

**الحل:** `!pip install -q ultralytics`

---

### المشكلة: `can't open file '/kaggle/working/fastdtw/...'`

**السبب:** الجلسة عملت ريستارت و `/kaggle/working` اتمسح بالكامل — الريبو
والكاش والمخرجات كلهم راحوا.

**الحل:** أعد خلية الـ clone من الأول.

---

### المشكلة: `الفيديوهات المتاحة (0)`

**السبب:** الـ keypoints مالها مسار.

**الحل:** اكتب:

```python
!python shared/paths.py
```

ثم شوف الخرج:

```
الـ keypoints جاية من: الريبو (جنب الكود)
```

لو كتب شيء تاني = في مشكلة.

---

### المشكلة: `FileNotFoundError`

**السبب:** ملف ناقص من الريبو.

**الحل:**

```python
!git status
!ls -la shared/keypoints/
```

شوف اللي في الـ ls.

---

## النتائج المتوقعة

```
⭐ الرقم الرئيسي:
   run_crossvideo.py = 41.7%
   (template من vidtest1، اختبار على vidtest2/3)

📊 أرقام تانية (متضخمة):
   run_vidtest2.py = 92.3%
   run_vidtest3.py = 16.1%
   run_vidtest4.py = اكتشاف 93.9% (بدون GT)
```

---

## الملفات المهمة في الـ Repo

```
fastdtw/
├── run_crossvideo.py          ⭐ الاختبار النظيف
├── run_vidtest2.py            متضخم
├── run_vidtest3.py            متضخم
├── run_vidtest4.py            أعمى
├── demo_oneshot.py            شرح تفصيلي
└── HANDOFF.md                 شرح شامل

shared/
├── paths.py                   المسارات
├── ground_truth.py            الإجابات الحقيقية
├── keypoints/                 البيانات المحفوظة
└── HANDOFF.md                 شرح مشترك
```

---

## الأوامر الكاملة (كلها مع بعض)

```python
# الخلية الأولى: Clone
!git clone -q -b hmdb51-frame-scale-experiment https://github.com/gradcs2027/Test1.git /kaggle/working/Test1
%cd /kaggle/working/Test1

# الخلية الثانية: تحقق
!python shared/paths.py

# الخلية الثالثة: اختبر
!python fastdtw/run_crossvideo.py

# الخلية الرابعة: اختياري
!python fastdtw/run_vidtest4.py
```

---

## ملاحظات مهمة

✓ **لا تحتاج GPU** — كل الأوامر CPU بس

✓ **لا تحتاج pip install للتصنيف** — fastdtw_core موجود في الريبو

⚠️ **لكن استخراج الـ pose محتاج `!pip install -q ultralytics`** — مش متثبّت
على Kaggle

✓ **المدة المتوقعة:**
   - run_crossvideo = 6 ثواني
   - run_vidtest4 = 29 ثانية
   - run_vidtest3 = 43 ثانية

✓ **الـ keypoints موجودة في الريبو** — مافيش حاجة بدون تحميل

---

## لو بتحب تشتغل محلي (بدون Kaggle)

```bash
cd Test1
python shared/paths.py
python fastdtw/run_crossvideo.py
```

نفس الأرقام بالضبط ✓

---

**أي سؤال؟ اقرا `fastdtw/HANDOFF.md` للتفاصيل الكاملة.**
