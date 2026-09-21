# 📖 كيفية تحميل الـ Notebook على Kaggle

---

## الملفات اللي محتاج ترفعها

### ✅ ملف واحد بس:

```
📄 fastdtw/fastdtw_kaggle_notebook.ipynb
```

ده ملف `.ipynb` كامل جاهز للـ Kaggle.

---

## الخطوات

### خطوة 1️⃣: اذهب لـ Kaggle

1. روح https://www.kaggle.com/
2. لو ما متسجّلش → سجّل دخول
3. اضغط على **Create** → **Notebook**
4. اختر **Python**

---

### خطوة 2️⃣: أضف الملف (طريقتان)

#### الطريقة الأولى: Upload مباشر

1. في الـ notebook اللي فتحته، اضغط على **+Code** أعلى الشاشة
2. ابحث عن زر **Upload**
3. اختر الملف: `fastdtw_kaggle_notebook.ipynb`
4. اختر **Import**

#### الطريقة الثانية: Copy-Paste الكود

1. اقتح الملف `fastdtw_kaggle_notebook.ipynb` (في جهازك)
2. Copy محتوى الـ cells
3. Paste في الـ notebook على Kaggle

---

### خطوة 3️⃣: شغّل الخلايا بالترتيب

اضغط **Run** على كل خلية:

#### ✅ الخلية الأولى (Clone)

```python
!git clone -q -b hmdb51-frame-scale-experiment https://github.com/gradcs2027/Test1.git /kaggle/working/Test1
%cd /kaggle/working/Test1
!git log --oneline -1
```

**الخرج:** آخر commit على الفرع. لو مش اللي انت مستنيه، يبقى اسم الفرع في
السطر اللي فوق بقى قديم — غيّره.

⚠️ اسم الفرع بيتغيّر مع الشغل. الفرع الحالي `hmdb51-frame-scale-experiment`.

---

#### ✅ الخلية الثانية (تحقق من المسارات)

```python
!python shared/paths.py
```

**الخرج الصح:**

```
الفيديوهات المتاحة (4):
  vidtest1    1125 فريم @  30.0 fps
  vidtest2     854 فريم @  30.0 fps
  vidtest3    1896 فريم @  15.0 fps
  vidtest4    2213 فريم @  29.4 fps
```

⚠️ **لو طلع (0):** في مشكلة، بس تعود للخطوة الأولى.

---

#### ✅ الخلية الثالثة (FastDTW — الرقم الرئيسي) ⭐

```python
!python fastdtw/run_crossvideo.py
```

**الخرج:**

```
🔬 الاختبار النظيف

✓ الصح: 5 من 12
📊 الدقة: 41.7%

بكل حركة:
  wave: 3/5 = 60%
  sit_down: 1/4 = 25%
  stand_up: 1/3 = 33%
```

---

#### ⚙️ الخلايا الاختيارية

اختر اللي تحب:

```python
# تجربة متضخمة
!python fastdtw/run_vidtest2.py

# تنبؤ أعمى
!python fastdtw/run_vidtest4.py

# شرح تفصيلي
!python fastdtw/demo_oneshot.py
```

---

## المدة المتوقعة

| الخلية | الوقت |
|---|---|
| Clone | 30 ثانية |
| تحقق من المسارات | 2 ثانية |
| run_crossvideo.py | 6 ثواني |
| run_vidtest4.py | 29 ثانية |
| run_vidtest3.py | 43 ثانية |

**الإجمالي:** ~1 دقيقة (للأساسي)

---

## إذا حصلت مشكلة

### مشكلة: `ModuleNotFoundError`

**الحل:**

```python
!pwd  # تحقق من المجلد الحالي
!ls -la
```

تأكد أنك في `/kaggle/working/Test1` — `shared/` و `fastdtw/` قاعدين في جذر
الريبو على طول، مافيش فولدر جوّاه.

---

### مشكلة: `الفيديوهات المتاحة (0)`

**الحل:**

```python
!ls -la shared/keypoints/
```

تأكد من وجود الملفات اللي فيها البيانات.

---

## تعديل الـ Notebook (اختياري)

لو بتحب تضيف شرح أو تعديلات:

1. اضغط **+ Text** لإضافة شرح
2. اضغط **+ Code** لإضافة كود

---

## حفظ النتائج

**Kaggle بيحفظ automatically** لكن تقدر:

1. اضغط **Save Version** أعلى الشاشة
2. أضف وصف (مثلاً "FastDTW Results 41.7%")

---

## Share مع الدكتور

1. اضغط على **Share** أعلى الشاشة
2. اختر **Share Notebook**
3. ابعت الـ link للدكتور

---

## ملخص الـ Commands

```python
# 1. Clone
!git clone -q -b hmdb51-frame-scale-experiment https://github.com/gradcs2027/Test1.git /kaggle/working/Test1
%cd /kaggle/working/Test1

# 2. تحقق
!python shared/paths.py

# 3. الرقم الرئيسي
!python fastdtw/run_crossvideo.py

# 4. اختياري
!python fastdtw/run_vidtest4.py
!python fastdtw/run_vidtest2.py
!python fastdtw/demo_oneshot.py
```

---

## النتيجة النهائية اللي تقول للدكتور

```
تشغلت على Kaggle بدون أي مشاكل.

الرقم الرئيسي: 41.7%
(اختبار نظيف، بدون تسريب بيانات)

الأرقام التانية:
- 92.3% على vidtest2 (متضخمة)
- 16.1% على vidtest3 (نظيفة)
- اكتشاف 93.9% على vidtest4 (أعمى)
```

---

**Done! الـ Notebook جاهز والـ Guide كامل! 🎉**
