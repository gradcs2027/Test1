"""
Ground Truth لـ vidtest2.mp4 — الملف المرجعي

المصدر: **صاحب المشروع، مشاهدة يدوية للفيديو** (2026-08-31).
ده أعلى مصدر عندنا — أعلى من أي contact sheet.

    3-5     wave
    5-6     قعد        ← التقسيم ده بموافقته صراحةً (2026-08-31)
    6-7     قام        ←
    14-16   بيجري
    18-22   مكالمة

⚠️⚠️ النسخة اللي كانت في `run_oneshot.py` كانت **فيها غلطتين حقيقيتين**:

  ١. **الـ wave و الـ running مكانوش موجودين خالص.** أسوأ من كده،
     الفترة 2.2–5.9 كانت متعلّمة `'other*'` يعني "سكون متأكّد منه" —
     وهي في الحقيقة فيها الـ wave (3–5). يعني أي كشف صحيح للتلويح في
     vidtest2 كان **بيتحسب إنذار كاذب**. الـ baselines كانت بتتعاقب
     على إجابة صح.

  ٢. `phone_call` كانت 18.2–**23.6**، والصح 18–**22**.

  ⚠️ أي رقم اتقاس على vidtest2 من `run_oneshot.py` لازم يتعاد.

⚠️ ليه الفترات 7–14 و 16–18 و 22–28.5 متعلّمة `'?'` مش `'other*'`؟
   لأن صاحب المشروع ماوصفهاش. `'?'` = تتستثنى من الحساب.
   لو حطّيناها `'other*'` (سكون متأكّد منه) نبقى **اخترعنا بيانات** —
   وده بالظبط اللي عمل الغلطة رقم ١ فوق.

اصطلاح اللابلز (متوثّق في HANDOFF قسم 6):
    'other*'  = سكون **متأكّد منه** — بيتحسب
    '?'       = مش موصوفة / مش صالحة — **بتتستثنى تماماً**
"""

# (بداية_ثانية, نهاية_ثانية, اللابل)
ground_truth_vidtest2 = [
    (0.0,   3.0,  '?'),           # مش موصوفة
    (3.0,   5.0,  'wave'),        # بيلوّح
    (5.0,   6.0,  'sit_down'),    # قعد        ← اتقسمت بموافقة صاحب المشروع
    (6.0,   7.0,  'stand_up'),    # قام        ←
    (7.0,  14.0,  '?'),           # مش موصوفة
    (14.0, 16.0,  'running'),     # بيجري
    (16.0, 18.0,  '?'),           # مش موصوفة
    (18.0, 22.0,  'phone_call'),  # مكالمة تليفون
    (22.0, 28.5,  '?'),           # مش موصوفة
]

# مافيش حركة بتتكرر في الفيديو ده — يعني **مافيش اختبار نظيف جوّاه**.
# الاختبار النظيف لـ vidtest2 لازم ييجي من بره: template من vidtest1
# أو vidtest3. شوف ground_truth.py:shared_labels()
REPEATED = []

video_info = {
    'filename': 'vidtest2.mp4',
    'duration': 28.45,
    'fps': 60,
    'effective_fps': 30.0,        # FRAME_SKIP = 2
    'n_frames': 854,
    'detect_rate': 0.966,
    'n_labelled_actions': 5,
    'annotated_until': 22.0,
    'verified_by_owner': True,    # ✅
}

# الحركات المعنونة بس
annotated_actions = [(s, e, l) for s, e, l in ground_truth_vidtest2 if l != '?']

# أسماء بديلة للتوافق مع الملفات القديمة
# (notebook_fastdtw.py:213 بيستورد الاسم ده وكان بيقع لأنه مش موجود)
ground_truth_segments_vidtest2 = ground_truth_vidtest2


def summary():
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    from ground_truth import print_video
    print_video('vidtest2')


if __name__ == '__main__':
    summary()
