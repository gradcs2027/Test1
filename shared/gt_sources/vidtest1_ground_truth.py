"""
Ground Truth لـ vidtest1.mp4 — محدّث (2026-09-01)

المصدر: **صاحب المشروع، وصف يدوي مصحّح** (2026-09-01).

⚠️ النسخة القديمة كانت فيها أخطاء — تم التصحيح:
   - أخطاء في تعريف الحركات
   - فترات زمنية غير دقيقة

الحركات الموجودة:
  • clapping (تصفيق)
  • wave (تلويح)
  • sitting (جلوس)
  • stand_up (قيام)
  • hugging (عناق)

اصطلاح اللابلز:
    'other*'  = سكون **متأكّد منه** — بيتحسب
    '?'       = مش موصوفة / مش صالحة — **بتتستثنى من الحساب تماماً**
"""

# (بداية_ثانية, نهاية_ثانية, اللابل)
ground_truth_vidtest1 = [
    (0.0,   3.0,  'clapping'),      # تصفيق
    (3.0,  10.0,  'wave'),          # تلويح
    (10.0, 12.0,  '?'),             # غير موصوف
    (12.0, 14.0,  'sitting'),       # جلوس
    (14.0, 18.0,  '?'),             # غير موصوف
    (18.0, 20.0,  'stand_up'),      # قيام
    (20.0, 21.0,  '?'),             # غير موصوف
    (21.0, 26.0,  'hugging'),       # عناق
    (26.0, 29.0,  '?'),             # غير موصوف
    (29.0, 31.0,  'sitting'),       # جلوس (مرة تانية)
    (31.0, 32.0,  'stand_up'),      # قيام (مرة تانية)
    (32.0, 33.0,  '?'),             # غير موصوف
    (33.0, 37.0,  'wave'),          # تلويح (مرة تانية)
    (37.0, 37.5,  '?'),             # آخر الفيديو
]

# الحركات اللي بتتكرر — دي بس اللي ينفع عليها اختبار نظيف داخل الفيديو
REPEATED = ['wave', 'sitting', 'stand_up']

video_info = {
    'filename': 'vidtest1.mp4',
    'duration': 37.48,
    'fps': 60,
    'effective_fps': 30.0,
    'n_frames': 1125,
    'detect_rate': 1.00,
    'n_labelled_actions': 5,
    'verified_by_owner': False,      # ⚠️ من وصف صاحب المشروع (محدّث)
}

# أسماء بديلة للتوافق مع الملفات القديمة
ground_truth_segments_vidtest1 = ground_truth_vidtest1
ground_truth_segments = ground_truth_vidtest1


def summary():
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    from ground_truth import print_video
    print_video('vidtest1')


if __name__ == '__main__':
    summary()
