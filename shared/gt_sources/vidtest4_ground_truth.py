"""
Ground Truth لـ vidtest4.mp4 — الملف الجديد

المصدر: **صاحب المشروع، وصف يدوي للفيديو** (2026-09-01).

الحركات الموجودة:
  • walking (مشي)
  • wave (تلويح)
  • sitting (جلوس/نوم)
  • stand_up (قيام)
  • clapping (تصفيق)
  • touch_head (لمس الرأس)
  • phone_call (مكالمة)
  • spray_perfume (رش برفان)
  • brush_hair (تسريح الشعر)

اصطلاح اللابلز:
    'other*'  = سكون **متأكّد منه** — بيتحسب
    '?'       = مش موصوفة / مش صالحة — **بتتستثنى من الحساب تماماً**
"""

# (بداية_ثانية, نهاية_ثانية, اللابل)
ground_truth_vidtest4 = [
    (0.0,   2.0,  'walking'),       # مشي
    (2.0,   4.0,  'wave'),          # تلويح
    (4.0,   9.0,  '?'),             # غير موصوف
    (9.0,  10.0,  'sitting'),       # جلوس/نوم
    (10.0, 18.0,  'lying'),         # نوم (lying)
    (18.0, 20.0,  '?'),             # غير موصوف
    (20.0, 21.0,  'stand_up'),      # قيام
    (21.0, 22.0,  '?'),             # غير موصوف
    (22.0, 26.0,  'clapping'),      # تصفيق
    (26.0, 29.0,  'touch_head'),    # لمس الرأس
    (29.0, 33.0,  '?'),             # غير موصوف
    (33.0, 40.0,  'phone_call'),    # مكالمة
    (40.0, 48.0,  '?'),             # غير موصوف
    (48.0, 51.0,  'spray_perfume'), # رش برفان
    (51.0, 53.0,  '?'),             # غير موصوف
    (53.0, 55.0,  'wave'),          # تلويح (مرة تانية)
    (55.0, 57.0,  '?'),             # غير موصوف
    (57.0, 62.0,  'spray_perfume'), # رش برفان (مرة تانية)
    (62.0, 68.0,  '?'),             # غير موصوف
    (68.0, 73.0,  'brush_hair'),    # تسريح الشعر
]

# الحركات اللي بتتكرر — دي بس اللي ينفع عليها اختبار نظيف داخل الفيديو
REPEATED = ['wave', 'spray_perfume']

video_info = {
    'filename': 'vidtest4.mp4',
    'duration': 73.0,
    'fps': 30,
    'effective_fps': 29.4,
    'n_frames': 2213,
    'detect_rate': 1.0,
    'n_labelled_actions': 9,
    'verified_by_owner': False,      # ⚠️ من وصف صاحب المشروع (لسه محتاج تأكيد)
}

# أسماء بديلة للتوافق
ground_truth_segments_vidtest4 = ground_truth_vidtest4
ground_truth_segments = ground_truth_vidtest4


def summary():
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    from ground_truth import print_video
    print_video('vidtest4')


if __name__ == '__main__':
    summary()
