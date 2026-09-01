"""
Ground Truth الحقيقي لـ vidtest3.mp4 — النسخة الصحيحة

⚠️⚠️ الملف القديم `vidtest3_ground_truth.py` **غلط تماماً**.
    كان بيقول إن الفيديو فيه sit_down مرتين و stand_up مرتين وباقيه سكون.
    الحقيقة إن الفيديو فيه **13 حركة مختلفة** ومعظمه حركة مش سكون.

    يعني أي رقم اتقاس على vidtest3 قبل كده (LSTM أو DTW أو FastDTW)
    كان متقاس على ground truth غلط ولازم يتعاد.

المصدر: صاحب المشروع، مشاهدة يدوية للفيديو (2026-08-31).

⚠️ الفترة من 88s لـ 126.5s **مش موصوفة** — الفيديو بيكمل بعدها
   (مكالمة تليفون، صاحبه بيقعد ويقوم، سلام، رجع نام) بس من غير
   توقيتات دقيقة. متعلّمة '?' يعني **تتستثنى من الحساب**، مش 'other'.
   حاطّها 'other' كان هيبقى اختراع بيانات.
"""

# (بداية_ثانية, نهاية_ثانية, اللابل)
#   '?' = مش موصوفة، تتستثنى من الحساب تماماً
ground_truth_vidtest3 = [
    (0.0,   2.0,  'wake_up'),        # نايم وبيصحى
    (2.0,   4.0,  'stand_up'),       # قام            ← نسخة ١ من ٢
    (4.0,   7.0,  '?'),
    (7.0,   9.0,  'wear_glasses'),   # لبس النضارة
    (9.0,  12.0,  '?'),
    (12.0, 21.0,  'drink_water'),    # بيشرب مية
    (21.0, 24.0,  'walking'),        # بيمشي          ← نسخة ١ من ٢
    (24.0, 25.0,  '?'),
    (25.0, 30.0,  'spray_perfume'),  # بيرش برفان
    (30.0, 31.0,  '?'),
    (31.0, 38.0,  'brush_hair'),     # بيسرّح شعره
    (38.0, 40.0,  'walking'),        # بيمشي          ← نسخة ٢ من ٢
    (40.0, 41.0,  '?'),
    (41.0, 58.0,  'sitting'),        # قاعد على الكرسي بيستخدم اللاب
    (58.0, 61.0,  'stand_up'),       # قام            ← نسخة ٢ من ٢
    (61.0, 63.0,  'wave'),           # بيلوّح
    (63.0, 66.0,  '?'),
    (66.0, 68.0,  'hand_shake'),     # بيسلّم على صاحبه
    (68.0, 72.0,  '?'),
    (72.0, 75.0,  'turn_on_light'),  # فتح النور
    (75.0, 84.0,  'play_pingpong'),  # بيلعب بينج بونج
    (84.0, 85.0,  '?'),
    (85.0, 88.0,  'clapping'),       # بيسقّف
    (88.0, 126.5, '?'),              # ⚠️ مش موصوفة — تتستثنى
]

# الحركات اللي بتتكرر — دي بس اللي نقدر نختبرها اختبار نظيف ١٠٠%
# (template من نسخة، اختبار على النسخة التانية)
REPEATED = ['stand_up', 'walking']

video_info = {
    'filename': 'vidtest3.mp4',
    'duration': 126.47,
    'fps': 29.98,
    'effective_fps': 14.99,       # FRAME_SKIP = 2
    'n_frames': 1896,
    'detect_rate': 1.00,
    'n_labelled_actions': 13,
    'annotated_until': 88.0,
    'verified_by_owner': True,    # ✅
}

# أسماء بديلة للتوافق مع الملفات القديمة
ground_truth_segments_vidtest3 = ground_truth_vidtest3
annotated_actions = [(s, e, l) for s, e, l in ground_truth_vidtest3 if l != '?']


def summary():
    import sys
    from collections import defaultdict
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    d = defaultdict(list)
    for s, e, lab in ground_truth_vidtest3:
        if lab != '?':
            d[lab].append((s, e))

    total = sum(e - s for s, e, lab in ground_truth_vidtest3 if lab != '?')
    unknown = sum(e - s for s, e, lab in ground_truth_vidtest3 if lab == '?')

    print(f'{"الحركة":<16} {"مرات":>5} {"ثواني":>7}   الفترات')
    print('-' * 62)
    for lab in sorted(d, key=lambda k: -sum(e - s for s, e in d[k])):
        spans = d[lab]
        dur = sum(e - s for s, e in spans)
        mark = ' ✅' if len(spans) > 1 else ''
        print(f'{lab:<16} {len(spans):>5} {dur:>7.1f}   '
              f'{", ".join(f"{s:.0f}-{e:.0f}" for s, e in spans)}{mark}')
    print('-' * 62)
    print(f'موصوف: {total:.1f}s   |   مش موصوف: {unknown:.1f}s   '
          f'|   المجموع: {total + unknown:.1f}s')
    print('✅ = بتتكرر، فينفع اختبار نظيف عليها')


if __name__ == '__main__':
    summary()
