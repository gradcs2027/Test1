"""
Ground Truth لـ vidtest5.mp4 — فيديو اختبار مخبّي (held-out)

المصدر: مسودة من صور فريم كل ثانية (`_scratch/contact/vidtest5_*.jpg`)،
**صاحب المشروع وافق عليها** (2026-09-30: "المسودة صح، اعتمدها").
⚠️ الحدود بدقة ±1 ثانية بس (فريم كل ثانية) — كفاية لتصنيف القصاصات،
   مش كفاية لقياس دقة الحدود.

الفيديو ده مادخلش في أي تدريب ولا ضبط ولا بنك FastDTW — عشان كده مش في
`VIDEOS` (فيديوهات التطوير)، هو في `HELDOUT` في `ground_truth.py`.

الحركات الموجودة:
  • walking (مشي لمكان الوقفة)
  • clapping (إيدين قدام الصدر) — مرتين
  • phone_call (إيد على الودن)

اصطلاح اللابلز:
    'other*'  = سكون **متأكّد منه** — بيتحسب
    '?'       = مش موصوفة / مش صالحة — **بتتستثنى من الحساب تماماً**
"""

# (بداية_ثانية, نهاية_ثانية, اللابل)
ground_truth_vidtest5 = [
    (0.0,   8.0,  'walking'),       # مشي لمكان الوقفة
    (8.0,  24.0,  '?'),             # إيدين عند الراس 9-19 — مش واضحة إيه
    (24.0, 30.0,  'clapping'),      # إيدين قدام الصدر
    (30.0, 36.0,  '?'),             # غير موصوف
    (36.0, 52.0,  'phone_call'),    # إيد على الودن
    (52.0, 72.0,  '?'),             # إيد عند الصدر/البق 60-67 — مش واضحة إيه
    (72.0, 77.0,  'clapping'),      # إيدين قدام الصدر (مرة تانية)
    (77.0, 78.0,  '?'),             # آخر الفيديو
]

REPEATED = ['clapping']

video_info = {
    'filename': 'vidtest5.mp4',
    'duration': 78.0,
    'fps': 60,
    'effective_fps': 30.0,
    'n_frames': 2339,
    'detect_rate': 1.0,
    'n_labelled_actions': 3,
    'verified_by_owner': True,       # مسودة من الصور، صاحب المشروع وافق عليها
    'timing_precision_s': 1.0,
}

ground_truth_segments_vidtest5 = ground_truth_vidtest5
ground_truth_segments = ground_truth_vidtest5


def summary():
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    from ground_truth import print_video
    print_video('vidtest5')


if __name__ == '__main__':
    summary()
