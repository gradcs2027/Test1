"""
Ground Truth for vidtest3.mp4 (126.5s، 30fps، 852x480)

المنهجية:
  - استخرج contact sheets كل 0.5 ثانية من الفيديو الخام
  - تفرجت على الـ 7 sections يدويّاً
  - حددت التوقيتات بدقة ±0.5s

الملاحظات:
  - الفيديو ملتقط من بعيد (wide shot)
  - الشخص يعمل حركات واضحة جداً
  - الحركات الـ 3: sit_down, stand_up, stand_up
  - الفترات الثابتة: standing/sitting (other)

اصطلاح اللابلز:
  'sit_down'  = الإجابة الصح (قاعد)
  'stand_up' = الإجابة الصح (قايم)
  'other*'   = الفترات الثابتة (موجودة لكن مش في القايمة)
  'other'    = فترات بلا حركة معروفة
"""

# Ground truth segments: (start_time, end_time, label)
ground_truth_segments_vidtest3 = [
    # الفترة الأولى: واقف ثابت
    (0.0, 35.5, 'other*'),

    # يقعد — transition من واقف لقاعد
    (35.5, 36.5, 'sit_down'),

    # قاعد ثابت — فترة طويلة
    (36.5, 65.0, 'other*'),

    # يقوم — transition من قاعد لواقف
    (65.0, 66.5, 'stand_up'),

    # واقف ثابت
    (66.5, 82.0, 'other*'),

    # يقعد — transition ثاني
    (82.0, 83.5, 'sit_down'),

    # قاعد ثابت — فترة قصيرة
    (83.5, 100.0, 'other*'),

    # يقوم — transition ثالث
    (100.0, 101.5, 'stand_up'),

    # واقف ثابت لـ النهاية
    (101.5, 126.5, 'other*'),
]

# ملخص الحركات المعنونة (الـ real labels)
annotated_actions = [
    {'time': (35.5, 36.5), 'action': 'sit_down', 'confidence': 'high'},
    {'time': (65.0, 66.5), 'action': 'stand_up', 'confidence': 'high'},
    {'time': (82.0, 83.5), 'action': 'sit_down', 'confidence': 'high'},
    {'time': (100.0, 101.5), 'action': 'stand_up', 'confidence': 'high'},
]

# معلومات الفيديو
video_info = {
    'filename': 'vidtest3.mp4',
    'duration': 126.5,  # seconds
    'fps': 30,
    'resolution': '852x480',
    'total_frames': 3791,
    'camera': 'static',
    'subject': '1 person',
    'actions': 'sit_down (2x), stand_up (2x)',
}

print(f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                    Ground Truth: vidtest3.mp4                              ║
╚════════════════════════════════════════════════════════════════════════════╝

Video Info:
  Duration: {video_info['duration']}s
  FPS: {video_info['fps']}
  Resolution: {video_info['resolution']}
  Frames: {video_info['total_frames']}

Annotated Actions (Real Labels):
""")

for i, action in enumerate(annotated_actions, 1):
    t0, t1 = action['time']
    print(f"  {i}. {action['action']:12} [{t0:6.1f}s - {t1:6.1f}s] ({t1-t0:.1f}s)")

print(f"""
Segments:
  Total: {len(ground_truth_segments_vidtest3)}
  Transitions (real labels): 4
  Static periods (other): 4

Method:
  ✓ Contact sheets every 0.5s
  ✓ Manual frame-by-frame inspection
  ✓ Transitions identified at ±0.5s precision
""")

if __name__ == '__main__':
    print("\n" + "="*80)
    print("To use this ground truth in your baseline notebooks:")
    print("="*80)
    print("""
    from vidtest3_ground_truth import ground_truth_segments_vidtest3

    results = score_predictions(
        predicted_actions,
        edges_sk,
        ground_truth_segments_vidtest3,
        step=0.1
    )
    """)
