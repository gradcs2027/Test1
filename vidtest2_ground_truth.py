"""
Ground Truth for vidtest2.mp4 (28.5s، 60fps، 576x1024)

المنهجية:
  - من الـ session السابقة: اتصححنا المشاكل في vidtest2 ground truth
  - sit_down: 5.9-6.6s (كان غلط قبل كده)
  - stand_up: 7.0-7.6s
  - phone_call: 18.2-23.6s (متواصل — كان غلط قبل كده)
  - closeups: 0-2.2s و 24.9-28.4s (مش صالحة)

ملاحظات:
  - الفيديو فيه closeups في البداية والنهاية (يجب استبعادها)
  - الجسم كامل موجود فقط في 2.2-24.9s
  - الشخص ظهره للكاميرا من 2.2-9s (partial view)
  - phone_call فترة واحدة متواصلة (18.2-23.6s)
"""

# Ground truth segments: (start_time, end_time, label)
ground_truth_segments_vidtest2 = [
    # Closeup / invalid frames — لا تحسب
    (0.0, 2.2, '?'),

    # Standing — idle period
    (2.2, 5.9, 'other*'),

    # Sit down — transition من واقف لقاعد
    (5.9, 6.6, 'sit_down'),

    # Stand up — transition من قاعد لواقف
    (7.0, 7.6, 'stand_up'),

    # Standing — gap بين stand_up والمكالمة
    (7.6, 18.2, 'other*'),

    # Phone call — فترة طويلة متواصلة
    (18.2, 23.6, 'phone_call'),

    # Standing — gap قبل closeup
    (23.6, 24.9, 'other*'),

    # Closeup / invalid frames — لا تحسب
    (24.9, 28.5, '?'),
]

# ملخص الحركات المعنونة
annotated_actions = [
    {'time': (5.9, 6.6), 'action': 'sit_down', 'confidence': 'high'},
    {'time': (7.0, 7.6), 'action': 'stand_up', 'confidence': 'high'},
    {'time': (18.2, 23.6), 'action': 'phone_call', 'confidence': 'high'},
]

# معلومات الفيديو
video_info = {
    'filename': 'vidtest2.mp4',
    'duration': 28.5,  # seconds
    'fps': 60,
    'resolution': '576x1024',
    'total_frames': int(28.5 * 60),  # 1710
    'camera': 'static wide shot',
    'subject': '1 person',
    'actions': 'sit_down (1x), stand_up (1x), phone_call (1x)',
    'notes': 'Back to camera 2.2-9s, closeups at start/end',
}

print(f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                    Ground Truth: vidtest2.mp4                              ║
╚════════════════════════════════════════════════════════════════════════════╝

Video Info:
  Duration: {video_info['duration']}s
  FPS: {video_info['fps']}
  Resolution: {video_info['resolution']}
  Frames: {video_info['total_frames']}
  Notes: {video_info['notes']}

Annotated Actions (Real Labels):
""")

for i, action in enumerate(annotated_actions, 1):
    t0, t1 = action['time']
    print(f"  {i}. {action['action']:12} [{t0:6.1f}s - {t1:6.1f}s] ({t1-t0:.1f}s)")

print(f"""
Segments:
  Total: {len(ground_truth_segments_vidtest2)}
  Transitions (real labels): 3
  Static periods (other): 2
  Invalid/excluded (?): 2

Known Issues (fixed in this version):
  ✓ phone_call was 20.0-21.7 + 22.6-23.4 (WRONG)
    Now: 18.2-23.6 (continuous, CORRECT)
  ✓ sit_down was 4.0-6.6 (WRONG)
    Now: 5.9-6.6 (CORRECT)
  ✓ Excluded closeups at 0-2.2s and 24.9-28.4s (marked as ?)

Method:
  ✓ From previous session corrections
  ✓ Multi-frame contact sheets
  ✓ Timing verified ±0.3s precision
""")

if __name__ == '__main__':
    print("\n" + "="*80)
    print("To use this ground truth in your baseline notebooks:")
    print("="*80)
    print("""
    from vidtest2_ground_truth import ground_truth_segments_vidtest2

    results = score_predictions(
        predicted_actions,
        edges_sk,
        ground_truth_segments_vidtest2,
        step=0.1
    )
    """)
