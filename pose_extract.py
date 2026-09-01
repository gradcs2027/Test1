"""
استخراج الـ keypoints من الفيديو بـ YOLOv8n-pose — بيشتغل محلياً.

الهدف: نطلّع مصفوفة (frames, 17, 2) لكل فيديو ونحفظها .npy، عشان مسار
الـ one-shot يشتغل من غير ما نعيد الاستخراج كل مرة ومن غير أي داتاسِت.

الاستخدام:
    python pose_extract.py                    # الـ 3 فيديوهات
    python pose_extract.py vidtest3           # فيديو واحد
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

VIDEO_DIR = Path(__file__).resolve().parent.parent / 'testvid_upload'
OUT_DIR = Path(__file__).resolve().parent / 'keypoints'
FRAME_SKIP = 2          # نفس اللي في cell7_video.py -> fps فعلي = fps/2


def extract(video_name, model):
    """بيرجّع (keypoints, meta). keypoints شكلها (frames, 17, 2)."""
    path = VIDEO_DIR / f'{video_name}.mp4'
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f'مش قادر أفتح {path}')

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f'\n🎬 {video_name}: {total} فريم @ {fps:.1f}fps '
          f'({total / fps:.1f}s) — بناخد كل فريم {FRAME_SKIP}')

    keypoints = []
    n_detected = 0
    frame_i = 0
    t0 = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_i % FRAME_SKIP == 0:
            res = model(frame, verbose=False)[0]

            if res.keypoints is not None and len(res.keypoints) > 0:
                # لو فيه أكتر من شخص، ناخد صاحب أكبر صندوق (الأقرب للكاميرا)
                if res.boxes is not None and len(res.boxes) > 1:
                    areas = (res.boxes.xywh[:, 2] * res.boxes.xywh[:, 3]).cpu().numpy()
                    person = int(np.argmax(areas))
                else:
                    person = 0
                kp = res.keypoints.xy[person].cpu().numpy()   # (17, 2)
                if kp.shape == (17, 2) and np.any(kp != 0):
                    keypoints.append(kp)
                    n_detected += 1
                else:
                    keypoints.append(np.zeros((17, 2), dtype=np.float32))
            else:
                keypoints.append(np.zeros((17, 2), dtype=np.float32))

            n = len(keypoints)
            if n % 250 == 0:
                el = time.perf_counter() - t0
                print(f'   {n} فريم — {el:.0f}s ({n / el:.1f} fps)')

        frame_i += 1

    cap.release()
    keypoints = np.array(keypoints, dtype=np.float32)

    meta = {
        'video': video_name,
        'source_fps': float(fps),
        'effective_fps': float(fps / FRAME_SKIP),
        'duration_s': float(total / fps),
        'n_frames': int(len(keypoints)),
        'n_detected': int(n_detected),
        'detect_rate': float(n_detected / max(1, len(keypoints))),
        'extract_seconds': float(time.perf_counter() - t0),
    }

    print(f'✅ {video_name}: {len(keypoints)} فريم | '
          f'اكتشاف {meta["detect_rate"] * 100:.1f}% | '
          f'{meta["extract_seconds"]:.0f}s')

    return keypoints, meta


def main():
    from ultralytics import YOLO

    videos = sys.argv[1:] or ['vidtest1', 'vidtest2', 'vidtest3']
    OUT_DIR.mkdir(exist_ok=True)

    print('📥 تحميل YOLOv8n-pose...')
    model = YOLO('yolov8n-pose.pt')

    for name in videos:
        kp, meta = extract(name, model)
        np.save(OUT_DIR / f'{name}_keypoints.npy', kp)
        np.save(OUT_DIR / f'{name}_meta.npy', meta, allow_pickle=True)
        print(f'💾 اتحفظ: keypoints/{name}_keypoints.npy  {kp.shape}')

    print('\n✅ خلص')


if __name__ == '__main__':
    main()
