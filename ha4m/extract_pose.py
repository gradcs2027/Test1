"""
بيطلّع هيكل COCO17 من فريمات HA4M الملونة بـ YOLO أو MediaPipe.

    python -I ha4m/extract_pose.py yolo <فولدر_فريمات أو zip> <Labels.txt> <out.npz>
    <mp_env>/python -I ha4m/extract_pose.py mp <فولدر_فريمات أو zip> <Labels.txt> <out.npz> <pose_landmarker.task>

  - فولدر: كل الـ PNG اللي فيه (download_color.py بينزّل كل فريم تاني). zip: فريمات Color كل فريم تاني (زي ha4m_yolo2.py).
  - اختيار الشخص (الاتنين): أكتر مفاصل واثقة (> 0.5)، والتعادل للأقرب لنص الكادر.
  - MediaPipe بيشتغل في mode الفيديو (بيتتبّع بين الفريمات). الثقة = visibility.
المخرج: kp (n، 17، 3) = x، y بالبكسل + الثقة، fid رقم الفريم (= index هيكل Kinect)، act من Labels.txt (-1 = مفيش).
MediaPipe كمان: kp33 (n، 33، 5) = x، y بالبكسل، z، visibility، presence، و world (n، 33، 3) بالمتر، لنفس الشخص المختار
(عشان نجرّب نقط الإيد أو 3D بعدين من غير ما نحمّل تاني).
"""
import re
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

BACK, SRC, LAB, OUT = sys.argv[1:5]
MP2COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
YOLO_PT = Path(__file__).resolve().parents[1] / 'yolov8n-pose.pt'


def frames():
    if SRC.endswith('.zip'):
        z = zipfile.ZipFile(SRC)
        names = sorted(n for n in z.namelist() if '/Color/' in n and n.endswith('.png'))[::2]
        read = lambda n: cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR)
    else:
        names = sorted(str(p) for p in Path(SRC).glob('*.png'))
        read = lambda n: cv2.imread(n, cv2.IMREAD_COLOR)
    for n in names:
        yield int(re.search(r'FrameID(\d+)', n).group(1)), read(n), len(names)


def pick(kx, kc, W):
    """kx: (P، 17، 2)، kc: (P، 17) → أحسن شخص (17، 3)، ورقمه (-1 = مفيش حد)."""
    out, j = np.zeros((17, 3), np.float32), -1
    if len(kx):
        cx = np.array([np.median(k[c > 0.3, 0]) if (c > 0.3).any() else W / 2 for k, c in zip(kx, kc)])
        j = int(np.argmax((kc > 0.5).sum(1) - np.abs(cx - W / 2) / W))
        out[:, :2], out[:, 2] = kx[j], kc[j]
    return out, j


def detector():
    if BACK == 'yolo':
        from ultralytics import YOLO
        m = YOLO(str(YOLO_PT))

        def run(img, t):
            r = m(img, verbose=False)[0]
            if r.keypoints is None or len(r.boxes) == 0:
                return np.zeros((0, 17, 2)), np.zeros((0, 17)), None
            return r.keypoints.xy.cpu().numpy(), r.keypoints.conf.cpu().numpy(), None
        return run
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    lm = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=sys.argv[5]), running_mode=vision.RunningMode.VIDEO, num_poses=3))

    def run(img, t):
        H, W = img.shape[:2]
        r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), t)
        kx = np.array([[(p[i].x * W, p[i].y * H) for i in MP2COCO] for p in r.pose_landmarks]).reshape(-1, 17, 2)
        kc = np.array([[p[i].visibility for i in MP2COCO] for p in r.pose_landmarks]).reshape(-1, 17)
        k33 = [np.array([(q.x * W, q.y * H, q.z, q.visibility, q.presence) for q in p], np.float32) for p in r.pose_landmarks]
        w33 = [np.array([(q.x, q.y, q.z) for q in p], np.float32) for p in r.pose_world_landmarks]
        return kx, kc, (k33, w33)
    return run


def main():
    lab = {}
    for l in Path(LAB).read_text(errors='replace').splitlines():
        p = l.split()
        if len(p) >= 3 and p[0].isdigit():
            lab[int(p[0])] = int(p[1])
    run = detector()
    kp, fid, act, k33, w33 = [], [], [], [], []
    for f, img, total in frames():
        kx, kc, extra = run(img, int(f * 1000 / 30))          # 30fps → timestamp بالملّي ثانية
        k, j = pick(kx, kc, img.shape[1])
        kp.append(k); fid.append(f); act.append(lab.get(f, -1))
        if extra is not None:
            k33.append(extra[0][j] if j >= 0 else np.zeros((33, 5), np.float32))
            w33.append(extra[1][j] if j >= 0 else np.zeros((33, 3), np.float32))
        if len(fid) % 200 == 0:
            print(len(fid), '/', total, flush=True)
    more = {'kp33': np.array(k33), 'world': np.array(w33)} if k33 else {}
    np.savez(OUT, kp=np.array(kp), fid=np.array(fid), act=np.array(act), **more)
    print('DONE', len(fid), 'detected', float((np.array(kp)[:, :, 2] > 0.5).any(1).mean()), flush=True)


if __name__ == '__main__':
    main()
