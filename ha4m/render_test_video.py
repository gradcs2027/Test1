"""
فيديو اختبار V001 (الشخص 1 = في الـ test، الموديل ما شافوش) عليه الـ annotation:
  - الحقيقي (من Labels.txt) والتوقع من هيكل Kinect والتوقع من هيكل YOLO (الموديل اتدرّب على Kinect بس)
  - شريط زمني تحت: الحقيقي / Kinect / YOLO بالألوان، وخط أبيض عند الفريم الحالي
  - هيكل YOLO أخضر، نقط Kinect صفرا

    python -I ha4m/render_test_video.py <IDU001V001.zip> <ملف_الإخراج.mp4>

الموديل: C (spatial attention) من train_att.py، seed 1، على نفس تقسيم results/split.json.
الـ zip بيانات خارجية: بنقرا منه الصور بس بـ zipfile.
"""
import json
import re
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prepare import OUT, SRC  # noqa: E402
from train_att import CFG, predict, train  # noqa: E402
from train_stream import NAMES, NONE, frame_pred, load_raw, norm17, smooth, windows  # noqa: E402

COL = [(230, 120, 40), (60, 180, 75), (255, 200, 0), (180, 60, 200), (0, 140, 255), (90, 90, 230),
       (200, 200, 60), (40, 200, 200), (140, 100, 50), (120, 220, 120), (220, 90, 160), (60, 60, 180), (150, 150, 150)]
EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
SCALE = 0.5


def main():
    zpath, outp = sys.argv[1], sys.argv[2]
    raw = load_raw()
    split = json.loads((HERE / 'results' / 'split.json').read_text())
    subj = {v: int(v[3:6]) for v in raw}
    W = {v: windows(*raw[v]) for v in raw}

    def pick(S, pur=0.8):
        L = [(W[v][0][W[v][2] >= pur], W[v][1][W[v][2] >= pur]) for v in raw if subj[v] in S]
        return np.concatenate([a for a, _ in L]), np.concatenate([b for _, b in L])

    Xtr, ytr = pick(split['train']); Xva, yva = pick(split['val'])
    m, prep, _ = train(CFG['C: spatial att'], Xtr, ytr, Xva, yva, 1)

    d = np.load(OUT.parent / 'ha4m_kp' / 'IDU001V001_kp2.npz')
    yolo, fid, act = d['new'], d['fid'], d['act']
    n = len(fid)
    gt = np.array([NONE if a <= 0 else a - 1 for a in act])
    kin = raw['IDU001V001'][0][:2 * n:2]
    st = np.arange(0, n - 45 + 1, 8)
    preds = {}
    for name, src in (('Kinect', kin), ('YOLO', yolo[:, :, :2])):
        P = predict(m, prep, np.stack([norm17(src[s:s + 45]) for s in st]))
        preds[name] = frame_pred(smooth(P, 5), st + 22, n).argmax(1)
    for k, p in preds.items():
        print(k, 'frame acc %.1f%%' % ((p == gt).mean() * 100))

    z = zipfile.ZipFile(zpath)
    names = sorted(x for x in z.namelist() if '/Color/' in x and x.endswith('.png'))[::2]
    assert len(names) == n and int(re.search(r'FrameID(\d+)', names[0]).group(1)) == fid[0]
    H, Wd = int(1536 * SCALE), int(2048 * SCALE)
    bar_h = 150
    vw = cv2.VideoWriter(outp, cv2.VideoWriter_fourcc(*'mp4v'), 15, (Wd, H + bar_h))
    for i, nm in enumerate(names):
        img = cv2.imdecode(np.frombuffer(z.read(nm), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (Wd, H))
        k = yolo[i, :, :2] * SCALE
        for a, b in EDGES:
            if yolo[i, a, 2] > 0.5 and yolo[i, b, 2] > 0.5:
                cv2.line(img, tuple(int(v) for v in k[a]), tuple(int(v) for v in k[b]), (0, 255, 0), 3)
        for j in range(17):
            kc = kin[i, j] * SCALE
            cv2.circle(img, (int(kc[0]), int(kc[1])), 4, (0, 255, 255), -1)
        canvas = np.full((H + bar_h, Wd, 3), 25, np.uint8)
        canvas[:H] = img
        rows = [('TRUE', gt), ('Kinect pred', preds['Kinect']), ('YOLO pred', preds['YOLO'])]
        for r, (lab, arr) in enumerate(rows):
            y0 = H + 10 + r * 34
            cv2.putText(canvas, lab, (8, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
            for x in range(0, Wd - 130):
                c = arr[min(n - 1, int(x / (Wd - 130) * n))]
                canvas[y0:y0 + 28, 130 + x] = COL[c]
            ok = arr[i] == gt[i]
            txt = NAMES[arr[i]] + ('' if lab == 'TRUE' else ('  OK' if ok else '  X'))
            cv2.putText(img if False else canvas, f'{lab}: {txt}', (12, 30 + r * 34), cv2.FONT_HERSHEY_SIMPLEX, 0.95,
                        (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(canvas, f'{lab}: {txt}', (12, 30 + r * 34), cv2.FONT_HERSHEY_SIMPLEX, 0.95,
                        (255, 255, 255) if lab == 'TRUE' else ((80, 255, 80) if ok else (80, 80, 255)), 2, cv2.LINE_AA)
        cx = 130 + int(i / n * (Wd - 130))
        cv2.line(canvas, (cx, H), (cx, H + bar_h), (255, 255, 255), 2)
        vw.write(canvas)
        if i % 200 == 0:
            print(i, '/', n, flush=True)
    vw.release()
    print('DONE', outp)


if __name__ == '__main__':
    main()
