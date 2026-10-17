"""صورة واحدة فيها فريم من منتصف كل فعل من الـ 12 (فيديو V001) + هيكل YOLO.
    python -I ha4m/action_sheet.py <IDU001V001.zip> <ملف.jpg>
"""
import re, sys, zipfile
from pathlib import Path
import cv2, numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prepare import OUT
NM = ['1 carrier', '2 gear bearings', '3 planet gears', '4 carrier shaft', '5 sun shaft', '6 sun gear',
      '7 sun gear bearing', '8 ring gear', '9 block2->block1', '10 cover', '11 screws', '12 allen key']
EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12)]
d = np.load(OUT.parent / 'ha4m_kp' / 'IDU001V001_kp2.npz')
kp, act = d['new'], d['act']
z = zipfile.ZipFile(sys.argv[1])
names = sorted(x for x in z.namelist() if '/Color/' in x and x.endswith('.png'))[::2]
tiles = []
for a in range(1, 13):
    idx = np.flatnonzero(act == a)
    first = idx[np.r_[True, np.diff(idx) > 1]][0]
    run = idx[idx >= first]; run = run[:np.argmax(np.diff(np.r_[run, 10**9]) > 1) + 1]
    i = int(run[len(run) // 2])
    img = cv2.imdecode(np.frombuffer(z.read(names[i]), np.uint8), cv2.IMREAD_COLOR)
    # قص على منطقة الشغل (نص الصورة تقريبا) عشان القطع تبان
    img = img[300:1400, 200:1900]
    k = kp[i, :, :2] - np.array([200, 300])
    for p, q in EDGES:
        if kp[i, p, 2] > .5 and kp[i, q, 2] > .5:
            cv2.line(img, tuple(int(v) for v in k[p]), tuple(int(v) for v in k[q]), (0, 255, 0), 5)
    img = cv2.resize(img, (560, 363))
    cv2.rectangle(img, (0, 0), (560, 36), (0, 0, 0), -1)
    cv2.putText(img, f'{NM[a-1]}  ({len(run)*2/30:.1f}s)', (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    tiles.append(img)
rows = [np.hstack(tiles[r * 4:(r + 1) * 4]) for r in range(3)]
cv2.imwrite(sys.argv[2], np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
print('saved')
