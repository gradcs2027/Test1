"""
HA4M: بيحوّل Skeletons.zip + Labels.txt (لكل فيديو) لقطاعات أفعال جاهزة للتدريب.

    python -I ha4m/prepare.py

المدخل  : Project/_scratch/ha4m_skel/<IDUxxVyyy>/{Skeletons.zip, Labels.txt}  (بيانات خارجية = غير موثوقة، بنقراها بـ zipfile بس)
المخرج  : Project/_scratch/ha4m_cache/segments.npz
  seg_full   (N، 30، 32، 5)  لكل مفصل: [x2d، y2d، X، Y، Z] — بعد التوسيط على الحوض والقسمة على طول الجذع
  seg_conf   (N، 30، 32)     ثقة Kinect (0 / 0.33 / 0.67)
  y          (N,)            الفعل 1..12 (بنستبعد 0 "don't care")
  subject    (N,)            رقم الشخص 1..41 (التقسيم بالشخص مش بالفيديو)
  video      (N,)

⚠️ كل فعل ليه مقابل في Kinect 32 مفصل. COCO17 = الـ 17 اللي موديلنا بيفهمها (YOLO).
"""
import itertools
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / '_scratch' / 'ha4m_skel'
OUT = Path(__file__).resolve().parents[1] / '_scratch' / 'ha4m_cache'
N = 30

# Azure Kinect body tracking → COCO-17 (nose، عينين، ودان، كتفين، كوعين، رسغين، حوضين، ركبتين، كعبين)
K2C = [27, 28, 30, 29, 31, 5, 12, 6, 13, 7, 14, 18, 22, 19, 23, 20, 24]
PELVIS, NECK = 0, 3


def read_video(vdir):
    z = zipfile.ZipFile(vdir / 'Skeletons.zip')
    frames = {}
    for i in z.infolist():
        m = re.search(r'FrameID(\d+)_.*\.txt$', i.filename)
        if not m:
            continue
        rows = [l.split('\t') for l in z.read(i).decode('utf-8', 'replace').splitlines()[1:]]
        rows = [r for r in rows if len(r) >= 12 and r[0] == rows[0][0]]     # أول جسم بس
        if len(rows) != 32:
            continue
        a = np.array([[float(r[2]), float(r[3]), float(r[4]), float(r[5]), float(r[10]), float(r[11])]
                      for r in rows], dtype=np.float32)
        frames[int(m.group(1))] = a[np.argsort([int(r[1]) for r in rows])]
    return frames


def labels(vdir):
    lab = {}
    for l in (vdir / 'Labels.txt').read_text(errors='replace').splitlines():
        p = l.split()
        if len(p) >= 3 and p[0].isdigit():
            lab[int(p[0])] = (int(p[1]), int(p[2]))
    return lab


def to_feat(seg):
    """seg: (T، 32، 6) [conf، X، Y، Z، x2d، y2d] → (30، 32، 5) متوسّط على الحوض ومقسوم على طول الجذع."""
    T = len(seg)
    idx = np.linspace(0, T - 1, N).astype(int)
    seg = seg[idx]
    conf = seg[..., 0]
    p3 = seg[..., 1:4] - seg[:, PELVIS:PELVIS + 1, 1:4]
    p2 = seg[..., 4:6] - seg[:, PELVIS:PELVIS + 1, 4:6]
    t3 = np.median(np.linalg.norm(seg[:, NECK, 1:4] - seg[:, PELVIS, 1:4], axis=-1)) + 1e-6
    t2 = np.median(np.linalg.norm(seg[:, NECK, 4:6] - seg[:, PELVIS, 4:6], axis=-1)) + 1e-6
    return np.concatenate([p2 / t2, p3 / t3], -1).astype(np.float32), conf.astype(np.float32)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    F, C, Y, S, V = [], [], [], [], []
    for vdir in sorted(p for p in SRC.iterdir() if p.is_dir() and (p / 'Skeletons.zip').exists() and (p / 'Labels.txt').exists()):
        fr, lab = read_video(vdir), labels(vdir)
        if not fr:
            print('skip', vdir.name)
            continue
        ids = sorted(fr)
        run = [(k, list(g)) for k, g in itertools.groupby([f for f in ids if f in lab], key=lambda f: lab[f])]
        for (act, rep), g in run:
            if act == 0 or len(g) < 10:
                continue
            f, c = to_feat(np.stack([fr[i] for i in g]))
            F.append(f); C.append(c); Y.append(act)
            S.append(int(re.search(r'IDU(\d+)', vdir.name).group(1))); V.append(vdir.name)
        print(vdir.name, len(fr), 'frames', flush=True)
    np.savez_compressed(OUT / 'segments.npz', seg_full=np.array(F), seg_conf=np.array(C), y=np.array(Y),
                        subject=np.array(S), video=np.array(V))
    print('segments', len(Y), '| subjects', len(set(S)), '| per-action', np.bincount(Y)[1:])


if __name__ == '__main__':
    main()
