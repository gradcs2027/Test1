"""
يرسم **توقّعات** الـ FastDTW على الفيديو

    python render_pred.py               # vidtest4

⚠️⚠️ الفرق بين الملف ده و `render_gt.py` — مهم جداً
────────────────────────────────────────────────────
`render_gt.py` بيرسم **الإجابة الصح** اللي إحنا كاتبينها بإيدنا.
الملف ده بيرسم **اللي الموديل قاله**. الاتنين شكلهم واحد على الشاشة،
فلو اتلخبطوا حد ممكن يفتكر توقّع غلط إنه ground truth.

عشان كده الهيدر هنا مكتوب فيه `FASTDTW PREDICTION` بالأحمر و
`NOT ground truth` تحتيها. متشيلهاش.

السياق: vidtest4 مالوش ground truth أصلاً، فالفيديو ده هو الطريقة
الوحيدة لمراجعة التوقّعات — بتتفرج وتقارن باللي بيحصل قدامك.

⚠️ اللي هتشوفه بيترجرج: 64 فترة في 75 ثانية (وسيط المدة 0.85s).
   ده مش عيب في الرسم، ده اللي الموديل طالعه فعلاً. شوف HANDOFF قسم 2-ج.

ملاحظة: cv2.putText مابيرسمش عربي، فكل النص على الفيديو إنجليزي.
"""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from render_gt import EDGES, FONT, PALETTE, draw_skeleton, open_writer
from run_vidtest4 import predict, segments

from paths import KP_DIR, OUT_DIR, VIDEO_DIR

FRAME_SKIP = 2

HEAD_H = 104            # أطول شوية من render_gt عشان سطر التحذير
BAR_H = 84

COLOR_GAP = (70, 70, 70)        # فترة اتشالت لأنها قصيرة أوي
COLOR_WARN = (60, 60, 235)      # أحمر — التحذير إن ده توقّع


def colors_for(segs):
    """لون ثابت لكل حركة متوقّعة."""
    labs = sorted({s['label'] for s in segs})
    return {l: PALETTE[i % len(PALETTE)] for i, l in enumerate(labs)}


def seg_at(segs, t):
    for s in segs:
        if s['t0'] <= t < s['t1']:
            return s
    return None


def draw_timeline(frame, segs, colors, t, dur, w, h):
    """
    شريط زمني تحت. الفجوات الرمادية = نوافذ اتشالت لأنها أقصر من
    MIN_SEGMENT — يعني الموديل كان بيترجرج فيها بسرعة.
    """
    y0 = h - BAR_H
    cv2.rectangle(frame, (0, y0), (w, h), (18, 18, 18), -1)
    cv2.rectangle(frame, (0, y0 + 22), (w, y0 + 52), COLOR_GAP, -1)

    for s in segs:
        x1, x2 = int(s['t0'] / dur * w), int(s['t1'] / dur * w)
        col = colors[s['label']]
        cv2.rectangle(frame, (x1, y0 + 22), (x2, y0 + 52), col, -1)
        cv2.rectangle(frame, (x1, y0 + 22), (x2, y0 + 52), (0, 0, 0), 1)

        tw = cv2.getTextSize(s['label'], FONT, 0.32, 1)[0][0]
        if x2 - x1 > tw + 6:
            cv2.putText(frame, s['label'],
                        (x1 + (x2 - x1 - tw) // 2, y0 + 43),
                        FONT, 0.32, (255, 255, 255), 1, cv2.LINE_AA)

    for sec in range(0, int(dur) + 1, 5):
        x = int(sec / dur * w)
        cv2.line(frame, (x, y0 + 52), (x, y0 + 58), (150, 150, 150), 1)
        cv2.putText(frame, str(sec), (max(1, x - 6), y0 + 72), FONT, 0.32,
                    (170, 170, 170), 1, cv2.LINE_AA)

    xc = int(t / dur * w)
    cv2.line(frame, (xc, y0 + 14), (xc, y0 + 58), (255, 255, 255), 2)
    cv2.circle(frame, (xc, y0 + 14), 4, (255, 255, 255), -1, cv2.LINE_AA)


def draw_header(frame, seg, t, dur, colors, w, n_pts):
    if seg is None:
        text = 'unstable'
        sub = 'flickered too fast to call - segment dropped'
        col = COLOR_GAP
    else:
        text = seg['label'].upper()
        sub = (f"[{seg['t0']:.1f} - {seg['t1']:.1f}]  "
               f"({seg['t1'] - seg['t0']:.1f}s)   dtw={seg['conf']:.3f}")
        col = colors[seg['label']]

    cv2.rectangle(frame, (0, 0), (w, HEAD_H), (18, 18, 18), -1)
    cv2.rectangle(frame, (0, 0), (10, HEAD_H), col, -1)
    cv2.putText(frame, text, (22, 42), FONT, 1.05, col, 2, cv2.LINE_AA)
    cv2.putText(frame, sub, (24, 68), FONT, 0.46, (200, 200, 200), 1,
                cv2.LINE_AA)

    clock = f'{t:5.2f}s / {dur:.1f}s'
    cv2.putText(frame, clock, (w - 168, 42), FONT, 0.52, (230, 230, 230), 1,
                cv2.LINE_AA)
    if n_pts == 0:
        cv2.putText(frame, 'NO POSE', (w - 168, 68), FONT, 0.52, (0, 0, 230),
                    2, cv2.LINE_AA)

    cv2.putText(frame, 'FASTDTW PREDICTION - NOT ground truth', (22, 94),
                FONT, 0.46, COLOR_WARN, 1, cv2.LINE_AA)


def render(video='vidtest4'):
    print(f'\n🔮 بحسب التوقّعات لـ {video}...')
    labels, conf, edges, templates, _ = predict(verbose=True)
    segs = segments(labels, edges, conf)
    colors = colors_for(segs)
    print(f'   {len(segs)} فترة، {len(set(s["label"] for s in segs))} حركة')

    kp_all = np.load(KP_DIR / f'{video}_keypoints.npy')

    cap = cv2.VideoCapture(str(VIDEO_DIR / f'{video}.mp4'))
    if not cap.isOpened():
        raise FileNotFoundError(f'مش قادر أفتح {VIDEO_DIR / video}.mp4')

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = total / fps

    w -= w % 2          # libx264 بـ yuv420p محتاج أبعاد زوجية
    vh -= vh % 2
    h = HEAD_H + vh + BAR_H

    out_path = OUT_DIR / f'pred_{video}.mp4'
    OUT_DIR.mkdir(exist_ok=True)
    _, write, close = open_writer(out_path, fps, w, h)

    print(f'🎬 {video}: {total} فريم @ {fps:.0f}fps → {out_path.name}')

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t = i / fps
        canvas = np.zeros((h, w, 3), np.uint8)
        canvas[HEAD_H:HEAD_H + vh] = frame[:vh, :w]

        k = min(i // FRAME_SKIP, len(kp_all) - 1)
        n_pts = draw_skeleton(canvas, kp_all[k], HEAD_H)

        draw_header(canvas, seg_at(segs, t), t, dur, colors, w, n_pts)
        draw_timeline(canvas, segs, colors, t, dur, w, h)

        write(canvas)
        i += 1
        if i % 500 == 0:
            print(f'   {i}/{total}')

    cap.release()
    close()
    mb = out_path.stat().st_size / 1e6
    print(f'✅ {out_path}  ({mb:.1f} MB)')
    return out_path


def main():
    videos = sys.argv[1:] or ['vidtest4']
    print('=' * 66)
    print('  رسم توقّعات FastDTW على الفيديو')
    print('=' * 66)
    print('  ⚠️ ده توقّع الموديل، مش الإجابة الصح.')

    made = [render(v) for v in videos]

    print(f'\n{"=" * 66}')
    for p in made:
        print(f'   {p}')
    print('\n  اتفرج وقارن: الاسم اللي فوق هو اللي الموديل قاله عند اللحظة دي.')
    print('  الفجوات الرمادية في الشريط = الموديل كان بيترجرج بسرعة أوي.')


if __name__ == '__main__':
    main()
