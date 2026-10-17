"""
تقرير HA4M — الخطوة 2: فيديو متعلّم لكل موديل + فيديو مقارنة لكل الموديلات (H.264، بيشتغل على واتساب و PowerPoint).

    python -I ha4m_report/build/render_videos.py <IDU001V001.zip>

بيقرا data/predictions.npz و data/summary.json (من make_predictions.py) وبيكتب في ha4m_report/videos/:
  00_all_models.mp4        كل الموديلات (بعد التحسين، مدخل YOLO) تحت بعض
  01_lstm.mp4 ... 08_fastdtw.mp4   لكل موديل: الحقيقي / قبل / بعد (YOLO) / بعد (Kinect) + الدقة وهي بتتحسب
  + صورة من كل فيديو في ha4m_report/figures/ عشان النوتبوك
الـ zip بيانات خارجية: بنقرا منه الصور بس بـ zipfile.
"""
import json
import re
import sys
import zipfile
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

REP = Path(__file__).resolve().parents[1]
DATA, VID, FIG = REP / 'data', REP / 'videos', REP / 'figures'

COL = [(230, 120, 40), (60, 180, 75), (255, 200, 0), (180, 60, 200), (0, 140, 255), (90, 90, 230),
       (200, 200, 60), (40, 200, 200), (140, 100, 50), (120, 220, 120), (220, 90, 160), (60, 60, 180), (150, 150, 150)]
EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
TITLE = {'lstm': 'BiLSTM (3 layers, last state)', 'attn': 'BiLSTM + attention pooling',
         'text': 'Text encoder (zero-shot style, MiniLM)', 'A: LSTM': 'A: small LSTM',
         'B: temporal att': 'B: LSTM + temporal attention', 'C: spatial att': 'C: LSTM + spatial attention',
         'D: both': 'D: LSTM + temporal & spatial att', 'fastdtw': 'FastDTW 1-NN (no training)'}
FILE = {'lstm': '01_lstm', 'attn': '02_bilstm_attention', 'text': '03_text_encoder', 'A: LSTM': '04_A_lstm',
        'B: temporal att': '05_B_temporal_att', 'C: spatial att': '06_C_spatial_att', 'D: both': '07_D_both_att',
        'fastdtw': '08_fastdtw'}
SHORT = {'lstm': 'BiLSTM (3 layers)', 'attn': 'BiLSTM + attention', 'text': 'Text encoder (MiniLM)', 'A: LSTM': 'A: small LSTM',
         'B: temporal att': 'B: + temporal att', 'C: spatial att': 'C: + spatial att', 'D: both': 'D: + both att',
         'fastdtw': 'FastDTW (no training)'}
FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN, RED, WHITE, GRAY = (80, 230, 80), (70, 70, 255), (240, 240, 240), (160, 160, 160)


def put(img, t, xy, s=0.55, c=WHITE, th=1, outline=False):
    if outline:                     # إطار أسود للكتابة فوق الصورة بس (على الخلفية الغامقة مش محتاج)
        cv2.putText(img, t, xy, FONT, s, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(img, t, xy, FONT, s, c, th, cv2.LINE_AA)


def bars(rows, n, half, w, lab_w=150, h=26, gap=6):
    """صورة ثابتة للشرايط الزمنية: صف لكل (اسم، توقعات)."""
    img = np.full((len(rows) * (h + gap) + gap + 18, w, 3), 25, np.uint8)
    bw = w - lab_w - 10
    for r, (lab, arr) in enumerate(rows):
        y0 = gap + r * (h + gap)
        put(img, lab, (6, y0 + 18), 0.45)
        x = (np.arange(bw) / bw * n).astype(int)
        img[y0:y0 + h, lab_w:lab_w + bw] = np.array(COL, np.uint8)[arr[x]]
    hx = lab_w + int(half / n * bw)
    cv2.line(img, (hx, 0), (hx, img.shape[0] - 18), WHITE, 1)
    put(img, '1st half: noise measured here (not scored)', (lab_w + 4, img.shape[0] - 4), 0.4, GRAY)
    put(img, 'scored part (2nd half)', (hx + 6, img.shape[0] - 4), 0.4, WHITE)
    return img, lab_w, bw


def legend(img, x0, y0, names, s=0.42, dy=19, first=0):
    for k, nm in enumerate(names):
        cv2.rectangle(img, (x0, y0 + k * dy - 11), (x0 + 22, y0 + k * dy + 3), COL[first + k], -1)
        put(img, nm, (x0 + 30, y0 + k * dy), s)


def skeleton(img, yolo_i, conf_i, kin_i, scale):
    k = yolo_i * scale
    for a, b in EDGES:
        if conf_i[a] > 0.5 and conf_i[b] > 0.5:
            cv2.line(img, tuple(int(v) for v in k[a]), tuple(int(v) for v in k[b]), (0, 255, 0), 3)
    for j in range(17):
        cv2.circle(img, (int(kin_i[j, 0] * scale), int(kin_i[j, 1] * scale)), 3, (0, 255, 255), -1)


class Writer:
    def __init__(self, path, w, h):
        self.g = imageio_ffmpeg.write_frames(str(path), (w, h), fps=15, codec='libx264', pix_fmt_out='yuv420p',
                                             quality=6, macro_block_size=16, ffmpeg_log_level='error')
        self.g.send(None)

    def write(self, bgr):
        self.g.send(np.ascontiguousarray(bgr[:, :, ::-1]).tobytes())

    def close(self):
        self.g.close()


def main():
    zpath = sys.argv[1]
    VID.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    S = json.loads((DATA / 'summary.json').read_text(encoding='utf-8'))
    D = np.load(DATA / 'predictions.npz')
    names = [x.replace('→', '->') for x in S['names']]
    gt, kin, yolo, conf, fid = D['gt'], D['kin'], D['yolo'], D['conf'], D['fid']
    n, half = len(gt), S['v001']['half']
    models = S['models']
    mean3 = {}
    for m in models:
        for cond in S['conds']:
            f = DATA / (f'run_all_{cond.split("_")[0]}_dtw.json' if m == 'fastdtw' else f'run_all_{cond}.json')
            r = json.loads(f.read_text())[m]
            mean3[m, cond] = {k: float(np.mean(v)) for k, v in r.items()}
    P = {(m, c, s): D[f'{m}|{c}|{s}'] for m in models for c in S['conds'] for s in ('kin', 'yolo')}
    before, after = S['conds']

    # ── فيديو لكل موديل: 1440×944
    FW, FH, PW = 1024, 768, 416
    sc1 = FW / 2048
    rows_of = {}
    for m in models:
        dtw = m == 'fastdtw'
        lb = 'full body' if dtw else 'full body, clean'
        la = 'upper body' if dtw else 'upper + YOLO noise'
        rows_of[m] = [('TRUE', gt), (f'BEFORE ({lb})', P[m, before, 'yolo']), (f'AFTER ({la})', P[m, after, 'yolo']),
                      ('AFTER, Kinect input', P[m, after, 'kin'])]
    tl = {m: bars([(r[0].split(' (')[0], r[1]) for r in rows_of[m]], n, half, FW + PW) for m in models}

    # ── فيديو المقارنة: 1440×816
    CW, CH = 768, 576
    sc2 = CW / 2048
    crow = [('TRUE', gt)] + [(TITLE[m].split(':')[0] if ':' in TITLE[m] else m, P[m, after, 'yolo']) for m in models]
    ctl = bars(crow, n, half, 1440, h=18, gap=4)

    out = VID if len(sys.argv) < 3 else VID / 'preview'      # argv[2] = عدد فريمات للتجربة بس
    out.mkdir(exist_ok=True)
    writers = {m: Writer(out / f'{FILE[m]}.mp4', FW + PW, 944) for m in models}
    cw = Writer(out / '00_all_models.mp4', 1440, 816)
    z = zipfile.ZipFile(zpath)
    imgs = sorted(x for x in z.namelist() if '/Color/' in x and x.endswith('.png'))[::2]
    assert len(imgs) == n and int(re.search(r'FrameID(\d+)', imgs[0]).group(1)) == fid[0]
    snap = int(n * 0.74)
    if len(sys.argv) > 2:
        imgs = imgs[:int(sys.argv[2])]
    for i, nm in enumerate(imgs):
        full = cv2.imdecode(np.frombuffer(z.read(nm), np.uint8), cv2.IMREAD_COLOR)
        sec = i / 15
        lo = slice(half, i + 1)
        for m in models:
            img = cv2.resize(full, (FW, FH), interpolation=cv2.INTER_AREA)
            skeleton(img, yolo[i], conf[i], kin[i], sc1)
            if i < half:
                put(img, '1st half: used to measure YOLO noise -> not scored', (14, FH - 16), 0.7, (0, 220, 255), 2, True)
            canvas = np.full((944, FW + PW, 3), 25, np.uint8)
            canvas[:FH, :FW] = img
            t, lab_w, bw = tl[m]
            canvas[FH:FH + t.shape[0]] = t
            cx = lab_w + int(i / n * bw)
            cv2.line(canvas, (cx, FH), (cx, FH + t.shape[0] - 18), WHITE, 2)
            x0, y = FW + 14, 34
            put(canvas, TITLE[m], (x0, y), 0.62, WHITE, 2); y += 24
            put(canvas, f'params: {S["params"][m]:,}' if S['params'][m] else 'params: 0 (templates only)', (x0, y), 0.45, GRAY); y += 20
            put(canvas, f'video V001 (subject 1, never seen)  t={sec:5.1f}s', (x0, y), 0.45, GRAY); y += 20
            put(canvas, 'green = YOLO skeleton (model input)', (x0, y), 0.42, (0, 255, 0)); y += 18
            put(canvas, 'yellow dots = Kinect (reference)', (x0, y), 0.42, (0, 255, 255)); y += 28
            for lab, arr in rows_of[m]:
                p = arr[i]
                if lab == 'TRUE':
                    put(canvas, f'TRUE: {names[p]}', (x0, y), 0.58, WHITE, 2); y += 26
                    continue
                ok = p == gt[i]
                put(canvas, lab, (x0, y), 0.45, GRAY); y += 20
                put(canvas, f'  {names[p]}  ' + ('OK' if ok else 'X'), (x0, y), 0.56, GREEN if ok else RED, 2); y += 22
                run = f'{(arr[lo] == gt[lo]).mean() * 100:5.1f}%' if i >= half else '  -  '
                put(canvas, f'  accuracy so far (2nd half): {run}', (x0, y), 0.45, WHITE); y += 26
            y += 2
            cb, ca = mean3[m, before], mean3[m, after]
            put(canvas, 'mean of 3 seeds (tables):', (x0, y), 0.45, GRAY); y += 19
            put(canvas, f'  test 8 people: before {cb["ktest"]:.1f}%  after {ca["ktest"]:.1f}%', (x0, y), 0.43); y += 19
            put(canvas, f'  V001 YOLO   : before {cb["v_yolo"]:.1f}%  after {ca["v_yolo"]:.1f}%', (x0, y), 0.43); y += 26
            legend(canvas, x0, y, names, 0.4, 17)
            writers[m].write(canvas)
            if i == snap:
                cv2.imwrite(str(FIG / f'snap_{FILE[m]}.jpg'), canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])

        # المقارنة
        img = cv2.resize(full, (CW, CH), interpolation=cv2.INTER_AREA)
        skeleton(img, yolo[i], conf[i], kin[i], sc2)
        if i < half:
            put(img, '1st half: not scored', (12, CH - 14), 0.65, (0, 220, 255), 2, True)
        canvas = np.full((816, 1440, 3), 25, np.uint8)
        canvas[:CH, :CW] = img
        t, lab_w, bw = ctl
        canvas[CH:CH + t.shape[0]] = t
        cx = lab_w + int(i / n * bw)
        cv2.line(canvas, (cx, CH), (cx, CH + t.shape[0] - 18), WHITE, 2)
        x0, y = CW + 16, 30
        put(canvas, 'All models on the SAME video, YOLO input', (x0, y), 0.62, WHITE, 2); y += 22
        put(canvas, '(best recipe: upper body + YOLO-noise training, seed 0)', (x0, y), 0.43, GRAY); y += 30
        put(canvas, f'TRUE: {names[gt[i]]}', (x0, y), 0.62, WHITE, 2); y += 30
        for lab, dx in (('model', 0), ('now', 262), ('so far', 425), ('final', 500)):
            put(canvas, lab, (x0 + dx, y), 0.42, GRAY)
        y += 22
        for m in models:
            arr = P[m, after, 'yolo']
            ok = arr[i] == gt[i]
            run = f'{(arr[lo] == gt[lo]).mean() * 100:5.1f}%' if i >= half else '   -  '
            fin = f'{(arr[half:] == gt[half:]).mean() * 100:5.1f}%'
            put(canvas, SHORT[m], (x0, y), 0.43)
            put(canvas, names[arr[i]][:16], (x0 + 262, y), 0.45, GREEN if ok else RED, 1)
            put(canvas, run, (x0 + 425, y), 0.45); put(canvas, fin, (x0 + 500, y), 0.45, GRAY)
            y += 25
        legend(canvas, x0, y + 18, names[:7], 0.38, 16)
        legend(canvas, x0 + 300, y + 18, names[7:], 0.38, 16, first=7)
        cw.write(canvas)
        if i == snap:
            cv2.imwrite(str(FIG / 'snap_00_all_models.jpg'), canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if i % 100 == 0:
            print(i, '/', n, flush=True)
    for w in list(writers.values()) + [cw]:
        w.close()
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
