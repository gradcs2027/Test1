"""
يبني فيديوهات متعلّم عليها توقّعات TIER 2 جنب الإجابة الصح

    python fastdtw/render_tier2_pred.py                  # الأربع فيديوهات
    python fastdtw/render_tier2_pred.py vidtest1         # فيديو واحد

⚠️⚠️ الفرق بين الملف ده و `render_pred.py` القديم
────────────────────────────────────────────────
`render_pred.py` القديم بيستورد من `run_vidtest4` — الملف ده **اتغيّر
اسمه** لـ `evaluate_vidtest4.py`، يعني الراندر القديم مكسور. وكمان كان
بيستخدم الخوارزمية القديمة (1-NN خام) وقبل ما نصلّح الـ ground truth.

الملف ده بيستخدم **TIER 2** — أحسن نسخة عندنا (24.1% مقابل 13.8%):
معايرة z لكل template + تصويت أقرب ٣ جيران + تنضيف البنك.

⚠️ إيه اللي بيتعرض بالظبط
─────────────────────────
بنرسم التوقّع على **فترات الحركة المكتوبة في الـ ground truth بس** —
نفس بروتوكول القصاصات اللي طلّع الـ 24.1%. يعني اللي هتشوفه في الفيديو
هو **بالظبط** الرقم اللي في التقرير، مش حاجة تانية.

ليه مش على الفيديو كله؟ عشان الفيديو كله محتاج "عتبة رفض" تقرر إمتى
الشخص ساكن — والعتبة دي معامل بيتظبط، وده الفخ اللي وقعنا فيه قبل كده
(HANDOFF قسم 6.7: "علّي الـ accuracy فخ"). فبنرسم اللي قِسناه وبس.

الفترات المعلّمة `?` (مش متعلّم عليها) بتتلوّن رمادي غامق ومكتوب عليها
`not evaluated` — دي مش داخلة في أي حساب.

⚠️ مافيش تسريب: templates كل فيديو جاية من **الفيديوهات التانية**،
   والمعايرة من قصاصات المصدر بس.

ملاحظة: cv2.putText مابيرسمش عربي، فكل النص على الفيديو إنجليزي.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import _bootstrap  # noqa: F401

from ground_truth import EXCLUDED, GROUND_TRUTH, STILL, VIDEOS
from render_gt import (COLOR_EXCL, COLOR_STILL, FONT, PALETTE, draw_skeleton,
                       open_writer)
from paths import KP_DIR, VIDEO_DIR, out_dir

from evaluate_crossvideo_tier2 import (SHARED, build_templates, calibrate,
                                       distance_matrix, featurize, load,
                                       predict, source_reference_clips)

OUT_DIR = out_dir(__file__)

FRAME_SKIP = 2          # رقم الـ keypoint = رقم فريم الفيديو ÷ 2

HEAD_H = 126            # شريط فوق: الصح + التوقّع + الحكم
BAR_H = 74              # ارتفاع كل شريط زمني (بنرسم اتنين)

COLOR_OK = (90, 210, 90)        # أخضر — توقّع صح
COLOR_BAD = (60, 60, 235)       # أحمر — توقّع غلط
COLOR_WARN = (60, 60, 235)


# ==============================================================================
# التوقّع — نفس مسار TIER 2 بالحرف
# ==============================================================================

def predictions_for(video, mode='vel', shape_norm=True):
    """
    بيرجّع [(بداية, نهاية, الصح, التوقّع)] لكل فترة حركة في الفيديو.

    نفس مسار `evaluate_crossvideo_tier2.py` بالظبط — templates من
    الفيديوهات التانية، ومعايرة من قصاصات المصدر بس.
    """
    sources = [o for o in VIDEOS if o != video]
    templates = build_templates(sources, mode, shape_norm)
    labels_avail = {t['label'] for t in templates}

    refs = source_reference_clips(sources, mode, shape_norm)
    Dref = distance_matrix(refs, templates)
    mu, sigma, good, bad, keep = calibrate(templates, refs, Dref)

    kp, fps = load(video)
    items, meta = [], []
    for s, e, lab in GROUND_TRUTH[video]:
        if lab not in SHARED or lab not in labels_avail:
            continue
        f = featurize(kp, fps, s, e, mode, shape_norm)
        if f is None:
            continue
        items.append({'feat': f})
        meta.append((s, e, lab))

    if not items:
        return [], 0, 0

    D = distance_matrix(items, templates)
    out = []
    for i, (s, e, lab) in enumerate(meta):
        pred = predict(D[i], templates, mu, sigma, keep,
                       use_z=True, use_knn=True, use_prune=True)
        out.append((s, e, lab, pred))

    hit = sum(1 for _, _, t, p in out if t == p)
    return out, hit, len(out)


# ==============================================================================
# الرسم
# ==============================================================================

def colors_for(video, preds):
    """لون ثابت لكل حركة — نفس اللون في الشريطين عشان المقارنة تبقى سهلة."""
    labs = sorted({l for _, _, l in GROUND_TRUTH[video]
                   if l not in (EXCLUDED, STILL)}
                  | {p for _, _, _, p in preds})
    out = {l: PALETTE[i % len(PALETTE)] for i, l in enumerate(labs)}
    out[STILL] = COLOR_STILL
    out[EXCLUDED] = COLOR_EXCL
    return out


def at(spans, t):
    """أول فترة شاملة للحظة t."""
    for item in spans:
        if item[0] <= t < item[1]:
            return item
    return None


def draw_bar(frame, y0, title, blocks, colors, t, dur, w):
    """
    شريط زمني واحد. blocks = [(بداية, نهاية, لابل, مرسوم_ولا_لأ)]

    الفترات اللي مش متقيّمة (`?` أو حركة مش مشتركة) بتتلوّن غامق —
    عشان يبان بالعين إن دي مش داخلة في الرقم.
    """
    cv2.rectangle(frame, (0, y0), (w, y0 + BAR_H), (18, 18, 18), -1)
    cv2.putText(frame, title, (6, y0 + 15), FONT, 0.42, (180, 180, 180), 1,
                cv2.LINE_AA)

    for s, e, lab, scored in blocks:
        x1, x2 = int(s / dur * w), int(e / dur * w)
        col = colors.get(lab, COLOR_STILL)
        if not scored:
            col = tuple(int(c * 0.35) for c in col)
        cv2.rectangle(frame, (x1, y0 + 22), (x2, y0 + 50), col, -1)
        cv2.rectangle(frame, (x1, y0 + 22), (x2, y0 + 50), (0, 0, 0), 1)

        name = 'not eval' if lab == EXCLUDED else lab
        tw = cv2.getTextSize(name, FONT, 0.32, 1)[0][0]
        if x2 - x1 > tw + 6:
            cv2.putText(frame, name, (x1 + (x2 - x1 - tw) // 2, y0 + 42),
                        FONT, 0.32, (255, 255, 255), 1, cv2.LINE_AA)

    for sec in range(0, int(dur) + 1, 5):
        x = int(sec / dur * w)
        cv2.line(frame, (x, y0 + 50), (x, y0 + 56), (150, 150, 150), 1)
        cv2.putText(frame, str(sec), (max(1, x - 6), y0 + 68), FONT, 0.30,
                    (170, 170, 170), 1, cv2.LINE_AA)

    xc = int(t / dur * w)
    cv2.line(frame, (xc, y0 + 16), (xc, y0 + 56), (255, 255, 255), 2)
    cv2.circle(frame, (xc, y0 + 16), 4, (255, 255, 255), -1, cv2.LINE_AA)


def draw_header(frame, cur, t, dur, colors, w, n_pts, hit, tot):
    """
    فوق: الحركة الصح، وتوقّع TIER 2، والحكم.

    cur = (بداية, نهاية, الصح, التوقّع) أو None لو اللحظة دي مش متقيّمة.
    """
    cv2.rectangle(frame, (0, 0), (w, HEAD_H), (18, 18, 18), -1)

    if cur is None:
        cv2.rectangle(frame, (0, 0), (10, HEAD_H), COLOR_EXCL, -1)
        cv2.putText(frame, 'NOT EVALUATED', (22, 44), FONT, 0.9,
                    (150, 150, 150), 2, cv2.LINE_AA)
        cv2.putText(frame, 'outside the annotated gesture spans',
                    (24, 72), FONT, 0.46, (140, 140, 140), 1, cv2.LINE_AA)
    else:
        s, e, truth, pred = cur
        ok = truth == pred
        col = COLOR_OK if ok else COLOR_BAD
        cv2.rectangle(frame, (0, 0), (10, HEAD_H), col, -1)

        cv2.putText(frame, 'TRUE', (22, 34), FONT, 0.42, (150, 150, 150), 1,
                    cv2.LINE_AA)
        cv2.putText(frame, truth.upper(), (90, 36), FONT, 0.8,
                    colors.get(truth, (255, 255, 255)), 2, cv2.LINE_AA)

        cv2.putText(frame, 'PRED', (22, 70), FONT, 0.42, (150, 150, 150), 1,
                    cv2.LINE_AA)
        cv2.putText(frame, pred.upper(), (90, 72), FONT, 0.8,
                    colors.get(pred, (255, 255, 255)), 2, cv2.LINE_AA)

        cv2.putText(frame, 'CORRECT' if ok else 'WRONG', (w - 300, 56),
                    FONT, 0.78, col, 2, cv2.LINE_AA)
        cv2.putText(frame, f'[{s:.1f} - {e:.1f}]', (24, 96), FONT, 0.42,
                    (170, 170, 170), 1, cv2.LINE_AA)

    clock = f'{t:5.2f}s / {dur:.1f}s'
    cv2.putText(frame, clock, (w - 300, 26), FONT, 0.5, (230, 230, 230), 1,
                cv2.LINE_AA)
    if n_pts == 0:
        cv2.putText(frame, 'NO POSE', (w - 130, 96), FONT, 0.5, (0, 0, 230),
                    2, cv2.LINE_AA)

    acc = hit / tot * 100 if tot else 0.0
    cv2.putText(frame, f'FASTDTW TIER 2  -  this video: {hit}/{tot} '
                       f'({acc:.0f}%)',
                (22, 118), FONT, 0.44, COLOR_WARN, 1, cv2.LINE_AA)


def render(video):
    print(f'\n🔮 بحسب توقّعات TIER 2 لـ {video} '
          f'(templates من الفيديوهات التانية)...')
    preds, hit, tot = predictions_for(video)
    if not preds:
        print(f'   ⚠️ {video}: مافيش أي فترة متقيّمة — اتخطّيناه')
        return None
    print(f'   {tot} فترة متقيّمة، {hit} صح ({hit / tot * 100:.1f}%)')

    colors = colors_for(video, preds)
    pred_by_span = {(s, e): p for s, e, _, p in preds}

    # الشريط العلوي = الإجابة الصح كاملة. السفلي = التوقّع.
    gt_blocks, pr_blocks = [], []
    for s, e, lab in GROUND_TRUTH[video]:
        scored = (s, e) in pred_by_span
        gt_blocks.append((s, e, lab, scored))
        pr_blocks.append((s, e, pred_by_span.get((s, e), EXCLUDED), scored))

    kp_all = np.load(KP_DIR / f'{video}_keypoints.npy')

    cap = cv2.VideoCapture(str(VIDEO_DIR / f'{video}.mp4'))
    if not cap.isOpened():
        raise FileNotFoundError(
            f'مش قادر أفتح {VIDEO_DIR / video}.mp4\n'
            f'   VIDEO_DIR = {VIDEO_DIR}\n'
            f'   شغّل `python shared/paths.py` تشوف المسارات')

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = total / fps

    w -= w % 2          # libx264 بـ yuv420p محتاج أبعاد زوجية
    vh -= vh % 2
    h = HEAD_H + vh + BAR_H * 2

    out_path = OUT_DIR / f'tier2_{video}.mp4'
    _, write, close = open_writer(out_path, fps, w, h)
    print(f'🎬 {total} فريم @ {fps:.0f}fps → {out_path.name}')

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

        draw_header(canvas, at(preds, t), t, dur, colors, w, n_pts, hit, tot)
        draw_bar(canvas, HEAD_H + vh, 'GROUND TRUTH', gt_blocks, colors,
                 t, dur, w)
        draw_bar(canvas, HEAD_H + vh + BAR_H, 'FASTDTW TIER 2 PREDICTION',
                 pr_blocks, colors, t, dur, w)

        write(canvas)
        i += 1
        if i % 500 == 0:
            print(f'   {i}/{total}')

    cap.release()
    close()
    mb = out_path.stat().st_size / 1e6
    print(f'✅ {out_path.name}  ({mb:.1f} MB)')
    return out_path, hit, tot


def main():
    videos = sys.argv[1:] or list(VIDEOS)
    print('=' * 72)
    print('  رسم توقّعات FastDTW TIER 2 جنب الإجابة الصح')
    print('=' * 72)
    print('  الشريط الفوقاني = الإجابة الصح، التحتاني = توقّع الموديل.')
    print('  الفترات الغامقة = مش داخلة في الحساب (`?` أو حركة مش مشتركة).')

    made, H, T = [], 0, 0
    for v in videos:
        r = render(v)
        if r:
            made.append(r[0])
            H += r[1]
            T += r[2]

    print(f'\n{"=" * 72}')
    for p in made:
        print(f'   {p}')
    if T:
        print(f'\n  الإجمالي: {H}/{T} = {H / T * 100:.1f}%  '
              f'(نفس رقم التقرير بالظبط)')
    print('\n  ⚠️ ده توقّع الموديل مش الإجابة الصح. الشريط الفوقاني هو الصح.')


if __name__ == '__main__':
    main()
