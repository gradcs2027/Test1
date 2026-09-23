"""
رسم التوقّعات على الفيديوهات الأربعة لكل أعداد الفريمات

النتيجة: 40 فيديو
- 4 فيديوهات × 10 إعدادات (10, 20, 30...100) بـ raw
- 4 فيديوهات × 10 إعدادات بـ Z-normalization
- = 40 فيديو

المسار:
outputs/
├─ raw/
│  ├─ 10_vidtest1.mp4
│  ├─ 10_vidtest2.mp4
│  ├─ 10_vidtest3.mp4
│  ├─ 10_vidtest4.mp4
│  ├─ 20_vidtest1.mp4
│  ...
│  └─ 100_vidtest4.mp4
├─ z_normalized/
│  ├─ 10_vidtest1.mp4
│  ...
│  └─ 100_vidtest4.mp4

التشغيل: python render_all_frame_configs.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace',
                           line_buffering=True)

import _bootstrap  # noqa: F401

from classifier import norm_distance, normalize_window, resample_linear, to_features
from ground_truth import GROUND_TRUTH, VIDEOS
from paths import KP_OUT, load_keypoints, video_path
from render_gt import FONT, PALETTE, draw_skeleton, open_writer

# الإعدادات من experiment_frame_scales.py
FRAME_CONFIGS = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
RADIUS = 1
GT_LABEL_ALIASES = {
    'sitting': ['sitting', 'sit_down'],
}

import os
OUT_DIR = Path(os.environ.get('RENDER_OUT', 'outputs'))
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / 'raw').mkdir(exist_ok=True)
(OUT_DIR / 'z_normalized').mkdir(exist_ok=True)


def load_external_templates():
    """اقرا البنك"""
    ext_dir = KP_OUT / 'external'
    manifest = np.load(ext_dir / 'manifest.npy', allow_pickle=True)
    templates = []
    for m in manifest:
        kp = np.load(ext_dir / m['file'])
        templates.append({
            'label': m['label'],
            'clip': m['clip'],
            'kp': kp
        })
    return templates


def build_templates(ext_templates, n_frames):
    """بناء القوالب"""
    out = []
    for t in ext_templates:
        seq = resample_linear(normalize_window(t['kp']), n=n_frames)
        feat = to_features(seq, mode='vel', shape_norm=True)
        out.append({'label': t['label'], 'feat': feat, 'clip': t['clip']})
    return out


def calibrate_templates(templates):
    """معايرة Z"""
    n = len(templates)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = norm_distance(templates[i]['feat'], templates[j]['feat'], radius=RADIUS)
            D[i, j] = D[j, i] = d

    mu = np.zeros(n)
    sigma = np.ones(n)
    for j in range(n):
        others = np.delete(D[:, j], j)
        mu[j] = others.mean()
        sigma[j] = others.std() + 1e-6
    return mu, sigma


def classify_z(window_feat, templates, mu, sigma):
    """التصنيف مع Z"""
    per_label_best = {}
    for idx, t in enumerate(templates):
        d = norm_distance(window_feat, t['feat'], radius=RADIUS)
        z = (d - mu[idx]) / sigma[idx]
        if t['label'] not in per_label_best or z < per_label_best[t['label']]:
            per_label_best[t['label']] = z

    ranked = sorted(per_label_best.items(), key=lambda kv: kv[1])
    return ranked[0][0]


def classify_raw(window_feat, templates):
    """التصنيف بدون Z"""
    per_label_best = {}
    for t in templates:
        d = norm_distance(window_feat, t['feat'], radius=RADIUS)
        if t['label'] not in per_label_best or d < per_label_best[t['label']]:
            per_label_best[t['label']] = d

    ranked = sorted(per_label_best.items(), key=lambda kv: kv[1])
    return ranked[0][0]


def segment_predictions(video, templates, mu_sigma, n_frames, use_z=True, min_duration=0.5, fps_out=10):
    """قسّم الفيديو لـ segments"""
    kp, fps = load_keypoints(video)
    duration = len(kp) / fps

    step = max(1, int(fps / fps_out))
    mu, sigma = mu_sigma if use_z else (None, None)

    # توقّع لكل نافذة منزلقة، منسوب لنص النافذة
    times, preds = [], []
    for frame_idx in range(0, len(kp) - n_frames + 1, step):
        clip = kp[frame_idx:frame_idx + n_frames]
        seq = resample_linear(normalize_window(clip), n=n_frames)
        feat = to_features(seq, mode='vel', shape_norm=True)
        preds.append(classify_z(feat, templates, mu, sigma) if use_z
                     else classify_raw(feat, templates))
        times.append((frame_idx + n_frames / 2) / fps)

    if not preds:
        return [], fps, duration

    # تنعيم بالأغلبية على ~1 ثانية عشان الشريط يبقى مقروء
    k = max(1, int(fps_out * min_duration))
    smooth = []
    for i in range(len(preds)):
        win = preds[max(0, i - k):i + k + 1]
        smooth.append(max(set(win), key=win.count))

    # كل توقّع بيغطي من نص المسافة للي قبله لنص المسافة للي بعده
    bounds = [0.0] + [(a + b) / 2 for a, b in zip(times, times[1:])] + [duration]
    segments = []
    for i, lab in enumerate(smooth):
        if segments and segments[-1]['label'] == lab:
            segments[-1]['t1'] = bounds[i + 1]
        else:
            segments.append({'label': lab, 't0': bounds[i], 't1': bounds[i + 1]})

    return segments, fps, duration


HEAD_H = 70                     # شريط التوقّع فوق
BAR_H = 110                     # شريطين تحت: التوقّع + الـ GT
COLOR_OTHER = (120, 120, 120)   # سكون / مستبعد في الـ GT


def label_colors(ext_templates):
    """لون ثابت لكل حركة عبر كل الـ 80 فيديو — عشان المقارنة بالعين."""
    labs = sorted({t['label'] for t in ext_templates})
    return {l: PALETTE[i % len(PALETTE)] if i < len(PALETTE)
            else tuple(int(c) for c in np.random.default_rng(i).integers(60, 230, 3))
            for i, l in enumerate(labs)}


def _bar(frame, y, segs, colors, dur, w, title):
    """صف واحد في الشريط الزمني. segs = [(بداية، نهاية، لابل)]."""
    cv2.putText(frame, title, (4, y + 20), FONT, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
    x_min = 44
    span = w - x_min
    for s, e, l in segs:
        x1, x2 = x_min + int(s / dur * span), x_min + int(e / dur * span)
        col = colors.get(l, COLOR_OTHER)
        cv2.rectangle(frame, (x1, y + 6), (x2, y + 30), col, -1)
        cv2.rectangle(frame, (x1, y + 6), (x2, y + 30), (0, 0, 0), 1)
        tw = cv2.getTextSize(l, FONT, 0.32, 1)[0][0]
        if x2 - x1 > tw + 6:
            cv2.putText(frame, l, (x1 + (x2 - x1 - tw) // 2, y + 23),
                        FONT, 0.32, (255, 255, 255), 1, cv2.LINE_AA)


def draw_timeline(frame, pred, gt, colors, t, dur, w, h):
    """التوقّع فوق والـ GT تحته — نفس محور الزمن."""
    y0 = h - BAR_H
    cv2.rectangle(frame, (0, y0), (w, h), (18, 18, 18), -1)
    _bar(frame, y0 + 4, [(s['t0'], s['t1'], s['label']) for s in pred],
         colors, dur, w, 'PRED')
    _bar(frame, y0 + 40, gt, colors, dur, w, 'GT')
    for sec in range(0, int(dur) + 1, 5):
        x = 44 + int(sec / dur * (w - 44))
        cv2.putText(frame, str(sec), (max(1, x - 6), y0 + 94), FONT, 0.32,
                    (170, 170, 170), 1, cv2.LINE_AA)
    xc = 44 + int(t / dur * (w - 44))
    cv2.line(frame, (xc, y0 + 4), (xc, y0 + 80), (255, 255, 255), 2)


def render_video(video_name, n_frames, templates, mu_sigma, colors, use_z=True):
    """رسم فيديو كامل: هيدر (عدد الفريمات + التوقّع) / الفيديو + الهيكل / شريطين زمن."""
    kp, kp_fps = load_keypoints(video_name)
    segs, _, _ = segment_predictions(
        video_name, templates, mu_sigma, n_frames, use_z=use_z)
    gt = [(s, e, l) for s, e, l in GROUND_TRUTH[video_name]]

    cap = cv2.VideoCapture(str(video_path(video_name)))
    if not cap.isOpened():
        print(f"   ⚠️ مالقتش {video_path(video_name)}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = total / fps
    w -= w % 2
    vh -= vh % 2
    h = HEAD_H + vh + BAR_H

    folder = 'z_normalized' if use_z else 'raw'
    out_path = OUT_DIR / folder / f'{n_frames}fr_{video_name}.mp4'
    _, write, close = open_writer(out_path, fps, w, h)

    mode = 'Z-NORM' if use_z else 'RAW'
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        canvas = np.zeros((h, w, 3), np.uint8)
        canvas[HEAD_H:HEAD_H + vh] = frame[:vh, :w]

        k = min(int(t * kp_fps), len(kp) - 1)   # الـ keypoints بـ fps أقل من الفيديو
        draw_skeleton(canvas, kp[k], HEAD_H)

        seg = next((s for s in segs if s['t0'] <= t < s['t1']), None)
        label = seg['label'] if seg else '-'
        truth = next((l for s, e, l in gt if s <= t < e), '-')
        col = colors.get(label, COLOR_OTHER)
        cv2.rectangle(canvas, (0, 0), (w, HEAD_H), (18, 18, 18), -1)
        cv2.rectangle(canvas, (0, 0), (10, HEAD_H), col, -1)
        cv2.putText(canvas, label.upper(), (22, 40), FONT, 1.0, col, 2, cv2.LINE_AA)
        cv2.putText(canvas, f'{n_frames} frames | {mode} | GT: {truth}',
                    (24, 62), FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, f'{t:5.1f}s', (w - 80, 40), FONT, 0.55,
                    (230, 230, 230), 1, cv2.LINE_AA)

        draw_timeline(canvas, segs, gt, colors, t, dur, w, h)
        write(canvas)
        i += 1

    cap.release()
    close()
    return True


def main():
    print("="*70)
    print("🎬 رسم التوقّعات لكل أعداد الفريمات")
    print("="*70)

    ext_templates = load_external_templates()
    colors = label_colors(ext_templates)

    print(f"\n📚 تحضير البيانات...")
    print(f"   {len(ext_templates)} قالب من 4 داتاسِتس")

    total_videos = len(FRAME_CONFIGS) * len(VIDEOS) * 2  # raw + Z
    current = 0

    # شغّل كل إعداد
    for n_frames in FRAME_CONFIGS:
        print(f"\n{'='*70}")
        print(f"🎬 الإعداد: {n_frames} فريم")
        print(f"{'='*70}")

        # بناء القوالب
        templates = build_templates(ext_templates, n_frames)
        mu, sigma = calibrate_templates(templates)
        mu_sigma = (mu, sigma)

        # رسم نسختين
        for use_z, folder in [(False, 'raw'), (True, 'z_normalized')]:
            mode = "Z" if use_z else "RAW"
            print(f"\n  📊 {mode}:")

            for video in VIDEOS:
                current += 1
                status = f"[{current}/{total_videos}]"

                print(f"    {status} {video}...", end='', flush=True)
                if render_video(video, n_frames, templates, mu_sigma, colors, use_z=use_z):
                    print(f" ✅")
                else:
                    print(f" ⚠️")

    print("\n" + "="*70)
    print("✅ انتهى")
    print(f"📁 {OUT_DIR.absolute()}")
    print("="*70)

    # ملخص
    raw_count = len(list((OUT_DIR / 'raw').glob('*.mp4')))
    z_count = len(list((OUT_DIR / 'z_normalized').glob('*.mp4')))

    print(f"\n📊 النتيجة:")
    print(f"   Raw:         {raw_count} فيديو")
    print(f"   Z-normalized: {z_count} فيديو")
    print(f"   الإجمالي:     {raw_count + z_count} فيديو")


if __name__ == '__main__':
    main()
