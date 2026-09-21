"""
رسم التوقّعات على الفيديوهات الأربعة

التفاوت عن render_pred.py:
- render_pred.py بتشتغل على vidtest4 (الأعمى فقط)
- الملف ده بتشتغل على vidtest1, 2, 3, 4 كلهم
- بتستخدم نفس الموديل (experiment_frame_scales الحالي)

التشغيل: python render_all_predictions.py
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
from ground_truth import VIDEOS, spans
from paths import KP_OUT, load_keypoints, out_dir
from render_gt import EDGES, FONT, PALETTE, draw_skeleton, open_writer

# نفس الإعدادات من experiment_frame_scales.py
RADIUS = 1
GT_LABEL_ALIASES = {
    'sitting': ['sitting', 'sit_down'],
}
BEST_N_FRAMES = 70  # أحسن إعداد من التجربة (بـ Z-normalization)

OUT_DIR = out_dir(__file__)


def load_external_templates():
    """اقرا البنك اللي اتبنى في build_external_templates.py"""
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
    """طبّع وحوّل القوالب"""
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


def classify_with_confidence_z(window_feat, templates, mu, sigma):
    """التصنيف مع Z-normalization"""
    per_label_best = {}
    for idx, t in enumerate(templates):
        d = norm_distance(window_feat, t['feat'], radius=RADIUS)
        z = (d - mu[idx]) / sigma[idx]
        if t['label'] not in per_label_best or z < per_label_best[t['label']]:
            per_label_best[t['label']] = z

    ranked = sorted(per_label_best.items(), key=lambda kv: kv[1])
    best_label, best_z = ranked[0]
    confidence = ranked[1][1] - best_z if len(ranked) > 1 else 1.0
    return best_label, confidence


def segment_predictions(video, ext_templates, mu, sigma, n_frames, min_duration=0.5, fps_out=10):
    """
    قسّم الفيديو لـ segments (فترات متتالية بنفس التصنيف)

    الفكرة: مش كل فريم، بل كل 0.1 ثانية نعمل تصنيف → معدّ للرسم بدون flicker
    """
    kp, fps = load_keypoints(video)
    duration = len(kp) / fps

    # شبابيك كل 0.1 ثانية
    step = int(fps / fps_out)  # fps/10 frames
    segments = []
    current_label = None
    t0 = 0

    for frame_idx in range(0, len(kp) - n_frames, step):
        clip = kp[frame_idx:frame_idx + n_frames]
        if len(clip) < n_frames:
            break

        seq = resample_linear(normalize_window(clip), n=n_frames)
        feat = to_features(seq, mode='vel', shape_norm=True)
        pred, conf = classify_with_confidence_z(feat, ext_templates, mu, sigma)

        t = frame_idx / fps

        if pred != current_label:
            if current_label is not None and t - t0 >= min_duration:
                segments.append({
                    'label': current_label,
                    't0': t0,
                    't1': t,
                })
            current_label = pred
            t0 = t

    if current_label is not None and duration - t0 >= min_duration:
        segments.append({
            'label': current_label,
            't0': t0,
            't1': duration,
        })

    return segments, fps, duration


def colors_for(segs):
    labs = sorted({s['label'] for s in segs})
    return {l: PALETTE[i % len(PALETTE)] for i, l in enumerate(labs)}


def draw_timeline(frame, segs, colors, t, dur, w, h):
    """رسم الشريط الزمني تحت"""
    y0 = h - 84
    cv2.rectangle(frame, (0, y0), (w, h), (18, 18, 18), -1)

    for s in segs:
        x1 = int(s['t0'] / dur * w)
        x2 = int(s['t1'] / dur * w)
        col = colors[s['label']]
        cv2.rectangle(frame, (x1, y0 + 22), (x2, y0 + 52), col, -1)

        tw = cv2.getTextSize(s['label'], FONT, 0.32, 1)[0][0]
        if x2 - x1 > tw + 6:
            cv2.putText(frame, s['label'],
                       (x1 + (x2 - x1 - tw) // 2, y0 + 43),
                       FONT, 0.32, (255, 255, 255), 1, cv2.LINE_AA)

    xc = int(t / dur * w)
    cv2.line(frame, (xc, y0 + 14), (xc, y0 + 58), (255, 255, 255), 2)


def render_video(video_name, ext_templates, mu, sigma):
    """رسم فيديو كامل"""
    kp, fps = load_keypoints(video_name)

    print(f"\n🎬 {video_name}:")
    print(f"   {len(kp)} فريم @ {fps} fps = {len(kp)/fps:.1f}s")

    # اقطع الـ segments
    segs, fps_actual, dur = segment_predictions(
        video_name, ext_templates, mu, sigma, BEST_N_FRAMES)
    print(f"   ✅ {len(segs)} segment متوقّع")

    colors = colors_for(segs)

    # قرا الفيديو الأصلي
    video_path = f'/kaggle/input/testvid_upload/{video_name}.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"   ❌ مالقتش {video_path}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)

    out_path = OUT_DIR / f'pred_{video_name}.mp4'
    writer = open_writer(out_path, w, h, fps_video)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_count / fps_video

        # رسم الـ skeleton
        try:
            kp_frame = kp[frame_count]
            frame = draw_skeleton(frame, kp_frame, EDGES, PALETTE[0])
        except Exception:
            pass

        # رسم الشريط الزمني
        draw_timeline(frame, segs, colors, t, dur, w, h)

        # رسم الـ label الحالي
        seg = next((s for s in segs if s['t0'] <= t < s['t1']), None)
        if seg:
            label = seg['label']
            cv2.rectangle(frame, (10, 10), (w - 10, 80), (50, 50, 50), -1)
            cv2.putText(frame, f'PREDICTION: {label}', (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

        writer.write(frame)
        frame_count += 1

    cap.release()
    writer.release()
    print(f"   ✅ {out_path}")


def main():
    print("="*70)
    print(f"🎬 رسم التوقّعات على الفيديوهات الأربعة")
    print(f"   الإعداد: {BEST_N_FRAMES} فريم + Z-normalization")
    print("="*70)

    ext_templates = load_external_templates()
    templates = build_templates(ext_templates, BEST_N_FRAMES)
    mu, sigma = calibrate_templates(templates)

    for video in VIDEOS:
        try:
            render_video(video, templates, mu, sigma)
        except Exception as e:
            print(f"   ❌ خطأ: {e}")

    print("\n" + "="*70)
    print("✅ انتهى")
    print("="*70)


if __name__ == '__main__':
    main()
