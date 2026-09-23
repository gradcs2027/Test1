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
from ground_truth import VIDEOS, spans
from paths import KP_OUT, load_keypoints, out_dir
from render_gt import EDGES, FONT, PALETTE, draw_skeleton, open_writer

# الإعدادات من experiment_frame_scales.py
FRAME_CONFIGS = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
RADIUS = 1
GT_LABEL_ALIASES = {
    'sitting': ['sitting', 'sit_down'],
}

OUT_DIR = Path('outputs')
OUT_DIR.mkdir(exist_ok=True)
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

    step = int(fps / fps_out)
    segments = []
    current_label = None
    t0 = 0

    mu, sigma = mu_sigma if use_z else (None, None)

    for frame_idx in range(0, len(kp) - n_frames, step):
        clip = kp[frame_idx:frame_idx + n_frames]
        if len(clip) < n_frames:
            break

        seq = resample_linear(normalize_window(clip), n=n_frames)
        feat = to_features(seq, mode='vel', shape_norm=True)

        if use_z:
            pred = classify_z(feat, templates, mu, sigma)
        else:
            pred = classify_raw(feat, templates)

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
    """رسم الشريط الزمني"""
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


def render_video(video_name, n_frames, templates, mu_sigma, use_z=True):
    """رسم فيديو كامل"""
    kp, fps = load_keypoints(video_name)

    # اقطع الـ segments
    segs, fps_actual, dur = segment_predictions(
        video_name, templates, mu_sigma, n_frames, use_z=use_z)

    colors = colors_for(segs)

    # قرا الفيديو الأصلي
    video_paths = [
        f'/kaggle/input/testvid_upload/{video_name}.mp4',
        f'/kaggle/input/testvid/{video_name}.mp4',
        Path('testvid_upload') / f'{video_name}.mp4',
        Path('testvid') / f'{video_name}.mp4',
    ]

    video_path = None
    for p in video_paths:
        if isinstance(p, str):
            p = Path(p)
        if p.exists():
            video_path = str(p)
            break

    if not video_path:
        print(f"   ⚠️ مالقتش {video_name}.mp4")
        return False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"   ❌ فشل الفتح")
        return False

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)

    # اعرف المسار
    folder = 'z_normalized' if use_z else 'raw'
    out_path = OUT_DIR / folder / f'{n_frames}_{video_name}.mp4'

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
            mode = "Z" if use_z else "RAW"
            cv2.rectangle(frame, (10, 10), (w - 10, 80), (50, 50, 50), -1)
            cv2.putText(frame, f'{n_frames}FR | {mode}: {label}',
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        writer.write(frame)
        frame_count += 1

    cap.release()
    writer.release()
    return True


def main():
    print("="*70)
    print("🎬 رسم التوقّعات لكل أعداد الفريمات")
    print("="*70)

    ext_templates = load_external_templates()

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
                if render_video(video, n_frames, templates, mu_sigma, use_z=use_z):
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
