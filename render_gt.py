"""
يرسم الـ ground truth على الفيديوهات عشان نتأكد منها بالعين

    python render_gt.py                 # التلاتة
    python render_gt.py vidtest1        # فيديو واحد

ليه؟
────
جدول vidtest1 مكتوب من contact sheets مش من مشاهدة فعلية
(`verified_by_owner: False`)، وهو **مصدر معظم الـ templates** في
الاختبار عبر-الفيديوهات. أي غلطة في توقيتاته بتنتقل لكل النتايج.

الفيديو ده بيخلّي المراجعة ممكنة: بتتفرج وتشوف الاسم مكتوب فوق، وتقول
"لأ، الوقوف بدأ قبل كده" — من غير ما تعدّ فريمات بإيدك.

⚠️ ده **مش** توقّعات موديل. ده الإجابة اللي إحنا كاتبينها، معروضة عشان
   تتراجع. لو غلط، الرقم اللي طلع من الاختبار غلط معاه.

ملاحظة: cv2.putText مابيرسمش عربي، فكل النص على الفيديو إنجليزي.
"""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from ground_truth import EXCLUDED, GROUND_TRUTH, STILL, VIDEO_INFO

VIDEO_DIR = Path(__file__).resolve().parent.parent / 'testvid_upload'
KP_DIR = Path(__file__).resolve().parent / 'keypoints'
OUT_DIR = Path(__file__).resolve().parent / 'results'
FRAME_SKIP = 2                      # نفس اللي في pose_extract.py
CRF = 23                            # 18=أنقى/أكبر، 28=أصغر/أوحش

# الفيديو بيتحط بين شريطين، فمفيش أي رسم فوق الصورة نفسها.
# قبل كده الهيدر كان مرسوم فوق الفيديو وبيغطي أول 96 بكسل منه.
HEAD_H = 96                         # شريط اللابل فوق
BAR_H = 84                          # الشريط الزمني تحت
FONT = cv2.FONT_HERSHEY_SIMPLEX

# هيكل COCO-17 اللي YOLOv8-pose بيطلّعه
EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
         (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
         (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6)]

# BGR. الرمادي للسكون المتأكّد، الأحمر الغامق للمستبعد.
PALETTE = [(80, 200, 80), (230, 160, 40), (60, 140, 240), (200, 90, 210),
           (40, 210, 210), (150, 120, 250), (90, 190, 130), (210, 190, 60)]
COLOR_STILL = (120, 120, 120)
COLOR_EXCL = (45, 45, 130)


def open_writer(path, fps, w, h):
    """
    كاتب فيديو H.264 عن طريق ffmpeg، وبيرجع لـ OpenCV لو مش موجود.

    🐛 ليه مش cv2.VideoWriter على طول: الـ OpenCV المتجمّع مع pip
       مافيهوش H.264 شغّال (libopenh264 بيفشل بـ -1313558101)، فبيقع
       على mp4v اللي ضغطه ضعيف جداً — التلات فيديوهات طلعوا 525 MB،
       وده فوق حد الـ 100 MB بتاع GitHub. بـ ffmpeg بينزلوا ~30×.

    بيرجّع (كائن_الكتابة, دالة_الكتابة, دالة_الإغلاق).
    """
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = None

    if exe is None:
        print('   ⚠️ مافيش ffmpeg — هيستخدم mp4v والملف هيبقى كبير')
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'),
                             fps, (w, h))
        return vw, vw.write, vw.release

    cmd = [exe, '-y', '-loglevel', 'error',
           '-f', 'rawvideo', '-pix_fmt', 'bgr24',
           '-s', f'{w}x{h}', '-r', f'{fps}', '-i', '-',
           '-an', '-c:v', 'libx264', '-preset', 'medium',
           '-crf', str(CRF), '-pix_fmt', 'yuv420p',
           '-movflags', '+faststart', str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def close():
        p.stdin.close()
        p.wait()

    return p, lambda f: p.stdin.write(f.tobytes()), close


def colors_for(video):
    """لون ثابت لكل حركة. السكون رمادي والمستبعد أحمر غامق."""
    labs = sorted({l for _, _, l in GROUND_TRUTH[video]
                   if l not in (EXCLUDED, STILL)})
    out = {l: PALETTE[i % len(PALETTE)] for i, l in enumerate(labs)}
    out[STILL] = COLOR_STILL
    out[EXCLUDED] = COLOR_EXCL
    return out


def seg_at(gt, t):
    """بيرجّع (بداية، نهاية، لابل) عند الثانية t، أو None."""
    for s, e, l in gt:
        if s <= t < e:
            return s, e, l
    return None


def draw_skeleton(canvas, kp, y_off):
    """
    يرسم الهيكل. النقط اللي (0,0) معناها مافيش اكتشاف، فبتتساب.

    ⚠️ `y_off` لازم يبقى ارتفاع الشريط اللي فوق. الـ keypoints
       بإحداثيات الفيديو الأصلي، والفيديو دلوقتي منزّل لتحت جوّه
       الـ canvas — من غير الإزاحة دي الهيكل بيتزنق في الشريط.

    بيرجّع عدد النقط المرسومة — لو صفر يبقى الفريم ده مافيهوش شخص،
    وده في حد ذاته معلومة مهمة للمراجعة.
    """
    ok = kp[:, 0] > 0
    pt = [(int(x), int(y) + y_off) for x, y in kp]

    for a, b in EDGES:
        if ok[a] and ok[b]:
            cv2.line(canvas, pt[a], pt[b], (255, 255, 255), 2, cv2.LINE_AA)
    for i in np.flatnonzero(ok):
        cv2.circle(canvas, pt[i], 4, (0, 215, 255), -1, cv2.LINE_AA)
    return int(ok.sum())


def draw_timeline(frame, gt, colors, t, dur, w, h):
    """
    شريط زمني تحت: كل فترة مستطيل بعرضها الحقيقي + مؤشر عند اللحظة دي.

    ده أهم حتة في الفيديو للمراجعة — بيخلّيك تشوف الحدود جاية إمتى
    قبل ما توصلها، فتقدر تحكم هي بدري ولا متأخرة.
    """
    y0 = h - BAR_H
    cv2.rectangle(frame, (0, y0), (w, h), (18, 18, 18), -1)

    for s, e, l in gt:
        x1, x2 = int(s / dur * w), int(e / dur * w)
        cv2.rectangle(frame, (x1, y0 + 22), (x2, y0 + 52), colors[l], -1)
        cv2.rectangle(frame, (x1, y0 + 22), (x2, y0 + 52), (0, 0, 0), 1)

        name = 'excl' if l == EXCLUDED else ('still' if l == STILL else l)
        tw = cv2.getTextSize(name, FONT, 0.34, 1)[0][0]
        if x2 - x1 > tw + 6:
            cv2.putText(frame, name, (x1 + (x2 - x1 - tw) // 2, y0 + 43),
                        FONT, 0.34, (255, 255, 255), 1, cv2.LINE_AA)

    for sec in range(0, int(dur) + 1, 5):
        x = int(sec / dur * w)
        cv2.line(frame, (x, y0 + 52), (x, y0 + 58), (150, 150, 150), 1)
        cv2.putText(frame, str(sec), (max(1, x - 6), y0 + 72), FONT, 0.32,
                    (170, 170, 170), 1, cv2.LINE_AA)

    xc = int(t / dur * w)
    cv2.line(frame, (xc, y0 + 14), (xc, y0 + 58), (255, 255, 255), 2)
    cv2.circle(frame, (xc, y0 + 14), 4, (255, 255, 255), -1, cv2.LINE_AA)


def draw_header(frame, seg, t, dur, colors, w, warn, n_pts):
    """اللابل الحالي + الفترة + الزمن، وتحذير لو الجدول مش متأكّد منه."""
    if seg is None:
        text, sub, col = 'NO LABEL', '', (0, 0, 200)
    else:
        s, e, l = seg
        col = colors[l]
        if l == EXCLUDED:
            text, sub = 'EXCLUDED', 'not annotated - ignored in scoring'
        elif l == STILL:
            text, sub = 'still', f'verified stillness  [{s:.1f} - {e:.1f}]'
        else:
            text, sub = l.upper(), f'[{s:.1f} - {e:.1f}]  ({e - s:.1f}s)'

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

    if warn:
        cv2.putText(frame, 'GROUND TRUTH - UNVERIFIED, PLEASE CHECK',
                    (22, 90), FONT, 0.42, (0, 180, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'GROUND TRUTH (not model output)', (22, 90),
                    FONT, 0.42, (140, 140, 140), 1, cv2.LINE_AA)


def render(video):
    gt = GROUND_TRUTH[video]
    info = VIDEO_INFO[video]
    colors = colors_for(video)
    warn = not info.get('verified_by_owner')

    kp_all = np.load(KP_DIR / f'{video}_keypoints.npy')

    cap = cv2.VideoCapture(str(VIDEO_DIR / f'{video}.mp4'))
    if not cap.isOpened():
        raise FileNotFoundError(f'مش قادر أفتح {VIDEO_DIR / video}.mp4')

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))     # ارتفاع الفيديو نفسه
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = total / fps

    # libx264 بـ yuv420p محتاج الأبعاد زوجية
    w -= w % 2
    vh -= vh % 2
    h = HEAD_H + vh + BAR_H                          # شريط + فيديو + شريط

    out_path = OUT_DIR / f'gt_{video}.mp4'
    OUT_DIR.mkdir(exist_ok=True)
    _, write, close = open_writer(out_path, fps, w, h)

    print(f'\n🎬 {video}: {total} فريم @ {fps:.0f}fps → {out_path.name}')
    if warn:
        print('   ⚠️ الجدول ده لسه مش متأكّد منه — ده الفيديو اللي محتاج مراجعة')

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t = i / fps
        canvas = np.zeros((h, w, 3), np.uint8)
        canvas[HEAD_H:HEAD_H + vh] = frame[:vh, :w]   # الفيديو كامل، مش متغطى

        k = min(i // FRAME_SKIP, len(kp_all) - 1)
        n_pts = draw_skeleton(canvas, kp_all[k], HEAD_H)

        draw_header(canvas, seg_at(gt, t), t, dur, colors, w, warn, n_pts)
        draw_timeline(canvas, gt, colors, t, dur, w, h)

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
    videos = sys.argv[1:] or list(GROUND_TRUTH)
    print('=' * 66)
    print('  رسم الـ ground truth على الفيديوهات للمراجعة بالعين')
    print('=' * 66)

    made = [render(v) for v in videos]

    print(f'\n{"=" * 66}')
    print('  خلص — افتح الملفات دي واتفرج')
    print('=' * 66)
    for p in made:
        print(f'   {p}')
    print('\n  الحاجة المهمة: gt_vidtest1.mp4. الجدول بتاعه مكتوب من')
    print('  contact sheets مش من مشاهدة، وهو مصدر معظم الـ templates.')
    print('  لو لقيت توقيت غلط، قول الرقم الصح وأنا أعدّله وأعيد الاختبار.')


if __name__ == '__main__':
    main()
