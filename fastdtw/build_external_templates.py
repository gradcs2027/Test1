"""
قصاصات خارجية لعشر حركات: تصفيق، تلويح، جلوس، قيام، عناق، مصافحة،
شرب مية، تسريح شعر، مشي، جري — من مصدرين مختلفين حسب طبيعة الحركة.

ليه الملف ده موجود؟
────────────────────
طلب صريح من صاحب المشروع: templates تجربة "عدد الفريمات" لازم تيجي من
**بره** فيديوهاتنا (vidtest1-4) — مش من نفس الفيديوهات اللي بنختبر
عليها، عشان الاختبار يبقى نظيف فعلاً (صفر تسريب، مش بس "نظيف عبر
الفيديوهات" زي run_crossvideo.py).

ليه مصدرين؟
────────────────────
أول تجربة (HMDB51 لكل الحركات العشرة) طلعت دقة واطية (10-14%) —
جزء كبير من السبب إن HMDB51 أفلام/يوتيوب بزوايا وقرب كاميرا مختلفة
تماماً عن فيديوهاتنا (اللي مصوّرة بمنظور كاميرا مراقبة/بعيدة). فحركات
"الجسم الكامل" (جلوس، قيام، مشي، جري) بقينا نجيبها من داتاسِت **CCTV
حقيقي** عشان يبقى نفس منظور الكاميرا تقريباً. حركات الإيد/الجزء العلوي
(تصفيق، تلويح، شرب، تسريح شعر) وكمان العناق/المصافحة سبناهم من HMDB51
زي ما هما، لأنهم أقل تأثر بمسافة الكاميرا.

المصدر الأول — HMDB51 (جامعة Brown)، نسخة "rawframes" (فريمات
مستخرجة مسبقاً كصور) مرفوعة على Kaggle باسم jizeyong/hmdb51 (16.5GB).
⚠️ المصدر ده رفعه فرد مش الفريق الأصلي، وماعندوش وصف/ترخيص واضح على
   صفحته. المحتوى نفسه (HMDB51) داتاسِت أكاديمي معروف ومسموح للبحث.

    HMDB51        →  حركتنا عندنا
    clap          →  clapping
    wave          →  wave
    hug           →  hugging
    shake_hands   →  hand_shake
    drink         →  drink_water
    brush_hair    →  brush_hair

المصدر التاني — CCTV Action Recognition Dataset (Kaggle، jonathannield/
cctv-action-recognition-dataset، 618MB): قصاصات حقيقية مجمّعة من
داتاسِتات كاميرات مراقبة فعلية + يوتيوب/جوجل. أسامي الملفات نفسها فيها
اسم الحركة، مثال: "NTU_fight0003_fall_2.mp4" (مصدر_اسم_حركة_رقم).

    CCTV          →  حركتنا عندنا
    sit           →  sitting      (وده كمان اللي بيتقاس عليه sit_down)
    stand         →  stand_up
    walk          →  walking
    run           →  running

3 قصاصات لكل حركة من كل مصدر (أول 3 بالترتيب الأبجدي — ثابتة وقابلة
للتكرار، مش عشوائية).

الاستخدام (على Kaggle بس، بعد ما jizeyong/hmdb51 و
jonathannield/cctv-action-recognition-dataset يتوصّلوا):
    python build_external_templates.py
"""
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import _bootstrap  # noqa: F401
from paths import KP_OUT, ON_KAGGLE

HMDB_TO_LABEL = {
    'clap': 'clapping',
    'wave': 'wave',
    'hug': 'hugging',
    'shake_hands': 'hand_shake',
    'drink': 'drink_water',
    'brush_hair': 'brush_hair',
}

CCTV_TO_LABEL = {
    'sit': 'sitting',
    'stand': 'stand_up',
    'walk': 'walking',
    'run': 'running',
}

CLIPS_PER_LABEL = 3
EXT_DIR = KP_OUT / 'external'

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def _find_rawframes_root():
    """
    بيدوّر على الفولدر اللي جواه فولدرات الحركات (clap/, wave/, ...).

    مش هاردكودينج للمسار عشان Kaggle بيحط اسم الداتاسِت في المسار،
    وممكن يبقى فيه تعشيش زيادة (rawframes/rawframes/...) زي ما شفنا
    في متصفّح الداتاسِت. بندوّر بالاسم مش بعدد المستويات.
    """
    if not ON_KAGGLE:
        raise RuntimeError('لازم تشغّل الملف ده على Kaggle — محتاج الداتاسِت المرفوع')

    for root in sorted(Path('/kaggle/input').iterdir()):
        for p in root.rglob('clap'):
            if p.is_dir():
                return p.parent
    raise FileNotFoundError(
        'مالقتش فولدر "clap" جوّه أي داتاسِت متوصّل بالنوتبوك — '
        'اتأكد إن jizeyong/hmdb51 مضاف في kernel-metadata.json وإنه فعلاً متوصّل')


def _pick_clips(class_dir, n=CLIPS_PER_LABEL):
    """أول n قصاصة بالترتيب الأبجدي — ثابت، مش عشوائي."""
    return sorted((p for p in class_dir.iterdir() if p.is_dir()))[:n]


def _find_cctv_clips():
    """
    بيدوّر على فيديوهات CCTV اللي اسمها بينتهي بـ ..._<حركة>_<رقم>.<امتداد>
    زي "NTU_fight0003_fall_2.mp4" — مش هاردكودينج لمسار الفولدر، بندوّر
    بالاسم زي _find_rawframes_root، عشان مش عارفين تعشيش الفولدرات بالظبط.
    """
    if not ON_KAGGLE:
        raise RuntimeError('لازم تشغّل الملف ده على Kaggle — محتاج الداتاسِت المرفوع')

    pattern = re.compile(
        r'_(' + '|'.join(CCTV_TO_LABEL) + r')_\d+\.(mp4|avi|mov|mkv)$', re.IGNORECASE)
    found = {k: [] for k in CCTV_TO_LABEL}
    all_names_sample = []
    for root in sorted(Path('/kaggle/input').iterdir()):
        if not root.is_dir():
            continue
        for p in root.rglob('*'):
            if not p.is_file():
                continue
            if len(all_names_sample) < 20:
                all_names_sample.append(p.name)
            m = pattern.search(p.name)
            if m:
                found[m.group(1).lower()].append(p)

    missing = [k for k, v in found.items() if not v]
    if missing:
        raise FileNotFoundError(
            f'مالقتش فيديوهات لـ {missing} — اتأكد إن '
            f'jonathannield/cctv-action-recognition-dataset متوصّل بالنوتبوك.\n'
            f'عينة من أسامي الملفات اللي لقيتها: {all_names_sample}')
    return found


def _extract_from_frames(frame_dir, model):
    """(frames, 17, 2) من فولدر صور مرتّبة — نفس منطق pose_extract.py
    بس بيقرا من ملفات صور جاهزة مش من فيديو."""
    frame_files = sorted(frame_dir.glob('*.jpg'))
    keypoints = []
    for f in frame_files:
        frame = cv2.imread(str(f))
        if frame is None:
            continue

        res = model(frame, verbose=False)[0]
        kp = None
        if res.keypoints is not None and len(res.keypoints) > 0:
            if res.boxes is not None and len(res.boxes) > 1:
                areas = (res.boxes.xywh[:, 2] * res.boxes.xywh[:, 3]).cpu().numpy()
                person = int(np.argmax(areas))
            else:
                person = 0
            cand = res.keypoints.xy[person].cpu().numpy()
            if cand.shape == (17, 2) and np.any(cand != 0):
                kp = cand

        keypoints.append(kp if kp is not None else np.zeros((17, 2), dtype=np.float32))

    return np.array(keypoints, dtype=np.float32)


def _extract_from_video(video_path, model):
    """(frames, 17, 2) من ملف فيديو — كل فريم (القصاصات قصيرة أصلاً)."""
    cap = cv2.VideoCapture(str(video_path))
    keypoints = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        res = model(frame, verbose=False)[0]
        kp = None
        if res.keypoints is not None and len(res.keypoints) > 0:
            if res.boxes is not None and len(res.boxes) > 1:
                areas = (res.boxes.xywh[:, 2] * res.boxes.xywh[:, 3]).cpu().numpy()
                person = int(np.argmax(areas))
            else:
                person = 0
            cand = res.keypoints.xy[person].cpu().numpy()
            if cand.shape == (17, 2) and np.any(cand != 0):
                kp = cand

        keypoints.append(kp if kp is not None else np.zeros((17, 2), dtype=np.float32))

    cap.release()
    return np.array(keypoints, dtype=np.float32)


def main():
    from ultralytics import YOLO

    EXT_DIR.mkdir(parents=True, exist_ok=True)
    print('📥 تحميل YOLOv8n-pose...')
    model = YOLO('yolov8n-pose.pt')

    manifest = []

    # ── المصدر الأول: HMDB51 (rawframes) ──
    root = _find_rawframes_root()
    print(f'📁 لقيت فولدرات حركات HMDB51 في: {root}')

    for hmdb_class, our_label in HMDB_TO_LABEL.items():
        class_dir = root / hmdb_class
        if not class_dir.is_dir():
            print(f'  ⚠️ مالقتش فولدر {hmdb_class} — اتخطّى')
            continue

        clips = _pick_clips(class_dir)
        print(f'\n🎬 [HMDB51] {hmdb_class} → {our_label}: {len(clips)} قصاصة مختارة')

        for i, clip_dir in enumerate(clips):
            t0 = time.perf_counter()
            kp = _extract_from_frames(clip_dir, model)
            if len(kp) < 4:
                print(f'   ✗ {clip_dir.name}: فريمات قليلة قوي ({len(kp)}) — اتخطّى')
                continue

            out_name = f'{our_label}_{i}.npy'
            np.save(EXT_DIR / out_name, kp)
            manifest.append({
                'label': our_label,
                'source': 'HMDB51',
                'clip': clip_dir.name,
                'file': out_name,
                'n_frames': int(len(kp)),
            })
            print(f'   ✓ {clip_dir.name}: {len(kp)} فريم '
                  f'({time.perf_counter() - t0:.1f}s)')

    # ── المصدر التاني: CCTV Action Recognition Dataset (فيديوهات حقيقية) ──
    cctv_clips = _find_cctv_clips()
    print(f'\n📁 لقيت فيديوهات CCTV لكل الحركات المطلوبة')

    for cctv_class, our_label in CCTV_TO_LABEL.items():
        clips = sorted(cctv_clips[cctv_class])[:CLIPS_PER_LABEL]
        print(f'\n🎬 [CCTV] {cctv_class} → {our_label}: {len(clips)} قصاصة مختارة')

        for i, clip_path in enumerate(clips):
            t0 = time.perf_counter()
            kp = _extract_from_video(clip_path, model)
            if len(kp) < 4:
                print(f'   ✗ {clip_path.name}: فريمات قليلة قوي ({len(kp)}) — اتخطّى')
                continue

            out_name = f'{our_label}_{i}.npy'
            np.save(EXT_DIR / out_name, kp)
            manifest.append({
                'label': our_label,
                'source': 'CCTV',
                'clip': clip_path.name,
                'file': out_name,
                'n_frames': int(len(kp)),
            })
            print(f'   ✓ {clip_path.name}: {len(kp)} فريم '
                  f'({time.perf_counter() - t0:.1f}s)')

    np.save(EXT_DIR / 'manifest.npy', manifest, allow_pickle=True)
    print(f'\n✅ خلص — {len(manifest)} قصاصة خارجية في {EXT_DIR}')


if __name__ == '__main__':
    main()
