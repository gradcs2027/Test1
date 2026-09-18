"""
قصاصات خارجية للـ18 حركة كلهم (كل الحركات الموجودة في vidtest1-4) —
حركتين بالظبط لكل حركة، من 5 مصادر حسب طبيعة الحركة.

ليه الملف ده موجود؟
────────────────────────
طلب صريح من صاحب المشروع: templates تجربة "عدد الفريمات" لازم تيجي من
**بره** فيديوهاتنا (vidtest1-4) — مش من نفس الفيديوهات اللي بنختبر
عليها، عشان الاختبار يبقى نظيف فعلاً (صفر تسريب). وطلب كمان إن كل حركة
تتغطى بالظبط بقصاصتين، مهما احتاج الأمر مصادر داتا مختلفة.

المصادر الخمسة
────────────────────
1. **محلي (USER + NTU60)** — مستخرج قبل كده على جهازك بـ
   build_local_templates.py ومسجّل في git تحت shared/keypoints/external_local/.
   ده بيغطّي: فيديوهاتك الحقيقية (تصفيق، تلويح، جلوس، وقوف، مكالمة،
   رش عطر) + لمس الدماغ ولبس النظارة (من NTU60، صيغة سكيلتون جاهزة).

2. **HMDB51** (جامعة Brown، نسخة "rawframes" مرفوعة على Kaggle باسم
   jizeyong/hmdb51، 16.5GB) — حركات الإيد/الجزء العلوي، أقل تأثر بمسافة
   الكاميرا: تصفيق، تلويح، عناق، مصافحة، شرب مية، تسريح شعر.
   ⚠️ مصدر فردي، ماعندوش وصف/ترخيص واضح، لكن HMDB51 نفسه داتاسِت أكاديمي
   معروف ومسموح للبحث.

3. **CCTV Action Recognition Dataset** (Kaggle، jonathannield/
   cctv-action-recognition-dataset، 618MB) — قصاصات حقيقية من كاميرات
   مراقبة فعلية، أقرب لمنظور فيديوهاتنا: جلوس، وقوف، مشي، جري، نوم/استلقاء
   (فئة "LyingDown" عندهم). أسامي الملفات فيها اسم الحركة، مثال:
   "NTU_fight0003_fall_2.mp4" (مصدر_اسم_حركة_رقم).

4. **Charades** (معهد Allen AI، نسخة rawframes مرفوعة على Kaggle باسم
   jizeyong/charades) — فيديوهات ناس بتعمل حركات يومية جوه البيت، كاميرا
   شبه ثابتة. بيغطّي 3 حركات كانت عالقة: يفتح النور (فئة "Turning on a
   light")، يصحى من النوم (فئة "awakening")، وقصاصة تانية لمكالمة تليفون
   (فئة "Talking on a phone"، بالإضافة لفيديوك). الفيديو الواحد فيه أكتر
   من حركة جوه بعض، فبنقرا ملف الـ annotations (CSV) ونقص بس الفترة
   الزمنية اللي فيها الحركة المطلوبة (24 فريم/ثانية، معدل استخراج Charades
   الرسمي).

5. **UCF101** — بيغطّي بينج بونج (فئة "TableTennisShot")، بنفس أسلوب
   البحث بالاسم اللي في CCTV (v_TableTennisShot_g01_c01.avi، تسمية UCF101
   الرسمية الثابتة في كل نسخه المرفوعة).

    HMDB51        →  حركتنا           CCTV        →  حركتنا
    clap          →  clapping         sit         →  sitting
    wave          →  wave             stand       →  stand_up
    hug           →  hugging          walk        →  walking
    shake_hands   →  hand_shake       run         →  running
    drink         →  drink_water      lyingdown   →  lying
    brush_hair    →  brush_hair

    Charades      →  حركتنا                       UCF101         →  حركتنا
    c019          →  phone_call                   tabletennisshot → play_pingpong
    c104          →  turn_on_light
    c133 / c146   →  wake_up

2 قصاصة بالظبط لكل حركة في النهاية (_cap_per_label). لو حركة عندها أكتر
من مصدر (زي phone_call: فيديوك + Charades)، بنفضّل فيديوك/NTU60 الأول
لأنهم حقيقيين ليك أو من داتاسِت مضبوط الفورمات مسبقاً، وبعدين نكمّل من
باقي المصادر بالترتيب اللي جوه main().

الاستخدام (على Kaggle، بعد ما build_local_templates.py يتشغّل محلياً
واتعمله commit، وبعد ما jizeyong/hmdb51، jonathannield/
cctv-action-recognition-dataset، jizeyong/charades، وداتاسِت UCF101
يتوصّلوا):
    python build_external_templates.py
"""
import csv
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import _bootstrap  # noqa: F401
from paths import KP_OUT, ON_KAGGLE, ROOT

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
    'lyingdown': 'lying',
}

CHARADES_TO_LABEL = {
    'c019': 'phone_call',      # Talking on a phone/camera
    'c104': 'turn_on_light',   # Turning on a light
    'c133': 'wake_up',         # Someone is awakening in bed
    'c146': 'wake_up',         # Someone is awakening somewhere
}
CHARADES_FPS = 24.0   # معدل استخراج فريمات Charades RGB الرسمي

UCF101_TO_LABEL = {
    'tabletennisshot': 'play_pingpong',
}

# ترتيب الأفضلية لما حركة توصلها قصاصات أكتر من CLIPS_PER_LABEL —
# المحلي (فيديوهاتك/NTU60) بييجي الأول، وبعدين المصادر الخارجية.
_SOURCE_PRIORITY = {'USER': 0, 'NTU60': 1, 'HMDB51': 2, 'CCTV': 2, 'CHARADES': 3, 'UCF101': 3}

CLIPS_PER_LABEL = 2
EXT_DIR = KP_OUT / 'external'
LOCAL_EXT_DIR = ROOT / 'shared' / 'keypoints' / 'external_local'

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def _dataset_dir(owner_slug):
    """
    مسار داتاسِت معيّن بالاسم (زي 'jizeyong/hmdb51') من غير ما ندوّر
    في كل الداتاسِتس التانية.

    ⚠️ ليه الدالة دي موجودة؟ الشكل القديم كان بيعمل
    `Path('/kaggle/input').iterdir()` ويمشي جوّه **كل حاجة تحتها**. لو
    Kaggle حاطط كل الداتاسِتس تحت فولدر واحد وسيط (`/kaggle/input/datasets/
    <owner>/<slug>/...` — ده اللي شفناه فعلاً في اللوج)، بقى `iterdir()`
    بيرجّع فولدر واحد بس ('datasets')، فأي دالة بتدوّر على حاجة كانت
    بتمشي في الـ100+ جيجا بتوع الخمس داتاسِتس **مع بعض** بدل ما تمشي في
    اللي محتاجاه بس — وده اللي خلّى خطوة CCTV تقعد شغالة من غير خلاص
    (بتمشي في الـ75 جيجا بتوع Charades من غير أي داعي وهي بتدوّر على
    CCTV بس). الدالة دي بتجرّب المسارين المعروفين (الجديد بالـ owner،
    والقديم من غيره) وترجع أول واحد موجود فعلاً — من غير أي مشي زيادة.
    """
    owner, slug = owner_slug.split('/')
    for candidate in (
        Path('/kaggle/input/datasets') / owner / slug,
        Path('/kaggle/input') / slug,
    ):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f'مالقتش داتاسِت {owner_slug} في /kaggle/input — اتأكد إنه مضاف '
        f'في kernel-metadata.json وإنه فعلاً متوصّل بالنوتبوك')


def _find_rawframes_root(marker_dir, owner_slug):
    """بيدوّر على الفولدر اللي جواه فولدرات الحركات (marker_dir بيتأكد
    بيه) **جوّه داتاسِت واحد بس** — مش كل /kaggle/input."""
    if not ON_KAGGLE:
        raise RuntimeError('لازم تشغّل الملف ده على Kaggle — محتاج الداتاسِت المرفوع')

    root = _dataset_dir(owner_slug)
    for p in root.rglob(marker_dir):
        if p.is_dir():
            return p.parent
    raise FileNotFoundError(
        f'مالقتش فولدر "{marker_dir}" جوّه داتاسِت {owner_slug} — '
        f'اتأكد إنه فعلاً بالبنية المتوقّعة')


def _pick_clips(class_dir, n=CLIPS_PER_LABEL):
    """أول n قصاصة بالترتيب الأبجدي — ثابت، مش عشوائي."""
    return sorted((p for p in class_dir.iterdir() if p.is_dir()))[:n]


def _find_clips_by_suffix(class_keys, suffix_pattern, owner_slug):
    """
    بيدوّر على فيديوهات اسمها بينتهي بـ ..._<فئة><suffix_pattern>.<امتداد>
    زي "NTU_fight0003_fall_2.mp4" أو "v_TableTennisShot_g01_c01.avi" —
    مش هاردكودينج لمسار الفولدر الداخلي، بندوّر بالاسم لأن مش عارفين
    تعشيش الفولدرات بالظبط جوه الداتاسِت. لكن **بنمشي جوّه الداتاسِت
    المطلوب بس** (owner_slug) — مش كل /kaggle/input زي الشكل القديم،
    عشان ميضطرش يمشي في داتاسِتس تانية ضخمة (Charades 75GB) وهو بيدوّر
    على حاجة صغيرة في داتاسِت تاني خالص.
    """
    if not ON_KAGGLE:
        raise RuntimeError('لازم تشغّل الملف ده على Kaggle — محتاج الداتاسِت المرفوع')

    root = _dataset_dir(owner_slug)
    pattern = re.compile(
        r'_(' + '|'.join(class_keys) + r')' + suffix_pattern + r'\.(mp4|avi|mov|mkv)$',
        re.IGNORECASE)
    found = {k: [] for k in class_keys}
    all_names_sample = []
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
            f'مالقتش فيديوهات لـ {missing} — اتأكد إن {owner_slug} متوصّل بالنوتبوك.\n'
            f'عينة من أسامي الملفات اللي لقيتها: {all_names_sample}')
    for k in found:
        found[k].sort()
    return found


def _find_charades_root():
    """بيدوّر على ملف Charades_v1_train.csv وفولدر فريمات الـ rgb جنبه —
    جوّه داتاسِت charades بس (75GB)، مش كل /kaggle/input."""
    if not ON_KAGGLE:
        raise RuntimeError('لازم تشغّل الملف ده على Kaggle — محتاج الداتاسِت المرفوع')

    root = _dataset_dir('jizeyong/charades')
    csvs = list(root.rglob('Charades_v1_train.csv'))
    if not csvs:
        raise FileNotFoundError(
            'مالقتش Charades_v1_train.csv جوّه jizeyong/charades — '
            'اتأكد إنه فعلاً بالبنية المتوقّعة')
    csv_path = csvs[0]

    # بندوّر على فولدر فريمات الـ rgb، بس من غير ما ننزل جوّه فولدرات
    # فريمات كل فيديو على حدة (فيه آلاف منها) — أول ما نلاقي فولدر اسمه
    # فيه 'rgb' منوقف وما بنكملش ننزل جواه.
    import os
    rgb_dir = None
    for dirpath, dirnames, _filenames in os.walk(root):
        for d in dirnames:
            if 'rgb' in d.lower():
                rgb_dir = Path(dirpath) / d
                break
        if rgb_dir is not None:
            break
    return csv_path, (rgb_dir if rgb_dir is not None else csv_path.parent)


def _charades_candidates(csv_path):
    """بيرجّع dict: كود الحركة -> [(video_id, start, end), ...] بترتيب ثابت."""
    found = {code: [] for code in CHARADES_TO_LABEL}
    with open(csv_path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            actions = (row.get('actions') or '').strip()
            if not actions:
                continue
            for triplet in actions.split(';'):
                parts = triplet.split()
                if len(parts) != 3:
                    continue
                code, start, end = parts
                if code in found:
                    found[code].append((row['id'], float(start), float(end)))
    for code in found:
        found[code].sort()
    return found


def _pose_from_bgr(frame, model):
    """(17, 2) أو None — نفس منطق اختيار أكبر صندوق شخص في كل مكان بالملف."""
    res = model(frame, verbose=False)[0]
    if res.keypoints is not None and len(res.keypoints) > 0:
        if res.boxes is not None and len(res.boxes) > 1:
            areas = (res.boxes.xywh[:, 2] * res.boxes.xywh[:, 3]).cpu().numpy()
            person = int(np.argmax(areas))
        else:
            person = 0
        cand = res.keypoints.xy[person].cpu().numpy()
        if cand.shape == (17, 2) and np.any(cand != 0):
            return cand
    return None


def _extract_frame_files(frame_files, model):
    """(frames, 17, 2) من قايمة ملفات صور بترتيب معيّن."""
    keypoints = []
    for f in frame_files:
        frame = cv2.imread(str(f))
        if frame is None:
            continue
        kp = _pose_from_bgr(frame, model)
        keypoints.append(kp if kp is not None else np.zeros((17, 2), dtype=np.float32))
    return np.array(keypoints, dtype=np.float32)


def _extract_from_frames(frame_dir, model):
    """(frames, 17, 2) من فولدر صور مرتّبة كامل."""
    return _extract_frame_files(sorted(frame_dir.glob('*.jpg')), model)


def _extract_charades_clip(rgb_dir, video_id, start, end, model):
    """بيقص فريمات الفترة [start, end] بس من فولدر فريمات الفيديو الكامل."""
    frame_dir = rgb_dir / video_id
    frame_files = sorted(frame_dir.glob('*.jpg'))
    if not frame_files:
        return np.zeros((0, 17, 2), dtype=np.float32)
    lo = max(0, int(round(start * CHARADES_FPS)))
    hi = min(len(frame_files), int(round(end * CHARADES_FPS)) + 1)
    return _extract_frame_files(frame_files[lo:hi], model)


def _extract_from_video(video_path, model):
    """(frames, 17, 2) من ملف فيديو — كل فريم (القصاصات قصيرة أصلاً)."""
    cap = cv2.VideoCapture(str(video_path))
    keypoints = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        kp = _pose_from_bgr(frame, model)
        keypoints.append(kp if kp is not None else np.zeros((17, 2), dtype=np.float32))
    cap.release()
    return np.array(keypoints, dtype=np.float32)


def _cap_per_label(manifest, cap=CLIPS_PER_LABEL):
    """بيقص كل حركة لـ cap قصاصة بالظبط، مفضّل المصادر الأعلى أفضلية."""
    by_label = {}
    for m in manifest:
        by_label.setdefault(m['label'], []).append(m)
    out = []
    for items in by_label.values():
        items.sort(key=lambda m: _SOURCE_PRIORITY.get(m['source'], 9))
        out.extend(items[:cap])
    return out


def _load_local_templates():
    """قصاصاتك + NTU60 — اتستخرجوا قبل كده محلياً بـ build_local_templates.py
    ومسجّلين في git، هنا بس بننسخهم لفولدر الإخراج الحالي."""
    manifest_path = LOCAL_EXT_DIR / 'manifest_local.npy'
    if not manifest_path.exists():
        print(f'⚠️ مفيش قصاصات محلية في {LOCAL_EXT_DIR} — تخطّي '
              f'(شغّل build_local_templates.py محلياً الأول لو ده مش مقصود)')
        return []

    local_manifest = np.load(manifest_path, allow_pickle=True)
    out = []
    for m in local_manifest:
        kp = np.load(LOCAL_EXT_DIR / m['file'])
        np.save(EXT_DIR / m['file'], kp)
        out.append(dict(m))
    print(f'📦 قصاصات محلية: {len(out)}')
    return out


def main():
    from ultralytics import YOLO

    EXT_DIR.mkdir(parents=True, exist_ok=True)
    print('📥 تحميل YOLOv8n-pose...')
    model = YOLO('yolov8n-pose.pt')

    manifest = []

    # ── مصدر محلي: فيديوهاتك + NTU60 ──
    manifest += _load_local_templates()

    # ── HMDB51 (rawframes) ──
    root = _find_rawframes_root('clap', 'jizeyong/hmdb51')
    print(f'\n📁 لقيت فولدرات حركات HMDB51 في: {root}')

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

            out_name = f'{our_label}_hmdb{i}.npy'
            np.save(EXT_DIR / out_name, kp)
            manifest.append({
                'label': our_label, 'source': 'HMDB51', 'clip': clip_dir.name,
                'file': out_name, 'n_frames': int(len(kp)),
            })
            print(f'   ✓ {clip_dir.name}: {len(kp)} فريم '
                  f'({time.perf_counter() - t0:.1f}s)')

    # ── CCTV Action Recognition Dataset (فيديوهات حقيقية) ──
    cctv_clips = _find_clips_by_suffix(
        CCTV_TO_LABEL, r'_\d+', 'jonathannield/cctv-action-recognition-dataset')
    print(f'\n📁 لقيت فيديوهات CCTV لكل الحركات المطلوبة')

    for cctv_class, our_label in CCTV_TO_LABEL.items():
        clips = cctv_clips[cctv_class][:CLIPS_PER_LABEL]
        print(f'\n🎬 [CCTV] {cctv_class} → {our_label}: {len(clips)} قصاصة مختارة')

        for i, clip_path in enumerate(clips):
            t0 = time.perf_counter()
            kp = _extract_from_video(clip_path, model)
            if len(kp) < 4:
                print(f'   ✗ {clip_path.name}: فريمات قليلة قوي ({len(kp)}) — اتخطّى')
                continue

            out_name = f'{our_label}_cctv{i}.npy'
            np.save(EXT_DIR / out_name, kp)
            manifest.append({
                'label': our_label, 'source': 'CCTV', 'clip': clip_path.name,
                'file': out_name, 'n_frames': int(len(kp)),
            })
            print(f'   ✓ {clip_path.name}: {len(kp)} فريم '
                  f'({time.perf_counter() - t0:.1f}s)')

    # ── Charades (rawframes + قص بالـ annotations) ──
    csv_path, rgb_dir = _find_charades_root()
    print(f'\n📁 لقيت Charades: {csv_path.name} + فريمات في {rgb_dir}')
    charades_candidates = _charades_candidates(csv_path)

    # فئتين كود بيتقاسوا نفس حركتنا (wake_up) — بنجمعهم قبل ما نقص لـ CLIPS_PER_LABEL
    by_our_label = {}
    for code, our_label in CHARADES_TO_LABEL.items():
        by_our_label.setdefault(our_label, []).extend(
            (code, vid, s, e) for vid, s, e in charades_candidates[code])

    for our_label, cands in by_our_label.items():
        cands = cands[:CLIPS_PER_LABEL]
        print(f'\n🎬 [Charades] → {our_label}: {len(cands)} قصاصة مختارة')

        for i, (code, video_id, start, end) in enumerate(cands):
            t0 = time.perf_counter()
            kp = _extract_charades_clip(rgb_dir, video_id, start, end, model)
            if len(kp) < 4:
                print(f'   ✗ {video_id} ({code}): فريمات قليلة قوي ({len(kp)}) — اتخطّى')
                continue

            out_name = f'{our_label}_charades{i}.npy'
            np.save(EXT_DIR / out_name, kp)
            manifest.append({
                'label': our_label, 'source': 'CHARADES',
                'clip': f'{video_id}_{code}_{start:.1f}-{end:.1f}',
                'file': out_name, 'n_frames': int(len(kp)),
            })
            print(f'   ✓ {video_id} ({code}, {start:.1f}s-{end:.1f}s): '
                  f'{len(kp)} فريم ({time.perf_counter() - t0:.1f}s)')

    # ── UCF101 (بينج بونج) ──
    ucf_clips = _find_clips_by_suffix(
        UCF101_TO_LABEL, r'_g\d+_c\d+', 'matthewjansen/ucf101-action-recognition')
    print(f'\n📁 لقيت فيديوهات UCF101 لكل الحركات المطلوبة')

    for ucf_class, our_label in UCF101_TO_LABEL.items():
        clips = ucf_clips[ucf_class][:CLIPS_PER_LABEL]
        print(f'\n🎬 [UCF101] {ucf_class} → {our_label}: {len(clips)} قصاصة مختارة')

        for i, clip_path in enumerate(clips):
            t0 = time.perf_counter()
            kp = _extract_from_video(clip_path, model)
            if len(kp) < 4:
                print(f'   ✗ {clip_path.name}: فريمات قليلة قوي ({len(kp)}) — اتخطّى')
                continue

            out_name = f'{our_label}_ucf{i}.npy'
            np.save(EXT_DIR / out_name, kp)
            manifest.append({
                'label': our_label, 'source': 'UCF101', 'clip': clip_path.name,
                'file': out_name, 'n_frames': int(len(kp)),
            })
            print(f'   ✓ {clip_path.name}: {len(kp)} فريم '
                  f'({time.perf_counter() - t0:.1f}s)')

    manifest = _cap_per_label(manifest, CLIPS_PER_LABEL)
    np.save(EXT_DIR / 'manifest.npy', manifest, allow_pickle=True)

    by_label_final = {}
    for m in manifest:
        by_label_final.setdefault(m['label'], 0)
        by_label_final[m['label']] += 1
    short = {l: n for l, n in by_label_final.items() if n < CLIPS_PER_LABEL}

    print(f'\n✅ خلص — {len(manifest)} قصاصة خارجية في {EXT_DIR} '
          f'({len(by_label_final)} حركة)')
    if short:
        print(f'⚠️ حركات ناقصة قصاصات ({CLIPS_PER_LABEL} مطلوبين): {short}')


if __name__ == '__main__':
    main()
