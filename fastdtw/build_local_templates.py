"""
قصاصات محلية: فيديوهاتك اللي بعتهالي + حركتين من NTU60 (لمس الدماغ، النظارة).

ليه سكريبت منفصل ومحلي (مش على Kaggle)؟
─────────────────────────────────────────
- الفيديوهات اللي بعتهالي (Clap.mp4, Wave.mp4, ...) موجودة على جهازك بس،
  جنب فولدر Project مباشرة — مش داتاسِت Kaggle ومفيش داعي نرفعها كواحد.
- ملف NTU60 (ntu60_2d.pkl) بيتحمّل بـ رابط عام مباشر (مش عن طريق Kaggle
  دخول datasets)، وحجمه صغير كفاية (~700MB) إنه يتحمّل محلياً مرة واحدة
  بس، فمفيش داعي نحمّله تاني كل مرة على Kaggle.

بيستخرج الـ keypoints (صغيرين، كيلوبايتات) ويسجّلهم في:
    shared/keypoints/external_local/

الفولدر ده **لازم يتسجّل في git** (زي shared/keypoints/ العادي) عشان
build_external_templates.py على Kaggle يلاقيه جاهز بعد git clone، ويضمّه
لباقي القصاصات (HMDB51 / CCTV / Charades / UCF101).

الاستخدام (محلياً بس):
    python build_local_templates.py
"""
import pickle
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from paths import ON_KAGGLE, ROOT
from build_external_templates import _extract_from_video

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

if ON_KAGGLE:
    raise RuntimeError(
        'الملف ده لازم يتشغّل محلياً بس — على Kaggle استخدم build_external_templates.py')

# ── فيديوهاتك — متسجّلة برا الريبو، جنب فولدر Project مباشرة ──
USER_CLIPS_DIR = ROOT.parent
USER_CLIPS = {
    'Clap.mp4': 'clapping',
    'Wave.mp4': 'wave',
    'Sitting.mp4': 'sitting',
    'standing.mp4': 'stand_up',
    'phone call.mp4': 'phone_call',
    'spray perfume.mp4': 'spray_perfume',
}

# ── NTU60: ملف سكيلتون 2D جاهز (مستخرج مسبقاً بترتيب COCO-17، زي YOLO
# بالظبط) — بنجيب منه بس الحركتين اللي مالقناش لهم مصدر فيديو حقيقي.
# الأرقام دي index صفر-أساس من ورقة NTU RGB+D 60 الرسمية:
#   A18 "wear on glasses"  (index 17) -> wear_glasses
#   A44 "headache"         (index 43) -> touch_head (نفس الفئة اللي
#        استخدمها مشروع lstm/lstm_ntu60.ipynb قبل كده بنجاح)
NTU60_URL = 'https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ntu60_2d.pkl'
NTU60_CACHE = ROOT / '_scratch' / 'ntu60_2d.pkl'
NTU60_TO_LABEL = {
    17: 'wear_glasses',
    43: 'touch_head',
}

LOCAL_EXT_DIR = ROOT / 'shared' / 'keypoints' / 'external_local'
CLIPS_PER_LABEL = 2


def _download_ntu60():
    NTU60_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if NTU60_CACHE.exists():
        print(f'📦 NTU60 موجود بالفعل: {NTU60_CACHE}')
        return
    print('📥 تحميل ntu60_2d.pkl (~700MB)... ده بيحصل مرة واحدة بس')
    urllib.request.urlretrieve(NTU60_URL, NTU60_CACHE)
    print('✅ خلص التحميل')


def _ntu60_clips_by_label():
    _download_ntu60()
    with open(NTU60_CACHE, 'rb') as f:
        data = pickle.load(f)

    by_label = {}
    for ann in data['annotations']:
        if ann['label'] in NTU60_TO_LABEL:
            by_label.setdefault(ann['label'], []).append(ann)
    for lbl in by_label:
        by_label[lbl].sort(key=lambda a: a['frame_dir'])
    return by_label


def main():
    from ultralytics import YOLO

    LOCAL_EXT_DIR.mkdir(parents=True, exist_ok=True)
    print('📥 تحميل YOLOv8n-pose...')
    model = YOLO('yolov8n-pose.pt')

    manifest = []

    print('\n🎬 فيديوهاتك:')
    for fname, label in USER_CLIPS.items():
        path = USER_CLIPS_DIR / fname
        if not path.exists():
            print(f'  ⚠️ مالقتش {path} — اتخطّى')
            continue

        t0 = time.perf_counter()
        kp = _extract_from_video(path, model)
        if len(kp) < 4:
            print(f'  ✗ {fname}: فريمات قليلة قوي ({len(kp)}) — اتخطّى')
            continue

        out_name = f'{label}_user.npy'
        np.save(LOCAL_EXT_DIR / out_name, kp)
        manifest.append({
            'label': label, 'source': 'USER', 'clip': fname,
            'file': out_name, 'n_frames': int(len(kp)),
        })
        print(f'  ✓ {fname} -> {label}: {len(kp)} فريم '
              f'({time.perf_counter() - t0:.1f}s)')

    print('\n🧬 NTU60 (لمس الدماغ / النظارة):')
    by_label = _ntu60_clips_by_label()
    for ntu_label, our_label in NTU60_TO_LABEL.items():
        clips = by_label.get(ntu_label, [])[:CLIPS_PER_LABEL]
        print(f'  فئة NTU60 رقم {ntu_label} -> {our_label}: {len(clips)} قصاصة')
        for i, ann in enumerate(clips):
            kp = np.asarray(ann['keypoint'][0], dtype=np.float32)  # أول شخص: (frames, 17, 2)
            if len(kp) < 4:
                print(f'    ✗ {ann["frame_dir"]}: فريمات قليلة قوي — اتخطّى')
                continue
            out_name = f'{our_label}_ntu{i}.npy'
            np.save(LOCAL_EXT_DIR / out_name, kp)
            manifest.append({
                'label': our_label, 'source': 'NTU60', 'clip': ann['frame_dir'],
                'file': out_name, 'n_frames': int(len(kp)),
            })
            print(f'    ✓ {ann["frame_dir"]}: {len(kp)} فريم')

    np.save(LOCAL_EXT_DIR / 'manifest_local.npy', manifest, allow_pickle=True)
    print(f'\n✅ خلص محلياً — {len(manifest)} قصاصة في {LOCAL_EXT_DIR}')
    print('دلوقتي لازم نعمل git add/commit للفولدر ده عشان Kaggle ياخده '
          'وقت ما build_external_templates.py يشتغل هناك.')


if __name__ == '__main__':
    main()
