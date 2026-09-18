"""
تشغيل build_external_templates.py **بالكامل** محلياً على شجرة مزيّفة.

الفرق بين الملف ده و test_dataset_finders.py:
  - ده بيختبر المسارات بس (بسرعة، بملفات فاضية)
  - الملف ده بيبني صور jpg وفيديوهات mp4/avi **حقيقية**، وبيشغّل main()
    كلها من أولها لآخرها بموديل YOLO حقيقي — نفس الكود اللي بيتشغّل على
    Kaggle بالحرف، بس على داتا صغيرة.

الهدف: نتأكد إن الـ 18 حركة بتطلع فعلاً بقصاصتين قبل ما نحرق تشغيلة
تانية على Kaggle. مش محتاج داتاسِت ولا كارت شاشة ولا فيديوهات المستخدم.

التشغيل:  python fastdtw/tests/test_end_to_end.py
"""
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent          # fastdtw/tests/
ROOT = HERE.parent.parent                       # فولدر المشروع
sys.path.insert(0, str(HERE.parent))            # fastdtw/

# YOLO('yolov8n-pose.pt') بيتنادى باسم نسبي جوّه main()، فلازم نبقى
# واقفين في فولدر المشروع عشان يلاقي الموديل مهما اتشغّل الاختبار منين
os.chdir(ROOT)

import build_external_templates as bet  # noqa: E402

MOCK = ROOT / '_scratch' / 'mock_e2e'
OUT = ROOT / '_scratch' / 'mock_e2e_out'

W, H, N_FRAMES = 160, 120, 10
CHARADES_VIDS = ['0BH84', '0L07S', '08Y62', '09F15', '107YZ', '1BVUA']

# نافذة الحركة قصيرة عن عمد: 0.05s→0.35s عند 24 فريم/ثانية = 8 فريمات
CHARADES_CSV_ROWS = [
    ('0BH84', 'c019 0.05 0.35'),
    ('0L07S', 'c019 0.05 0.35'),
    ('08Y62', 'c104 0.05 0.35'),
    ('09F15', 'c104 0.05 0.35'),
    ('107YZ', 'c133 0.05 0.35'),
    ('1BVUA', 'c146 0.05 0.35'),
    ('ZZZZZ', ''),
]


def _person_frame(t):
    """رسمة بدائية لشخص بتتحرك — مش مهم YOLO يلاقيها، المهم إن الصورة
    حقيقية وبتتقرا زي أي فريم."""
    img = np.full((H, W, 3), 210, dtype=np.uint8)
    x = 50 + int(20 * np.sin(t))
    cv2.circle(img, (x, 30), 9, (60, 60, 60), -1)
    cv2.line(img, (x, 39), (x, 75), (60, 60, 60), 4)
    cv2.line(img, (x, 48), (x - 16, 62 + int(8 * np.sin(t))), (60, 60, 60), 3)
    cv2.line(img, (x, 48), (x + 16, 62 - int(8 * np.sin(t))), (60, 60, 60), 3)
    cv2.line(img, (x, 75), (x - 11, 100), (60, 60, 60), 3)
    cv2.line(img, (x, 75), (x + 11, 100), (60, 60, 60), 3)
    return img


def _write_frames(folder, prefix, n=N_FRAMES):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        cv2.imwrite(str(folder / f'{prefix}-{i + 1:06d}.jpg'), _person_frame(i * 0.6))


def _write_video(path, n=N_FRAMES):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*('MJPG' if path.suffix == '.avi' else 'mp4v'))
    vw = cv2.VideoWriter(str(path), fourcc, 24.0, (W, H))
    for i in range(n):
        vw.write(_person_frame(i * 0.6))
    vw.release()
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f'ماقدرتش اكتب الفيديو {path} — codec مش متاح')


def build_tree():
    if MOCK.exists():
        shutil.rmtree(MOCK)
    ds = MOCK / 'datasets'

    hmdb = ds / 'jizeyong' / 'hmdb51' / 'rawframes' / 'rawframes'
    for cls in bet.HMDB_TO_LABEL:
        for i in range(3):
            _write_frames(hmdb / cls / f'{cls}_clip_{i}', 'img')

    cctv = ds / 'jonathannield' / 'cctv-action-recognition-dataset' / 'Videos' / 'Videos'
    for cls in bet.CCTV_TO_LABEL:
        for i in (1, 2, 3):
            _write_video(cctv / cls / f'NTU_fight00{i:02d}_{cls}_{i}.mp4')

    charades = ds / 'jizeyong' / 'charades'
    header = ('id,subject,scene,quality,relevance,verified,script,objects,'
              'descriptions,actions,length\n')
    rows = ''.join(
        f'{vid},SUBJ,Bedroom,6,6,Yes,script,obj,desc,{acts},12.00\n'
        for vid, acts in CHARADES_CSV_ROWS)
    (charades / 'Charades_v1_train.csv').parent.mkdir(parents=True, exist_ok=True)
    (charades / 'Charades_v1_train.csv').write_text(header + rows, encoding='utf-8')

    # ⭐ التعشيش المكرر اللي وقّع التشغيلة الأخيرة على Kaggle
    rgb = charades / 'Charades_v1_rgb' / 'Charades_v1_rgb'
    for vid in CHARADES_VIDS:
        _write_frames(rgb / vid, vid)

    ucf = ds / 'matthewjansen' / 'ucf101-action-recognition' / 'train' / 'TableTennisShot'
    for g, c in ((2, 3), (4, 6), (1, 1)):
        _write_video(ucf / f'v_TableTennisShot_g{g:02d}_c{c:02d}.avi')

    return ds


def main():
    print('🔧 ببني شجرة فيها صور وفيديوهات حقيقية...')
    build_tree()
    if OUT.exists():
        shutil.rmtree(OUT)

    bet.KAGGLE_INPUT = MOCK
    bet.ON_KAGGLE = True
    bet.EXT_DIR = OUT

    print('🚀 بشغّل main() بالكامل — نفس الكود اللي بيتشغّل على Kaggle\n')
    print('─' * 64)
    bet.main()
    print('─' * 64)

    manifest = np.load(OUT / 'manifest.npy', allow_pickle=True)
    counts = {}
    for m in manifest:
        counts[m['label']] = counts.get(m['label'], 0) + 1

    expected = (
        {'clapping', 'phone_call', 'sitting', 'spray_perfume', 'stand_up',
         'touch_head', 'wave', 'wear_glasses'}
        | set(bet.HMDB_TO_LABEL.values()) | set(bet.CCTV_TO_LABEL.values())
        | set(bet.CHARADES_TO_LABEL.values()) | set(bet.UCF101_TO_LABEL.values())
    )

    print('\n📊 المراجعة النهائية')
    fails = []

    missing = sorted(expected - set(counts))
    ok = not missing
    print(f'  {"✓" if ok else "✗"} كل الـ {len(expected)} حركة موجودة'
          f'{"" if ok else f" — ناقص: {missing}"}')
    fails += [] if ok else ['حركات ناقصة خالص']

    short = {l: n for l, n in sorted(counts.items()) if n < bet.CLIPS_PER_LABEL}
    ok = set(short) <= {'spray_perfume'}
    print(f'  {"✓" if ok else "✗"} مفيش حركة ناقصة قصاصات غير spray_perfume'
          f' — الناقص: {short}')
    fails += [] if ok else ['حركات ناقصة قصاصات']

    over = {l: n for l, n in counts.items() if n > bet.CLIPS_PER_LABEL}
    print(f'  {"✓" if not over else "✗"} مفيش حركة أخدت أكتر من قصاصتين'
          f'{"" if not over else f" — {over}"}')
    fails += [] if not over else ['قصاصات زيادة']

    charades_rows = [m for m in manifest if m['source'] == 'CHARADES']
    ok = len(charades_rows) >= 4 and all(m['n_frames'] >= 4 for m in charades_rows)
    print(f'  {"✓" if ok else "✗"} ⭐ Charades طلّعت فريمات فعلاً (مش صفر): '
          f'{[(m["label"], m["n_frames"]) for m in charades_rows]}')
    fails += [] if ok else ['Charades لسه بتطلّع صفر فريم']

    phone = sorted(m['source'] for m in manifest if m['label'] == 'phone_call')
    ok = 'USER' in phone
    print(f'  {"✓" if ok else "✗"} phone_call فيها فيديو المستخدم + مصدر تاني: {phone}')
    fails += [] if ok else ['أولوية المصادر']

    files_on_disk = {p.name for p in OUT.glob('*.npy')} - {'manifest.npy'}
    ok = files_on_disk >= {m['file'] for m in manifest}
    print(f'  {"✓" if ok else "✗"} كل ملفات الـ npy المذكورة موجودة على الديسك '
          f'({len(files_on_disk)} ملف)')
    fails += [] if ok else ['ملفات ناقصة']

    shapes_ok = True
    for m in manifest:
        kp = np.load(OUT / m['file'])
        if kp.ndim != 3 or kp.shape[1:] != (17, 2) or len(kp) != m['n_frames']:
            shapes_ok = False
            print(f'      ✗ {m["file"]}: شكل غلط {kp.shape} (المفروض {m["n_frames"]}, 17, 2)')
    print(f'  {"✓" if shapes_ok else "✗"} كل القصاصات شكلها (فريمات, 17, 2)')
    fails += [] if shapes_ok else ['شكل المصفوفات']

    print(f'\n{"=" * 64}')
    if fails:
        print(f'❌ فشل: {", ".join(fails)}')
    else:
        print(f'✅ التشغيلة الكاملة نجحت — {len(manifest)} قصاصة، '
              f'{len(counts)} حركة، وكله بالشكل المتوقّع')
    shutil.rmtree(MOCK, ignore_errors=True)
    shutil.rmtree(OUT, ignore_errors=True)
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
