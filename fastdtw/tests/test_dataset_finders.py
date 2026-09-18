"""
اختبار محلي لدوال تحديد المسارات في build_external_templates.py.

ليه الملف ده موجود؟
────────────────────
تلات بقّات ورا بعض طلعوا من نفس المكان بالظبط (دوال التدوير على
الداتاسِتس)، وكل مرة كنا بنكتشفها بتشغيلة كاملة على Kaggle بتاخد دقايق
وتحرق محاولة. الملف ده بيبني شجرة فولدرات **مزيّفة** بنفس شكل Kaggle
الحقيقي (بالتعشيش المكرر اللي شفناه بعنينا) وبيشغّل الدوال عليها محلياً
في أقل من ثانية — من غير داتاسِت ومن غير GPU ومن غير الفيديوهات.

بيختبر حاجتين:
  1. الصح: الدوال بترجّع المسارات المضبوطة؟
  2. الأداء: الدوال بتمشي جوّه الداتاسِت المطلوب **بس**؟ (بنسجّل كل
     فولدر اتقرا فعلاً ونتأكد إن الداتاسِتس التانية ماتلمستش خالص)

التشغيل:  python fastdtw/tests/test_dataset_finders.py
"""
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent          # fastdtw/tests/
sys.path.insert(0, str(HERE.parent))            # fastdtw/

import build_external_templates as bet  # noqa: E402

# الشجرة المزيّفة بتتبني في _scratch (متجاهَل في git) مش جنب الاختبار
MOCK = HERE.parent.parent / '_scratch' / 'mock_kaggle'

CHARADES_CSV = (
    'id,subject,scene,quality,relevance,verified,script,objects,descriptions,actions,length\n'
    '0BH84,FZTQ,Bedroom,6,6,Yes,script one,phone,desc one,c019 11.90 21.20;c092 0.00 5.00,24.83\n'
    '08Y62,ABCD,Kitchen,5,6,Yes,script two,light,desc two,c104 2.50 7.75,20.10\n'
    '107YZ,EFGH,Bedroom,7,6,Yes,script three,bed,desc three,c133 0.00 6.40;c019 30.00 33.00,31.20\n'
    '0L07S,IJKL,Living,6,5,Yes,script four,phone,desc four,c019 4.00 9.10,15.00\n'
    '09F15,MNOP,Hallway,6,6,Yes,script five,light,desc five,c104 1.00 4.20,12.40\n'
    '1BVUA,QRST,Bedroom,6,6,Yes,script six,bed,desc six,c146 3.30 8.80,19.90\n'
    'ZZZZZ,UVWX,Garage,4,4,Yes,nothing here,car,desc seven,,10.00\n'
)


def _touch(path, text='x'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def build_mock_tree(charades_double_nested=True, decoy_files=1500):
    """
    شجرة بنفس شكل Kaggle الحقيقي اللي شفناه في اللوج:
      - HMDB51 اسم الفولدر مكرر مرتين (rawframes/rawframes) — ده حقيقي
      - CCTV كمان مكرر (Videos/Videos) — ده حقيقي
      - Charades: بنقدر نبنيه بتعشيش مكرر أو عادي عشان نختبر الاتنين
      - decoy: ملفات كتير جوّه charades عشان نمسك أي دالة بتمشي فيها بالغلط
    """
    if MOCK.exists():
        shutil.rmtree(MOCK)
    ds = MOCK / 'datasets'

    hmdb = ds / 'jizeyong' / 'hmdb51' / 'rawframes' / 'rawframes'
    for cls in ('clap', 'wave', 'hug', 'shake_hands', 'drink', 'brush_hair'):
        for i in range(3):
            _touch(hmdb / cls / f'{cls}_clip_{i}' / '00001.jpg')

    cctv = ds / 'jonathannield' / 'cctv-action-recognition-dataset' / 'Videos' / 'Videos'
    for cls in ('sit', 'stand', 'walk', 'run', 'lying_down'):
        for i in (1, 2):
            _touch(cctv / cls / f'NTU_fight000{i}_{cls}_{i}.mp4')
    _touch(cctv.parent / 'train_split.txt')

    charades = ds / 'jizeyong' / 'charades'
    _touch(charades / 'Charades_v1_train.csv', CHARADES_CSV)
    rgb = charades / 'Charades_v1_rgb'
    if charades_double_nested:
        rgb = rgb / 'Charades_v1_rgb'
    for vid in ('0BH84', '08Y62', '107YZ', '0L07S', '09F15', '1BVUA'):
        for i in range(1, 4):
            _touch(rgb / vid / f'{vid}-{i:06d}.jpg')

    # الفخ: ملفات كتير جوّه Charades. أي دالة بتدوّر على HMDB51/CCTV/UCF101
    # ولمست الفولدر ده يبقى رجعت لنفس بق الأداء القديم.
    decoy = charades / 'Charades_v1_flow'
    for i in range(decoy_files):
        _touch(decoy / f'bucket_{i % 30}' / f'frame_{i:06d}.jpg')

    ucf = ds / 'matthewjansen' / 'ucf101-action-recognition' / 'train' / 'TableTennisShot'
    for g, c in ((2, 3), (4, 6), (1, 1)):
        _touch(ucf / f'v_TableTennisShot_g{g:02d}_c{c:02d}.avi')

    return ds


class ScanRecorder:
    """بيسجّل كل فولدر اتقرا فعلاً من الملفات — عشان نقيس الأداء بدقة
    بدل ما نقيس بالثواني (اللي بتختلف من جهاز للتاني)."""

    def __init__(self):
        self.dirs = []

    def __enter__(self):
        self._scandir, self._listdir = os.scandir, os.listdir

        def scandir(path='.'):
            self.dirs.append(str(path))
            return self._scandir(path)

        def listdir(path=None):
            self.dirs.append(str(path))
            return self._listdir(path)

        os.scandir, os.listdir = scandir, listdir
        return self

    def __exit__(self, *exc):
        os.scandir, os.listdir = self._scandir, self._listdir

    def touched(self, needle):
        return sum(1 for d in self.dirs if needle in d.replace('\\', '/'))


PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"✓" if cond else "✗"} {name}{(" — " + detail) if detail else ""}')


def main():
    print('🔧 ببني شجرة Kaggle مزيّفة...')
    t0 = time.perf_counter()
    build_mock_tree()
    print(f'   اتبنت في {time.perf_counter() - t0:.1f}s → {MOCK}\n')

    bet.KAGGLE_INPUT = MOCK
    bet.ON_KAGGLE = True

    print('١) _dataset_dir — بيلاقي الداتاسِت من غير ما يمشي في التانيين')
    with ScanRecorder() as rec:
        hmdb_ds = bet._dataset_dir('jizeyong/hmdb51')
    check('بيرجّع مسار hmdb51 الصح', hmdb_ds.name == 'hmdb51', str(hmdb_ds))
    check('مالمسش charades خالص', rec.touched('/charades') == 0,
          f'{rec.touched("/charades")} قراءة')

    print('\n٢) _find_rawframes_root — تعشيش rawframes/rawframes')
    with ScanRecorder() as rec:
        root = bet._find_rawframes_root('clap', 'jizeyong/hmdb51')
    check('لقى فولدر الحركات المتعشّش',
          root.name == 'rawframes' and (root / 'clap').is_dir(), str(root))
    check('مالمسش charades خالص', rec.touched('/charades') == 0,
          f'{rec.touched("/charades")} قراءة')

    print('\n٣) _find_clips_by_suffix — CCTV (بالمفتاح lying_down المصحّح)')
    with ScanRecorder() as rec:
        cctv = bet._find_clips_by_suffix(
            bet.CCTV_TO_LABEL, r'_\d+', 'jonathannield/cctv-action-recognition-dataset')
    check('لقى كل الخمس فئات', sorted(cctv) == sorted(bet.CCTV_TO_LABEL),
          str(sorted(cctv)))
    check('lying_down فيها فيديوهات', len(cctv.get('lying_down', [])) == 2)
    check('مالمسش charades خالص', rec.touched('/charades') == 0,
          f'{rec.touched("/charades")} قراءة')

    print('\n٤) _find_clips_by_suffix — UCF101')
    with ScanRecorder() as rec:
        ucf = bet._find_clips_by_suffix(
            bet.UCF101_TO_LABEL, r'_g\d+_c\d+', 'matthewjansen/ucf101-action-recognition')
    check('لقى 3 فيديوهات بينج بونج', len(ucf['tabletennisshot']) == 3)
    check('مرتّبة أبجدي (ثابتة مش عشوائية)',
          ucf['tabletennisshot'] == sorted(ucf['tabletennisshot']))
    check('مالمسش charades خالص', rec.touched('/charades') == 0)

    print('\n٥) _find_charades_root — بيلاقي الـ CSV من غير ما يمشي الشجرة كلها')
    with ScanRecorder() as rec:
        csv_path, rgb_start = bet._find_charades_root()
    check('لقى Charades_v1_train.csv', csv_path.name == 'Charades_v1_train.csv')
    check('مادخلش جوّه فولدرات الفريمات الضخمة',
          rec.touched('Charades_v1_flow/bucket') == 0,
          f'{rec.touched("Charades_v1_flow/bucket")} قراءة')

    print('\n٦) _charades_candidates — قراءة الـ CSV')
    cands = bet._charades_candidates(csv_path)
    check('c019 (مكالمة) لقى 3 فيديوهات', len(cands['c019']) == 3, str(cands['c019']))
    check('c104 (نور) لقى 2', len(cands['c104']) == 2)
    check('c133 + c146 (صحيان) لقى 1 + 1',
          len(cands['c133']) == 1 and len(cands['c146']) == 1)
    check('مرتّبة ثابتة', cands['c019'] == sorted(cands['c019']))
    check('السطر الفاضي (ZZZZZ) ماكسرش حاجة',
          all(vid != 'ZZZZZ' for v in cands.values() for vid, _, _ in v))

    print('\n٧) _resolve_frames_dir — ⭐ البق اللي دوّر اللوج على صفر فريم')
    sample_ids = [vid for c in cands.values() for vid, _s, _e in c[:3]]
    rgb_dir = bet._resolve_frames_dir(rgb_start, sample_ids)
    check('نزل جوّه التعشيش المكرر Charades_v1_rgb/Charades_v1_rgb',
          rgb_dir.parent.name == 'Charades_v1_rgb', str(rgb_dir))
    check('فولدر الفيديو فيه فريمات فعلاً',
          len(sorted((rgb_dir / '0BH84').glob('*.jpg'))) == 3)

    print('\n٨) _resolve_frames_dir — نفس الكود لو التعشيش مش مكرر')
    build_mock_tree(charades_double_nested=False, decoy_files=200)
    csv2, start2 = bet._find_charades_root()
    cands2 = bet._charades_candidates(csv2)
    ids2 = [vid for c in cands2.values() for vid, _s, _e in c[:3]]
    rgb2 = bet._resolve_frames_dir(start2, ids2)
    check('لقاه من غير ما ينزل مستوى زيادة',
          rgb2.name == 'Charades_v1_rgb' and (rgb2 / '0BH84').is_dir(), str(rgb2))

    print('\n٩) _resolve_frames_dir — بيقع بصوت عالي لو الفريمات مش موجودة')
    shutil.rmtree(rgb2)
    (rgb2 / 'wrong_layout').mkdir(parents=True)
    try:
        bet._resolve_frames_dir(start2, ids2)
        check('رمى FileNotFoundError', False, 'عدّى من غير خطأ!')
    except FileNotFoundError as e:
        check('رمى FileNotFoundError', True)
        check('الرسالة فيها المسار اللي جرّبه', 'Charades_v1_rgb' in str(e))
        check('الرسالة فيها الـ video_id اللي دوّر عليه', '0BH84' in str(e))

    print('\n١٠) _resolve_frames_dir — فولدر الفيديو موجود بس الفريمات امتداد تاني')
    build_mock_tree(charades_double_nested=False, decoy_files=100)
    csv3, start3 = bet._find_charades_root()
    ids3 = [vid for c in bet._charades_candidates(csv3).values()
            for vid, _s, _e in c[:3]]
    for jpg in (start3 / '0BH84').glob('*.jpg'):
        jpg.rename(jpg.with_suffix('.png'))
    try:
        bet._resolve_frames_dir(start3, ids3)
        check('رمى FileNotFoundError بدل صفر فريم بالسكوت', False, 'عدّى!')
    except FileNotFoundError as e:
        check('رمى FileNotFoundError بدل صفر فريم بالسكوت', True)
        check('الرسالة بتقول الملفات الحقيقية اللي جوّه الفولدر',
              '.png' in str(e), str(e).split('اللي جواه فعلاً:')[-1].strip()[:60])

    print('\n١١) _cap_per_label — الأولوية للمحلي')
    manifest = [
        {'label': 'phone_call', 'source': 'CHARADES', 'file': 'a.npy'},
        {'label': 'phone_call', 'source': 'USER', 'file': 'b.npy'},
        {'label': 'phone_call', 'source': 'CHARADES', 'file': 'c.npy'},
        {'label': 'walking', 'source': 'CCTV', 'file': 'd.npy'},
    ]
    capped = bet._cap_per_label(manifest, cap=2)
    phone = [m['file'] for m in capped if m['label'] == 'phone_call']
    check('قصّ لاتنين بالظبط', len(phone) == 2, str(phone))
    check('فيديو المستخدم جه الأول', phone[0] == 'b.npy')
    check('walking فضلت واحدة', len([m for m in capped if m['label'] == 'walking']) == 1)

    print(f'\n{"=" * 60}')
    print(f'✅ نجح: {len(PASS)}   ❌ فشل: {len(FAIL)}')
    if FAIL:
        print('الفاشل: ' + ', '.join(FAIL))
    shutil.rmtree(MOCK, ignore_errors=True)
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
