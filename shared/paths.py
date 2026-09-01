"""
مصدر واحد لمسارات الملفات — بيشتغل محلياً وعلى Kaggle من غير تعديل

    from paths import load_keypoints, KP_DIR, VIDEO_DIR
    kp, fps = load_keypoints('vidtest4')

ليه الملف ده موجود؟
──────────────────
مسار `keypoints/` كان مكتوب بالإيد في **6 ملفات**، وكل واحد فيهم كاتب
دالة `load()` بتاعته بنفس السطرين بالظبط. ده معناه إن أي نقلة للكود
(Kaggle مثلاً) محتاجة 6 تعديلات، وأي واحد منهم يتنسي = بق.

نفس المشكلة اللي `ground_truth.py` اتعمل عشانها: التكرار بيخلّي
الغلطات تعيش.

ترتيب البحث عن الـ keypoints
────────────────────────────
    ١. متغيّر البيئة KEYPOINTS_DIR          (للتجارب — بيكسر كل اللي تحت)
    ٢. الفولدر اللي جنب الكود                keypoints/
    ٣. الداتاسِت المرفوع على Kaggle          /kaggle/input/*/keypoints/
    ٤. /kaggle/working/keypoints             (لو اتولدت في نفس الجلسة)

⚠️ رقم ٢ قبل رقم ٣ **بنيّة**: الـ keypoints متسجّلة في git (828 KB)، يعني
   `git clone` بيجيبها معاه. عشان كده FastDTW بيشتغل على Kaggle **من غير
   أي داتاسِت خالص** — الداتاسِت محتاجينه بس لو عايزين نعيد استخراج الـ
   pose أو نرسم فيديو، لأن دول اللي محتاجين ملفات الـ mp4.

⚠️ على Kaggle الـ input **للقراءة بس**. `pose_extract.py` بيكتب، فبيستخدم
   `KP_OUT` اللي بيروح `/kaggle/working` هناك — مش `KP_DIR`.
"""

import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# ==============================================================================
# اكتشاف البيئة
# ==============================================================================

ON_KAGGLE = Path('/kaggle/input').is_dir()


def _find_kaggle_dataset():
    """
    بيدوّر على فولدر الداتاسِت جوّه /kaggle/input.

    مش بنكتب 'testvid' صريح لأن Kaggle بيسمّي الفولدر باسم الداتاسِت،
    ولو الاسم اتغيّر الكود بيقع. بندوّر على اللي فيه فيديوهاتنا.
    بيرجّع None لو مالقاش — الاختيار بيتساب للترتيب تحت.
    """
    if not ON_KAGGLE:
        return None
    for d in sorted(Path('/kaggle/input').iterdir()):
        if (d / 'vidtest1.mp4').exists() or (d / 'keypoints').is_dir():
            return d
    return None


def _first_dir(*candidates):
    """أول مرشّح موجود وفيه ملفات فعلاً. آخر واحد بيترجع حتى لو مش موجود."""
    for c in candidates:
        if c is not None and Path(c).is_dir() and any(Path(c).iterdir()):
            return Path(c)
    return Path(candidates[-1])


_DATASET = _find_kaggle_dataset()

# ⚠️ الترتيب ده مقصود: الفولدر اللي **جنب الكود** له الأولوية على الداتاسِت.
#
# ليه؟ عشان لو الكود اتعمله `git clone` على Kaggle، الـ keypoints بتيجي مع
# الريبو (828 KB، متسجّلة في git). لو بدأنا بالداتاسِت هنتجاهل الملفات اللي
# قدامنا ونروح ندوّر على داتاسِت ممكن ما يكونش متوصّل أصلاً.
#
# ده اللي بيخلّي FastDTW يشتغل على Kaggle **من غير أي داتاسِت خالص**.
KP_DIR = _first_dir(
    HERE / 'keypoints',                                 # جنب الكود (git clone)
    _DATASET / 'keypoints' if _DATASET else None,       # الداتاسِت المرفوع
    Path('/kaggle/working/keypoints') if ON_KAGGLE else HERE / 'keypoints',
)

# الفيديوهات محتاجينها للاستخراج والرسم بس، مش للتصنيف
VIDEO_DIR = _first_dir(
    HERE.parent / 'testvid_upload',
    _DATASET,
    HERE.parent / 'testvid_upload',
)

if ON_KAGGLE:
    KP_OUT = Path('/kaggle/working/keypoints')     # الكتابة لازم تروح working
    OUT_DIR = Path('/kaggle/working/results')
    CACHE_DIR = Path('/kaggle/working/_scratch/dtw_cache')
else:
    KP_OUT = HERE / 'keypoints'                    # محلياً القراءة والكتابة واحد
    OUT_DIR = HERE / 'results'
    CACHE_DIR = HERE / '_scratch' / 'dtw_cache'

# متغيّر بيئة بيكسر أي حاجة فوق — للتجارب
KP_DIR = Path(os.environ.get('KEYPOINTS_DIR', KP_DIR))
VIDEO_DIR = Path(os.environ.get('VIDEO_DIR', VIDEO_DIR))


# ==============================================================================
# التحميل
# ==============================================================================

def load_meta(video):
    """meta الفيديو. بيقع بصوت عالي لو مش موجود بدل ما يرجّع قيم افتراضية."""
    p = KP_DIR / f'{video}_meta.npy'
    if not p.exists():
        raise FileNotFoundError(
            f'مافيش meta لـ {video} في {KP_DIR}\n'
            f'  محلياً:  python pose_extract.py {video}\n'
            f'  Kaggle: اتأكد إن فولدر keypoints/ مرفوع مع الداتاسِت')
    return np.load(p, allow_pickle=True).item()


def load_keypoints(video):
    """
    بيرجّع (keypoints, effective_fps).

    ⚠️ effective_fps مش fps الفيديو — هو fps/FRAME_SKIP لأن الاستخراج
       بياخد فريم من كل اتنين. أي حساب زمني لازم يستخدم ده.
    """
    p = KP_DIR / f'{video}_keypoints.npy'
    if not p.exists():
        raise FileNotFoundError(
            f'مافيش keypoints لـ {video} في {KP_DIR}\n'
            f'  محلياً:  python pose_extract.py {video}\n'
            f'  Kaggle: اتأكد إن فولدر keypoints/ مرفوع مع الداتاسِت')
    return np.load(p), float(load_meta(video)['effective_fps'])


def video_path(video):
    """مسار ملف الـ mp4."""
    return VIDEO_DIR / f'{video}.mp4'


def available():
    """أسماء الفيديوهات اللي لها keypoints متحمّلة."""
    if not KP_DIR.is_dir():
        return []
    return sorted(p.name.replace('_keypoints.npy', '')
                  for p in KP_DIR.glob('*_keypoints.npy'))


if __name__ == '__main__':
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    ok = lambda p: '✅' if Path(p).is_dir() else '❌ مش موجود'

    print(f'البيئة    : {"Kaggle" if ON_KAGGLE else "محلي"}')
    if ON_KAGGLE:
        print(f'الداتاسِت  : {_DATASET or "مالقيتش — شغّالين من الريبو بس"}')
    print(f'KP_DIR    : {KP_DIR}   {ok(KP_DIR)}')
    print(f'VIDEO_DIR : {VIDEO_DIR}   {ok(VIDEO_DIR)}')
    print(f'KP_OUT    : {KP_OUT}')
    print(f'OUT_DIR   : {OUT_DIR}')
    print(f'CACHE_DIR : {CACHE_DIR}')

    # مهم يبان إذا الـ keypoints جاية من الريبو ولا من الداتاسِت — لو حد
    # فكّر إنه بيقرا من الداتاسِت وهو بيقرا من الريبو هيتلخبط في تفسير النتايج
    src = ('الريبو (جنب الكود)' if KP_DIR == HERE / 'keypoints'
           else 'الداتاسِت' if _DATASET and KP_DIR == _DATASET / 'keypoints'
           else 'مكان تاني')
    print(f'\nالـ keypoints جاية من: {src}')

    vids = available()
    print(f'الفيديوهات المتاحة ({len(vids)}):')
    for v in vids:
        kp, fps = load_keypoints(v)
        m = load_meta(v)
        print(f'  {v:<10} {len(kp):>5} فريم @ {fps:5.1f} fps فعلي   '
              f'اكتشاف {m["detect_rate"] * 100:5.1f}%')
    if not vids:
        print('  ⚠️ مافيش! شوف KP_DIR فوق — الملفات مش في المكان ده.')
