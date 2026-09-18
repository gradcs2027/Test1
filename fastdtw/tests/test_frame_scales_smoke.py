"""
تشغيل experiment_frame_scales.py محلياً على بنك قصاصات حقيقي (جزئي).

ليه؟ ده السكريبت اللي بعد build_external_templates.py على طول. مش عايزين
نكتشف إنه بيقع بعد ما نستنى البناء كله على Kaggle تاني.

البنك هنا = الـ10 قصاصات المحلية الحقيقية (فيديوهات المستخدم + NTU60)
اللي متسجّلة في git — كيبوينتس حقيقية مش مولّدة. بتغطّي 8 حركات من الـ18،
وده كفاية عشان نتأكد إن الحساب كله ماشي من غير ما يقع.

⚠️ الأرقام اللي هتطلع من هنا **مش** نتيجة التجربة الحقيقية — البنك ناقص
10 حركات. الأرقام الحقيقية بتطلع من Kaggle بالبنك الكامل.

التشغيل:  python fastdtw/tests/test_frame_scales_smoke.py
"""
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # fastdtw/tests/
ROOT = HERE.parent.parent                       # فولدر المشروع
sys.path.insert(0, str(HERE.parent))            # fastdtw/

import build_external_templates as bet  # noqa: E402
import experiment_frame_scales as efs  # noqa: E402

EXT = ROOT / '_scratch' / 'smoke_external'
OUT = ROOT / '_scratch' / 'smoke_out'


def build_bank():
    """بنسخ القصاصات المحلية الحقيقية ونعمل manifest بنفس شكل اللي
    build_external_templates.py بيطلّعه."""
    for d in (EXT, OUT):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    local = np.load(bet.LOCAL_EXT_DIR / 'manifest_local.npy', allow_pickle=True)
    manifest = []
    for m in local:
        shutil.copy(bet.LOCAL_EXT_DIR / m['file'], EXT / m['file'])
        manifest.append(dict(m))
    np.save(EXT / 'manifest.npy', manifest, allow_pickle=True)
    return manifest


def main():
    manifest = build_bank()
    labels = sorted({m['label'] for m in manifest})
    print(f'📚 بنك مؤقت: {len(manifest)} قصاصة حقيقية، {len(labels)} حركة')
    print(f'   {labels}\n')

    efs.EXT_DIR = EXT
    efs.OUT_DIR = OUT
    # عيّنة من الإعدادات بس — الهدف نمسك أي وقوع، مش نطلّع أرقام نهائية
    efs.FRAME_CONFIGS = (10, 50, 100)

    t0 = time.perf_counter()
    efs.main()
    elapsed = time.perf_counter() - t0

    print(f'\n{"=" * 70}')
    fails = []

    for name in ('frame_scale_summary.npy', 'frame_scale_summary_zscore.npy'):
        ok = (OUT / name).exists()
        print(f'  {"✓" if ok else "✗"} اتحفظ {name}')
        fails += [] if ok else [name]

    s = np.load(OUT / 'frame_scale_summary.npy', allow_pickle=True)
    sz = np.load(OUT / 'frame_scale_summary_zscore.npy', allow_pickle=True)

    ok = len(s) == len(efs.FRAME_CONFIGS) == len(sz)
    print(f'  {"✓" if ok else "✗"} في صف لكل إعداد ({len(s)} و {len(sz)})')
    fails += [] if ok else ['عدد الصفوف']

    ok = all(r['n'] > 0 for r in s)
    print(f'  {"✓" if ok else "✗"} كل إعداد لقى نوافذ اختبار فعلاً: '
          f'{[int(r["n"]) for r in s]}')
    fails += [] if ok else ['نوافذ فاضية']

    ok = all(0.0 <= r['accuracy'] <= 100.0 and np.isfinite(r['avg_confidence'])
             for r in s)
    print(f'  {"✓" if ok else "✗"} الدقة والثقة أرقام سليمة (مش NaN)')
    fails += [] if ok else ['أرقام مش سليمة']

    ok = all(np.isfinite(r['avg_confidence']) for r in sz)
    print(f'  {"✓" if ok else "✗"} نسخة التطبيع كمان أرقامها سليمة')
    fails += [] if ok else ['NaN في التطبيع']

    print(f'\n  ⏱️ {elapsed:.0f} ثانية لـ {len(efs.FRAME_CONFIGS)} إعدادات '
          f'× نسختين (خام + مطبّع)')
    print(f'{"=" * 70}')
    if fails:
        print(f'❌ فشل: {", ".join(fails)}')
    else:
        print('✅ experiment_frame_scales.py بيشتغل من غير وقوع، '
              'والمخرجات بالشكل المتوقّع')
        print('⚠️ الأرقام دي مش النتيجة الحقيقية — البنك هنا 8 حركات بس من 18')

    shutil.rmtree(EXT, ignore_errors=True)
    shutil.rmtree(OUT, ignore_errors=True)
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
