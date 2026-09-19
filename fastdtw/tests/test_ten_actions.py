"""
تشغيل experiment_ten_actions.py كامل محلياً — من غير Kaggle ولا HMDB51.

إزاي؟ السكريبت بيخزّن الـ keypoints في CACHE_DIR وبيتخطّى الاستخراج
لأي قصاصة متخزّنة. فبنملا التخزين بنفسنا بكيبوينتس **حقيقية** من
الفيديوهات المسجّلة في git، والسكريبت بيمشي في كل مساره التاني (بناء
التمثيل، المقارنة، التقرير، الحفظ) من غير ما يدوّر على /kaggle/input.

البنك هنا: الستة فيديوهات المصوّرة في البيت، كل واحد متقسّم 7 قِطع
متتالية — قطعتين تدريب و5 اختبار، بالظبط زي التجربة الحقيقية.

⚠️ الدقة اللي هتطلع من هنا **مش** نتيجة التجربة. القطع كلها من نفس
   الفيديو، يعني نفس الشخص ونفس الكاميرا — أسهل حالة ممكنة. الأرقام
   الحقيقية بتطلع من Kaggle بقصاصات HMDB51 المختلفة.

   لكن الرقم ده **مفيد كفحص صحّة**: لو طلع قريب من الصدفة يبقى في حاجة
   مكسورة في الأنبوب نفسه (التمثيل، التطبيع، أو المقارنة) — لأن تصنيف
   قطعة من فيديو بقطعة تانية من نفس الفيديو المفروض يبقى سهل جداً.
   اتقاس 2026-09-19 بنفس التمثيل ده على نص الفيديو: 90.5%.

التشغيل:  python fastdtw/tests/test_ten_actions.py
"""
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # fastdtw/tests/
ROOT = HERE.parent.parent                       # فولدر المشروع
sys.path.insert(0, str(HERE.parent))            # fastdtw/

import build_external_templates as bet  # noqa: E402
import experiment_ten_actions as eta  # noqa: E402

SCRATCH = ROOT / '_scratch' / 'ten_actions'
CACHE = SCRATCH / 'cache'
OUT = SCRATCH / 'out'

ARABIC = {
    'clapping': 'تصفيق', 'wave': 'تلويح', 'sitting': 'قعود',
    'stand_up': 'وقوف', 'phone_call': 'مكالمة', 'spray_perfume': 'رش عطر',
}


def fill_cache():
    """بيقسّم كل فيديو محلي لـ N_TRAIN+N_TEST قطعة ويحطّها في التخزين
    بنفس أسامي الملفات اللي السكريبت بيدوّر عليها."""
    for d in (CACHE, OUT):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    manifest = np.load(bet.LOCAL_EXT_DIR / 'manifest_local.npy', allow_pickle=True)
    user = {m['label']: bet.LOCAL_EXT_DIR / m['file']
            for m in manifest if m['source'] == 'USER'}

    n_want = eta.N_TRAIN + eta.N_TEST
    actions, sources = {}, {}
    for label in sorted(user):
        kp = np.load(user[label])
        step = len(kp) // n_want
        if step < 4:
            continue
        for i in range(n_want):
            np.save(CACHE / f'{label}_{i:02d}.npy', kp[i * step:(i + 1) * step])
            # كل قطعة بتاخد اسم فيديو أصلي مختلف بقصد: القطع فعلاً من نفس
            # الفيديو، والسكريبت بيرفض ده ويرمي خطأ. إحنا عايزين نعدّيه
            # هنا عشان نختبر باقي الأنبوب. حارس التسريب نفسه بيتختبر
            # لوحده تحت في check_leak_guard().
            sources[f'{label}_{i:02d}'] = f'{label}_source{i}'
        actions[label] = ARABIC[label]
    np.save(CACHE / 'sources.npy', sources, allow_pickle=True)
    return actions


def check_leak_guard(actions):
    """الحارس المفروض يرمي خطأ لو نفس الفيديو الأصلي في التدريب والاختبار.

    حارس ساكت مش حارس. بنخلّي كل القطع من "فيديو أصلي" واحد وبنشوف
    السكريبت بيقع فعلاً ولا بيكمّل وهو مبسوط.
    """
    good = np.load(CACHE / 'sources.npy', allow_pickle=True).item()
    np.save(CACHE / 'sources.npy', {k: 'نفس_الفيديو' for k in good},
            allow_pickle=True)
    try:
        eta.main()
    except RuntimeError as e:
        return 'تسريب' in str(e)
    except Exception:
        return False
    finally:
        np.save(CACHE / 'sources.npy', good, allow_pickle=True)
    return False


def check_amp_modes(kp):
    """تطبيع السعة: كل وضع بيعمل اللي مكتوب عليه بالظبط؟

    أهم فحص هنا هو **عدم التأثر بالسعة**: لو ضربنا الحركة في 3 وزوّدنا
    إزاحة ثابتة، `ناقص الوسط ÷ السعة` لازم يطلّع نفس الأرقام. ده بالظبط
    اللي بيمنع قالب حركته صغيرة إنه يبقى أقرب حاجة لكل حاجة.
    """
    problems = []
    joints = eta.JOINT_SETS['الجسم كله (17)']
    raw = eta.pose_features(kp, joints, 'خام')

    centered = eta.pose_features(kp, joints, 'ناقص الوسط')
    if not np.allclose(centered.mean(axis=0), 0, atol=1e-9):
        problems.append('"ناقص الوسط" سايب وسط مش صفر')

    shape = eta.pose_features(kp, joints, 'ناقص الوسط ÷ السعة')
    amp = np.sqrt((shape ** 2).sum(axis=1).mean())
    if not np.isclose(amp, 1.0, atol=1e-6):
        problems.append(f'"÷ السعة" المفروض سعته 1 وطلعت {amp:.4f}')

    louder = eta.AMP_MODES['ناقص الوسط ÷ السعة'](raw * 3.0 + 7.0)
    if not np.allclose(louder, shape, atol=1e-6):
        problems.append('نفس الحركة بسعة مختلفة طلّعت أرقام مختلفة — '
                        'التطبيع مش شايل السعة')

    quiet = eta.AMP_MODES['ناقص الوسط ÷ السعة'](np.zeros_like(raw))
    if not np.isfinite(quiet).all():
        problems.append('قصاصة ساكنة تماماً طلّعت NaN أو لانهاية')

    return problems


def check_mirror(kp):
    """القلب يمين/شمال: قلبتين لازم يرجّعوا الأصل، وقلبة واحدة لازم
    تغيّر فعلاً (مش تبقى بلا فايدة)."""
    problems = []
    once = eta.mirror_kp(kp)
    twice = eta.mirror_kp(once)

    if not np.array_equal(twice, np.asarray(kp)):
        problems.append('قلبتين مارجّعوش الأصل')

    if np.array_equal(once, np.asarray(kp)):
        problems.append('القلب مغيّرش حاجة خالص')

    if not np.isclose(eta.detect_rate(once), eta.detect_rate(kp)):
        problems.append('القلب غيّر نسبة الكشف — يعني ضيّع مفاصل')

    joints = eta.JOINT_SETS['الجسم كله (17)']
    if np.allclose(eta.pose_features(once, joints),
                   eta.pose_features(kp, joints)):
        problems.append('التمثيل المقلوب مطابق للأصلي — القالب الزيادة '
                        'مش بيضيف أي تغطية')

    return problems


def check_source_parser():
    """تفكيك أسامي HMDB51 للوصول لعنوان الفيديو الأصلي."""
    cases = [
        ('April_09_brush_hair_u_nm_np1_ba_goo_0', 'brush_hair', 'April_09'),
        ('April_09_brush_hair_u_nm_np1_ba_goo_1', 'brush_hair', 'April_09'),
        ('50_FIRST_DATES_walk_f_cm_np1_ba_med_10', 'walk', '50_FIRST_DATES'),
        ('Faith_Rewarded_clap_u_nm_np1_fr_med_21', 'clap', 'Faith_Rewarded'),
        ('اسم_من_غير_علامة', 'walk', 'اسم_من_غير_علامة'),
    ]
    return [(name, eta._source_video(name, act), want)
            for name, act, want in cases
            if eta._source_video(name, act) != want]


def check_diverse_picker():
    """اختيار القصاصات على شجرة HMDB51 مزيّفة بأسامي حقيقية الشكل.

    ثلاث فيديوهات أصلية × ثلاث قصاصات من كل واحد = 9 فولدر. الترتيب
    الأبجدي بيحطّ قصاصات نفس الفيديو ورا بعض، فـ`_pick_clips` العادية
    كانت هتاخد تلاتة كلهم من ApplauseApplause.
    """
    class_dir = SCRATCH / 'fake_hmdb' / 'clap'
    class_dir.mkdir(parents=True, exist_ok=True)
    for src in ('ApplauseApplause', 'Faith_Rewarded', 'winKey'):
        for i in range(3):
            (class_dir / f'{src}_clap_u_nm_np1_fr_med_{i}').mkdir(exist_ok=True)
    (class_dir / 'ملف_مش_فولدر.jpg').write_text('x', encoding='utf-8')

    problems = []
    picked = eta.pick_diverse_clips(class_dir, 'clap', 3)
    srcs = [eta._source_video(p.name, 'clap') for p in picked]
    if len(set(srcs)) != 3:
        problems.append(f'طلبت 3 من فيديوهات مختلفة وجات من {len(set(srcs))}: {srcs}')

    # أكتر من المتاح: بيرجّع اللي موجود بس، والسكريبت هو اللي بيرمي الخطأ
    if len(eta.pick_diverse_clips(class_dir, 'clap', 7)) != 3:
        problems.append('المفروض يرجّع 3 بس لما أطلب 7 ومفيش غير 3 فيديوهات')

    return problems


def main():
    actions = fill_cache()
    print(f'📦 تخزين مؤقت: {len(actions)} حركة × {eta.N_TRAIN + eta.N_TEST} قطعة '
          f'من كيبوينتس حقيقية')
    print(f'   {sorted(actions)}\n')

    eta.TEN_ACTIONS = actions
    eta.CACHE_DIR = CACHE
    eta.OUT_DIR = OUT

    eta.main()

    print(f'\n{"=" * 74}')
    fails = []

    res_path = OUT / 'ten_actions_results.npy'
    ok = res_path.exists()
    print(f'  {"✓" if ok else "✗"} اتحفظ ten_actions_results.npy')
    fails += [] if ok else ['مفيش ملف نتيجة']
    if not ok:
        return 1

    res = np.load(res_path, allow_pickle=True).item()

    ok = sum(v[1] for v in res['per_action'].values()) == len(actions) * eta.N_TEST
    print(f'  {"✓" if ok else "✗"} كل حركة اتختبرت بـ {eta.N_TEST} قطع')
    fails += [] if ok else ['عدد الاختبارات غلط']

    ok = np.isfinite(res['accuracy']) and 0.0 <= res['accuracy'] <= 1.0
    print(f'  {"✓" if ok else "✗"} الدقة رقم سليم (مش NaN): {res["accuracy"]:.3f}')
    fails += [] if ok else ['دقة مش سليمة']

    # فحص الأنبوب: القطع من نفس الفيديو، فالمفروض تبقى سهلة جداً
    chance = 1.0 / len(actions)
    ok = res['accuracy'] > 0.5
    print(f'  {"✓" if ok else "✗"} الدقة فوق 50% على قطع من نفس الفيديو '
          f'(الصدفة {chance * 100:.0f}%) — فحص صحّة الأنبوب')
    fails += [] if ok else ['الأنبوب مكسور: تصنيف نفس الفيديو فشل']

    # التمثيل نفسه: الشكل والأرقام
    kp = np.load(CACHE / f'{sorted(actions)[0]}_00.npy')
    bad = []
    for jname, joints in eta.JOINT_SETS.items():
        for aname in eta.AMP_MODES:
            f = eta.pose_features(kp, joints, aname)
            if f.shape != (eta.N_FRAMES, len(joints) * 2) or not np.isfinite(f).all():
                bad.append(f'{jname} | {aname} → {f.shape}')
    n_variants = len(eta.JOINT_SETS) * len(eta.AMP_MODES)
    ok = not bad
    print(f'  {"✓" if ok else "✗"} pose_features بيطلّع الشكل الصح من غير NaN '
          f'لكل الـ{n_variants} تمثيل')
    for p in bad:
        print(f'      {p}')
    fails += [] if ok else ['شكل التمثيل غلط']

    bad = check_amp_modes(kp)
    ok = not bad
    print(f'  {"✓" if ok else "✗"} تطبيع السعة بيعمل اللي المفروض يعمله')
    for p in bad:
        print(f'      {p}')
    fails += [] if ok else ['تطبيع السعة غلط']

    bad = check_mirror(kp)
    ok = not bad
    print(f'  {"✓" if ok else "✗"} القلب يمين/شمال سليم (قلبتين = الأصل)')
    for p in bad:
        print(f'      {p}')
    fails += [] if ok else ['القلب غلط']

    # النتيجة المحفوظة لازم تقول اتحسبت بأنهي احتمال، وإلا الرقم مش
    # قابل لإعادة الإنتاج
    n_expected = n_variants * 2          # × معكوس / مش معكوس
    ok = len(res.get('by_variant', {})) == n_expected and bool(res.get('variant'))
    print(f'  {"✓" if ok else "✗"} النتيجة بتسجّل دقة كل الـ{n_expected} احتمال '
          f'والفايز فيهم ({res.get("variant")})')
    fails += [] if ok else ['الاحتمالات مش متسجّلة']

    # مقارنة القصاصة بنفسها لازم تدّي صفر
    feat = eta.pose_features(kp, res['joints'], res['amp'])
    d = eta.norm_distance(feat, feat, radius=eta.RADIUS)
    ok = d < 1e-9
    print(f'  {"✓" if ok else "✗"} مسافة القصاصة عن نفسها = {d:.2e}')
    fails += [] if ok else ['المسافة مش صفر مع نفسها']

    bad = check_source_parser()
    ok = not bad
    print(f'  {"✓" if ok else "✗"} _source_video بيفكّك أسامي HMDB51 صح')
    if bad:
        for name, got, want in bad:
            print(f'      {name} → "{got}" والمفروض "{want}"')
    fails += [] if ok else ['تفكيك الاسم غلط']

    bad = check_diverse_picker()
    ok = not bad
    print(f'  {"✓" if ok else "✗"} pick_diverse_clips بياخد قصاصة واحدة '
          f'من كل فيديو أصلي')
    for p in bad:
        print(f'      {p}')
    fails += [] if ok else ['اختيار القصاصات مش متنوّع']

    ok = check_leak_guard(actions)
    print(f'  {"✓" if ok else "✗"} حارس التسريب بيرمي خطأ لما التدريب '
          f'والاختبار من نفس الفيديو')
    fails += [] if ok else ['حارس التسريب ساكت']

    print(f'{"=" * 74}')
    if fails:
        print(f'❌ فشل: {", ".join(fails)}')
    else:
        print('✅ experiment_ten_actions.py بيشتغل كامل من غير وقوع')
        print('⚠️ الدقة دي مش نتيجة التجربة — القطع كلها من نفس الفيديو')

    shutil.rmtree(SCRATCH, ignore_errors=True)
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
