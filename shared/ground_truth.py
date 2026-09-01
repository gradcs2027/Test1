"""
مصدر الحقيقة الواحد للـ ground truth بتاع الفيديوهات التلاتة

    python ground_truth.py          # جدول كامل + فحص سلامة + التقاطعات

ليه الملف ده موجود؟
──────────────────
قبل 2026-08-31 كان الـ ground truth **متكرر في تسع أماكن**:
  • vidtest1: منسوخ حرفياً في 5 ملفات، ومفيش ملف مرجعي أصلاً
  • vidtest2: ملف + نسخة تانية مختلفة جوّه run_oneshot.py
  • vidtest3: ملفين (واحد غلط تماماً) + نسخة تالتة جوّه run_oneshot.py

والنتيجة كانت غلطتين حقيقيتين اتكشفوا:
  ١. vidtest3 كله كان متوصّف غلط (4 حركات بدل 13) — HANDOFF قسم 6.8
  ٢. الـ wave بتاع vidtest2 كان متعلّم 'other*' (سكون) — يعني الكشف
     الصحيح ليه كان بيتحسب إنذار كاذب

**أي كود جديد يستورد من هنا، مش من الملفات فرادى.**

الاصطلاح
────────
    'other*'  = سكون **متأكّد منه** بالمشاهدة — بيتحسب في التقييم
    '?'       = مش موصوفة / مش صالحة — **بتتستثنى من الحساب تماماً**

الفرق بين الاتنين مش شكلي: حطّ فترة مش موصوفة على إنها 'other*' =
اختراع بيانات، وده اللي عمل الغلطة رقم ٢ فوق.
"""

import sys
from collections import defaultdict

from vidtest1_ground_truth import (REPEATED as REPEATED1,
                                   ground_truth_vidtest1, video_info as INFO1)
from vidtest2_ground_truth import (REPEATED as REPEATED2,
                                   ground_truth_vidtest2, video_info as INFO2)
from vidtest3_ground_truth_v2 import (REPEATED as REPEATED3,
                                      ground_truth_vidtest3, video_info as INFO3)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

EXCLUDED = '?'          # تتستثنى من الحساب
STILL = 'other*'        # سكون متأكّد منه

GROUND_TRUTH = {
    'vidtest1': ground_truth_vidtest1,
    'vidtest2': ground_truth_vidtest2,
    'vidtest3': ground_truth_vidtest3,
}

VIDEO_INFO = {'vidtest1': INFO1, 'vidtest2': INFO2, 'vidtest3': INFO3}
REPEATED = {'vidtest1': REPEATED1, 'vidtest2': REPEATED2, 'vidtest3': REPEATED3}

VIDEOS = list(GROUND_TRUTH)


# ==============================================================================
# استعلامات
# ==============================================================================

def actions(video):
    """أسماء الحركات الحقيقية في الفيديو (من غير 'other*' ولا '?')."""
    return sorted({l for _, _, l in GROUND_TRUTH[video]
                   if l not in (EXCLUDED, STILL)})


def spans(video, label):
    """كل الفترات اللي فيها الحركة دي."""
    return [(s, e) for s, e, l in GROUND_TRUTH[video] if l == label]


def label_at(video, t):
    """اللابل عند الثانية t، أو None لو بره الفيديو."""
    for s, e, l in GROUND_TRUTH[video]:
        if s <= t < e:
            return l
    return None


def duration_of(video, label):
    return sum(e - s for s, e in spans(video, label))


def project(video, keep, rest=STILL):
    """
    يرجّع نسخة من الـ ground truth بمفردات مضيّقة.

    مثال: تجربة sit/stand بكلاسين عايزة تشوف الفيديو بعيون كلاسين بس.
    بدل ما تكتب جدول تاني بإيدها (وده اللي عمل كل مشاكلنا)، بتاخد
    الجدول المرجعي وتسقطه:

        project('vidtest3', keep=('sit_down', 'stand_up'))

    أي حركة مش في `keep` بتبقى `rest`. الـ '?' بتفضل '?' على طول —
    فترة مش موصوفة تفضل مش موصوفة مهما ضيّقنا المفردات.

    ⚠️ الافتراضي `rest='other*'` صح **بس** لو الـ ground truth بتاع
       الفيديو كامل. لو فيه حركات مش موصوفة، استخدم `rest='?'`.
    """
    keep = set(keep)
    return [(s, e, l if l in keep else (EXCLUDED if l == EXCLUDED else rest))
            for s, e, l in GROUND_TRUTH[video]]


def shared_labels(min_videos=2):
    """
    الحركات اللي بتظهر في أكتر من فيديو — **دي مفتاح الاختبار النظيف**.

    مشكلتنا الكبيرة إن الـ template والاختبار بييجوا من نفس الفيديو،
    فالرقم بيبقى متضخّم. الحركة اللي في فيديوهين بتخلّينا ناخد الـ
    template من واحد ونختبر على التاني: كاميرا تانية، لبس تاني، يوم تاني.

    بيرجّع {اللابل: [(الفيديو, عدد المرات, الثواني), ...]}
    """
    out = defaultdict(list)
    for v in VIDEOS:
        for lab in actions(v):
            out[lab].append((v, len(spans(v, lab)), duration_of(v, lab)))
    return {k: v for k, v in sorted(out.items()) if len(v) >= min_videos}


# ==============================================================================
# فحص السلامة — عشان غلطة زي بتاعة vidtest2 ماتعديش تاني
# ==============================================================================

def check(video, tol=0.05):
    """
    بيدوّر على مشاكل بنيوية في الجدول. بيرجّع ليستة رسايل (فاضية = تمام).

    بيفحص: تداخل فترات · فجوات · بداية مش من صفر · نهاية مش عند مدة
    الفيديو · فترات بطول صفر أو سالب.
    """
    gt = GROUND_TRUTH[video]
    dur = VIDEO_INFO[video]['duration']
    problems = []

    for s, e, l in gt:
        if e <= s:
            problems.append(f'فترة بطول صفر أو سالب: [{s}-{e}] {l}')

    ordered = sorted(gt, key=lambda x: x[0])
    if ordered != list(gt):
        problems.append('الفترات مش مترتبة زمنياً')

    for (s1, e1, l1), (s2, e2, l2) in zip(ordered, ordered[1:]):
        if s2 < e1 - tol:
            problems.append(f'تداخل: [{s1}-{e1}] {l1} مع [{s2}-{e2}] {l2}')
        elif s2 > e1 + tol:
            problems.append(f'فجوة {s2 - e1:.1f}s بين {e1} و {s2} '
                            f'(بين {l1} و {l2}) — مش متغطية بأي لابل')

    if ordered and ordered[0][0] > tol:
        problems.append(f'مش بيبدأ من 0 — أول لابل عند {ordered[0][0]}')
    if ordered and abs(ordered[-1][1] - dur) > 0.5:
        problems.append(f'بينتهي عند {ordered[-1][1]} بس مدة الفيديو {dur}')

    return problems


# ==============================================================================
# عرض
# ==============================================================================

def print_video(video):
    gt, info = GROUND_TRUTH[video], VIDEO_INFO[video]
    real = sum(e - s for s, e, l in gt if l not in (EXCLUDED, STILL))
    still = sum(e - s for s, e, l in gt if l == STILL)
    unknown = sum(e - s for s, e, l in gt if l == EXCLUDED)

    mark = '✅ متأكّد من صاحب المشروع' if info.get('verified_by_owner') \
        else '⚠️ لسه مش متأكّد من صاحب المشروع'
    print(f'\n{"=" * 66}')
    print(f'  {video}  —  {info["duration"]:.1f}s @ {info["effective_fps"]:.0f} '
          f'eff-fps  —  {mark}')
    print('=' * 66)

    print(f'  {"الحركة":<16} {"مرات":>5} {"ثواني":>7}   الفترات')
    print('  ' + '-' * 62)
    for lab in sorted(actions(video), key=lambda k: -duration_of(video, k)):
        sp = spans(video, lab)
        rep = ' 🔁' if lab in REPEATED[video] else ''
        print(f'  {lab:<16} {len(sp):>5} {duration_of(video, lab):>7.1f}   '
              f'{", ".join(f"{s:.0f}-{e:.0f}" for s, e in sp)}{rep}')
    print('  ' + '-' * 62)
    print(f'  حركة: {real:.1f}s  |  سكون متأكّد: {still:.1f}s  |  '
          f'مش موصوف: {unknown:.1f}s')

    problems = check(video)
    if problems:
        print(f'  ⚠️ فحص السلامة — {len(problems)} ملاحظة:')
        for p in problems:
            print(f'      • {p}')
    else:
        print('  ✅ فحص السلامة: مافيش تداخل ولا فجوات، التغطية كاملة')


def main():
    print('=' * 66)
    print('  الـ Ground Truth — مصدر الحقيقة الواحد')
    print('=' * 66)

    for v in VIDEOS:
        print_video(v)

    print(f'\n{"=" * 66}')
    print('  🔑 الحركات المشتركة — دي اللي ينفع عليها اختبار عبر-الفيديوهات')
    print('=' * 66)
    print('  template من فيديو، اختبار على فيديو تاني = مافيش أي تسريب.')
    print('  ده أنضف من الاختبار جوّه الفيديو الواحد.\n')

    sh = shared_labels()
    if not sh:
        print('  مافيش حركة مشتركة ❌')
    for lab, where in sh.items():
        loc = '   '.join(f'{v}({n}×, {d:.0f}s)' for v, n, d in where)
        print(f'  {lab:<14} في {len(where)} فيديوهات:  {loc}')

    print(f'\n  الحركات اللي في فيديو واحد بس (مافيش اختبار نظيف ليها):')
    solo = {lab for v in VIDEOS for lab in actions(v)} - set(sh)
    print(f'    {", ".join(sorted(solo))}')

    print(f'\n{"=" * 66}')
    print('  🔁 الحركات اللي بتتكرر جوّه نفس الفيديو')
    print('=' * 66)
    for v in VIDEOS:
        r = REPEATED[v] or ['— مافيش —']
        print(f'  {v}: {", ".join(r)}')


if __name__ == '__main__':
    main()
