"""
Few-Shot FastDTW على vidtest3 — 13 حركة، من غير أي داتاسِت

    python run_vidtest3.py

المهمة: الفيديو فيه 13 حركة مختلفة. بناخد **قصاصة قصيرة واحدة** من كل
حركة كـ template، وبنصنّف باقي الفيديو بأقرب template بمسافة FastDTW.

⚠️ المنهجية — تلات نقاط لازم الدكتور يشوفها:

  ١. **القياس الأساسي مافيهوش أي معامل بيتظبط.** أقرب template وخلاص،
     من غير عتبة رفض ولا بوابة. يعني مافيش حاجة نقدر نغشّ بيها.
     الصدفة = 1/13 = 7.7%.

  ٢. **الـ template بيتستثنى من التقييم.** بناخد أول جزء من الحركة
     template، والباقي بس هو اللي بيتحسب. من غير كده كنا هنختبر على
     نفس البيانات اللي أخدنا منها.

  ٣. **stand_up و walking بيتكرروا مرتين** في الفيديو. دول بس اللي
     بنقدر نختبرهم اختبار نظيف ١٠٠٪: الـ template من النسخة الأولى
     والاختبار على النسخة **التانية** اللي مالهاش أي علاقة بيه.
     الرقم ده بيتقال لوحده لأنه الأصدق.

⚠️ الفترة 88s-126.5s مش موصوفة، فبتتستثنى تماماً — مش بتتحسب 'other'.
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from oneshot_core import (build_multiscale_windows, cut_templates,
                          distance_cache, majority_smooth, norm_distance,
                          window_edges)
from ground_truth import GROUND_TRUTH, REPEATED as _REP

GT = GROUND_TRUTH['vidtest3']
REPEATED = _REP['vidtest3']

SCALES = (1.5, 2.0, 3.0)
STRIDE = 5
RADIUS = 1
SMOOTH_K = 3

# طول قصاصة الـ template: نص الحركة بحد أقصى 2.5s وحد أدنى 1.0s
TMPL_MAX, TMPL_MIN, TMPL_FRAC = 2.5, 1.0, 0.5


def load():
    kp = np.load('keypoints/vidtest3_keypoints.npy')
    meta = np.load('keypoints/vidtest3_meta.npy', allow_pickle=True).item()
    return kp, float(meta['effective_fps'])


def template_spans():
    """
    لكل حركة: أول جزء من **أول** ظهور ليها.

    بنرجّع كمان الفترات دي عشان نستثنيها من التقييم — من غيرها بنكون
    بنختبر على نفس البيانات اللي أخدنا منها الـ template.
    """
    first = {}
    for s, e, lab in GT:
        if lab != '?' and lab not in first:
            first[lab] = (s, e)

    clips = []
    for lab, (s, e) in first.items():
        dur = min(TMPL_MAX, max(TMPL_MIN, (e - s) * TMPL_FRAC))
        clips.append((s, min(e, s + dur), lab))
    return clips


def evaluate(labels, edges, clips, step=0.1):
    """
    تقييم على مستوى الزمن. بيستثني:
      • '?'            = مش موصوفة
      • فترات الـ template = اللي أخدنا منها، مينفعش نختبر عليها
    """
    tmpl_spans = [(s, e) for s, e, _ in clips]
    per_class = defaultdict(lambda: {'n': 0, 'hit': 0})
    confusion = Counter()
    n = hit = 0
    strict_n = strict_hit = 0

    for s, e, truth in GT:
        if truth == '?':
            continue
        # هل الفترة دي هي نفسها الـ template؟
        is_first = any(abs(s - ts) < 1e-6 for ts, te, _ in clips)

        for t in np.arange(s, e, step):
            if any(ts <= t < te for ts, te in tmpl_spans):
                continue                       # جوه الـ template، استثنِ
            w = int(np.searchsorted(edges, t, side='right') - 1)
            if not (0 <= w < len(labels)):
                continue

            pred = labels[w]
            ok = (pred == truth)
            n += 1
            hit += ok
            per_class[truth]['n'] += 1
            per_class[truth]['hit'] += ok
            if not ok:
                confusion[(truth, pred)] += 1

            # الاختبار النظيف ١٠٠٪: حركة بتتكرر، والظهور ده مش اللي
            # أخدنا منه الـ template
            if truth in REPEATED and not is_first:
                strict_n += 1
                strict_hit += ok

    return {'n': n, 'hit': hit, 'per_class': per_class,
            'confusion': confusion, 'strict_n': strict_n,
            'strict_hit': strict_hit}


def hubness_correct(D):
    """
    تصحيح الـ hubness — كل template يتقاس بمقياسه هو.

    🐛 المشكلة اللي بيحلها (اتقاست تلات مرات في المشروع ده):
       بعض الـ templates بتبقى "قريبة من كل حاجة" فبتكسب في كل النوافذ.
       آخر تشغيلة: play_pingpong خد 100% على نفسه (64/64) بس كمان كان
       التوقّع الغلط لـ sitting 133 مرة و brush_hair 41 مرة —
       يعني الـ 100% دي **وهمية**، هو بس بيبلع كل حاجة.

       ده اسمه hubness: في الأبعاد العالية بعض النقاط بتبقى "أقرب جار"
       لعدد كبير من النقاط التانية لأسباب هندسية مالهاش علاقة بالتشابه.

    الحل: بدل ما نقارن المسافات الخام، نشوف المسافة دي **قد إيه غير
    عادية بالنسبة للـ template ده**. لكل template بنحسب الوسيط و MAD
    على كل نوافذ الفيديو، وبعدين:

        z = (d - وسيط_الـtemplate) / MAD_الـtemplate

    الـ template اللي قريب من كل حاجة وسيطه واطي، فالقرب منه مابقاش
    ميزة. اللي بيكسب هو اللي المسافة ليه **أقل من المعتاد بالنسبة له**.

    ⚠️ ده مش تسريب: بيتحسب من مسافات الفيديو نفسه من غير أي ground
       truth — نفس فكرة تطبيع الإضاءة في الصور.
    """
    med = np.median(D, axis=0, keepdims=True)                  # لكل (مقاس، template)
    mad = np.median(np.abs(D - med), axis=0, keepdims=True)
    return (D - med) / np.maximum(mad, 1e-6)


def run(mode='vel', shape_norm=True, smooth=SMOOTH_K, hubness=True,
        verbose=True):
    kp, fps = load()
    clips = template_spans()

    templates = cut_templates(kp, fps, clips, source='vidtest3', mode=mode,
                              shape_norm=shape_norm, scales=SCALES)
    tl = [t['label'] for t in templates]

    centers = np.arange(0, len(kp), STRIDE)
    edges = window_edges(centers, fps, len(kp))
    feats, E = build_multiscale_windows(kp, fps, centers, SCALES,
                                        mode=mode, shape_norm=shape_norm)

    key = f'v3_{mode}_{int(shape_norm)}_s{STRIDE}_r{RADIUS}'
    cache = Path('_scratch/dtw_cache')
    cache.mkdir(parents=True, exist_ok=True)
    cf = cache / f'{key}.npy'

    if cf.exists():
        D = np.load(cf)
    else:
        if verbose:
            print(f'  حساب {len(centers) * len(SCALES) * len(templates):,} '
                  f'مسافة DTW...')
        D = distance_cache(feats, templates, SCALES, radius=RADIUS)
        np.save(cf, D)

    if hubness:
        D = hubness_correct(D)

    # أقرب template — من غير أي عتبة ولا بوابة، مافيش حاجة تتظبط
    flat = D.reshape(len(centers), -1)
    best = flat.argmin(axis=1)
    raw = [tl[i % len(templates)] for i in best]
    labels = majority_smooth(raw, smooth) if smooth > 1 else raw

    return evaluate(labels, edges, clips), labels, edges, clips, templates


def main():
    print('=' * 72)
    print('  Few-Shot FastDTW — vidtest3، 13 حركة، من غير أي داتاسِت')
    print('=' * 72)

    clips = template_spans()
    print(f'\n📚 الـ templates ({len(clips)} حركة، قصاصة واحدة لكل واحدة):')
    for s, e, lab in clips:
        print(f'    {lab:<15} [{s:5.1f} - {e:5.1f}]  ({e - s:.1f}s)')

    r, labels, edges, _, templates = run()

    acc = r['hit'] / max(1, r['n'])
    chance = 1 / len(clips)

    print(f'\n{"=" * 72}')
    print('  النتيجة — أقرب template، من غير أي عتبة أو معامل مظبوط')
    print('=' * 72)
    print(f'  الدقة        {acc * 100:5.1f}%   على {r["n"]} عيّنة زمنية')
    print(f'  الصدفة       {chance * 100:5.1f}%   (1 من {len(clips)})')
    print(f'  ↑ أحسن من الصدفة بـ {acc / chance:.1f}×')

    print(f'\n  --- الاختبار النظيف ١٠٠٪ (نسخة تانية، مالهاش علاقة بالـ template) ---')
    if r['strict_n']:
        sa = r['strict_hit'] / r['strict_n']
        print(f'  stand_up + walking (الظهور التاني): '
              f'{sa * 100:.1f}%  ({r["strict_hit"]}/{r["strict_n"]})')
    else:
        print('  مفيش عيّنات')

    print(f'\n  --- الدقة لكل حركة ---')
    pc = r['per_class']
    for lab in sorted(pc, key=lambda k: -pc[k]['hit'] / max(1, pc[k]['n'])):
        v = pc[lab]
        a = v['hit'] / max(1, v['n'])
        bar = '█' * int(a * 24)
        star = ' ✅' if lab in REPEATED else ''
        print(f'    {lab:<15} {a * 100:5.1f}%  {bar:<24} '
              f'({v["hit"]}/{v["n"]}){star}')

    print(f'\n  --- أكتر 8 التباسات ---')
    for (t, p), c in r['confusion'].most_common(8):
        print(f'    {t:<15} → {p:<15} {c:4}×')

    # ---- تجارب الحذف: إيه اللي بيفرق فعلاً ----
    print(f'\n{"=" * 72}\n  تجارب الحذف\n{"=" * 72}')
    variants = [
        ('كامل',                    dict()),
        ('من غير تصحيح hubness',    dict(hubness=False)),
        ('مواضع بدل الفروق',        dict(mode='pos')),
        ('مواضع + من غير hubness',  dict(mode='pos', hubness=False)),
        ('من غير تطبيع الشكل',      dict(shape_norm=False)),
        ('من غير تنعيم',            dict(smooth=1)),
    ]
    for name, kw in variants:
        rv, *_ = run(verbose=False, **kw)
        a = rv['hit'] / max(1, rv['n'])
        s = (rv['strict_hit'] / rv['strict_n'] * 100) if rv['strict_n'] else 0
        print(f'  {name:<26} دقة={a * 100:5.1f}%   نظيف={s:5.1f}%')


if __name__ == '__main__':
    main()
