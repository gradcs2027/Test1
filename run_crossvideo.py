"""
الاختبار عبر-الفيديوهات — أنضف اختبار في المشروع كله

    python run_crossvideo.py

الفكرة
──────
كل اختبار عملناه قبل كده كان الـ template والاختبار من **نفس الفيديو**،
فالرقم كان متضخّم. هنا: **الـ template من فيديو، الاختبار على فيديو تاني**.
كاميرا تانية، لبس تاني، يوم تاني، وضع جسم تاني. مافيش أي تسريب ممكن.

ده اتاح لأول مرة بعد ما وحّدنا الـ ground truth في `ground_truth.py` —
قبل كده مكانش واضح إن فيه حركات مشتركة أصلاً.

الحركات المشتركة (من `ground_truth.shared_labels()`):
    stand_up   vidtest1(2×)  vidtest2(1×)  vidtest3(2×)
    wave       vidtest1(2×)  vidtest2(1×)  vidtest3(1×)
    sit_down   vidtest1(2×)  vidtest2(1×)

⚠️ المنهجية — أربع نقاط لازم الدكتور يشوفها
────────────────────────────────────────────
  ١. **مافيش أي معامل بيتظبط.** أقرب template وخلاص — لا عتبة رفض،
     لا بوابة طاقة، لا حاجة. يعني مافيش أي مساحة نغشّ فيها.

  ٢. **بنقيّم على قصاصات الحركة بس** (closed-set)، مش على الفيديو كله.
     السبب: لو حطّينا فترات السكون، هنحتاج عتبة رفض — والعتبة معامل
     بيتظبط، وده اللي وقعنا فيه قبل كده (شوف HANDOFF قسم 6.7:
     "علّي الـ accuracy فخ"). الصدفة هنا = 1/3 = 33.3%.

  ٣. **بنقارن بخط أساس الأغلبية مش بالصدفة بس.** لو 42% من القصاصات
     `stand_up`، يبقى "قول stand_up على طول" بيدّي 42% — وده الرقم
     الحقيقي اللي لازم نكسره، مش الـ 33.3%.

  ٤. **مقارنة جوّه-الفيديو مقابل عبر-الفيديوهات.** نفس الكود بالظبط،
     الفرق الوحيد مصدر الـ template. الفرق بين الرقمين بيقيس **قد إيه
     كانت أرقامنا القديمة متضخّمة**.

بروتوكولين
──────────
  • **قصاصة**: كل ظهور للحركة = قصاصة واحدة بحدودها الحقيقية -> أقرب
    template. ده البروتوكول الكلاسيكي لتصنيف القصاصات المقسّمة مسبقاً،
    وأعدل اختبار ممكن للـ DTW (مافيش مشكلة حدود نوافذ).
  • **نافذة**: نوافذ منزلقة جوّه فترات الحركة. عيّنات أكتر، وبيختبر
    التحمّل لما الحدود مش مظبوطة.
"""

import sys
from collections import Counter, defaultdict

import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from ground_truth import GROUND_TRUTH, VIDEOS, shared_labels, spans
from oneshot_core import (NUM_FRAMES, balance_templates, cut_templates,
                          normalize_window, norm_distance, resample_linear,
                          to_features)

SCALES = (1.0, 1.5, 2.0, 3.0)   # مقاسات تقسيم الـ templates الطويلة
STRIDE = 5                      # خطوة النافذة بالفريمات
RADIUS = 1                      # نصف قطر FastDTW
MAX_PER_LABEL = 8               # سقف templates لكل حركة — شوف balance_templates

SHARED = tuple(shared_labels())  # ('sit_down', 'stand_up', 'wave')


def load(video):
    kp = np.load(f'keypoints/{video}_keypoints.npy')
    meta = np.load(f'keypoints/{video}_meta.npy', allow_pickle=True).item()
    return kp, float(meta['effective_fps'])


# ==============================================================================
# بنك الـ templates
# ==============================================================================

def build_templates(sources, mode='vel', shape_norm=True, exclude=()):
    """
    بيبني بنك templates من الفيديوهات المصدر.

    exclude: فترات (فيديو, بداية, نهاية) تتستثنى — بتُستخدم في اختبار
             جوّه-الفيديو عشان مانختبرش على نفس الظهور اللي أخدنا منه.
    """
    out = []
    for v in sources:
        kp, fps = load(v)
        clips = [(s, e, lab) for lab in SHARED for s, e in spans(v, lab)
                 if (v, s, e) not in exclude]
        out += cut_templates(kp, fps, clips, source=v, mode=mode,
                             shape_norm=shape_norm, scales=SCALES)
    return balance_templates(out, MAX_PER_LABEL)


def featurize(kp, fps, t0, t1, mode, shape_norm):
    """قصاصة زمنية -> متجه ملامح جاهز للمقارنة."""
    lo, hi = int(round(t0 * fps)), int(round(t1 * fps))
    clip = kp[max(0, lo):min(len(kp), hi)]
    if len(clip) < 4:
        return None
    seq = resample_linear(normalize_window(clip), NUM_FRAMES)
    return to_features(seq, mode=mode, shape_norm=shape_norm)


def nearest(feat, templates):
    """أقرب template — من غير أي عتبة. بيرجّع (اللابل, المسافة)."""
    d = [norm_distance(feat, t['feat'], radius=RADIUS) for t in templates]
    i = int(np.argmin(d))
    return templates[i]['label'], d[i]


# ==============================================================================
# البروتوكول الأول: قصاصة كاملة
# ==============================================================================

def segment_protocol(mode='vel', shape_norm=True, within=False):
    """
    كل ظهور للحركة = قصاصة واحدة بحدودها الحقيقية -> أقرب template.

    within=False : الـ templates من الفيديوهات **التانية** (عبر-الفيديوهات)
    within=True  : من **نفس** الفيديو، بس من ظهور تاني (جوّه-الفيديو)
                   ده الضابط العلمي — بيقيس قد إيه أرقامنا القديمة
                   كانت متضخّمة.
    """
    rows = []
    for v in VIDEOS:
        kp, fps = load(v)
        for lab in SHARED:
            for (s, e) in spans(v, lab):
                if within:
                    # نفس الفيديو، من غير الظهور ده بالذات
                    srcs, excl = [v], {(v, s, e)}
                    tmpl = build_templates(srcs, mode, shape_norm, exclude=excl)
                else:
                    srcs = [o for o in VIDEOS if o != v]
                    tmpl = build_templates(srcs, mode, shape_norm)

                if not tmpl:
                    continue                    # مافيش مثال تاني للحركة دي
                if not any(t['label'] == lab for t in tmpl):
                    continue                    # الحركة دي مش موجودة في المصدر

                feat = featurize(kp, fps, s, e, mode, shape_norm)
                if feat is None:
                    continue
                pred, dist = nearest(feat, tmpl)
                rows.append({'video': v, 'span': (s, e), 'truth': lab,
                             'pred': pred, 'dist': dist, 'n_tmpl': len(tmpl)})
    return rows


# ==============================================================================
# البروتوكول التاني: نوافذ منزلقة
# ==============================================================================

FRACTIONS = (0.6, 0.8, 1.0)     # مقاس النافذة كنسبة من طول الحركة


def window_protocol(mode='vel', shape_norm=True):
    """
    نوافذ منزلقة جوّه فترة الحركة، بمقاسات مختلفة.

    الهدف: نشوف هل النتيجة متينة لما حدود الحركة مش مظبوطة — مش نزوّد
    حجم العيّنة.

    ⚠️⚠️ **العيّنات دي مترابطة، مش مستقلة.** إحنا عندنا 12 ظهور حقيقي
       للحركة وخلاص. النوافذ دي بتتقص من نفس الـ 12، فـ "300 عيّنة"
       هنا **ماتتعاملش معاملة 300 قياس مستقل** في أي حساب دلالة
       إحصائية. الرقم اللي يتقال في أي اختبار دلالة هو 12.

       بكتب ده صريح عشان مايتقريش غلط: زيادة عدد النوافذ **مش** بتزوّد
       المعلومة، بتزوّد التفاصيل عن نفس المعلومة.

    مقاس النافذة نسبة من طول الحركة مش رقم ثابت — عشان مانرجعش لمشكلة
    "النافذة 3s والحركة 2s فبتشرشح على حركتين" (HANDOFF قسم 6.9).
    """
    rows = []
    for v in VIDEOS:
        kp, fps = load(v)
        srcs = [o for o in VIDEOS if o != v]
        tmpl = build_templates(srcs, mode, shape_norm)
        labels_avail = {t['label'] for t in tmpl}

        for lab in SHARED:
            if lab not in labels_avail:
                continue
            for (s, e) in spans(v, lab):
                dur = e - s
                for frac in FRACTIONS:
                    win = dur * frac
                    half = win / 2.0
                    lo_c, hi_c = s + half, e - half
                    step = STRIDE / fps
                    centers = (np.arange(lo_c, hi_c + 1e-9, step)
                               if hi_c > lo_c else np.array([(s + e) / 2]))
                    for c in centers:
                        feat = featurize(kp, fps, c - half, c + half,
                                         mode, shape_norm)
                        if feat is None:
                            continue
                        pred, dist = nearest(feat, tmpl)
                        rows.append({'video': v, 'truth': lab, 'pred': pred,
                                     'dist': dist, 'span': (s, e)})
    return rows


def binom_tail(k, n, p):
    """P(X >= k) لتوزيع ذي الحدين — دلالة النتيجة مقابل خط أساس."""
    from math import comb
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


# ==============================================================================
# التقييم
# ==============================================================================

def score(rows):
    """بيرجّع الدقة + خط أساس الأغلبية + الصدفة."""
    if not rows:
        return None
    n = len(rows)
    hit = sum(r['pred'] == r['truth'] for r in rows)
    truths = Counter(r['truth'] for r in rows)
    majority_lab, majority_n = truths.most_common(1)[0]
    return {
        'n': n, 'hit': hit, 'acc': hit / n,
        'majority': majority_n / n, 'majority_lab': majority_lab,
        'chance': 1 / len(truths),
        'per_class': {lab: (sum(r['pred'] == r['truth'] for r in rows
                                if r['truth'] == lab), c)
                      for lab, c in truths.items()},
        'confusion': Counter((r['truth'], r['pred']) for r in rows
                             if r['pred'] != r['truth']),
    }


def print_score(title, s, indent='  '):
    if s is None:
        print(f'{indent}{title}: مافيش عيّنات')
        return
    verdict = '✅' if s['acc'] > s['majority'] else '❌'
    print(f'{indent}{title}')
    print(f'{indent}  الدقة            {s["acc"] * 100:5.1f}%  '
          f'({s["hit"]}/{s["n"]})')
    print(f'{indent}  خط أساس الأغلبية {s["majority"] * 100:5.1f}%  '
          f'("قول {s["majority_lab"]} على طول")  {verdict}')
    print(f'{indent}  الصدفة           {s["chance"] * 100:5.1f}%')


def main():
    print('=' * 72)
    print('  الاختبار عبر-الفيديوهات — template من فيديو، اختبار على فيديو تاني')
    print('=' * 72)
    print(f'\n  الحركات المشتركة: {", ".join(SHARED)}')

    # ---------- البروتوكول الأول: قصاصات ----------
    print(f'\n{"=" * 72}')
    print('  البروتوكول ١: قصاصة كاملة بحدودها الحقيقية')
    print('=' * 72)

    cross = segment_protocol()
    s_cross = score(cross)

    print(f'\n  --- كل قصاصة على حدة ---')
    print(f'  {"فيديو":<10} {"الفترة":>12}  {"الصح":<10} {"التوقّع":<10} مسافة')
    print('  ' + '-' * 60)
    for r in cross:
        mark = '✅' if r['pred'] == r['truth'] else '❌'
        s, e = r['span']
        print(f'  {r["video"]:<10} {f"{s:.1f}-{e:.1f}":>12}  '
              f'{r["truth"]:<10} {r["pred"]:<10} {r["dist"]:.3f} {mark}')

    print()
    print_score('⭐ عبر-الفيديوهات (مافيش أي تسريب)', s_cross)

    within = segment_protocol(within=True)
    s_within = score(within)
    print()
    print_score('   جوّه الفيديو الواحد (الضابط العلمي)', s_within)

    if s_cross and s_within:
        gap = (s_within['acc'] - s_cross['acc']) * 100
        print(f'\n  📊 الفرق = {gap:+.1f} نقطة.')
        if gap > 5:
            print('     يعني الأرقام القديمة (جوّه الفيديو) كانت متضخّمة '
                  'بالمقدار ده.')
        elif gap < -5:
            print('     عبر-الفيديوهات أحسن — الـ templates من فيديو تاني '
                  'أنفع من ظهور تاني لنفس الحركة.')
        else:
            print('     الاتنين قريبين — الحركة بتنتقل بين الفيديوهات كويس.')

    # ---------- البروتوكول التاني: نوافذ ----------
    print(f'\n{"=" * 72}')
    print('  البروتوكول ٢: نوافذ منزلقة (عيّنات أكتر)')
    print('=' * 72)

    win = window_protocol()
    s_win = score(win)
    print()
    print_score('⭐ عبر-الفيديوهات', s_win)

    if s_win:
        print(f'\n  --- الدقة لكل حركة ---')
        for lab, (h, c) in sorted(s_win['per_class'].items(),
                                  key=lambda kv: -kv[1][0] / max(1, kv[1][1])):
            a = h / max(1, c)
            print(f'    {lab:<12} {a * 100:5.1f}%  {"█" * int(a * 24):<24} '
                  f'({h}/{c})')

        if s_win['confusion']:
            print(f'\n  --- الالتباسات ---')
            for (t, p), c in s_win['confusion'].most_common(6):
                print(f'    {t:<12} → {p:<12} {c:4}×')

        print(f'\n  --- لكل فيديو اختبار ---')
        for v in VIDEOS:
            sv = score([r for r in win if r['video'] == v])
            if sv:
                print(f'    {v}: {sv["acc"] * 100:5.1f}% ({sv["hit"]}/'
                      f'{sv["n"]})   أغلبية={sv["majority"] * 100:.1f}%')

    # ---------- تجارب الحذف ----------
    print(f'\n{"=" * 72}\n  تجارب الحذف — إيه اللي بيفرق فعلاً\n{"=" * 72}')
    print('  ⚠️ الإعداد الافتراضي (فروق + تطبيع شكل) هو اللي كان مختار')
    print('     **قبل** التجربة دي. أي نسخة تانية تطلع أحسن = اختيار')
    print('     بعد رؤية النتيجة، ولازم تتقال كده مش كأنها نتيجة نظيفة.')
    print()
    print(f'  {"النسخة":<26} {"قصاصة":>9} {"دلالة":>8} {"نافذة":>9}')
    print('  ' + '-' * 56)

    p0 = s_cross['majority']
    for name, kw in [('فروق + تطبيع شكل ⭐', dict()),
                     ('مواضع بدل الفروق', dict(mode='pos')),
                     ('من غير تطبيع الشكل', dict(shape_norm=False)),
                     ('مواضع من غير تطبيع', dict(mode='pos',
                                                 shape_norm=False))]:
        a = score(segment_protocol(**kw))
        b = score(window_protocol(**kw))
        pv = binom_tail(a['hit'], a['n'], p0)
        print(f'  {name:<26} {a["acc"] * 100:8.1f}% {f"p={pv:.2f}":>8} '
              f'{b["acc"] * 100:8.1f}%')

    print(f'\n  خط أساس الأغلبية: {p0 * 100:.1f}% (قصاصة) / '
          f'{s_win["majority"] * 100:.1f}% (نافذة)')
    print(f'  عمود الدلالة = احتمال إنك توصل للرقم ده أو أحسن **بالصدفة**')
    print(f'  لو الموديل مالوش أي قدرة وبيقول {s_cross["majority_lab"]} '
          f'على طول. n={s_cross["n"]} بس.')

    # ---------- الخلاصة الصريحة ----------
    print(f'\n{"=" * 72}\n  الخلاصة\n{"=" * 72}')
    best = max([('فروق', score(segment_protocol())),
                ('مواضع', score(segment_protocol(mode='pos')))],
               key=lambda kv: kv[1]['acc'])
    pv = binom_tail(best[1]['hit'], best[1]['n'], p0)
    print(f'  أحسن نسخة ({best[0]}): {best[1]["acc"] * 100:.1f}% '
          f'({best[1]["hit"]}/{best[1]["n"]}) مقابل {p0 * 100:.1f}% '
          f'خط أساس الأغلبية')
    print(f'  الاحتمال إن ده يحصل بالصدفة = {pv:.2f}')
    if pv > 0.05:
        print(f'  ⚠️ **مش دال إحصائياً.** بـ {best[1]["n"]} عيّنة بس، الفرق')
        print('     ده مايتفرقش عن الحظ. مينفعش نقول إن الطريقة "نجحت".')
    else:
        print('  ✅ دال إحصائياً عند 0.05')
    print(f'\n  🔬 القيد الحقيقي: عندنا **{s_cross["n"]} ظهور حقيقي للحركة**')
    print('     في التلات فيديوهات كلهم. ده سقف الإحصاء، ومفيش بروتوكول')
    print('     بيزوّده. أي رقم من هنا لازم يتقال ومعاه العدد ده.')


if __name__ == '__main__':
    main()
