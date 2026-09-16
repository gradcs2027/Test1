"""
TIER 2 — تحسين بنك الـ templates (FastDTW بس، مافيش أي شبكة عصبية)

    python fastdtw/evaluate_crossvideo_tier2.py

ليه TIER 2 أصلاً؟
──────────────────
TIER 1 (تظبيط RADIUS + عتبة ثقة) فشل فشل تام: كل قيم RADIUS من ١ لـ ٥
طلّعت نفس الرقم بالظبط. السبب اتكشف في debug_confidence.py — المسافات
بين الـ templates متقاربة جداً (1.21 لـ 1.42)، يعني ترتيب الـ argmin
مابيتغيّرش مهما وسّعنا شريط البحث.

فالمشكلة **مش** في حساب المسافة. المشكلة في إن فيه templates "بلّاعة"
(hubs): template واحد بيطلع أقرب واحد لكل حاجة. من آخر تشغيلة:

    sitting        → stand_up      38×
    spray_perfume  → brush_hair    25×
    wave           → brush_hair    22×
    phone_call     → brush_hair    22×
    clapping       → brush_hair    20×

brush_hair لوحده بلع ٨٩ نافذة غلط. ده مش إن brush_hair شبه كل الحركات —
ده إن الـ templates بتاعته واقعة في نُص فضاء الملامح، فمسافتها لأي حاجة
صغيرة. الظاهرة دي اسمها **hubness** وهي معروفة في تصنيف السلاسل الزمنية.

TIER 2 بيهاجم ده بتلات خطوات، كل واحدة بتتقاس لوحدها:

  ١. **معايرة كل template (z-normalisation)**
     لكل template بنحسب متوسط وانحراف مسافته لقصاصات **مرجعية من
     فيديوهات المصدر**. بعدها المسافة اللي بنقارن بيها مش الخام، دي:

         z = (d - متوسط_الـtemplate) / انحراف_الـtemplate

     يعني السؤال بقى "هل القصاصة دي قريبة من الـ template ده **بالنسبة
     لعادته**؟" مش "هل هي قريبة منه بشكل مطلق؟". الـ template البلّاع
     متوسطه صغير أصلاً، فالمعايرة بتشيل ميزته الوهمية.

  ٢. **تصويت k أقرب جيران بدل أقرب واحد**
     1-NN بياخد قراره من template واحد — لو ده شاذ، خلاص ضاعت. بناخد
     أقرب ٣ ونخليهم يصوّتوا بوزن حسب الترتيب (١، ١/٢، ١/٣).

  ٣. **تنضيف بنك الـ templates**
     على قصاصات المصدر بنعدّ لكل template: كام مرة كان أقرب واحد لقصاصة
     **من نفس حركته** (نافع) وكام مرة لقصاصة **من حركة تانية** (ضار).
     الـ template اللي ضرره أكتر من نفعه ومانفعش ولا مرة → يتشال.

⚠️⚠️ نقطة المنهجية الأهم — مافيش تسريب
───────────────────────────────────────
كل حاجة "بنتعلّمها" هنا (المتوسط، الانحراف، عدّاد النفع والضرر، قرار
الشطب) بتتحسب من **فيديوهات المصدر بس**. فيديو الاختبار مابيتلمسش خالص
لا في المعايرة ولا في التنضيف.

وكمان: لما بنعاير template جاي من فيديو s، بنستثني القصاصات المرجعية
اللي من فيديو s نفسه — عشان ماياخدش متوسط متفائل من قصاصات هو شايفها
أصلاً.

⚠️ وبرضه: الأرقام اللي هتطلع لازم تتقري ومعاها n=29. ده كل اللي عندنا.
"""

import os
import sys
import time
from collections import Counter

import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import _bootstrap  # noqa: F401

from ground_truth import VIDEOS, shared_labels, spans
from classifier import (NUM_FRAMES, balance_templates, cut_templates,
                        normalize_window, norm_distance, resample_linear,
                        to_features)
from paths import load_keypoints

# نفس معاملات الأساس بالظبط — عشان المقارنة تبقى عادلة
SCALES = (1.0, 1.5, 2.0, 3.0)
STRIDE = 5
RADIUS = 1
MAX_PER_LABEL = 8
FRACTIONS = (0.6, 0.8, 1.0)

K_NEIGHBORS = 3          # عدد الجيران في التصويت
EPS = 1e-9

SHARED = tuple(shared_labels())

HERE = os.path.dirname(os.path.abspath(__file__))

_KP_CACHE = {}


def load(video):
    if video not in _KP_CACHE:
        _KP_CACHE[video] = load_keypoints(video)
    return _KP_CACHE[video]


def featurize(kp, fps, t0, t1, mode, shape_norm):
    """قصاصة زمنية -> متجه ملامح."""
    lo, hi = int(round(t0 * fps)), int(round(t1 * fps))
    clip = kp[max(0, lo):min(len(kp), hi)]
    if len(clip) < 4:
        return None
    seq = resample_linear(normalize_window(clip), NUM_FRAMES)
    return to_features(seq, mode=mode, shape_norm=shape_norm)


def build_templates(sources, mode, shape_norm):
    out = []
    for v in sources:
        kp, fps = load(v)
        clips = [(s, e, lab) for lab in SHARED for s, e in spans(v, lab)]
        out += cut_templates(kp, fps, clips, source=v, mode=mode,
                             shape_norm=shape_norm, scales=SCALES)
    return balance_templates(out, MAX_PER_LABEL)


def source_reference_clips(sources, mode, shape_norm):
    """
    القصاصات المرجعية اللي بنعاير عليها — من فيديوهات المصدر بس.

    دي "المسطرة" اللي بنقيس بيها عادة كل template. مش بيانات اختبار،
    وفيديو الاختبار مش فيها.
    """
    refs = []
    for v in sources:
        kp, fps = load(v)
        for lab in SHARED:
            for (s, e) in spans(v, lab):
                f = featurize(kp, fps, s, e, mode, shape_norm)
                if f is not None:
                    refs.append({'video': v, 'label': lab, 'feat': f})
    return refs


def distance_matrix(items, templates):
    """مصفوفة المسافات الخام: صف لكل عيّنة، عمود لكل template."""
    D = np.empty((len(items), len(templates)), dtype=float)
    for i, it in enumerate(items):
        for j, t in enumerate(templates):
            D[i, j] = norm_distance(it['feat'], t['feat'], radius=RADIUS)
    return D


# ==============================================================================
# الخطوة ١+٣: المعايرة وتنضيف البنك — كل ده من المصدر بس
# ==============================================================================

def calibrate(templates, refs, Dref):
    """
    بيرجّع (mu, sigma, keep) لكل template — كلها متحسبة من المصدر بس.

    mu/sigma : عادة الـ template (متوسط وانحراف مسافته للمرجع)
    keep     : هل نسيبه في البنك ولا نشطبه

    القصاصات المرجعية اللي من نفس فيديو الـ template بتتستثنى، عشان
    مايتعايرش على بيانات هو جاي منها.
    """
    n_t = len(templates)
    mu = np.zeros(n_t)
    sigma = np.ones(n_t)
    good = np.zeros(n_t, dtype=int)   # كان الأقرب لقصاصة من نفس حركته
    bad = np.zeros(n_t, dtype=int)    # كان الأقرب لقصاصة من حركة تانية

    ref_videos = np.array([r['video'] for r in refs])

    # --- عادة كل template ---
    for j, t in enumerate(templates):
        own = ref_videos == t['source']
        col = Dref[~own, j]
        if len(col) >= 2:
            mu[j] = col.mean()
            sigma[j] = col.std() + EPS
        elif len(col) == 1:
            mu[j] = col[0]
            sigma[j] = 1.0

    # --- نفع وضرر: مين بيكسب المرجع، وبعد المعايرة ---
    Z = (Dref - mu) / sigma
    for i, r in enumerate(refs):
        # ماينفعش template من نفس فيديو القصاصة يتحاسب — ده تسريب داخلي
        mask = np.array([t['source'] != r['video'] for t in templates])
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        j = idx[int(np.argmin(Z[i, idx]))]
        if templates[j]['label'] == r['label']:
            good[j] += 1
        else:
            bad[j] += 1

    # نشطب اللي ضرره أكتر من نفعه ومانفعش ولا مرة — دول الـ hubs الصريحة
    keep = ~((good == 0) & (bad >= 2))
    if keep.sum() < 2 or len({templates[j]['label']
                              for j in np.where(keep)[0]}) < 2:
        keep = np.ones(n_t, dtype=bool)   # ماينفعش نفضّي البنك

    return mu, sigma, good, bad, keep


# ==============================================================================
# الخطوة ٢: القرار — أربع طرق نقارنها
# ==============================================================================

def predict(Drow, templates, mu, sigma, keep, use_z, use_knn, use_prune):
    """
    بيرجّع اللابل المتوقّع لعيّنة واحدة.

    use_z     : نعاير المسافة بعادة كل template
    use_knn   : نصوّت بأقرب K بدل أقرب واحد
    use_prune : نتجاهل الـ templates المشطوبة
    """
    idx = np.where(keep)[0] if use_prune else np.arange(len(templates))
    if len(idx) == 0:
        idx = np.arange(len(templates))

    d = (Drow[idx] - mu[idx]) / sigma[idx] if use_z else Drow[idx]

    if not use_knn:
        return templates[idx[int(np.argmin(d))]]['label']

    k = min(K_NEIGHBORS, len(idx))
    order = np.argsort(d)[:k]
    votes = Counter()
    for rank, o in enumerate(order):
        votes[templates[idx[o]]['label']] += 1.0 / (rank + 1)
    best = max(votes.values())
    tied = [lab for lab, val in votes.items() if val == best]
    if len(tied) == 1:
        return tied[0]
    # تعادل -> اللي أقرب واحد فيه أقرب
    return min(tied, key=lambda lab: min(
        d[o] for o in order if templates[idx[o]]['label'] == lab))


METHODS = [
    # (الاسم, use_z, use_knn, use_prune)
    ('الأساس: أقرب template (1-NN)',      False, False, False),
    ('+ معايرة كل template (z)',           True,  False, False),
    ('+ تصويت أقرب ٣ جيران',               True,  True,  False),
    ('+ تنضيف البنك (TIER 2 كامل) ⭐',      True,  True,  True),
]


# ==============================================================================
# البروتوكولات
# ==============================================================================

def test_items(video, protocol, mode, shape_norm, labels_avail):
    """عيّنات الاختبار من فيديو واحد، حسب البروتوكول."""
    kp, fps = load(video)
    items = []
    for lab in SHARED:
        if lab not in labels_avail:
            continue
        for (s, e) in spans(video, lab):
            if protocol == 'segment':
                f = featurize(kp, fps, s, e, mode, shape_norm)
                if f is not None:
                    items.append({'video': video, 'truth': lab,
                                  'span': (s, e), 'feat': f})
            else:
                dur = e - s
                for frac in FRACTIONS:
                    half = dur * frac / 2.0
                    lo_c, hi_c = s + half, e - half
                    step = STRIDE / fps
                    centers = (np.arange(lo_c, hi_c + 1e-9, step)
                               if hi_c > lo_c else np.array([(s + e) / 2]))
                    for c in centers:
                        f = featurize(kp, fps, c - half, c + half,
                                      mode, shape_norm)
                        if f is not None:
                            items.append({'video': video, 'truth': lab,
                                          'span': (s, e), 'feat': f})
    return items


def run_all(protocol, mode='vel', shape_norm=True, verbose=False):
    """
    بيشغّل كل الطرق الأربعة على نفس العيّنات بالظبط.

    مهم: مصفوفة المسافات بتتحسب **مرة واحدة** لكل فيديو، والطرق الأربعة
    مجرد قراءات مختلفة لنفس المصفوفة. يعني الفرق بينهم هو الطريقة وبس،
    مافيش أي عشوائية بينهم.
    """
    rows = {name: [] for name, *_ in METHODS}
    pruned_total = kept_total = 0

    for v in VIDEOS:
        sources = [o for o in VIDEOS if o != v]
        templates = build_templates(sources, mode, shape_norm)
        if len(templates) < 2:
            continue
        labels_avail = {t['label'] for t in templates}

        refs = source_reference_clips(sources, mode, shape_norm)
        if len(refs) < 3:
            continue
        Dref = distance_matrix(refs, templates)
        mu, sigma, good, bad, keep = calibrate(templates, refs, Dref)
        pruned_total += int((~keep).sum())
        kept_total += int(keep.sum())

        if verbose:
            dropped = [f'{templates[j]["label"]}@{templates[j]["source"]}'
                       for j in np.where(~keep)[0]]
            print(f'    {v}: {len(templates)} template، '
                  f'اتشطب {(~keep).sum()}'
                  + (f' ({", ".join(dropped[:5])})' if dropped else ''))

        items = test_items(v, protocol, mode, shape_norm, labels_avail)
        if not items:
            continue
        Dtest = distance_matrix(items, templates)

        for i, it in enumerate(items):
            for name, uz, uk, up in METHODS:
                pred = predict(Dtest[i], templates, mu, sigma, keep,
                               uz, uk, up)
                rows[name].append({'video': v, 'truth': it['truth'],
                                   'span': it['span'], 'pred': pred})

    return rows, pruned_total, kept_total


# ==============================================================================
# التقييم
# ==============================================================================

def binom_tail(k, n, p):
    """P(X >= k) — احتمال توصل للرقم ده أو أحسن بالصدفة."""
    from math import comb
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def score(rows):
    if not rows:
        return None
    n = len(rows)
    hit = sum(r['pred'] == r['truth'] for r in rows)
    truths = Counter(r['truth'] for r in rows)
    lab, cnt = truths.most_common(1)[0]
    return {
        'n': n, 'hit': hit, 'acc': hit / n,
        'majority': cnt / n, 'majority_lab': lab,
        'chance': 1 / len(truths),
        'per_class': {l: (sum(r['pred'] == r['truth'] for r in rows
                              if r['truth'] == l), c)
                      for l, c in truths.items()},
        'confusion': Counter((r['truth'], r['pred']) for r in rows
                             if r['pred'] != r['truth']),
    }


def table(title, rows_by_method, note=''):
    print(f'\n  {title}')
    if note:
        print(f'  {note}')
    print(f'  {"الطريقة":<34} {"الدقة":>16} {"مقابل الأساس":>14} {"دلالة":>9}')
    print('  ' + '-' * 78)

    base = score(rows_by_method[METHODS[0][0]])
    out = {}
    for name, *_ in METHODS:
        s = score(rows_by_method[name])
        out[name] = s
        if s is None:
            continue
        delta = (s['acc'] - base['acc']) * 100
        arrow = '▲' if delta > 0.05 else ('▼' if delta < -0.05 else '=')
        pv = binom_tail(s['hit'], s['n'], base['majority'])
        print(f'  {name:<34} {s["acc"] * 100:7.1f}% ({s["hit"]:>3}/{s["n"]:<3})'
              f' {arrow}{delta:+6.1f} نقطة {f"p={pv:.2f}":>9}')

    print(f'\n  خط أساس الأغلبية: {base["majority"] * 100:.1f}% '
          f'("قول {base["majority_lab"]} على طول")'
          f'   |   الصدفة: {base["chance"] * 100:.1f}%')
    return out


def main():
    t_start = time.time()
    print('=' * 82)
    print('  TIER 2 — تحسين بنك الـ templates (FastDTW بس)')
    print('=' * 82)
    print(f'\n  الحركات المشتركة ({len(SHARED)}): {", ".join(SHARED)}')
    print(f'  الفيديوهات: {", ".join(VIDEOS)}')
    print(f'  المعاملات: RADIUS={RADIUS}، K={K_NEIGHBORS}، '
          f'أقصى {MAX_PER_LABEL} template لكل حركة')
    print('\n  ⚠️ المعايرة والتنضيف بيتحسبوا من فيديوهات المصدر بس.')
    print('     فيديو الاختبار مابيتلمسش في أي خطوة تعلّم.')

    # ---------- البروتوكول ١: قصاصات ----------
    print(f'\n{"=" * 82}')
    print('  البروتوكول ١: قصاصة كاملة بحدودها الحقيقية  ← الرقم الأساسي')
    print('=' * 82)
    print('\n  تنضيف البنك لكل فيديو اختبار:')
    seg_rows, pruned, kept = run_all('segment', verbose=True)
    print(f'\n  الإجمالي: اتشطب {pruned} template، فضل {kept}')

    seg = table('النتيجة — قصاصات (عبر-الفيديوهات، مافيش تسريب)', seg_rows)

    # تفاصيل أحسن طريقة
    best_name = METHODS[-1][0]
    s_best = seg[best_name]
    s_base = seg[METHODS[0][0]]

    print(f'\n  --- الدقة لكل حركة (قبل / بعد TIER 2) ---')
    print(f'  {"الحركة":<16} {"الأساس":>12} {"TIER 2":>12}')
    print('  ' + '-' * 44)
    for lab in sorted(s_base['per_class']):
        h0, c0 = s_base['per_class'][lab]
        h1, c1 = s_best['per_class'].get(lab, (0, c0))
        mark = ' ▲' if h1 > h0 else (' ▼' if h1 < h0 else '')
        print(f'  {lab:<16} {h0}/{c0:<10} {h1}/{c1:<10}{mark}')

    if s_best['confusion']:
        print(f'\n  --- الالتباسات اللي فضلت بعد TIER 2 ---')
        for (t, p), c in s_best['confusion'].most_common(6):
            print(f'    {t:<16} → {p:<16} {c:3}×')

    print(f'\n  --- كل قصاصة على حدة (TIER 2) ---')
    print(f'  {"فيديو":<10} {"الفترة":>12}  {"الصح":<16} {"التوقّع":<16}')
    print('  ' + '-' * 60)
    for r in seg_rows[best_name]:
        mark = '✅' if r['pred'] == r['truth'] else '❌'
        s, e = r['span']
        print(f'  {r["video"]:<10} {f"{s:.1f}-{e:.1f}":>12}  '
              f'{r["truth"]:<16} {r["pred"]:<16} {mark}')

    # ---------- البروتوكول ٢: نوافذ ----------
    print(f'\n{"=" * 82}')
    print('  البروتوكول ٢: نوافذ منزلقة')
    print('=' * 82)
    win_rows, _, _ = run_all('window')
    win = table('النتيجة — نوافذ',
                win_rows,
                note='⚠️ النوافذ دي مقصوصة من نفس الـ '
                     f'{s_base["n"]} ظهور — مش عيّنات مستقلة، '
                     'وماتتحسبش في الدلالة.')

    s_win_best = win[best_name]
    print(f'\n  --- الدقة لكل حركة (نوافذ، TIER 2) ---')
    for lab, (h, c) in sorted(s_win_best['per_class'].items(),
                              key=lambda kv: -kv[1][0] / max(1, kv[1][1])):
        a = h / max(1, c)
        print(f'    {lab:<16} {a * 100:5.1f}%  {"█" * int(a * 24):<24} ({h}/{c})')

    print(f'\n  --- الالتباسات (نوافذ، TIER 2) ---')
    for (t, p), c in s_win_best['confusion'].most_common(6):
        print(f'    {t:<16} → {p:<16} {c:4}×')

    # ---------- الخلاصة ----------
    print(f'\n{"=" * 82}\n  الخلاصة\n{"=" * 82}')
    p0 = s_base['majority']
    pv = binom_tail(s_best['hit'], s_best['n'], p0)
    delta = (s_best['acc'] - s_base['acc']) * 100

    print(f'\n  قبل TIER 2 : {s_base["acc"] * 100:5.1f}%  '
          f'({s_base["hit"]}/{s_base["n"]})')
    print(f'  بعد TIER 2 : {s_best["acc"] * 100:5.1f}%  '
          f'({s_best["hit"]}/{s_best["n"]})   {delta:+.1f} نقطة')
    print(f'  خط الأساس  : {p0 * 100:5.1f}%  '
          f'("قول {s_base["majority_lab"]} على طول")')
    print(f'\n  احتمال إن رقم TIER 2 يحصل بالصدفة لو الموديل مالوش قدرة: '
          f'p = {pv:.3f}')

    if s_best['acc'] <= p0:
        print('\n  ❌ لسه **تحت خط أساس الأغلبية**. يعني "قول الحركة الأشهر')
        print('     على طول" أحسن من الخوارزمية. ماينفعش نقول إنها نجحت.')
    elif pv > 0.05:
        print(f'\n  ⚠️ فوق خط الأساس بس **مش دال إحصائياً** (p={pv:.2f} > 0.05).')
        print(f'     بـ {s_best["n"]} عيّنة بس، الفرق ده مايتفرقش عن الحظ.')
    else:
        print(f'\n  ✅ فوق خط الأساس و**دال إحصائياً** (p={pv:.3f} ≤ 0.05).')

    print(f'\n  🔬 القيد اللي مافيش منه فكاك: {s_base["n"]} ظهور حقيقي للحركة '
          f'في الـ {len(VIDEOS)} فيديوهات كلهم.')
    print('     مهما حسّنّا الخوارزمية، ده سقف الإحصاء. الحل الوحيد فيديوهات أكتر.')

    # ---------- الحفظ ----------
    save(seg, win, seg_rows, s_base, s_best, pruned, kept, pv)
    print(f'\n  ⏱️ الوقت: {time.time() - t_start:.0f} ثانية')


def save(seg, win, seg_rows, s_base, s_best, pruned, kept, pv):
    path = os.path.join(HERE, 'results', 'tier2_results_annotated.txt')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    best_name = METHODS[-1][0]

    with open(path, 'w', encoding='utf-8') as f:
        f.write('═' * 82 + '\n')
        f.write('TIER 2 — تحسين بنك الـ templates لـ FastDTW\n')
        f.write('═' * 82 + '\n\n')

        f.write('ما الذي تم عمله\n')
        f.write('─' * 82 + '\n')
        f.write('١. معايرة كل template بمتوسط وانحراف مسافته لقصاصات مرجعية\n')
        f.write('   من فيديوهات المصدر (علاج ظاهرة الـ hubness).\n')
        f.write('٢. تصويت أقرب ٣ جيران بوزن حسب الترتيب بدل أقرب واحد.\n')
        f.write('٣. شطب الـ templates اللي ضررها أكتر من نفعها على المصدر.\n\n')
        f.write(f'   عدد الـ templates المشطوبة: {pruned} من {pruned + kept}\n\n')

        f.write('لا يوجد تسريب بيانات: كل خطوات المعايرة والشطب محسوبة من\n')
        f.write('فيديوهات المصدر فقط. فيديو الاختبار غير مستخدم في أي منها.\n\n')

        f.write('═' * 82 + '\n')
        f.write('النتائج — بروتوكول القصاصات (الرقم الأساسي)\n')
        f.write('═' * 82 + '\n\n')
        f.write(f'{"الطريقة":<36} {"الدقة":>18} {"p-value":>10}\n')
        f.write('-' * 82 + '\n')
        for name, *_ in METHODS:
            s = seg[name]
            p = binom_tail(s['hit'], s['n'], s_base['majority'])
            f.write(f'{name:<36} {s["acc"] * 100:7.1f}% '
                    f'({s["hit"]:>3}/{s["n"]:<3}) {p:>9.3f}\n')
        f.write(f'\nخط أساس الأغلبية: {s_base["majority"] * 100:.1f}% '
                f'("قول {s_base["majority_lab"]} على طول")\n')
        f.write(f'الصدفة: {s_base["chance"] * 100:.1f}%\n')
        f.write(f'عدد العينات: {s_base["n"]} ظهور حقيقي للحركة\n\n')

        f.write('كيف تُقرأ هذه الأرقام\n')
        f.write('─' * 82 + '\n')
        f.write('• الدقة وحدها لا تعني شيئاً. الرقم الذي يجب كسره هو خط أساس\n')
        f.write('  الأغلبية، وليس الصدفة.\n')
        f.write('• p-value = احتمال الوصول لهذه الدقة أو أفضل بالصدفة لو كان\n')
        f.write('  النموذج بلا أي قدرة. p > 0.05 يعني النتيجة غير دالة.\n')
        f.write(f'• العينة {s_base["n"]} فقط — أي فرق أقل من عدة نقاط لا معنى له.\n\n')

        f.write('═' * 82 + '\n')
        f.write('الدقة لكل حركة (قبل / بعد)\n')
        f.write('═' * 82 + '\n\n')
        for lab in sorted(s_base['per_class']):
            h0, c0 = s_base['per_class'][lab]
            h1, c1 = s_best['per_class'].get(lab, (0, c0))
            f.write(f'{lab:<18} الأساس {h0}/{c0:<6} TIER 2 {h1}/{c1}\n')

        f.write('\nالالتباسات المتبقية بعد TIER 2:\n')
        for (t, p), c in s_best['confusion'].most_common(8):
            f.write(f'  {t:<18} → {p:<18} {c:3}×\n')

        f.write('\n' + '═' * 82 + '\n')
        f.write('النتائج — بروتوكول النوافذ المنزلقة\n')
        f.write('═' * 82 + '\n')
        f.write('تحذير: النوافذ مقصوصة من نفس الظهورات، فهي عينات مترابطة\n')
        f.write('وليست مستقلة. لا تُستخدم في أي حساب دلالة إحصائية.\n\n')
        for name, *_ in METHODS:
            s = win[name]
            f.write(f'{name:<36} {s["acc"] * 100:7.1f}% '
                    f'({s["hit"]}/{s["n"]})\n')
        f.write(f'\nخط أساس الأغلبية (نوافذ): '
                f'{win[METHODS[0][0]]["majority"] * 100:.1f}%\n')

        f.write('\n' + '═' * 82 + '\n')
        f.write('التفاصيل — كل قصاصة على حدة (TIER 2)\n')
        f.write('═' * 82 + '\n\n')
        f.write(f'{"الفيديو":<12} {"الفترة":<16} {"الصحيح":<18} '
                f'{"التوقع":<18} {"الحالة"}\n')
        f.write('-' * 82 + '\n')
        for r in seg_rows[best_name]:
            s, e = r['span']
            mark = 'صح' if r['pred'] == r['truth'] else 'خطأ'
            f.write(f'{r["video"]:<12} {f"{s:.1f}-{e:.1f}":<16} '
                    f'{r["truth"]:<18} {r["pred"]:<18} {mark}\n')

        f.write('\n' + '═' * 82 + '\n')
        f.write('الخلاصة\n')
        f.write('═' * 82 + '\n\n')
        f.write(f'قبل TIER 2: {s_base["acc"] * 100:.1f}%\n')
        f.write(f'بعد TIER 2: {s_best["acc"] * 100:.1f}%\n')
        f.write(f'خط الأساس : {s_base["majority"] * 100:.1f}%\n')
        f.write(f'p-value   : {pv:.3f}\n\n')
        if s_best['acc'] <= s_base['majority']:
            f.write('الحكم: النتيجة ما زالت تحت خط أساس الأغلبية. لا يمكن\n')
            f.write('القول بأن الطريقة نجحت.\n')
        elif pv > 0.05:
            f.write('الحكم: أعلى من خط الأساس لكن غير دالة إحصائياً.\n')
        else:
            f.write('الحكم: أعلى من خط الأساس ودالة إحصائياً عند 0.05.\n')
        f.write(f'\nالقيد الأساسي: {s_base["n"]} ظهور حقيقي فقط في '
                f'{len(VIDEOS)} فيديوهات.\n')

    print(f'\n  ✅ اتحفظت النتائج مع الشرح في:')
    print(f'     {path}')


if __name__ == '__main__':
    main()
