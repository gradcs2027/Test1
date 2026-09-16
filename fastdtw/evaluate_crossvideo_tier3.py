"""
TIER 3 — تحسين مسافة الـ DTW نفسها (FastDTW بس، مافيش أي شبكة عصبية)

    python fastdtw/evaluate_crossvideo_tier3.py

⚠️ ملاحظة على التسمية: ملف `ALGORITHM_ANALYSIS.md` القديم كان كاتب
   TIER 3 = تدريب LSTM. ده اتلغى — إحنا شغّالين على FastDTW بس.
   TIER 3 هنا = شريط Sakoe-Chiba + DTW كامل + أوزان مفاصل متعلَّمة.

فين وصلنا
─────────
  TIER 1 (تظبيط RADIUS + عتبة ثقة)     فشل تماماً — نفس الرقم بالحرف
  TIER 2 (معايرة z + تصويت + تنضيف)     13.8% → 24.1%  (خط الأساس 20.7%)

TIER 2 صلّح **إزاي بنقارن** المسافات. TIER 3 بيصلّح **إزاي بنحسبها**.

التشخيص اللي بنى TIER 3
────────────────────────
بصّ على الحركات اللي لسه فاشلة بعد TIER 2 (بروتوكول النوافذ):

    phone_call     90.7%  ✅        wave            1.3%  ❌
    clapping       69.4%  ✅        walking         0.0%  ❌
    stand_up       35.0%           spray_perfume   2.4%  ❌
                                    sitting        10.5%  ❌

والالتباسات: wave→clapping، wave→spray_perfume، brush_hair→phone_call،
spray_perfume→phone_call، sitting→stand_up.

**كل الالتباسات دي جوّه نفس مجموعة الجسم.** الحركات اللي بالدراع
(wave/clapping/spray_perfume/brush_hair/phone_call) بتتلخبط في بعض،
والحركات اللي بالرجل والجذع (walking/sitting/stand_up) بتتلخبط في بعض.

يعني الـ DTW شايف "الجسم كله اتحرك" بس مش شايف **مين** اتحرك. السبب
إن المسافة بتدي وزن متساوي لكل الـ ١٧ مفصل — الودن والمنخار بياخدوا
نفس وزن الرسغ. الودن ضوضاء صافية هنا، وبيغرق إشارة الرسغ.

التلات خطوات
────────────
  أ. **DTW كامل بدل تقريب FastDTW**
     المتتاليات بتتعمّلها resample لـ ٣٠ فريم، يعني المصفوفة ٢٩×٢٩ = ٨٤١
     خانة بس. ده رخيص جداً إننا نحسبه بالظبط. فليه نستخدم تقريب أصلاً؟
     الخطوة دي بتقيس: **هل تقريب FastDTW كان بياكل من الدقة؟**

  ب. **شريط Sakoe-Chiba**
     الـ DTW الحر ممكن يمطّ فريم واحد على عشرين فريم عشان يلاقي تطابق.
     ده بيخلي أي حركة تقدر "تتلوى" لحد ما تشبه أي حركة تانية — وده
     بالظبط اللي بيخلي wave تطلع clapping. الشريط بيمنع المسار إنه
     يبعد أكتر من B خانة عن القُطر، فالتمطيط بيفضل معقول.

  ج. **أوزان المفاصل المتعلَّمة (Weighted DTW)**
     بدل ما كل مفصل ياخد وزن ١، بنحسب لكل مفصل **نسبة فيشر**:

         فيشر = تباين_بين_الحركات / تباين_داخل_الحركة

     المفصل اللي بيفرق بين الحركات وثابت جوّه الحركة الواحدة ياخد وزن
     أكبر. بنخلطها مع التوزيع المتساوي بنسبة ALPHA عشان مانبالغش في
     الثقة بـ ٢٩ عيّنة بس.

مافيش تسريب
───────────
أوزان المفاصل بتتحسب من **templates فيديوهات المصدر بس**، ومعاها
المعايرة والتنضيف بتوع TIER 2. فيديو الاختبار مابيتلمسش في أي خطوة.

⚠️ وبرضه: n=29. أي رقم هنا لازم يتقري ومعاه العدد ده.
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

# --- نفس معاملات TIER 2 بالظبط عشان المقارنة تبقى عادلة ---
SCALES = (1.0, 1.5, 2.0, 3.0)
STRIDE = 5
RADIUS = 1
MAX_PER_LABEL = 8
FRACTIONS = (0.6, 0.8, 1.0)
K_NEIGHBORS = 3

# --- الجديد في TIER 3 ---
BAND = 4        # نص عرض شريط Sakoe-Chiba (٤ من ٢٩ ≈ ١٤٪)
ALPHA = 0.5     # قد إيه نثق في أوزان فيشر مقابل التوزيع المتساوي
N_JOINTS = 17   # COCO-17

EPS = 1e-9
SHARED = tuple(shared_labels())
HERE = os.path.dirname(os.path.abspath(__file__))

# أسماء مفاصل COCO-17 — للعرض بس
JOINT_NAMES = ['أنف', 'عين ش', 'عين ي', 'ودن ش', 'ودن ي',
               'كتف ش', 'كتف ي', 'كوع ش', 'كوع ي', 'رسغ ش', 'رسغ ي',
               'ورك ش', 'ورك ي', 'ركبة ش', 'ركبة ي', 'كاحل ش', 'كاحل ي']

_KP_CACHE = {}


def load(video):
    if video not in _KP_CACHE:
        _KP_CACHE[video] = load_keypoints(video)
    return _KP_CACHE[video]


# ==============================================================================
# الخطوات أ + ب: DTW كامل بشريط Sakoe-Chiba وأوزان
# ==============================================================================

def dtw_banded(a, b, band=None, w=None):
    """
    DTW كامل (مش تقريب) بشريط Sakoe-Chiba اختياري وأوزان مفاصل اختيارية.

    a, b : (T, 34) — الاتنين بنفس الطول لأن كل حاجة بتتعمّلها resample
                     لـ NUM_FRAMES قبل ما توصل هنا
    band : نص عرض الشريط. None = DTW حر (زي التقليدي)
    w    : (34,) أوزان لكل بُعد. None = وزن متساوي

    بيرجّع المسافة **مقسومة على طول المسار** — بالظبط زي `norm_distance`
    عشان الأرقام تبقى قابلة للمقارنة مع TIER 2.

    ⚠️ ليه بنحسب طول المسار مع التكلفة؟ عشان القسمة عليه. من غيرها
       المسارات الأطول بتاخد مسافات أكبر تلقائياً والمقارنة مش عادلة.
    """
    n, m = len(a), len(b)

    # مصفوفة التكلفة المحلية — دي الجزء الغالي، فبنتّجهها بـ numpy
    diff = a[:, None, :] - b[None, :, :]
    if w is not None:
        cost = np.sqrt((diff * diff * w).sum(axis=-1))
    else:
        cost = np.sqrt((diff * diff).sum(axis=-1))

    INF = np.inf
    C = np.full((n + 1, m + 1), INF)      # التكلفة المتراكمة
    L = np.zeros((n + 1, m + 1))          # طول المسار المتراكم
    C[0, 0] = 0.0

    for i in range(1, n + 1):
        if band is None:
            j0, j1 = 1, m
        else:
            # الشريط بيتحسب على القُطر المتناسب مع الطولين
            centre = (i - 1) * m / max(1, n)
            j0 = max(1, int(centre - band) + 1)
            j1 = min(m, int(centre + band) + 1)
        row_c, row_l = C[i], L[i]
        prev_c, prev_l = C[i - 1], L[i - 1]
        ci = cost[i - 1]
        for j in range(j0, j1 + 1):
            # الاختيار بين التلات اتجاهات
            d, l = prev_c[j - 1], prev_l[j - 1]        # قُطري
            if prev_c[j] < d:
                d, l = prev_c[j], prev_l[j]            # رأسي
            if row_c[j - 1] < d:
                d, l = row_c[j - 1], row_l[j - 1]      # أفقي
            if d == INF:
                continue
            row_c[j] = d + ci[j - 1]
            row_l[j] = l + 1.0

    total, steps = C[n, m], L[n, m]
    if not np.isfinite(total) or steps < 1:
        return float('inf')
    return float(total / steps)


# ==============================================================================
# الخطوة ج: أوزان المفاصل المتعلَّمة (نسبة فيشر)
# ==============================================================================

def learn_joint_weights(templates):
    """
    وزن لكل مفصل = تباين_بين_الحركات / تباين_داخل_الحركة  (نسبة فيشر).

    بتتحسب من **templates المصدر بس** — مافيش أي عيّنة اختبار هنا.

    كل template بنلخّصه في متجه طاقة بطول ١٧: طاقة كل مفصل =
    متوسط (x² + y²) عبر الفريمات. بعدها نقيس المفصل ده بيفرّق بين
    الحركات قد إيه.

    ⚠️ ليه بنخلط مع التوزيع المتساوي (ALPHA)؟ عندنا عيّنة صغيرة جداً،
       والأوزان الخام ممكن تتشكّل من الضوضاء وتقفل على مفصل واحد.
       الخلط بيخلي أسوأ حالة إننا نرجع لـ TIER 2 مش أوحش منه.

    بيرجّع (أوزان_٣٤_بُعد, أوزان_١٧_مفصل) — التانية للعرض بس.
    """
    labels = sorted({t['label'] for t in templates})
    if len(labels) < 2:
        return None, None

    # طاقة كل مفصل في كل template
    E, y = [], []
    for t in templates:
        x = t['feat']                                   # (T, 34)
        xy = x.reshape(len(x), N_JOINTS, 2)
        E.append((xy ** 2).sum(axis=2).mean(axis=0))    # (17,)
        y.append(t['label'])
    E = np.asarray(E)
    y = np.asarray(y)

    grand = E.mean(axis=0)
    between = np.zeros(N_JOINTS)
    within = np.zeros(N_JOINTS)
    n_used = 0
    for lab in labels:
        g = E[y == lab]
        if len(g) < 2:
            continue
        between += len(g) * (g.mean(axis=0) - grand) ** 2
        within += len(g) * g.var(axis=0)
        n_used += len(g)
    if n_used == 0:
        return None, None

    fisher = (between / max(1, n_used)) / (within / max(1, n_used) + EPS)

    # نطبّع لمتوسط ١، وبعدين نخلط مع المتساوي
    f = fisher / (fisher.mean() + EPS)
    f = np.clip(f, 0.0, 5.0)                 # نمنع مفصل واحد يبلع كل الوزن
    w_joint = (1.0 - ALPHA) + ALPHA * f
    w_joint = w_joint / w_joint.mean()

    w_dim = np.repeat(w_joint, 2)            # كل مفصل ليه x و y
    return w_dim, w_joint


# ==============================================================================
# بناء البنك والعيّنات (نفس TIER 2)
# ==============================================================================

def featurize(kp, fps, t0, t1, mode, shape_norm):
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
    """قصاصات المعايرة — من المصدر بس، مش من فيديو الاختبار."""
    refs = []
    for v in sources:
        kp, fps = load(v)
        for lab in SHARED:
            for (s, e) in spans(v, lab):
                f = featurize(kp, fps, s, e, mode, shape_norm)
                if f is not None:
                    refs.append({'video': v, 'label': lab, 'feat': f})
    return refs


def test_items(video, protocol, mode, shape_norm, labels_avail):
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


# ==============================================================================
# المسافات — أربع نسخ نقارنها
# ==============================================================================

VARIANTS = [
    # (الاسم, نوع, شريط, أوزان)
    ('TIER 2 (تقريب FastDTW)',        'fast',  None, False),
    ('أ. DTW كامل بدل التقريب',        'exact', None, False),
    ('ب. + شريط Sakoe-Chiba',          'exact', BAND, False),
    ('ج. + أوزان المفاصل ⭐ TIER 3',    'exact', BAND, True),
]


def distance_matrix(items, templates, kind, band, w):
    D = np.empty((len(items), len(templates)), dtype=float)
    for i, it in enumerate(items):
        fa = it['feat']
        for j, t in enumerate(templates):
            if kind == 'fast':
                D[i, j] = norm_distance(fa, t['feat'], radius=RADIUS)
            else:
                D[i, j] = dtw_banded(fa, t['feat'], band=band, w=w)
    return D


# ==============================================================================
# قرار TIER 2 (معايرة z + تصويت ٣ جيران + تنضيف) — بنبني فوقه
# ==============================================================================

def calibrate(templates, refs, Dref):
    """نفس دالة TIER 2 — كل حاجة متحسبة من المصدر بس."""
    n_t = len(templates)
    mu, sigma = np.zeros(n_t), np.ones(n_t)
    good = np.zeros(n_t, dtype=int)
    bad = np.zeros(n_t, dtype=int)
    ref_videos = np.array([r['video'] for r in refs])

    for j, t in enumerate(templates):
        col = Dref[ref_videos != t['source'], j]
        col = col[np.isfinite(col)]
        if len(col) >= 2:
            mu[j], sigma[j] = col.mean(), col.std() + EPS
        elif len(col) == 1:
            mu[j] = col[0]

    Z = (Dref - mu) / sigma
    for i, r in enumerate(refs):
        mask = np.array([t['source'] != r['video'] for t in templates])
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        j = idx[int(np.argmin(Z[i, idx]))]
        if templates[j]['label'] == r['label']:
            good[j] += 1
        else:
            bad[j] += 1

    keep = ~((good == 0) & (bad >= 2))
    if keep.sum() < 2 or len({templates[j]['label']
                              for j in np.where(keep)[0]}) < 2:
        keep = np.ones(n_t, dtype=bool)
    return mu, sigma, keep


def predict(Drow, templates, mu, sigma, keep):
    """قرار TIER 2 الكامل: معايرة z + تصويت أقرب ٣ بوزن الترتيب."""
    idx = np.where(keep)[0]
    if len(idx) == 0:
        idx = np.arange(len(templates))
    d = (Drow[idx] - mu[idx]) / sigma[idx]
    d = np.where(np.isfinite(d), d, np.inf)

    k = min(K_NEIGHBORS, len(idx))
    order = np.argsort(d)[:k]
    votes = Counter()
    for rank, o in enumerate(order):
        votes[templates[idx[o]]['label']] += 1.0 / (rank + 1)
    best = max(votes.values())
    tied = [lab for lab, val in votes.items() if val == best]
    if len(tied) == 1:
        return tied[0]
    return min(tied, key=lambda lab: min(
        d[o] for o in order if templates[idx[o]]['label'] == lab))


# ==============================================================================
# التشغيل
# ==============================================================================

def run_all(protocol, mode='vel', shape_norm=True, verbose=False):
    """
    كل النسخ على نفس العيّنات بالظبط.

    الأوزان والمعايرة بتتحسب لكل فيديو اختبار على حدة من مصادره بس.
    """
    rows = {name: [] for name, *_ in VARIANTS}
    weight_log = {}

    for v in VIDEOS:
        sources = [o for o in VIDEOS if o != v]
        templates = build_templates(sources, mode, shape_norm)
        if len(templates) < 2:
            continue
        labels_avail = {t['label'] for t in templates}
        refs = source_reference_clips(sources, mode, shape_norm)
        if len(refs) < 3:
            continue
        items = test_items(v, protocol, mode, shape_norm, labels_avail)
        if not items:
            continue

        w_dim, w_joint = learn_joint_weights(templates)
        weight_log[v] = w_joint

        for name, kind, band, use_w in VARIANTS:
            w = w_dim if use_w else None
            Dref = distance_matrix(refs, templates, kind, band, w)
            mu, sigma, keep = calibrate(templates, refs, Dref)
            Dtest = distance_matrix(items, templates, kind, band, w)
            for i, it in enumerate(items):
                rows[name].append({
                    'video': v, 'truth': it['truth'], 'span': it['span'],
                    'pred': predict(Dtest[i], templates, mu, sigma, keep)})
        if verbose:
            print(f'    {v}: {len(templates)} template، '
                  f'{len(refs)} قصاصة معايرة، {len(items)} عيّنة اختبار')

    return rows, weight_log


# ==============================================================================
# التقييم
# ==============================================================================

def binom_tail(k, n, p):
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


def table(title, rows_by_variant, note=''):
    print(f'\n  {title}')
    if note:
        print(f'  {note}')
    print(f'  {"النسخة":<34} {"الدقة":>17} {"مقابل TIER 2":>15} {"دلالة":>9}')
    print('  ' + '-' * 80)
    base = score(rows_by_variant[VARIANTS[0][0]])
    out = {}
    for name, *_ in VARIANTS:
        s = score(rows_by_variant[name])
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
    t0 = time.time()
    print('=' * 84)
    print('  TIER 3 — تحسين مسافة الـ DTW نفسها (FastDTW بس)')
    print('=' * 84)
    print(f'\n  الحركات المشتركة ({len(SHARED)}): {", ".join(SHARED)}')
    print(f'  المعاملات الجديدة: شريط Sakoe-Chiba = ±{BAND} من ٢٩، '
          f'خلط الأوزان ALPHA = {ALPHA}')
    print('  فوق قرار TIER 2 (معايرة z + تصويت ٣ جيران + تنضيف البنك)')
    print('\n  ⚠️ الأوزان والمعايرة بتتحسب من فيديوهات المصدر بس.')
    print('     فيديو الاختبار مابيتلمسش في أي خطوة تعلّم.')

    # ---------- البروتوكول ١ ----------
    print(f'\n{"=" * 84}')
    print('  البروتوكول ١: قصاصة كاملة بحدودها الحقيقية  ← الرقم الأساسي')
    print('=' * 84 + '\n')
    seg_rows, wlog = run_all('segment', verbose=True)
    seg = table('النتيجة — قصاصات (عبر-الفيديوهات، مافيش تسريب)', seg_rows)

    base_name, best_name = VARIANTS[0][0], VARIANTS[-1][0]
    s_base, s_best = seg[base_name], seg[best_name]

    # ---------- الأوزان اللي اتعلمناها ----------
    print(f'\n  --- أوزان المفاصل المتعلَّمة (متوسط الأربع تجارب) ---')
    print('      الوزن > 1 = المفصل ده بيفرّق بين الحركات، '
          'أقل من 1 = ضوضاء')
    stack = np.array([w for w in wlog.values() if w is not None])
    if len(stack):
        avg = stack.mean(axis=0)
        for k in np.argsort(-avg):
            bar = '█' * int(avg[k] * 12)
            print(f'    {JOINT_NAMES[k]:<10} {avg[k]:5.2f}  {bar}')

    print(f'\n  --- الدقة لكل حركة (TIER 2 / TIER 3) ---')
    print(f'  {"الحركة":<16} {"TIER 2":>12} {"TIER 3":>12}')
    print('  ' + '-' * 44)
    for lab in sorted(s_base['per_class']):
        h0, c0 = s_base['per_class'][lab]
        h1, c1 = s_best['per_class'].get(lab, (0, c0))
        mark = ' ▲' if h1 > h0 else (' ▼' if h1 < h0 else '')
        print(f'  {lab:<16} {f"{h0}/{c0}":>12} {f"{h1}/{c1}":>12}{mark}')

    if s_best['confusion']:
        print(f'\n  --- الالتباسات اللي فضلت بعد TIER 3 ---')
        for (t, p), c in s_best['confusion'].most_common(6):
            print(f'    {t:<16} → {p:<16} {c:3}×')

    print(f'\n  --- كل قصاصة على حدة (TIER 3) ---')
    print(f'  {"فيديو":<10} {"الفترة":>12}  {"الصح":<16} {"التوقّع":<16}')
    print('  ' + '-' * 62)
    for r in seg_rows[best_name]:
        mark = '✅' if r['pred'] == r['truth'] else '❌'
        s, e = r['span']
        print(f'  {r["video"]:<10} {f"{s:.1f}-{e:.1f}":>12}  '
              f'{r["truth"]:<16} {r["pred"]:<16} {mark}')

    # ---------- البروتوكول ٢ ----------
    print(f'\n{"=" * 84}')
    print('  البروتوكول ٢: نوافذ منزلقة')
    print('=' * 84)
    win_rows, _ = run_all('window')
    win = table('النتيجة — نوافذ', win_rows,
                note=f'⚠️ مقصوصة من نفس الـ {s_base["n"]} ظهور — '
                     'عيّنات مترابطة، ماتتحسبش في الدلالة.')

    s_win_best, s_win_base = win[best_name], win[base_name]
    print(f'\n  --- الدقة لكل حركة (نوافذ) ---')
    print(f'  {"الحركة":<16} {"TIER 2":>9} {"TIER 3":>9}')
    print('  ' + '-' * 40)
    for lab, (h, c) in sorted(s_win_best['per_class'].items(),
                              key=lambda kv: -kv[1][0] / max(1, kv[1][1])):
        h0, c0 = s_win_base['per_class'].get(lab, (0, c))
        mark = ' ▲' if h / max(1, c) > h0 / max(1, c0) else (
            ' ▼' if h / max(1, c) < h0 / max(1, c0) else '')
        print(f'  {lab:<16} {h0 / max(1, c0) * 100:8.1f}% '
              f'{h / max(1, c) * 100:8.1f}%{mark}')

    print(f'\n  --- الالتباسات (نوافذ، TIER 3) ---')
    for (t, p), c in s_win_best['confusion'].most_common(6):
        print(f'    {t:<16} → {p:<16} {c:4}×')

    # ---------- الخلاصة ----------
    print(f'\n{"=" * 84}\n  الخلاصة\n{"=" * 84}')
    p0 = s_base['majority']
    pv = binom_tail(s_best['hit'], s_best['n'], p0)
    print(f'\n  الأساس 1-NN (قبل TIER 2)  :  13.8%   ← من التشغيلة السابقة')
    print(f'  TIER 2                     : {s_base["acc"] * 100:5.1f}%  '
          f'({s_base["hit"]}/{s_base["n"]})')
    print(f'  TIER 3                     : {s_best["acc"] * 100:5.1f}%  '
          f'({s_best["hit"]}/{s_best["n"]})   '
          f'{(s_best["acc"] - s_base["acc"]) * 100:+.1f} نقطة')
    print(f'  خط أساس الأغلبية           : {p0 * 100:5.1f}%')
    print(f'\n  احتمال إن رقم TIER 3 يحصل بالصدفة = {pv:.3f}')

    if s_best['acc'] <= p0:
        print('\n  ❌ تحت خط أساس الأغلبية. ماينفعش نقول إنها نجحت.')
    elif pv > 0.05:
        print(f'\n  ⚠️ فوق خط الأساس بس **مش دال إحصائياً** (p={pv:.2f} > 0.05).')
        print(f'     بـ {s_best["n"]} عيّنة بس، الفرق ده مايتفرقش عن الحظ.')
    else:
        print(f'\n  ✅ فوق خط الأساس و**دال إحصائياً** (p={pv:.3f} ≤ 0.05).')

    print(f'\n  🔬 السقف: {s_base["n"]} ظهور حقيقي في {len(VIDEOS)} فيديوهات.')
    print('     ده سقف **الإحصاء** مش سقف الخوارزمية. حتى لو TIER 4 طلّع')
    print('     40%، هيفضل p > 0.05 على العدد ده. الحل فيديوهات أكتر.')

    save(seg, win, seg_rows, s_base, s_best, wlog, pv)
    print(f'\n  ⏱️ الوقت: {time.time() - t0:.0f} ثانية')


def save(seg, win, seg_rows, s_base, s_best, wlog, pv):
    path = os.path.join(HERE, 'results', 'tier3_results_annotated.txt')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    best_name = VARIANTS[-1][0]

    with open(path, 'w', encoding='utf-8') as f:
        f.write('═' * 84 + '\n')
        f.write('TIER 3 — تحسين مسافة الـ DTW نفسها (FastDTW فقط)\n')
        f.write('═' * 84 + '\n\n')

        f.write('ما الذي تم عمله\n')
        f.write('─' * 84 + '\n')
        f.write('TIER 2 أصلح طريقة *مقارنة* المسافات. TIER 3 يصلح طريقة\n')
        f.write('*حسابها*. ثلاث خطوات فوق قرار TIER 2:\n\n')
        f.write('أ. DTW كامل بدل تقريب FastDTW — المتتاليات 30 فريم فقط،\n')
        f.write('   أي 841 خانة، فالحساب الدقيق رخيص. الخطوة تقيس هل كان\n')
        f.write('   التقريب يكلفنا دقة.\n')
        f.write(f'ب. شريط Sakoe-Chiba بعرض ±{BAND} — يمنع مسار الـ warping\n')
        f.write('   من الابتعاد عن القطر، فلا تستطيع حركة أن "تتلوى" حتى\n')
        f.write('   تشبه حركة أخرى.\n')
        f.write('ج. أوزان مفاصل متعلَّمة بنسبة فيشر (تباين بين الحركات على\n')
        f.write(f'   تباين داخلها)، مخلوطة مع التوزيع المتساوي بنسبة {ALPHA}.\n\n')

        f.write('لا يوجد تسريب: الأوزان والمعايرة والتنضيف كلها محسوبة من\n')
        f.write('فيديوهات المصدر فقط. فيديو الاختبار غير مستخدم في أي منها.\n\n')

        f.write('═' * 84 + '\n')
        f.write('النتائج — بروتوكول القصاصات (الرقم الأساسي)\n')
        f.write('═' * 84 + '\n\n')
        f.write(f'{"النسخة":<36} {"الدقة":>18} {"p-value":>10}\n')
        f.write('-' * 84 + '\n')
        for name, *_ in VARIANTS:
            s = seg[name]
            p = binom_tail(s['hit'], s['n'], s_base['majority'])
            f.write(f'{name:<36} {s["acc"] * 100:7.1f}% '
                    f'({s["hit"]:>3}/{s["n"]:<3}) {p:>9.3f}\n')
        f.write(f'\nخط أساس الأغلبية: {s_base["majority"] * 100:.1f}% '
                f'("قول {s_base["majority_lab"]} على طول")\n')
        f.write(f'الصدفة: {s_base["chance"] * 100:.1f}%\n')
        f.write(f'عدد العينات: {s_base["n"]} ظهور حقيقي\n\n')

        f.write('كيف تُقرأ هذه الأرقام\n')
        f.write('─' * 84 + '\n')
        f.write('• الرقم الذي يجب كسره هو خط أساس الأغلبية، وليس الصدفة.\n')
        f.write('• p-value = احتمال الوصول لهذه الدقة أو أفضل بالصدفة لو كان\n')
        f.write('  النموذج بلا أي قدرة. p > 0.05 يعني النتيجة غير دالة.\n')
        f.write(f'• العينة {s_base["n"]} فقط — أي فرق أقل من عدة نقاط لا معنى له.\n\n')

        f.write('═' * 84 + '\n')
        f.write('أوزان المفاصل المتعلَّمة (متوسط التجارب الأربع)\n')
        f.write('═' * 84 + '\n')
        f.write('الوزن > 1 يعني أن هذا المفصل يميّز بين الحركات.\n')
        f.write('الوزن < 1 يعني أنه ضوضاء بالنسبة لهذه المجموعة من الحركات.\n\n')
        stack = np.array([w for w in wlog.values() if w is not None])
        if len(stack):
            avg = stack.mean(axis=0)
            for k in np.argsort(-avg):
                f.write(f'  {JOINT_NAMES[k]:<12} {avg[k]:5.2f}\n')

        f.write('\n' + '═' * 84 + '\n')
        f.write('الدقة لكل حركة (TIER 2 / TIER 3)\n')
        f.write('═' * 84 + '\n\n')
        for lab in sorted(s_base['per_class']):
            h0, c0 = s_base['per_class'][lab]
            h1, c1 = s_best['per_class'].get(lab, (0, c0))
            f.write(f'{lab:<18} TIER 2 {h0}/{c0:<8} TIER 3 {h1}/{c1}\n')

        f.write('\nالالتباسات المتبقية بعد TIER 3:\n')
        for (t, p), c in s_best['confusion'].most_common(8):
            f.write(f'  {t:<18} → {p:<18} {c:3}×\n')

        f.write('\n' + '═' * 84 + '\n')
        f.write('النتائج — بروتوكول النوافذ المنزلقة\n')
        f.write('═' * 84 + '\n')
        f.write('تحذير: النوافذ مقصوصة من نفس الظهورات، فهي عينات مترابطة\n')
        f.write('وليست مستقلة. لا تُستخدم في أي حساب دلالة إحصائية.\n\n')
        for name, *_ in VARIANTS:
            s = win[name]
            f.write(f'{name:<36} {s["acc"] * 100:7.1f}% '
                    f'({s["hit"]}/{s["n"]})\n')
        f.write(f'\nخط أساس الأغلبية (نوافذ): '
                f'{win[VARIANTS[0][0]]["majority"] * 100:.1f}%\n')

        f.write('\n' + '═' * 84 + '\n')
        f.write('التفاصيل — كل قصاصة على حدة (TIER 3)\n')
        f.write('═' * 84 + '\n\n')
        f.write(f'{"الفيديو":<12} {"الفترة":<16} {"الصحيح":<18} '
                f'{"التوقع":<18} {"الحالة"}\n')
        f.write('-' * 84 + '\n')
        for r in seg_rows[best_name]:
            s, e = r['span']
            f.write(f'{r["video"]:<12} {f"{s:.1f}-{e:.1f}":<16} '
                    f'{r["truth"]:<18} {r["pred"]:<18} '
                    f'{"صح" if r["pred"] == r["truth"] else "خطأ"}\n')

        f.write('\n' + '═' * 84 + '\n')
        f.write('الخلاصة\n')
        f.write('═' * 84 + '\n\n')
        f.write('الأساس 1-NN : 13.8%\n')
        f.write(f'TIER 2      : {s_base["acc"] * 100:.1f}%\n')
        f.write(f'TIER 3      : {s_best["acc"] * 100:.1f}%\n')
        f.write(f'خط الأساس   : {s_base["majority"] * 100:.1f}%\n')
        f.write(f'p-value     : {pv:.3f}\n\n')
        if s_best['acc'] <= s_base['majority']:
            f.write('الحكم: تحت خط أساس الأغلبية. لا يمكن القول بأنها نجحت.\n')
        elif pv > 0.05:
            f.write('الحكم: أعلى من خط الأساس لكن غير دالة إحصائياً.\n')
        else:
            f.write('الحكم: أعلى من خط الأساس ودالة إحصائياً عند 0.05.\n')
        f.write(f'\nالقيد الأساسي: {s_base["n"]} ظهور حقيقي فقط في '
                f'{len(VIDEOS)} فيديوهات.\nهذا سقف إحصائي وليس سقف خوارزمية.\n')

    print(f'\n  ✅ اتحفظت النتائج مع الشرح في:\n     {path}')


if __name__ == '__main__':
    main()
