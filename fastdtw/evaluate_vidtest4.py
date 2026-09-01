"""
FastDTW على vidtest4 — تنبؤ أعمى، من غير ground truth ومن غير أي داتاسِت

    python run_vidtest4.py

⚠️ الفرق بين الملف ده وكل اللي قبله
────────────────────────────────────
`run_vidtest2.py` و `run_vidtest3.py` بياخدوا الـ template من **نفس
الفيديو** اللي بيختبروا عليه. ده بيدّي أرقام عالية (83.3% على vidtest2)
بس فيه ضعف معروف: الـ template والاختبار من نفس ظهور الحركة.

هنا الـ templates كلها من vidtest1 + vidtest2 + vidtest3، و vidtest4
**مشافش ولا template ولا معامل واحد منه**. ده أنضف إعداد في المشروع.

⚠️ مافيش ground truth لـ vidtest4 لحد دلوقتي، **فمافيش رقم دقة**.
   اللي بيطلع هو الخط الزمني المتوقَّع عشان يتشاف بالعين على الفيديو.
   أول ما التوقيتات الحقيقية تتحط في `ground_truth.py` الرقم بيتحسب.

⚠️ المفردات محدودة بـ 15 حركة — اللي موجودة في الفيديوهات التلاتة بس.
   أي حركة في vidtest4 مش منهم **مستحيل** تتظبط، هتتنسب لأقرب حاجة.
   ده قيد في الإعداد مش فشل في الخوارزمية، ولازم يتقال مع أي نتيجة.
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from ground_truth import GROUND_TRUTH
from oneshot_core import (build_multiscale_windows, cut_templates,
                          distance_cache, majority_smooth, window_edges)
from paths import CACHE_DIR, load_keypoints, load_meta

TARGET = 'vidtest4'
SOURCES = ('vidtest1', 'vidtest2', 'vidtest3')

SCALES = (1.5, 2.0, 3.0)
STRIDE = 5
RADIUS = 1
SMOOTH_K = 5            # أطول من 3 لأن التنبؤ الأعمى أكتر ضجيجاً
MAX_PER_LABEL = 6
MIN_SEGMENT = 3         # نوافذ — أقل من كده يتبلع في اللي جنبه

TMPL_MAX, TMPL_MIN, TMPL_FRAC = 2.5, 1.0, 0.5


def load(video):
    return load_keypoints(video)


def build_bank(mode='vel', shape_norm=True):
    """
    بنك templates من الفيديوهات التلاتة.

    بناخد **أول ظهور** لكل حركة في كل فيديو، وقصاصة قصيرة منه. لو الحركة
    موجودة في أكتر من فيديو بياخد منهم كلهم — ده كويس، بيدّي تنوّع في
    الكاميرا والشخص.
    """
    from oneshot_core import balance_templates

    out = []
    for v in SOURCES:
        kp, fps = load(v)
        first = {}
        for s, e, lab in GROUND_TRUTH[v]:
            if lab not in ('?',) and lab not in first:
                first[lab] = (s, e)

        clips = []
        for lab, (s, e) in first.items():
            dur = min(TMPL_MAX, max(TMPL_MIN, (e - s) * TMPL_FRAC))
            clips.append((s, min(e, s + dur), lab))

        out += cut_templates(kp, fps, clips, source=v, mode=mode,
                             shape_norm=shape_norm, scales=SCALES)
    return balance_templates(out, MAX_PER_LABEL)


def hubness_correct(D):
    """كل template يتقاس بمقياسه هو — شوف run_vidtest3.py للتفاصيل."""
    med = np.median(D, axis=0, keepdims=True)
    mad = np.median(np.abs(D - med), axis=0, keepdims=True)
    return (D - med) / np.maximum(mad, 1e-6)


def predict(mode='vel', shape_norm=True, smooth=SMOOTH_K, hubness=True,
            verbose=True):
    kp, fps = load(TARGET)
    templates = build_bank(mode, shape_norm)
    tl = [t['label'] for t in templates]

    centers = np.arange(0, len(kp), STRIDE)
    edges = window_edges(centers, fps, len(kp))
    feats, E = build_multiscale_windows(kp, fps, centers, SCALES,
                                        mode=mode, shape_norm=shape_norm)

    key = f'v4_{mode}_{int(shape_norm)}_s{STRIDE}_r{RADIUS}_m{MAX_PER_LABEL}'
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cf = CACHE_DIR / f'{key}.npy'

    if cf.exists():
        D = np.load(cf)
    else:
        if verbose:
            print(f'  حساب {len(centers) * len(SCALES) * len(templates):,} '
                  f'مسافة DTW...')
        D = distance_cache(feats, templates, SCALES, radius=RADIUS)
        np.save(cf, D)

    Dz = hubness_correct(D) if hubness else D
    flat = Dz.reshape(len(centers), -1)
    best = flat.argmin(axis=1)

    raw = [tl[i % len(templates)] for i in best]
    # المسافة الخام للفايز — مؤشر ثقة، أصغر = أوثق
    conf = D.reshape(len(centers), -1)[np.arange(len(centers)), best]

    labels = majority_smooth(raw, smooth) if smooth > 1 else raw
    return labels, conf, edges, templates, fps


def segments(labels, edges, conf, min_len=MIN_SEGMENT):
    """بيلمّ النوافذ المتتالية اللي ليها نفس اللابل في فترات."""
    segs = []
    i = 0
    while i < len(labels):
        j = i
        while j + 1 < len(labels) and labels[j + 1] == labels[i]:
            j += 1
        if j - i + 1 >= min_len:
            segs.append({'label': labels[i],
                         't0': float(edges[i]),
                         't1': float(edges[min(j + 1, len(edges) - 1)]),
                         'n': j - i + 1,
                         'conf': float(np.mean(conf[i:j + 1]))})
        i = j + 1
    return segs


def main():
    print('=' * 74)
    print('  FastDTW على vidtest4 — تنبؤ أعمى، الفيديو مشافش ولا template منه')
    print('=' * 74)

    kp, fps = load(TARGET)
    meta = load_meta(TARGET)
    # 🐛 المفتاح اسمه detect_rate مش detection_rate. كان مكتوب
    #    .get('detection_rate', 0) فكان بيطبع "اكتشاف 0.0%" بدل 93.9%.
    #    الوصول بالأقواس المربعة بيقع لو الاسم غلط بدل ما يكدب في التقرير.
    print(f'\n🎬 {TARGET}: {len(kp)} فريم @ {fps:.1f} fps فعلي '
          f'({len(kp) / fps:.1f}s)   '
          f'اكتشاف {meta["detect_rate"] * 100:.1f}% '
          f'({meta["n_detected"]}/{meta["n_frames"]})')

    labels, conf, edges, templates, _ = predict()

    vocab = sorted(set(t['label'] for t in templates))
    print(f'\n📚 بنك الـ templates: {len(templates)} template، '
          f'{len(vocab)} حركة، من {", ".join(SOURCES)}')
    by_lab = Counter(t['label'] for t in templates)
    for lab in vocab:
        srcs = sorted({t['source'][-1] for t in templates if t['label'] == lab})
        print(f'    {lab:<18} {by_lab[lab]:>2} template   '
              f'(من فيديو {", ".join(srcs)})')

    segs = segments(labels, edges, conf)

    print(f'\n{"=" * 74}')
    print('  الخط الزمني المتوقَّع — قارنه بالفيديو بعينك')
    print('=' * 74)
    print(f'  {"الفترة":<16} {"الحركة":<18} {"مدة":>6}  {"مسافة":>7}')
    print('  ' + '-' * 56)
    for s in segs:
        dur = s['t1'] - s['t0']
        print(f'  {s["t0"]:5.1f} - {s["t1"]:5.1f}s   {s["label"]:<18} '
              f'{dur:5.1f}s  {s["conf"]:7.3f}')

    print(f'\n  إجمالي {len(segs)} فترة على {len(kp) / fps:.1f} ثانية')

    print(f'\n  --- نصيب كل حركة من الزمن ---')
    tot = Counter()
    for s in segs:
        tot[s['label']] += s['t1'] - s['t0']
    span = sum(tot.values()) or 1
    for lab, d in tot.most_common():
        bar = '█' * int(d / span * 34)
        print(f'    {lab:<18} {d:5.1f}s  {d / span * 100:4.1f}%  {bar}')

    print(f'\n{"=" * 74}')
    print('  ⚠️ مافيش رقم دقة — vidtest4 لسه من غير ground truth.')
    print('     حط التوقيتات الحقيقية في ground_truth.py والرقم هيتحسب.')
    print('=' * 74)


if __name__ == '__main__':
    main()
