"""
Few-Shot FastDTW — التجربة الكاملة (من غير أي داتاسِت)

    python run_oneshot.py            # المعايرة + التقييم على الـ 3 فيديوهات
    python run_oneshot.py --ablate   # كمان تجارب الحذف (إيه اللي ساعد فعلاً)

المنهجية — ثلاث نقاط لازم تتقري قبل أي رقم:

  ١. مفيش داتاسِت خالص. الـ templates متقصوصة من الفيديوهات نفسها.

  ٢. **مافيش تسريب**: كل فيديو بيتقيّم بـ templates من الفيديوهات
     **التانية** بس (leave-one-video-out). العتبات كمان متعايرة على
     الفيديوهات التانية. يعني vidtest3 مشافش ولا template ولا عتبة
     جايين منه.

  ٣. كل رقم بيتقارن بخط الأساس الغبي "قول 'other' على طول". في vidtest3
     الخط ده لوحده 95.7% لأن 96% من الفيديو سكون. أي رقم دقة إجمالي من
     غير الخط ده جنبه هو رقم مضلّل.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# طرفية الويندوز افتراضيّاً cp1252 وبتقع مع العربي والإيموجي
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from oneshot_core import (balance_templates, build_multiscale_windows,
                          classify, cut_templates, distance_cache,
                          drop_short_segments, event_metrics, frame_metrics,
                          majority_smooth, window_edges)

# ==============================================================================
# الإعدادات
# ==============================================================================

SCALES = (1.5, 2.0, 3.0)      # مقاسات النوافذ بالثواني
STRIDE = 5                    # خطوة المركز بالفريمات (~0.33s عند 15fps)
RADIUS = 1                    # نصف قطر FastDTW
SMOOTH_K = 3                  # تصويت الأغلبية (مش متوسط — بص القسم في oneshot_core)
MIN_SEGMENT = 2               # أقصر قطعة مسموح بيها
TOLERANCE = 1.0               # سماحية التطابق الزمني للحدث (ثانية)
MAX_PER_LABEL = 4             # سقف templates لكل حركة — بص balance_templates

# ==============================================================================
# البيانات — الـ ground truth للفيديوهات التلاتة
# ==============================================================================

# ⚠️⚠️ الجدول ده كان مكتوب هنا بإيد، وكان **فيه تلات أغلاط**:
#
#   ١. vidtest3 كان متوصّف 4 حركات + 120 ثانية سكون. الحقيقة 13 حركة.
#      كل رقم اتقاس عليه لاغي.
#   ٢. vidtest2 كان ناقص الـ wave و الـ running. وأسوأ: الفترة 2.2-5.9
#      كانت متعلّمة 'other*' (سكون متأكّد) وهي فيها الـ wave (3-5) —
#      يعني الكشف الصح للتلويح كان بيتحسب إنذار كاذب.
#   ٣. phone_call كانت 18.2-23.6 والصح 18-22.
#
# اتشال 2026-08-31 وبقى بييجي من `ground_truth.py` (المصدر الواحد).
# التفاصيل في HANDOFF قسم 6.8.
#
# ⚠️ التجربة دي بكلاسين (sit/stand)، فبنسقط المفردات الكاملة عليها
#    بـ project(). أي حركة تانية بتبقى 'other*'، و '?' بتفضل '?'.
#    الإسقاط ده بيحصل من الجدول المرجعي مش بجدول تاني مكتوب بإيد —
#    وده بالظبط اللي بيمنع الأغلاط الفوق دي إنها ترجع.
from ground_truth import GROUND_TRUTH, project, spans

KEEP = ('rub_hands', 'wave', 'sit_down', 'stand_up', 'phone_call')

GT = {v: project(v, KEEP, rest='other*') for v in GROUND_TRUTH}

# القصاصات اللي بنقص منها الـ templates — نفس مقاطع الـ ground truth
# ⚠️ vidtest3 فاضية عمداً: هو فيديو الاختبار، مابناخدش منه templates خالص
TEMPLATE_SOURCES = ('vidtest1', 'vidtest2')

TEMPLATE_CLIPS = {
    v: sorted((s, e, lab) for lab in KEEP for s, e in spans(v, lab))
    if v in TEMPLATE_SOURCES else []
    for v in GROUND_TRUTH
}


def load(name):
    kp = np.load(f'keypoints/{name}_keypoints.npy')
    meta = np.load(f'keypoints/{name}_meta.npy', allow_pickle=True).item()
    return kp, float(meta['effective_fps'])


# ==============================================================================
# مقاييس مستوى الحدث — ده اللي بنعاير عليه
# ==============================================================================

def event_f1(labels, edges, gt):
    """F1 على مستوى الحدث — للعرض في التقرير."""
    ev = event_metrics(labels, edges, gt, tolerance=TOLERANCE)
    tp, fa = ev['n_detected'], ev['n_false_alarms']
    if tp == 0:
        return 0.0, ev
    prec = tp / (tp + fa)
    rec = tp / ev['n_events']
    return 2 * prec * rec / (prec + rec), ev


def calib_score(labels, edges, gt):
    """
    مقياس المعايرة = المتوسط التوافقي بين:
      • استدعاء الأحداث  = مسكنا كام حركة من الحركات الحقيقية؟
      • دقة السكون       = قعدنا ساكتين كام % من وقت السكون؟

    🐛 ليه مش الدقة الإجمالية ولا F1 الحدث لوحده؟ الاتنين **قابلين للغش**
       في اتجاهين متعاكسين، وإحنا وقعنا في التانية فعلاً:

       • الدقة الإجمالية: "قول other على طول" -> 95.7% على vidtest3
         من غير ما تمسك ولا حركة.

       • F1 الحدث لوحده: الإنذار الكاذب بيتحسب **قطعة** مش بمدته. يعني
         إنذار طوله 8 ثواني تكلفته زي إنذار طوله نص ثانية. فالبحث اكتشف
         إن "حط لابل على كل حاجة" بيدّي استدعاء عالي بتكلفة شبه صفر.
         التشغيلة الأولى عملت كده حرفياً: 2/4 أحداث بس دقة سكون 0/1210.

       المقياس ده مايتغشّش في الاتجاهين: لو لبّست كل حاجة -> دقة السكون
       صفر. لو سكتّ خالص -> الاستدعاء صفر. المتوسط التوافقي بيعاقب
       بشدة لو أي طرف واطي.
    """
    ev = event_metrics(labels, edges, gt, tolerance=TOLERANCE)
    fm = frame_metrics(labels, edges, gt)

    rec = ev['n_detected'] / max(1, ev['n_events'])
    spec = fm['other_hit'] / max(1, fm['other_total'])

    if rec <= 0 or spec <= 0:
        return 0.0
    return 2 * rec * spec / (rec + spec)


def gate_of(E, mult):
    """
    بوابة الطاقة = وسيط طاقة **الفيديو نفسه** × مُعامل.

    ⚠️ ليه نسبية مش رقم مطلق؟ اتقاس: وسيط الطاقة في vidtest1 = 0.371
       وفي vidtest2 = 1.601 — فرق 4.3× رغم إن الاتنين **نفس الكاميرا
       ونفس المعدل ونفس الشخص**. يعني الفرق من خصائص المشهد نفسه (زاوية
       الجسم، جودة اكتشاف المفاصل)، مش حاجة نقدر نلغيها بالتطبيع.

       نتيجة كده: أي بوابة مطلقة مستحيل تنتقل. الرقم 0.527 اللي اتعاير
       عليه كان فوق q65 بتاعة vidtest1 وتحت q05 بتاعة vidtest2 —
       يعني حرفياً بيعمل حاجتين مختلفتين على الفيديوهين.

       المُعامل النسبي هو اللي بينتقل. وده **مش تسريب**: الوسيط
       بيتحسب من الـ keypoints الخام بتاعة الفيديو، من غير أي ground
       truth — نفس فكرة تطبيع الإضاءة في معالجة الصور.
    """
    return float(np.median(E)) * mult


def postprocess(raw):
    return drop_short_segments(majority_smooth(raw, SMOOTH_K), MIN_SEGMENT)


def grid_search(folds, gates, thresholds):
    """
    بيدوّر على أحسن (بوابة، عتبة) بمتوسط F1 عبر كل الـ folds.

    folds: list من (D, energy, tmpl_labels, edges, gt)
    البحث ده سريع لأن الـ D متحسوبة قبل كده — مفيش DTW هنا خالص.
    """
    best = (-1.0, None, None)
    table = []

    for g in gates:
        for th in thresholds:
            scores = []
            for D, E, tl, edges, gt in folds:
                labels, _, _ = classify(D, E, tl, gate_of(E, g), th)
                scores.append(calib_score(postprocess(labels), edges, gt))
            m = float(np.mean(scores))
            table.append((g, th, m, scores))
            if m > best[0]:
                best = (m, g, th)

    return best, table


# ==============================================================================
# التشغيل
# ==============================================================================

def prepare(name, template_source_videos, mode='vel', shape_norm=True,
            scales=SCALES, verbose=True):
    """بيجهّز فيديو واحد: نوافذ + templates من فيديوهات تانية + مصفوفة المسافات."""
    kp, fps = load(name)
    centers = np.arange(0, len(kp), STRIDE)
    edges = window_edges(centers, fps, len(kp))

    templates = []
    for src in template_source_videos:
        skp, sfps = load(src)
        templates += cut_templates(skp, sfps, TEMPLATE_CLIPS[src], source=src,
                                   mode=mode, shape_norm=shape_norm,
                                   scales=scales)
    templates = balance_templates(templates, MAX_PER_LABEL)

    feats, energy = build_multiscale_windows(kp, fps, centers, scales,
                                             mode=mode, shape_norm=shape_norm)

    # تخزين المسافات على القرص — حسابها بياخد دقايق، والمعايرة بتتكرر كتير.
    # المفتاح فيه كل حاجة بتغيّر النتيجة، فأي تعديل في الإعدادات بيبطّل الكاش.
    key = (f'{name}_{"-".join(template_source_videos)}_{mode}_'
           f'{int(shape_norm)}_{"x".join(map(str, scales))}_'
           f's{STRIDE}_r{RADIUS}_m{MAX_PER_LABEL}')
    cache_dir = Path('_scratch/dtw_cache')
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f'{key}.npy'

    if cache_file.exists():
        D = np.load(cache_file)
        if verbose:
            print(f'  {name}: {D.size:,} مسافة من الكاش')
    else:
        t0 = time.perf_counter()
        D = distance_cache(feats, templates, scales, radius=RADIUS)
        el = time.perf_counter() - t0
        np.save(cache_file, D)
        if verbose:
            print(f'  {name}: {len(centers)} نافذة × {len(scales)} مقاس × '
                  f'{len(templates)} template = {D.size:,} مقارنة DTW '
                  f'في {el:.0f}s ({el / D.size * 1000:.2f}ms للواحدة)')

    # ⚠️ GT خاصة بالمعايرة: الأحداث اللي مفيش لها template أصلاً بتتعلّم '?'
    #    (يعني تتستثنى). في fold الأول مثلاً، vidtest1 بياخد templates من
    #    vidtest2 بس — فمفيش عنده 'wave' ولا 'rub_hands' خالص. لو حسبناهم
    #    ضمن الأحداث الفايتة، بنعاقب النظام على حاجة **مستحيلة هيكلياً**،
    #    وده بيشوّه اختيار العتبة. التقييم النهائي بيستخدم الـ GT الكاملة.
    have = set(t['label'] for t in templates)
    gt_cal = [(s, e, lab if (lab in have or lab == 'other*' or lab == '?')
               else '?') for s, e, lab in GT[name]]

    return {'name': name, 'D': D, 'E': energy, 'edges': edges,
            'tl': [t['label'] for t in templates], 'templates': templates,
            'gt': GT[name], 'gt_cal': gt_cal, 'fps': fps, 'n_frames': len(kp)}


def report(p, gate_mult, th, title):
    g = gate_of(p['E'], gate_mult)
    labels = postprocess(classify(p['D'], p['E'], p['tl'], g, th)[0])
    fm = frame_metrics(labels, p['edges'], p['gt'])
    f1, ev = event_f1(labels, p['edges'], p['gt'])

    base = fm['other_total'] / max(1, fm['total'])
    acc = fm['hit'] / max(1, fm['total'])

    print(f'\n{"=" * 72}\n  {title}\n{"=" * 72}')
    print(f'  البوابة الفعلية = {g:.3f}  (وسيط الفيديو '
          f'{np.median(p["E"]):.3f} × {gate_mult})')
    print(f'  الدقة الإجمالية        {acc * 100:5.1f}%   '
          f'(خط الأساس "other دايماً" = {base * 100:.1f}%)')
    print(f'  {"↑ فوق خط الأساس" if acc > base else "↓ تحت خط الأساس"} '
          f'بـ {abs(acc - base) * 100:.1f} نقطة')
    print(f'  استدعاء الحركات        {fm["real_hit"]}/{fm["real_total"]} '
          f'({fm["real_hit"] / max(1, fm["real_total"]) * 100:.1f}%)')
    print(f'  دقة السكون             {fm["other_hit"]}/{fm["other_total"]} '
          f'({fm["other_hit"] / max(1, fm["other_total"]) * 100:.1f}%)')
    print(f'\n  --- مستوى الحدث (الأهم) ---')
    print(f'  حركات متكشّفة           {ev["n_detected"]}/{ev["n_events"]}')
    print(f'  إنذارات كاذبة          {ev["n_false_alarms"]}')
    print(f'  F1                     {f1:.3f}')
    for e in ev['events']:
        s, en = e['span']
        print(f'    {"✅" if e["detected"] else "❌"} {e["label"]:11} '
              f'[{s:6.1f} - {en:6.1f}]')
    if ev['false_alarms']:
        print('  الإنذارات الكاذبة:')
        for s, en, lab in ev['false_alarms'][:8]:
            print(f'    ⚠️  {lab:11} [{s:6.1f} - {en:6.1f}]')
        if len(ev['false_alarms']) > 8:
            print(f'    ... و{len(ev["false_alarms"]) - 8} كمان')
    return {'acc': acc, 'base': base, 'f1': f1, 'ev': ev, 'fm': fm,
            'labels': labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ablate', action='store_true')
    args = ap.parse_args()

    print('=' * 72)
    print('  Few-Shot FastDTW — من غير أي داتاسِت')
    print('=' * 72)

    # ---- تجهيز الـ folds: كل فيديو بـ templates من التانيين -------------
    print('\n📐 تجهيز النوافذ وحساب المسافات...')
    p1 = prepare('vidtest1', ['vidtest2'])
    p2 = prepare('vidtest2', ['vidtest1'])
    p3 = prepare('vidtest3', ['vidtest1', 'vidtest2'])

    n_by_label = {}
    for t in p3['templates']:
        n_by_label[t['label']] = n_by_label.get(t['label'], 0) + 1
    print(f'\n📚 بنك الـ templates لـ vidtest3: {n_by_label} '
          f'(المجموع {len(p3["templates"])})')

    # ---- المعايرة على vidtest1 + vidtest2 بس ---------------------------
    all_e = np.concatenate([p1['E'].ravel(), p2['E'].ravel()])
    all_d = np.concatenate([p1['D'].ravel(), p2['D'].ravel()])
    # البوابة **معامل نسبي** لوسيط طاقة كل فيديو (بص gate_of).
    # العتبة تفضل مطلقة — توزيعات المسافة اتظبطت بعد إصلاح الاستيفاء
    # (vidtest3 من 0.800 لـ 1.039 مقابل 1.084 و 1.147).
    gates = np.array([0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4, 2.8, 3.5])
    thresholds = np.unique(np.percentile(all_d, [0.2, 0.5, 1, 2, 3, 5, 8, 12,
                                                 18, 25, 35, 50]))

    print(f'\n🔍 معايرة على vidtest1+vidtest2 '
          f'({len(gates)}×{len(thresholds)} تركيبة)...')
    folds = [(p['D'], p['E'], p['tl'], p['edges'], p['gt_cal'])
             for p in (p1, p2)]
    (best_f1, gate, th), table = grid_search(folds, gates, thresholds)

    print(f'  ✓ أحسن معامل بوابة = {gate:.2f} × وسيط طاقة الفيديو')
    print(f'  ✓ أحسن عتبة مسافة  = {th:.3f}')
    print(f'  ✓ أحسن نتيجة معايرة = {best_f1:.3f}')
    print(f'  أحسن 5 تركيبات (بوابة، عتبة، النتيجة):')
    for g, t, m, sc in sorted(table, key=lambda r: -r[2])[:5]:
        edge = ''
        if g in (gates[0], gates[-1]) or t in (thresholds[0], thresholds[-1]):
            edge = '  ⚠️ على طرف النطاق'
        print(f'    {g:7.3f}  {t:7.3f}  {m:.3f}  '
              f'[{", ".join(f"{s:.2f}" for s in sc)}]{edge}')

    # ---- النتايج -------------------------------------------------------
    report(p1, gate, th, 'vidtest1  (templates من vidtest2)')
    report(p2, gate, th, 'vidtest2  (templates من vidtest1)')
    r3 = report(p3, gate, th,
                '⭐ vidtest3  (templates من vidtest1+2 — مشافش نفسه خالص)')

    print(f'\n{"=" * 72}')
    print('  الخلاصة — vidtest3')
    print('=' * 72)
    print(f'  مسكنا {r3["ev"]["n_detected"]} من {r3["ev"]["n_events"]} حركات، '
          f'{r3["ev"]["n_false_alarms"]} إنذار كاذب')
    print(f'  الدقة {r3["acc"] * 100:.1f}% مقابل خط أساس '
          f'{r3["base"] * 100:.1f}%')

    if args.ablate:
        ablate(gate, th)


def ablate(gate, th):
    """إيه اللي ساعد فعلاً؟ نشيل مكوّن ونشوف vidtest3 بيحصله إيه."""
    print(f'\n{"=" * 72}\n  تجارب الحذف — إيه اللي ساعد فعلاً\n{"=" * 72}')

    variants = [
        ('كامل',                    dict(mode='vel', shape_norm=True,  scales=SCALES)),
        ('من غير فروق (مواضع)',      dict(mode='pos', shape_norm=True,  scales=SCALES)),
        ('من غير تطبيع الشكل',       dict(mode='vel', shape_norm=False, scales=SCALES)),
        ('مقاس واحد 3s بس',          dict(mode='vel', shape_norm=True,  scales=(3.0,))),
    ]

    for name, kw in variants:
        p = prepare('vidtest3', ['vidtest1', 'vidtest2'], verbose=False, **kw)
        labels = postprocess(classify(p['D'], p['E'], p['tl'], gate, th)[0])
        f1, ev = event_f1(labels, p['edges'], p['gt'])
        fm = frame_metrics(labels, p['edges'], p['gt'])
        print(f'  {name:24} F1={f1:.3f}  '
              f'كشف={ev["n_detected"]}/{ev["n_events"]}  '
              f'كاذب={ev["n_false_alarms"]}  '
              f'دقة={fm["hit"] / max(1, fm["total"]) * 100:.1f}%')

    print('\n  ⚠️ البوابة والعتبة هنا متثبّتة من الإعداد الكامل، فالمتغيّرات')
    print('     التانية مش معايرة لنفسها — ده بيقلّل أرقامها شوية. الغرض')
    print('     ترتيب الأهمية، مش رقم نهائي لكل متغيّر.')


if __name__ == '__main__':
    main()
