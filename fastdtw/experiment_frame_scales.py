"""
تجربة: عدد الفريمات في الـ template/window بيأثر إزاي على "ثقة" FastDTW؟

10 إعدادات متوازية بالظبط زي بعض في كل حاجة إلا عدد الفريمات:
10، 20، 30 ... لحد 100 فريم (فاصل 10). الـ templates كلها **خارجية** جايه من
5 مصادر مختلفة (build_external_templates.py) — مفيش ولا فريم من نفس
الفيديو اللي بنقيس عليه، عشان صفر تسريب. الاختبار على القصاصات الحقيقية
من vidtest1-4 عند الأزمنة المعروفة (ground truth) بس للحركات اللي عندها
template خارجي مطابق (18 حركة، قصاصتين لكل واحدة — راجع docstring
build_external_templates.py للتفاصيل).

الثقة (confidence) بتتحسب بطريقة الـ margin:
    confidence = (المسافة للحركة التانية - المسافة لأقرب حركة) / المسافة للحركة التانية
- قريب من 1  → واثق جداً (أقرب حركة أبعد بكتير عن أي حركة تانية).
- قريب من 0  → مش واثق (أقرب حركتين قريبين من بعض، ممكن يبقى في لخبطة).

ملحوظة مهمة: 'sitting' و 'sit_down' بيتعاملوا كحركة واحدة هنا، لأن
HMDB51 عنده فئة عامة واحدة بس اسمها 'sit' — مفيش تمييز عندهم بين
"يجلس" و"قاعد فعلاً"، فمنطقياً بيتقاسوا على نفس الـ template.

الاستخدام (على Kaggle، بعد ما build_external_templates.py يتشغّل):
    python experiment_frame_scales.py
"""
import sys
import time

import numpy as np

import _bootstrap  # noqa: F401
from classifier import norm_distance, normalize_window, resample_linear, to_features
from ground_truth import VIDEOS, spans
from paths import KP_OUT, load_keypoints, out_dir

if hasattr(sys.stdout, 'reconfigure'):
    # ⚠️ line_buffering=True مش رفاهية: من غيرها بايثون بيجمّع المخرجات في
    # ذاكرة مؤقتة (8KB) لما تكون رايحة لأنبوبة مش لشاشة — زي `!python x.py`
    # في نوتبوك Kaggle. السكريبت ده بيطبع ~400 بايت بعد كل إعداد، يعني كان
    # بيخلّص الجدول الأول كله قبل ما يظهر حرف واحد، والتشغيلة شكلها واقفة
    # عشر دقايق وهي شغّالة عادي. حصل فعلاً 2026-09-18.
    sys.stdout.reconfigure(encoding='utf-8', errors='replace',
                           line_buffering=True)

FRAME_CONFIGS = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
RADIUS = 1
EXT_DIR = KP_OUT / 'external'
OUT_DIR = out_dir(__file__)

# حركتنا → أسامي الـ ground truth اللي بتتحسب عليها (فيه استثناء واحد:
# sitting بتتقاس كمان على sit_down لأنهم نفس الحركة الفيزيائية).
GT_LABEL_ALIASES = {
    'sitting': ['sitting', 'sit_down'],
}


def load_external_templates():
    manifest = np.load(EXT_DIR / 'manifest.npy', allow_pickle=True)
    templates = []
    for m in manifest:
        kp = np.load(EXT_DIR / m['file'])
        templates.append({'label': m['label'], 'clip': m['clip'], 'kp': kp})
    return templates


def build_templates(ext_templates, n_frames):
    out = []
    for t in ext_templates:
        seq = resample_linear(normalize_window(t['kp']), n=n_frames)
        feat = to_features(seq, mode='vel', shape_norm=True)
        out.append({'label': t['label'], 'feat': feat, 'clip': t['clip']})
    return out


def build_test_windows(labels_needed, n_frames):
    """نافذة واحدة لكل ظهور فعلي للحركة، بحجم الفترة الحقيقية من الـ ground truth."""
    windows = []
    for v in VIDEOS:
        kp, fps = load_keypoints(v)
        for label in labels_needed:
            for gt_label in GT_LABEL_ALIASES.get(label, [label]):
                for t0, t1 in spans(v, gt_label):
                    lo, hi = int(round(t0 * fps)), int(round(t1 * fps))
                    clip = kp[max(0, lo):min(len(kp), hi)]
                    if len(clip) < 4:
                        continue
                    seq = resample_linear(normalize_window(clip), n=n_frames)
                    feat = to_features(seq, mode='vel', shape_norm=True)
                    windows.append({
                        'true': label, 'gt_label': gt_label, 'feat': feat,
                        'video': v, 'span': (t0, t1),
                    })
    return windows


def classify_with_confidence(window_feat, templates, radius=RADIUS):
    per_label_best = {}
    for t in templates:
        d = norm_distance(window_feat, t['feat'], radius=radius)
        if t['label'] not in per_label_best or d < per_label_best[t['label']]:
            per_label_best[t['label']] = d

    ranked = sorted(per_label_best.items(), key=lambda kv: kv[1])
    best_label, best_d = ranked[0]
    if len(ranked) > 1 and ranked[1][1] > 1e-9:
        second_d = ranked[1][1]
        confidence = (second_d - best_d) / second_d
    else:
        confidence = 1.0
    return best_label, best_d, confidence


def calibrate_templates(templates, radius=RADIUS):
    """
    لكل template بنحسب "معدل بعده الطبيعي" عن باقي الـ29 template التانية
    (leave-one-out بين الـ templates الخارجية نفسها — مفيش أي تسريب من
    vidtest). نفس فكرة تطبيع TIER2 القديمة: بعض القصاصات (زي تسريح شعر
    مصوّر قريب) مسافتها عن أي حاجة بتبقى كبيرة/صغيرة بشكل عام مش لأنها
    قريبة فعلاً من الحركة الصح، فبنطبّعها بمتوسطها وانحرافها الخاص قبل
    ما نقارن.
    """
    n = len(templates)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = norm_distance(templates[i]['feat'], templates[j]['feat'], radius=radius)
            D[i, j] = D[j, i] = d

    mu = np.zeros(n)
    sigma = np.ones(n)
    for j in range(n):
        others = np.delete(D[:, j], j)
        mu[j] = others.mean()
        sigma[j] = others.std() + 1e-6
    return mu, sigma


def classify_with_confidence_z(window_feat, templates, mu, sigma, radius=RADIUS):
    """زي classify_with_confidence بس بيقارن بالمسافة المطبّعة (Z) مش الخام."""
    per_label_best = {}
    for idx, t in enumerate(templates):
        d = norm_distance(window_feat, t['feat'], radius=radius)
        z = (d - mu[idx]) / sigma[idx]
        if t['label'] not in per_label_best or z < per_label_best[t['label']]:
            per_label_best[t['label']] = z

    ranked = sorted(per_label_best.items(), key=lambda kv: kv[1])
    best_label, best_z = ranked[0]
    # الفرق مباشرة (مش نسبة) لأن Z ممكن يبقى بالسالب
    confidence = ranked[1][1] - best_z if len(ranked) > 1 else 1.0
    return best_label, best_z, confidence


def run_config(n_frames, ext_templates, labels_needed):
    templates = build_templates(ext_templates, n_frames)
    windows = build_test_windows(labels_needed, n_frames)

    results = []
    for w in windows:
        pred, dist, conf = classify_with_confidence(w['feat'], templates)
        results.append({**w, 'pred': pred, 'dist': dist, 'confidence': conf})
    return results


def run_config_z(n_frames, ext_templates, labels_needed):
    templates = build_templates(ext_templates, n_frames)
    mu, sigma = calibrate_templates(templates)
    windows = build_test_windows(labels_needed, n_frames)

    results = []
    for w in windows:
        pred, dist, conf = classify_with_confidence_z(w['feat'], templates, mu, sigma)
        results.append({**w, 'pred': pred, 'dist': dist, 'confidence': conf})
    return results


def summarize(n_frames, results, title='📐'):
    n = len(results)
    correct = sum(1 for r in results if r['pred'] == r['true'])
    acc = correct / n * 100 if n else 0.0
    wrong = [r for r in results if r['pred'] != r['true']]

    avg_conf = float(np.mean([r['confidence'] for r in results])) if n else 0.0
    avg_conf_correct = float(np.mean([r['confidence'] for r in results if r['pred'] == r['true']])) if correct else 0.0
    avg_conf_wrong = float(np.mean([r['confidence'] for r in wrong])) if wrong else 0.0

    print(f'\n{"=" * 70}')
    print(f'  {title} {n_frames} فريم')
    print(f'{"=" * 70}')
    print(f'  الدقة: {correct}/{n} = {acc:.1f}%')
    print(f'  متوسط الثقة (كل النوافذ): {avg_conf:.3f}')
    print(f'  متوسط الثقة (لما التصنيف يصح): {avg_conf_correct:.3f}')
    print(f'  متوسط الثقة (لما التصنيف يغلط): {avg_conf_wrong:.3f}')

    return {
        'n_frames': n_frames, 'n': n, 'correct': correct, 'accuracy': acc,
        'avg_confidence': avg_conf,
        'avg_confidence_correct': avg_conf_correct,
        'avg_confidence_wrong': avg_conf_wrong,
    }


def main():
    print('=' * 70)
    print('  🎬 تجربة عدد الفريمات: من 10 لحد 100 (فاصل 10)')
    print('  (templates خارجية من HMDB51 — اختبار على vidtest1-4)')
    print('=' * 70)

    ext_templates = load_external_templates()
    labels_needed = sorted({t['label'] for t in ext_templates})
    print(f'\n📚 القصاصات الخارجية: {len(ext_templates)} '
          f'({len(labels_needed)} حركة: {", ".join(labels_needed)})')

    # حجم الشغل قدّام المستخدم من أول لحظة — أرخص من إنه يقعد يخمّن هيقف
    # امتى. النسخة المطبّعة بتزوّد معايرة (كل template ضد كل template).
    n_windows = len(build_test_windows(labels_needed, FRAME_CONFIGS[0]))
    n_tpl = len(ext_templates)
    per_config = 2 * n_windows * n_tpl + n_tpl * (n_tpl - 1) // 2
    print(f'🧮 {n_windows} نافذة اختبار × {n_tpl} قصاصة × '
          f'{len(FRAME_CONFIGS)} إعداد × نسختين (خام + مطبّع) = '
          f'{per_config * len(FRAME_CONFIGS):,} مقارنة DTW')
    print('   الإعدادات الكبيرة أبطأ بكتير من الصغيرة — التقدّم مش بالتساوي')

    t_start = time.perf_counter()
    summaries = []
    for i, n_frames in enumerate(FRAME_CONFIGS, 1):
        t0 = time.perf_counter()
        print(f'\n⏳ [{i}/{len(FRAME_CONFIGS)}] {n_frames} فريم (خام)...',
              end='', flush=True)
        results = run_config(n_frames, ext_templates, labels_needed)
        print(f' تمّ في {time.perf_counter() - t0:.0f}s '
              f'(إجمالي {time.perf_counter() - t_start:.0f}s)')
        summaries.append(summarize(n_frames, results))

    print(f'\n{"=" * 70}')
    print('  📊 المقارنة النهائية (بدون تطبيع)')
    print(f'{"=" * 70}')
    print(f'  {"فريمات":<10}{"دقة%":<10}{"ثقة عامة":<12}{"ثقة (صح)":<12}{"ثقة (غلط)":<12}')
    for s in summaries:
        print(f'  {s["n_frames"]:<10}{s["accuracy"]:<10.1f}'
              f'{s["avg_confidence"]:<12.3f}{s["avg_confidence_correct"]:<12.3f}'
              f'{s["avg_confidence_wrong"]:<12.3f}')

    # ── نفس الشيء بس بتطبيع Z (كل template يتقاس بمعدل بعده الطبيعي عن
    # باقي الـ templates الخارجية) — تجربة TIER2-style لمحاولة رفع الدقة.
    print(f'\n{"=" * 70}')
    print('  🧪 نفس التجربة بس بتطبيع Z (z-normalization) على كل template')
    print(f'{"=" * 70}')
    summaries_z = []
    for i, n_frames in enumerate(FRAME_CONFIGS, 1):
        t0 = time.perf_counter()
        print(f'\n⏳ [{i}/{len(FRAME_CONFIGS)}] {n_frames} فريم (مطبّع)...',
              end='', flush=True)
        results_z = run_config_z(n_frames, ext_templates, labels_needed)
        print(f' تمّ في {time.perf_counter() - t0:.0f}s '
              f'(إجمالي {time.perf_counter() - t_start:.0f}s)')
        summaries_z.append(summarize(n_frames, results_z, title='🧪 (مطبّع)'))

    print(f'\n{"=" * 70}')
    print('  📊 المقارنة النهائية: بدون تطبيع مقابل بتطبيع Z')
    print(f'{"=" * 70}')
    print(f'  {"فريمات":<10}{"دقة% (خام)":<14}{"دقة% (مطبّع)":<14}')
    for s, sz in zip(summaries, summaries_z):
        print(f'  {s["n_frames"]:<10}{s["accuracy"]:<14.1f}{sz["accuracy"]:<14.1f}')

    np.save(OUT_DIR / 'frame_scale_summary.npy', summaries, allow_pickle=True)
    np.save(OUT_DIR / 'frame_scale_summary_zscore.npy', summaries_z, allow_pickle=True)
    print(f'\n✅ النتيجة اتحفظت في {OUT_DIR / "frame_scale_summary.npy"}')
    print(f'✅ نتيجة التطبيع اتحفظت في {OUT_DIR / "frame_scale_summary_zscore.npy"}')


if __name__ == '__main__':
    main()
