"""
تجربة: عدد الفريمات في الـ template/window بيأثر إزاي على "ثقة" FastDTW؟

3 إعدادات متوازية بالظبط زي بعض في كل حاجة إلا عدد الفريمات:
10 فريم، 20 فريم، 30 فريم. الـ templates كلها **خارجية** جايه من
HMDB51 (build_external_templates.py) — مفيش ولا فريم من نفس الفيديو
اللي بنقيس عليه، عشان صفر تسريب. الاختبار على القصاصات الحقيقية من
vidtest1-4 عند الأزمنة المعروفة (ground truth) بس للحركات العشرة اللي
عندها template خارجي مطابق.

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

import numpy as np

import _bootstrap  # noqa: F401
from classifier import norm_distance, normalize_window, resample_linear, to_features
from ground_truth import VIDEOS, spans
from paths import KP_OUT, load_keypoints, out_dir

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

FRAME_CONFIGS = (10, 20, 30)
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


def run_config(n_frames, ext_templates, labels_needed):
    templates = build_templates(ext_templates, n_frames)
    windows = build_test_windows(labels_needed, n_frames)

    results = []
    for w in windows:
        pred, dist, conf = classify_with_confidence(w['feat'], templates)
        results.append({**w, 'pred': pred, 'dist': dist, 'confidence': conf})
    return results


def summarize(n_frames, results):
    n = len(results)
    correct = sum(1 for r in results if r['pred'] == r['true'])
    acc = correct / n * 100 if n else 0.0
    wrong = [r for r in results if r['pred'] != r['true']]

    avg_conf = float(np.mean([r['confidence'] for r in results])) if n else 0.0
    avg_conf_correct = float(np.mean([r['confidence'] for r in results if r['pred'] == r['true']])) if correct else 0.0
    avg_conf_wrong = float(np.mean([r['confidence'] for r in wrong])) if wrong else 0.0

    print(f'\n{"=" * 70}')
    print(f'  📐 {n_frames} فريم')
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
    print('  🎬 تجربة عدد الفريمات: 10 مقابل 20 مقابل 30')
    print('  (templates خارجية من HMDB51 — اختبار على vidtest1-4)')
    print('=' * 70)

    ext_templates = load_external_templates()
    labels_needed = sorted({t['label'] for t in ext_templates})
    print(f'\n📚 القصاصات الخارجية: {len(ext_templates)} '
          f'({len(labels_needed)} حركة: {", ".join(labels_needed)})')

    summaries = []
    for n_frames in FRAME_CONFIGS:
        results = run_config(n_frames, ext_templates, labels_needed)
        summaries.append(summarize(n_frames, results))

    print(f'\n{"=" * 70}')
    print('  📊 المقارنة النهائية')
    print(f'{"=" * 70}')
    print(f'  {"فريمات":<10}{"دقة%":<10}{"ثقة عامة":<12}{"ثقة (صح)":<12}{"ثقة (غلط)":<12}')
    for s in summaries:
        print(f'  {s["n_frames"]:<10}{s["accuracy"]:<10.1f}'
              f'{s["avg_confidence"]:<12.3f}{s["avg_confidence_correct"]:<12.3f}'
              f'{s["avg_confidence_wrong"]:<12.3f}')

    np.save(OUT_DIR / 'frame_scale_summary.npy', summaries, allow_pickle=True)
    print(f'\n✅ النتيجة اتحفظت في {OUT_DIR / "frame_scale_summary.npy"}')


if __name__ == '__main__':
    main()
