"""
تجربة: Ensemble بالتصويت الموزون بين الـ 10 classifiers (10, 20 ... 100 فريم).

الفكرة: كل classifier (عدد فريمات) بيشوف الحركة بمقياس مختلف — الصغير
بيلقط الحركات القصيرة، والكبير بيلقط الطويلة. بدل ما نختار واحد، كلهم
بيصوّتوا وكل صوت ليه وزن.

⚠️ الأوزان بتتحسب بـ leave-one-video-out: وإحنا بنقيس على vidtest1، الأوزان
   جاية من دقة كل classifier على vidtest2-4 بس — عمرها ما شافت الفيديو اللي
   بنقيس عليه. لو الأوزان اتحسبت من نفس الدقة اللي بنقيسها يبقى غش.

جزئين:
  أ) نفس اختبار experiment_frame_scales.py بالظبط (40 قصاصة GT) — عشان
     الرقم يتقارن مباشرة بأحسن classifier لوحده (70 فريم Z = 30%).
     هنا "عدد الفريمات" = دقة إعادة التعيين للقصاصة.
  ب) نافذة منزلقة على الفيديو كله (زي الفيديوهات المرسومة) — هنا "عدد
     الفريمات" = طول النافذة فعلاً (قصيرة ← → طويلة)، والدقة = نسبة
     اللحظات المتعلّمة اللي اتصنّفت صح. وبترسم 4 فيديوهات ensemble.

طرق الدمج (كلها Z-normalized):
  majority       كل classifier صوت واحد
  weighted       الوزن = دقة الـ classifier على الفيديوهات التانية
  weighted_conf  الوزن × الثقة (الفرق بين أقرب حركتين)
  per_class      وزن لكل (classifier، حركة): دقته لما بيقول الحركة دي —
                 يعني كل مقياس بيتصدّق في الحركات اللي بيعرفها بس
  soft           متوسط Z موزون لكل حركة بدل الأصوات

التشغيل:
    python experiment_ensemble.py            # أ + ب + رسم الفيديوهات
    python experiment_ensemble.py --no-render
متغيرات: RENDER_OUT (فولدر الفيديوهات)، WORKERS
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace',
                           line_buffering=True)

import _bootstrap  # noqa: F401
import experiment_frame_scales as E
from classifier import norm_distance, normalize_window, resample_linear, to_features
from ground_truth import VIDEOS, label_at
from paths import load_keypoints, out_dir

FRAME_CONFIGS = E.FRAME_CONFIGS
RADIUS = E.RADIUS
OUT_DIR = out_dir(__file__)
STEPS_PER_SEC = 5               # نقطة تقييم كل 0.2 ثانية في الجزء ب
PRIOR = 2                       # تنعيم أوزان per_class (عدد أصوات وهمية)
GT_ALIAS = {'sit_down': 'sitting'}
RENDER_METHOD = 'weighted'      # الطريقة اللي بتترسم في الفيديوهات

METHODS = [
    ('best_single', 'أحسن classifier لوحده (متختار من الفيديوهات التانية)'),
    ('majority', 'تصويت عادي (كل classifier صوت)'),
    ('weighted', 'تصويت موزون (الوزن = دقة الـ classifier)'),
    ('weighted_conf', 'تصويت موزون × الثقة'),
    ('per_class', 'وزن لكل حركة (كل مقياس في اللي بيعرفه)'),
    ('soft', 'تصويت ناعم (متوسط Z موزون)'),
]


def _label_z(D, mu, sigma, labels, uniq):
    """مسافات (نوافذ × قوالب) → أحسن Z لكل حركة (نوافذ × حركات). NaN يفضل NaN."""
    Z = (D - mu) / sigma
    return np.stack([Z[:, [i for i, l in enumerate(labels) if l == u]].min(axis=1)
                     for u in uniq], axis=1)


def _setup(n):
    ext = E.load_external_templates()
    tpl = E.build_templates(ext, n)
    mu, sigma = E.calibrate_templates(tpl)
    labels = [t['label'] for t in tpl]
    return tpl, mu, sigma, labels, sorted(set(labels))


# ─────────────────── أ) الـ 40 قصاصة GT ───────────────────

def gt_windows_job(n):
    tpl, mu, sigma, labels, uniq = _setup(n)
    wins = E.build_test_windows(uniq, n)
    D = np.array([[norm_distance(w['feat'], t['feat'], radius=RADIUS) for t in tpl]
                  for w in wins])
    return (n, _label_z(D, mu, sigma, labels, uniq),
            [w['true'] for w in wins], [w['video'] for w in wins])


# ─────────────────── ب) نافذة منزلقة على الفيديو ───────────────────

def centers_of(video):
    kp, fps = load_keypoints(video)
    step = max(1, int(round(fps / STEPS_PER_SEC)))
    return np.arange(0, len(kp), step), fps, len(kp)


def sliding_job(args):
    """نافذة طولها n فريم متمركزة عند كل نقطة تقييم. بره الفيديو = NaN (مابيصوّتش)."""
    video, n = args
    tpl, mu, sigma, labels, uniq = _setup(n)
    kp, _ = load_keypoints(video)
    centers, _, _ = centers_of(video)
    D = np.full((len(centers), len(tpl)), np.nan)
    for i, c in enumerate(centers):
        s = c - n // 2
        if s < 0 or s + n > len(kp):
            continue
        seq = resample_linear(normalize_window(kp[s:s + n]), n=n)
        feat = to_features(seq, mode='vel', shape_norm=True)
        D[i] = [norm_distance(feat, t['feat'], radius=RADIUS) for t in tpl]
    return video, n, _label_z(D, mu, sigma, labels, uniq)


# ─────────────────── الدمج ───────────────────

def combine(S, valid, pk, W, method):
    """S: (K مقياس، N نافذة، L حركة) Z. بيرجّع رقم الحركة لكل نافذة (-1 = مفيش أصوات)."""
    K, N, L = S.shape
    Wk = W if W.ndim == 1 else W.mean(axis=1)
    num = (np.where(valid[..., None], S, 0) * Wk[:, None, None]).sum(0)
    den = (valid * Wk[:, None]).sum(0)
    soft = -num / np.maximum(den, 1e-9)[:, None]

    if method == 'soft':
        score = soft
    else:
        wk = W[np.arange(K)[:, None], pk] if W.ndim == 2 else np.repeat(W[:, None], N, 1)
        if method == 'weighted_conf':
            with np.errstate(invalid='ignore'):
                srt = np.sort(np.where(valid[..., None], S, np.inf), axis=2)
                wk = wk * np.where(valid, srt[:, :, 1] - srt[:, :, 0], 0)
        wk = np.where(valid, wk, 0)
        score = np.zeros((N, L))
        for k in range(K):
            np.add.at(score, (np.arange(N), pk[k]), wk[k])
        score = score + 1e-6 * soft          # كسر التعادل بالـ Z

    pred = score.argmax(axis=1)
    pred[~valid.any(axis=0)] = -1
    return pred


def lovo_predict(S, y, vids, method):
    """
    لكل فيديو: الأوزان من الفيديوهات التانية بس، وبعدين التوقّع عليه.
    بيرجّع (التوقّعات، الأوزان المستخدمة لكل فيديو).
    """
    K, N, L = S.shape
    valid = ~np.isnan(S[..., 0])
    pk = np.where(valid[..., None], S, np.inf).argmin(axis=2)
    pred = np.full(N, -1)
    weights = {}
    for v in np.unique(vids):
        test = vids == v
        train = ~test & (y >= 0)
        acc = np.array([(pk[k][train] == y[train]).mean() if train.any() else 0.0
                        for k in range(K)])     # نافذة مالهاش صوت = غلط
        if method == 'best_single':
            k = int(acc.argmax())
            p = np.where(valid[k], pk[k], -1)
            pred[test] = p[test]
            weights[v] = FRAME_CONFIGS[k]
            continue
        if method == 'majority':
            W = np.ones(K)
        elif method == 'per_class':
            W = np.zeros((K, L))
            for k in range(K):
                m = train & valid[k]
                for l in range(L):
                    said = m & (pk[k] == l)
                    W[k, l] = ((y[said] == l).sum() + PRIOR * acc[k]) / (said.sum() + PRIOR)
        else:
            W = acc + 1e-3
        pred[test] = combine(S[:, test], valid[:, test], pk[:, test], W, method)
        weights[v] = W
    return pred, weights


def accuracy(pred, y):
    m = y >= 0
    return float((pred[m] == y[m]).mean() * 100) if m.any() else 0.0


def per_scale_acc(S, y):
    valid = ~np.isnan(S[..., 0])
    pk = np.where(valid[..., None], S, np.inf).argmin(axis=2)
    return [accuracy(np.where(valid[k], pk[k], -1), y) for k in range(len(S))]


def report(title, S, y, vids, lines):
    def out(s=''):
        print(s)
        lines.append(s)

    out(f'\n{"=" * 70}\n  {title}\n{"=" * 70}')
    out(f'  عدد نقاط التقييم: {(y >= 0).sum()}')
    accs = per_scale_acc(S, y)
    out('\n  كل classifier لوحده (Z):')
    out('  ' + '  '.join(f'{n}fr={a:.1f}%' for n, a in zip(FRAME_CONFIGS, accs)))
    best = int(np.argmax(accs))
    out(f'  ↳ الأحسن: {FRAME_CONFIGS[best]} فريم = {accs[best]:.1f}%  '
        f'(ده متختار على الاختبار نفسه — رقم متفائل)')

    out(f'\n  {"الطريقة":<52}{"الكل":>8}' + ''.join(f'{v[-5:]:>9}' for v in VIDEOS))
    results = {}
    for key, name in METHODS:
        pred, w = lovo_predict(S, y, vids, key)
        per_v = [accuracy(pred[vids == v], y[vids == v]) for v in VIDEOS]
        results[key] = (pred, w)
        out(f'  {name:<52}{accuracy(pred, y):>7.1f}%'
            + ''.join(f'{a:>8.1f}%' for a in per_v))
    return results


def main():
    render = '--no-render' not in sys.argv
    workers = int(os.environ.get('WORKERS', max(1, (os.cpu_count() or 2) - 2)))
    ext = E.load_external_templates()
    uniq = sorted({t['label'] for t in ext})
    lab_idx = {l: i for i, l in enumerate(uniq)}
    lines = []

    print('=' * 70)
    print(f'  🗳️  Ensemble بين {len(FRAME_CONFIGS)} classifiers: '
          + ' · '.join(map(str, FRAME_CONFIGS)))
    print(f'  {len(ext)} قالب خارجي، {len(uniq)} حركة، {workers} عملية بالتوازي')
    print('=' * 70)
    t0 = time.perf_counter()

    with Pool(workers) as pool:
        # أ) الـ 40 قصاصة
        res = sorted(pool.map(gt_windows_job, FRAME_CONFIGS))
        S_a = np.stack([r[1] for r in res])
        y_a = np.array([lab_idx[l] for l in res[0][2]])
        v_a = np.array(res[0][3])
        print(f'⏳ أ) خلص في {time.perf_counter() - t0:.0f}s')

        # ب) نافذة منزلقة
        jobs = [(v, n) for n in sorted(FRAME_CONFIGS, reverse=True) for v in VIDEOS]
        got = {(v, n): z for v, n, z in pool.imap_unordered(sliding_job, jobs)}
        print(f'⏳ ب) خلص في {time.perf_counter() - t0:.0f}s')

    S_b = np.concatenate([np.stack([got[(v, n)] for n in FRAME_CONFIGS]) for v in VIDEOS],
                         axis=1)
    y_b, v_b, t_b = [], [], []
    for v in VIDEOS:
        centers, fps, _ = centers_of(v)
        for c in centers:
            gt = label_at(v, c / fps)
            gt = GT_ALIAS.get(gt, gt)
            y_b.append(lab_idx.get(gt, -1))
            v_b.append(v)
            t_b.append(c / fps)
    y_b, v_b, t_b = np.array(y_b), np.array(v_b), np.array(t_b)

    report('أ) الـ 40 قصاصة GT — نفس اختبار experiment_frame_scales', S_a, y_a, v_a, lines)
    res_b = report('ب) نافذة منزلقة على الفيديو كله (دقة لكل 0.2 ثانية)', S_b, y_b, v_b, lines)

    (OUT_DIR / 'ensemble_results.txt').write_text('\n'.join(lines), encoding='utf-8')
    print(f'\n✅ الجدول اتحفظ في {OUT_DIR / "ensemble_results.txt"}')

    if render:
        render_videos(res_b[RENDER_METHOD][0], v_b, t_b, uniq, ext, workers, lines)


def _render_one(args):
    import render_all_frame_configs as R
    video, times, labels, out_path, tag = args
    _, _, n_kp = centers_of(video)
    _, fps = load_keypoints(video)
    segs = R.preds_to_segments(times, labels, n_kp / fps, fps_out=STEPS_PER_SEC)
    colors = R.label_colors(E.load_external_templates())
    return video, R.render_video(video, 0, segs, colors, out_path=out_path, tag=tag)


def render_videos(pred, vids, times, uniq, ext, workers, lines):
    from pathlib import Path
    out = Path(os.environ.get('RENDER_OUT', 'outputs')) / 'ensemble'
    out.mkdir(parents=True, exist_ok=True)
    tag = f'ENSEMBLE {FRAME_CONFIGS[0]}-{FRAME_CONFIGS[-1]}fr | weighted vote'
    jobs = [(v, list(times[vids == v]),
             [uniq[p] if p >= 0 else '-' for p in pred[vids == v]],
             out / f'ensemble_{v}.mp4', tag) for v in VIDEOS]
    print(f'\n🎬 رسم {len(jobs)} فيديو ensemble...')
    with Pool(min(workers, len(jobs))) as pool:
        for video, ok in pool.imap_unordered(_render_one, jobs):
            print(f'   {video} {"✅" if ok else "⚠️"}')
    (out / 'ensemble_results.txt').write_text('\n'.join(lines), encoding='utf-8')
    print(f'📁 {out.resolve()}')


if __name__ == '__main__':
    main()
