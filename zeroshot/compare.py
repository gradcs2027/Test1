"""
الخطوة 6: مقارنة الـ zero shot بـ FastDTW على نفس الـ 40 قصاصة GT.

⚠️ المقارنة مش متساوية: FastDTW عنده قوالب (فيديوهات أمثلة) للـ 18 حركة كلهم،
   يعني هو مش zero shot. الموديل ماشافش ولا فيديو من الـ 13 حركة المش متشافة.
   عشان كده الجدول مقسوم: القصاصات المتشافة / المش متشافة.

أرقام FastDTW محسوبة من الكاش بتاع experiment_ensemble.py بنفس دوالّه
(leave-one-video-out)، من غير DTW تاني.

    python zeroshot/compare.py              بعد train.py
"""
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'fastdtw'))
import experiment_ensemble as X  # noqa: E402

FASTDTW = [('FastDTW بنك قديم', 'ensemble_scores.npz'),
           ('FastDTW بنك كبير 171', 'ensemble_scores_external_big.npz')]
METHODS = [('best_single', 'أحسن مقياس'), ('weighted_conf', 'موزون×ثقة'),
           ('per_class', 'وزن لكل حركة')]


def main():
    tag = '_quick' if '--quick' in sys.argv else ''
    z = np.load(HERE / 'results' / f'ours_scores{tag}.npz')
    sim, y, vid, names, gamma = z['sim'], z['y'], z['vid'], list(z['names']), float(z['gamma'])
    seen_lab = z['seen']
    seen = seen_lab[y]
    lines = []

    def out(s=''):
        print(s)
        lines.append(s)

    def row(name, pred, only_unseen=False):
        a = lambda m: 100.0 * (pred[m] == y[m]).mean()
        cnt = np.bincount(pred[pred >= 0], minlength=len(names))
        top = int(cnt.argmax())
        out(f'  {name:<34}{"—" if only_unseen else f"{a(np.ones_like(seen)):.1f}%":>8}'
            f'{"—" if only_unseen else f"{a(seen):.1f}%":>10}{a(~seen):>10.1f}%'
            f'   {names[top]} ×{cnt[top]}، {(cnt > 0).sum()} حركة مختلفة')

    out(f'🆚 نفس الـ 40 قصاصة: {seen.sum()} متشافة ({", ".join(np.array(names)[seen_lab])})'
        f' + {(~seen).sum()} مش متشافة')
    out(f'  {"الطريقة":<34}{"الكل":>8}{"متشافة":>10}{"مش متشافة":>11}   أكتر حركة اتقالت')
    out(f'  {"صدفة (1 من 18)":<34}{100 / 18:>7.1f}%{100 / 18:>9.1f}%{100 / 18:>10.1f}%')
    top_y = np.bincount(y).argmax()
    row(f'دايماً «{names[top_y]}» (أكتر حركة)', np.full_like(y, top_y))

    out('\n  — FastDTW (عنده قوالب للـ 18، مش zero shot) —')
    for title, f in FASTDTW:
        c = np.load(X.OUT_DIR / f)
        assert (c['y_a'] == y).all() and (c['v_a'] == vid).all(), f'ترتيب القصاصات مختلف في {f}'
        for key, name in METHODS:
            row(f'{title} · {name}', X.lovo_predict(c['S_a'], y, vid, key)[0])

    out('\n  — Zero shot (BiLSTM + MiniLM، ماشافش المش متشافة) —')
    p0 = sim.argmax(1)
    pg = (sim - seen_lab * gamma).argmax(1)
    row('من غير تصحيح (18 وصف)', p0)
    row(f'بتصحيح gamma={gamma:.3f} من NTU', pg)
    un = [i for i, s in enumerate(seen_lab) if not s]
    pz = np.full_like(y, -1)
    pz[~seen] = np.array(un)[sim[~seen][:, un].argmax(1)]
    row('zsl: المش متشافة من 13 وصف بس', pz, only_unseen=True)
    out(f'     (صدفة الـ 13 = {100 / 13:.1f}%)')

    (HERE / 'results' / f'compare{tag}.txt').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
