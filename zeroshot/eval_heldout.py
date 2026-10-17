"""
القياس على الاختبار المخبّي (vidtest5 + vidtest6): الـ zero shot مقابل FastDTW.

الفيديوهين دول مادخلوش في أي حاجة: لا تدريب الـ BiLSTM، ولا ضبط gamma،
ولا بنك FastDTW، ولا أوزان الـ ensemble (دي بتتحسب من الـ 40 قصاصة بتوع
vidtest1-4 بس). يعني ده **أنضف رقم** عندنا للطريقتين.

القصاصات = كل فترة متعلّمة في الـ GT (من غير '?')، بنفس طريقة الـ 40 قصاصة.
`ironing` مش من الـ 18: FastDTW مالوش قالب ليها، فبتتقاس في سطر لوحدها
(الـ zero shot بوصف نص بس من EXTRA في predict_video.py).

    python zeroshot/eval_heldout.py            # أول مرة: FastDTW ~دقايق، بعد كده من الكاش
    python zeroshot/eval_heldout.py --redo     # يعيد حساب FastDTW
"""
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# البنك الكبير (171 قصاصة) — لازم يتحدد قبل import experiment_frame_scales
os.environ.setdefault('BANK_DIR', str(ROOT / 'shared' / 'keypoints' / 'external_big'))
sys.path.insert(0, str(ROOT / 'fastdtw'))

# ⚠️ torch و MiniLM بيتعملّهم import جوّه main() بس: على ويندوز كل worker
#    في الـ Pool بيعيد import الملف ده، ومش عايزين 10 نسخ من torch.
import _bootstrap  # noqa: F401,E402
import experiment_ensemble as X  # noqa: E402
import experiment_frame_scales as E  # noqa: E402
from classifier import norm_distance, normalize_window, resample_linear, to_features  # noqa: E402
from ground_truth import EXCLUDED, GROUND_TRUTH, HELDOUT, STILL, label_at  # noqa: E402
from paths import load_keypoints  # noqa: E402
from skeleton_norm import normalize_skeleton, resample_to  # noqa: E402

import descriptions as D  # noqa: E402

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

CACHE = ROOT / '_scratch' / 'zeroshot' / 'heldout_fastdtw.npz'
DEV_CACHE = X.OUT_DIR / 'ensemble_scores_external_big.npz'
NAMES = list(D.OURS)                       # الـ 18 بنفس ترتيب FastDTW (أبجدي)


def heldout_clips():
    """[(فيديو، من، لحد، لابل، keypoints)] لكل فترة متعلّمة."""
    out = []
    for v in HELDOUT:
        kp, fps = load_keypoints(v)
        kp = kp[..., :2]
        for s, e, lab in GROUND_TRUTH[v]:
            if lab in (EXCLUDED, STILL):
                continue
            clip = kp[max(0, int(round(s * fps))):min(len(kp), int(round(e * fps)))]
            if len(clip) >= 4:
                out.append((v, s, e, lab, clip))
    return out


def fastdtw_job(n):
    """نفس gt_windows_job في experiment_ensemble بس على قصاصات الاختبار المخبّي."""
    tpl, mu, sigma, labels, uniq = X._setup(n)
    assert uniq == NAMES, 'ترتيب حركات البنك مختلف عن الـ 18'
    feats = [to_features(resample_linear(normalize_window(c[4]), n=n), mode='vel', shape_norm=True)
             for c in heldout_clips() if c[3] in NAMES]
    Dm = np.array([[norm_distance(f, t['feat'], radius=X.RADIUS) for t in tpl] for f in feats])
    return n, X._label_z(Dm, mu, sigma, labels, uniq)


def fastdtw_scores(redo):
    if CACHE.exists() and not redo:
        return np.load(CACHE)['S']
    print(f'⏳ FastDTW على البنك {E.EXT_DIR.name} (10 مقاييس)...')
    with Pool(min(10, os.cpu_count() or 1)) as pool:
        res = sorted(pool.map(fastdtw_job, E.FRAME_CONFIGS))
    S = np.stack([r[1] for r in res])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, S=S)
    return S


def fastdtw_predict(S_h, y_h, method):
    """الأوزان من الـ 40 قصاصة بتوع التطوير بس، والتوقّع على المخبّي."""
    dev = np.load(DEV_CACHE)
    S = np.concatenate([dev['S_a'], S_h], axis=1)
    y = np.concatenate([dev['y_a'], y_h])
    vids = np.concatenate([dev['v_a'], np.full(len(y_h), 'heldout')])
    return X.lovo_predict(S, y, vids, method)[0][-len(y_h):]


def main():
    import torch
    from predict_video import CKPT, EXTRA, text_codes
    from zsl_model import SkeletonEncoder, embed

    clips = heldout_clips()
    labs = [c[3] for c in clips]
    in18 = np.array([l in NAMES for l in labs])
    y18 = np.array([NAMES.index(l) for l in labs if l in NAMES])
    seen = np.array([NAMES[i] in D.SEEN for i in y18])

    # ── zero shot ──
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SkeletonEncoder().to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    names19, txt = text_codes()
    Xh = np.stack([resample_to(normalize_skeleton(c[4])) for c in clips]).astype(np.float32)
    sim = embed(model, Xh, device) @ txt.T               # (قصاصات، 19)
    sim18 = sim[in18][:, :len(NAMES)]
    gamma = float(np.load(HERE / 'results' / 'ours_scores.npz')['gamma'])
    seen_lab = np.array([k in D.SEEN for k in NAMES])

    # ── FastDTW ──
    S_h = fastdtw_scores('--redo' in sys.argv)
    assert S_h.shape[1] == len(y18)

    lines = []

    def out(s=''):
        print(s)
        lines.append(s)

    def row(name, pred, y=y18, m_seen=seen, only_unseen=False):
        a = lambda m: f'{100.0 * (pred[m] == y[m]).mean():.1f}%' if m.any() else '—'
        ok = int((pred == y).sum()) if not only_unseen else int((pred[~m_seen] == y[~m_seen]).sum())
        tot = len(y) if not only_unseen else int((~m_seen).sum())
        cnt = np.bincount(pred[pred >= 0], minlength=len(names19))
        top = int(cnt.argmax())
        out(f'  {name:<36}{"—" if only_unseen else a(np.ones_like(m_seen)):>8}'
            f'{"—" if only_unseen else a(m_seen):>9}{a(~m_seen):>11}   {ok}/{tot}'
            f'   {names19[top]} ×{cnt[top]}')

    n_by = {l: labs.count(l) for l in dict.fromkeys(labs)}
    out(f'🔒 اختبار مخبّي: {", ".join(HELDOUT)} — {len(clips)} قصاصة '
        f'({", ".join(f"{k} {v}" for k, v in n_by.items())})')
    out(f'   من الـ 18: {len(y18)} قصاصة = {seen.sum()} متشافة ({", ".join(sorted({NAMES[i] for i in y18[seen]}))})'
        f' + {(~seen).sum()} مش متشافة. قصاصة واحدة = {100 / len(y18):.1f}%')
    out(f'\n  {"الطريقة (18 حركة)":<36}{"الكل":>8}{"متشافة":>9}{"مش متشافة":>11}   صح     أكتر حركة اتقالت')
    out(f'  {"صدفة (1 من 18)":<36}{100 / 18:>7.1f}%{100 / 18:>8.1f}%{100 / 18:>10.1f}%')
    top_dev = int(np.bincount(np.load(DEV_CACHE)['y_a']).argmax())
    row(f'دايماً «{NAMES[top_dev]}» (أكتر حركة في التطوير)', np.full_like(y18, top_dev))
    top_h = int(np.bincount(y18).argmax())
    row(f'دايماً «{NAMES[top_h]}» (أكتر حركة هنا — غش)', np.full_like(y18, top_h))

    out('\n  — FastDTW بنك كبير 171 (عنده قوالب للـ 18، أوزان من الـ 40 قصاصة) —')
    fd = {}
    for key, name in [('best_single', 'أحسن مقياس'), ('weighted_conf', 'موزون×ثقة'),
                      ('per_class', 'وزن لكل حركة')]:
        fd[key] = fastdtw_predict(S_h, y18, key)
        row(f'FastDTW · {name}', fd[key])

    out('\n  — Zero shot (BiLSTM + MiniLM) —')
    p0 = sim18.argmax(1)
    row('من غير تصحيح (18 وصف)', p0)
    row(f'بتصحيح gamma={gamma:.3f} من NTU', (sim18 - seen_lab * gamma).argmax(1))
    un = [i for i, s in enumerate(seen_lab) if not s]
    pz = np.full_like(y18, -1)
    pz[~seen] = np.array(un)[sim18[~seen][:, un].argmax(1)]
    row('zsl: المش متشافة من 13 وصف بس', pz, only_unseen=True)
    out(f'     (صدفة الـ 13 = {100 / 13:.1f}%)')

    # ironing: 19 وصف على الـ 13 قصاصة كلها
    y19 = np.array([names19.index(l) for l in labs])
    p19 = sim.argmax(1)
    new = np.array([l in EXTRA for l in labs])
    out(f'\n  — بالوصف الجديد (19 وصف، {len(labs)} قصاصة) —')
    out(f'  الكل {100 * (p19 == y19).mean():.1f}% ({(p19 == y19).sum()}/{len(y19)})'
        f' | ironing: {(p19[new] == y19[new]).sum()}/{new.sum()}')
    for i in np.where(new)[0]:
        r = int((sim[i] > sim[i, y19[i]]).sum()) + 1
        out(f'  ironing {clips[i][1]:.0f}-{clips[i][2]:.0f}s: ترتيب الوصف الصح {r} من 19 '
            f'(cosine {sim[i, y19[i]]:.2f}، الأعلى {names19[p19[i]]} {sim[i].max():.2f})')

    # كل قصاصة لوحدها
    out(f'\n  {"الفيديو":<9}{"الفترة":>9}  {"الحقيقة":<12}{"zero shot (أعلى 3، cosine)":<50}FastDTW موزون×ثقة')
    j = 0
    for i, (v, s, e, lab, _) in enumerate(clips):
        top3 = np.argsort(-sim[i])[:3]
        z = '  '.join(f'{"✓" if k == y19[i] else ""}{names19[k]} {sim[i, k]:.2f}' for k in top3)
        f = '—'
        if lab in NAMES:
            pf = fd['weighted_conf'][j]
            f = f'{"✓" if pf == y18[j] else "✗"} {NAMES[pf]}'
            j += 1
        out(f'  {v:<9}{s:>4.0f}-{e:<4.0f}  {lab:<12}{z:<50}{f}')

    # لـ render_heldout.py: كل قصاصة + درجاتها + توقّع FastDTW ('' = مالوش قالب)
    it = iter(fd['weighted_conf'])
    np.savez(HERE / 'results' / 'heldout_clips.npz',
             video=[c[0] for c in clips], s=[c[1] for c in clips], e=[c[2] for c in clips],
             label=labs, sim=sim, names=names19,
             fastdtw=[NAMES[next(it)] if m else '' for m in in18])

    # النوافذ المنزلقة (من predict_video.py) — كل نص ثانية
    out('\n  — النوافذ المنزلقة (3s كل 0.5s، من blind_*.npz) — نص النافذة جوّه فترة متعلّمة —')
    for v in HELDOUT:
        b = np.load(HERE / 'results' / f'blind_{v}.npz')
        assert list(b['names']) == names19
        mid = (b['t0'] + b['t1']) / 2
        gt = [label_at(v, t) for t in mid]
        m = np.array([g not in (None, EXCLUDED, STILL) for g in gt])
        yv = np.array([names19.index(g) for g, k in zip(gt, m) if k])
        pv = b['sim'][m].argmax(1)
        m18 = yv < len(NAMES)
        pv18 = b['sim'][m][m18][:, :len(NAMES)].argmax(1)
        out(f'  {v}: {m.sum()} نافذة | 19 وصف {100 * (pv == yv).mean():.1f}% | '
            f'18 وصف (من غير ironing) {100 * (pv18 == yv[m18]).mean():.1f}% | صدفة {100 / 18:.1f}%')

    (HERE / 'results' / 'heldout.txt').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
