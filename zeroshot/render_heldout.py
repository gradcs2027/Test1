"""
فيديوهات الاختبار المخبّي (vidtest5 و vidtest6) متعلّم عليها كل حاجة:
الإجابة الصح، توقّع الـ zero shot لحظة بلحظة، توقّع كل قصاصة (zero shot و
FastDTW)، أعلى 5 أوصاف دلوقتي، والنسب.

    python zeroshot/predict_video.py vidtest5 vidtest6   # blind_*.npz
    python zeroshot/eval_heldout.py                      # heldout_clips.npz
    python zeroshot/render_heldout.py [vidtest5] [vidtest6]

الناتج: results/annotated_{video}.mp4

⚠️ مفيش موديل بيشتغل هنا — ده رسم بس للملفين اللي فوق، يعني كل رقم على
   الفيديو هو نفس الرقم اللي في heldout.txt. التوقّع بالـ 19 وصف (الـ 18 +
   ironing)، ودي نفس الـ 6 من 12 على القصاصات.

الأشرطة تحت (من فوق لتحت):
  1. الإجابة الصح (الأحمر الغامق = '?' مش متعلّمة، مش داخلة في الحساب)
  2. zero shot لكل قصاصة (ده الـ 50%) — تحته خط أخضر = صح، أحمر = غلط
  3. FastDTW لكل قصاصة (ده الـ 33%)
  4. zero shot لكل نافذة 3 ثواني كل نص ثانية (ده الـ ~37%)

بنكتب فريم من كل اتنين: الـ keypoints اتطلّعت من الفريمات الزوجية، فالهيكل
بيبقى على الصورة بالظبط والملف أصغر.

ملاحظة: cv2.putText مابيرسمش عربي، فكل النص على الفيديو إنجليزي.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

import _bootstrap  # noqa: F401
from ground_truth import EXCLUDED, GROUND_TRUTH, HELDOUT, STILL, label_at
from paths import KP_DIR, VIDEO_DIR
from render_gt import COLOR_EXCL, COLOR_STILL, FONT, draw_skeleton, open_writer

import descriptions as D

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

HERE = Path(__file__).resolve().parent
RES = HERE / 'results'
NAMES = list(D.OURS)
FRAME_SKIP = 2
AA = cv2.LINE_AA

PANEL_W = 336           # لوحة يمين الفيديو: أعلى 5 أوصاف + النسب
HEAD_H = 104            # فوق: الصح + التوقّع + الحكم
BAR_H = 54              # كل شريط زمني (4 أشرطة)
TICKS_H = 14            # أرقام الثواني تحت آخر شريط
COS_MAX = 0.4           # طول عمود الـ cosine الكامل

GREEN, RED, GRAY, WHITE = (90, 210, 90), (60, 60, 235), (150, 150, 150), (235, 235, 235)
BG = (18, 18, 18)

# لون لكل وصف (BGR) — متختار بالإيد عشان الحركات اللي بتتلخبط في بعض (الصح
# والتوقّع الغلط بتاعه) تبان مختلفة. من غير أحمر ولا رمادي: دول لـ «غلط» و«مش متقيّم».
COLORS = {
    'walking': (255, 255, 0), 'clapping': (0, 215, 255), 'phone_call': (14, 127, 255),
    'wave': (255, 0, 255), 'touch_head': (210, 182, 247), 'ironing': (189, 103, 148),
    'brush_hair': (180, 119, 31), 'drink_water': (213, 176, 197), 'hand_shake': (138, 223, 152),
    'hugging': (141, 219, 219), 'play_pingpong': (150, 152, 255), 'running': (232, 199, 174),
    'sitting': (34, 189, 188), 'spray_perfume': (75, 86, 140), 'stand_up': (148, 156, 196),
    'turn_on_light': (229, 218, 158), 'lying': (194, 119, 227), 'wake_up': (120, 187, 255),
    'wear_glasses': (44, 160, 44),
}


# ==============================================================================
# البيانات — من الملفات اللي طلّعت الأرقام، مش حساب جديد
# ==============================================================================

def load_all():
    c = np.load(RES / 'heldout_clips.npz')
    names = [str(n) for n in c['names']]
    clips = [dict(video=str(v), s=float(s), e=float(e), label=str(l),
                  zs=names[int(r.argmax())], cos=float(r.max()), fd=str(f))
             for v, s, e, l, r, f in zip(c['video'], c['s'], c['e'], c['label'], c['sim'], c['fastdtw'])]

    wins = {}
    for v in HELDOUT:
        b = np.load(RES / f'blind_{v}.npz')
        assert [str(n) for n in b['names']] == names, 'blind_*.npz قديم — شغّل predict_video.py تاني'
        mid = (b['t0'] + b['t1']) / 2
        gt = np.array([label_at(v, t) or EXCLUDED for t in mid])
        scored = np.array([g not in (EXCLUDED, STILL) for g in gt])
        pred = b['sim'].argmax(1)
        ok = scored & (np.array(names)[pred] == gt)
        in18 = np.isin(gt, NAMES)
        ok18 = in18 & (np.array(NAMES)[b['sim'][:, :len(NAMES)].argmax(1)] == gt)
        wins[v] = dict(sim=b['sim'], mid=mid, step=float(b['step']), win=float(b['win']),
                       gt=gt, scored=scored, pred=pred, ok=ok, cum=np.cumsum(ok), cum_n=np.cumsum(scored),
                       hit=int(ok.sum()), n=int(scored.sum()), acc18=ok18.sum() / in18.sum())
    return names, clips, wins


def clip_score(clips, key, video=None):
    """(صح، العدد) على قصاصات الـ 18 — نفس الـ 12 اللي طلّعوا الـ 50%."""
    cs = [c for c in clips if c['label'] in NAMES and video in (None, c['video'])]
    return sum(c[key] == c['label'] for c in cs), len(cs)


def tag_of(label):
    if label in D.SEEN:
        return 'seen in training'
    if label in NAMES:
        return 'never seen in training'
    return 'NEW action: text description only'


# ==============================================================================
# الرسم
# ==============================================================================

def text(img, s, x, y, col=WHITE, scale=0.4, th=1):
    cv2.putText(img, s, (int(x), int(y)), FONT, scale, col, th, AA)
    return x + cv2.getTextSize(s, FONT, scale, th)[0][0]


def row(img, x0, x1, y, left, right, col=WHITE, scale=0.4):
    """سطر في اللوحة: كلام شمال ورقم يمين."""
    text(img, left, x0, y, col, scale)
    text(img, right, x1 - cv2.getTextSize(right, FONT, scale, 1)[0][0], y, col, scale)


def ink(col):
    """لون كلام مقروء فوق الخلفية دي."""
    b, g, r = col
    return (0, 0, 0) if 0.114 * b + 0.587 * g + 0.299 * r > 150 else (255, 255, 255)


def draw_bar(img, y0, title, blocks, dur, W, outline=True):
    """blocks = [(من، لحد، لون، اسم أو ''، صح True/False/None=مش متقيّم)]"""
    text(img, title, 6, y0 + 13, (190, 190, 190), 0.4)
    for s, e, col, name, ok in blocks:
        x1, x2 = int(s / dur * W), max(int(s / dur * W) + 1, int(e / dur * W))
        cv2.rectangle(img, (x1, y0 + 17), (x2 - 1, y0 + 37), col, -1)
        if outline:
            cv2.rectangle(img, (x1, y0 + 17), (x2 - 1, y0 + 37), (0, 0, 0), 1)
        if ok is not None:
            cv2.rectangle(img, (x1, y0 + 39), (x2 - 1, y0 + 43), GREEN if ok else RED, -1)
        tw = cv2.getTextSize(name, FONT, 0.34, 1)[0][0]
        if name and x2 - x1 > tw + 6:
            text(img, name, x1 + (x2 - x1 - tw) // 2, y0 + 32, ink(col), 0.34)


def dim(col, k=0.35):
    return tuple(int(c * k) for c in col)


def build_base(v, names, colors, clips, wins, dur, VW, VH):
    """كل اللي مابيتغيّرش من فريم لفريم: الأشرطة + أرقام اللوحة."""
    W, H = VW + PANEL_W, HEAD_H + VH + 4 * BAR_H + TICKS_H
    img = np.full((H, W, 3), BG, np.uint8)
    y = HEAD_H + VH
    mine = {(c['s'], c['e']): c for c in clips if c['video'] == v}

    def gt_col(l):
        return COLOR_EXCL if l == EXCLUDED else COLOR_STILL if l == STILL else colors[l]

    gt_blocks, zs_blocks, fd_blocks = [], [], []
    for s, e, l in GROUND_TRUTH[v]:
        gt_blocks.append((s, e, gt_col(l), 'not annotated' if l == EXCLUDED else l, None))
        c = mine.get((s, e))
        if c is None:
            zs_blocks.append((s, e, dim(gt_col(l)), '', None))
            fd_blocks.append((s, e, dim(gt_col(l)), '', None))
            continue
        zs_blocks.append((s, e, colors[c['zs']], c['zs'], c['zs'] == l))
        fd_blocks.append((s, e, colors[c['fd']], c['fd'], c['fd'] == l) if c['fd']
                         else (s, e, dim(WHITE, 0.25), 'no template', None))

    w = wins[v]
    win_blocks = [(m - w['step'] / 2, m + w['step'] / 2,
                   colors[names[p]] if sc else dim(colors[names[p]]), '', bool(ok) if sc else None)
                  for m, p, sc, ok in zip(w['mid'], w['pred'], w['scored'], w['ok'])]

    h5, n5 = clip_score(clips, 'zs', v)
    hf, nf = clip_score(clips, 'fd', v)
    draw_bar(img, y, 'GROUND TRUTH', gt_blocks, dur, W)
    draw_bar(img, y + BAR_H, f'ZERO-SHOT, whole clip  ->  {h5}/{n5} correct', zs_blocks, dur, W)
    draw_bar(img, y + 2 * BAR_H, f'FASTDTW (big bank), whole clip  ->  {hf}/{nf} correct', fd_blocks, dur, W)
    draw_bar(img, y + 3 * BAR_H, f'ZERO-SHOT, every {w["step"]:g}s ({w["win"]:g}s window)  ->  '
             f'{100 * w["hit"] / w["n"]:.1f}% of {w["n"]} windows', win_blocks, dur, W, outline=False)
    for sec in range(0, int(dur) + 1, 10):
        x = int(sec / dur * W)
        cv2.line(img, (x, y + 4 * BAR_H - 10), (x, y + 4 * BAR_H - 4), GRAY, 1)
        text(img, f'{sec}s', max(1, x - 8), y + 4 * BAR_H + 10, (170, 170, 170), 0.32)

    # اللوحة: الأجزاء الثابتة
    x0, x1, top = VW + 12, VW + PANEL_W - 12, HEAD_H
    text(img, f'HELD-OUT TEST: {v}', x0, top + 22, WHITE, 0.55)
    text(img, 'never used in training or tuning', x0, top + 42, GRAY, 0.38)
    text(img, 'zero-shot = BiLSTM skeleton + MiniLM text', x0, top + 58, GRAY, 0.36)
    for yy in (68, 250, 376):
        cv2.line(img, (x0, top + yy), (x1, top + yy), (60, 60, 60), 1)
    text(img, f'NOW: top 5 of {len(names)} descriptions (cosine)', x0, top + 88, WHITE, 0.4)

    text(img, 'THIS VIDEO', x0, top + 272, WHITE, 0.45)
    row(img, x0, x1, top + 294, 'clips, zero-shot', f'{h5}/{n5}  {100 * h5 / n5:.0f}%')
    row(img, x0, x1, top + 314, 'clips, FastDTW', f'{hf}/{nf}  {100 * hf / nf:.0f}%')
    row(img, x0, x1, top + 334, 'windows, zero-shot', f'{100 * w["hit"] / w["n"]:.1f}%')
    text(img, f'  ({100 * w["acc18"]:.1f}% without the ironing description)', x0, top + 350, GRAY, 0.33)

    hz, nz = clip_score(clips, 'zs')
    hd, nd = clip_score(clips, 'fd')
    labs = [c['label'] for c in clips if c['label'] in NAMES]
    top_lab = max(set(labs), key=labs.count)
    hit_all = sum(x['hit'] for x in wins.values())
    n_all = sum(x['n'] for x in wins.values())
    new = [c for c in clips if c['label'] not in NAMES]
    text(img, f'BOTH TEST VIDEOS ({nz} clips)', x0, top + 396, WHITE, 0.45)
    cv2.rectangle(img, (x0 - 4, top + 403), (x1 + 4, top + 423), (40, 70, 40), -1)
    row(img, x0, x1, top + 418, 'zero-shot (our model)', f'{hz}/{nz}  {100 * hz / nz:.1f}%', GREEN)
    row(img, x0, x1, top + 440, 'FastDTW', f'{hd}/{nd}  {100 * hd / nd:.1f}%')
    row(img, x0, x1, top + 460, f"always guess '{top_lab}'",
        f'{labs.count(top_lab)}/{nz}  {100 * labs.count(top_lab) / nz:.1f}%')
    row(img, x0, x1, top + 480, f'random chance (1 of {len(NAMES)})', f'{100 / len(NAMES):.1f}%')
    row(img, x0, x1, top + 500, 'windows, zero-shot', f'{100 * hit_all / n_all:.1f}%')
    row(img, x0, x1, top + 520, 'ironing (new, text only)',
        f'{sum(c["zs"] == c["label"] for c in new)}/{len(new)}')
    text(img, 'clip = one annotated span   window = 3s', x0, top + 544, GRAY, 0.33)
    text(img, 'green line = correct   red line = wrong', x0, top + 558, GRAY, 0.33)
    text(img, 'dark = not annotated, not scored', x0, top + 572, GRAY, 0.33)
    return img


def draw_live(img, v, t, names, colors, clips, wins, dur, VW):
    """الهيدر + أعلى 5 + العدّاد + المؤشر — اللي بيتغيّر كل فريم."""
    W = VW + PANEL_W
    w = wins[v]
    i = int(np.clip(round((t - w['mid'][0]) / w['step']), 0, len(w['mid']) - 1))
    gt, sc, sim = str(w['gt'][i]), bool(w['scored'][i]), w['sim'][i]
    p = names[w['pred'][i]]

    # ── الهيدر ──
    state = GRAY if not sc else GREEN if p == gt else RED
    cv2.rectangle(img, (0, 0), (10, HEAD_H), state, -1)
    text(img, 'TRUE', 22, 32, GRAY, 0.42)
    if sc:
        x = text(img, gt.upper(), 92, 34, colors[gt], 0.8, 2)
        text(img, f'({tag_of(gt)})', x + 14, 32, GRAY, 0.42)
    else:
        x = text(img, 'NOT ANNOTATED', 92, 34, GRAY, 0.8, 2)
        text(img, '(not scored)', x + 14, 32, GRAY, 0.42)
    text(img, 'MODEL', 22, 66, GRAY, 0.42)
    x = text(img, p.upper(), 92, 68, colors[p], 0.8, 2)
    text(img, f'cos {sim.max():.2f}   (zero-shot, {w["win"]:g}s window)', x + 14, 66, GRAY, 0.42)
    text(img, 'CORRECT' if state == GREEN else 'WRONG' if state == RED else 'NOT SCORED',
         W - 300, 68, state, 0.95, 2)
    text(img, f'{t:6.2f}s / {dur:.1f}s', W - 300, 30, WHITE, 0.5)

    c = next((c for c in clips if c['video'] == v and c['s'] <= t < c['e']), None)
    if c:
        x = text(img, f'WHOLE CLIP {c["s"]:.0f}-{c["e"]:.0f}s:', 22, 94, GRAY, 0.42)
        x = text(img, f'zero-shot {c["zs"]}', x + 12, 94, WHITE, 0.42)
        x = text(img, ' RIGHT' if c['zs'] == c['label'] else ' WRONG', x, 94,
                 GREEN if c['zs'] == c['label'] else RED, 0.42, 2)
        x = text(img, '|  FastDTW ' + (c['fd'] or 'no template for this action'), x + 16, 94, WHITE, 0.42)
        if c['fd']:
            text(img, ' RIGHT' if c['fd'] == c['label'] else ' WRONG', x, 94,
                 GREEN if c['fd'] == c['label'] else RED, 0.42, 2)

    # ── أعلى 5 أوصاف ──
    x0, top = VW + 12, HEAD_H
    for r, k in enumerate(np.argsort(-sim)[:5]):
        yy = top + 98 + r * 24
        is_gt = names[k] == gt
        text(img, names[k], x0, yy + 14, GREEN if is_gt else WHITE, 0.4, 2 if is_gt else 1)
        bw = int(np.clip(sim[k] / COS_MAX, 0, 1) * 110)
        cv2.rectangle(img, (x0 + 140, yy + 3), (x0 + 140 + max(bw, 1), yy + 17), colors[names[k]], -1)
        text(img, f'{sim[k]:.2f}', x0 + 140 + bw + 6, yy + 15, WHITE, 0.38)
    if sc:
        rank = int((sim > sim[names.index(gt)]).sum()) + 1
        text(img, f'true label rank: {rank} of {len(names)}', x0, top + 236,
             GREEN if rank == 1 else (200, 200, 200), 0.4)

    j = int(np.searchsorted(w['mid'], t, 'right'))
    if j:
        text(img, f'  so far: {w["cum"][j - 1]} of {w["cum_n"][j - 1]} windows correct',
             x0, top + 366, (200, 200, 200), 0.36)

    # ── المؤشر على الأشرطة ──
    xc = int(t / dur * W)
    y = img.shape[0] - 4 * BAR_H - TICKS_H
    cv2.line(img, (xc, y + 14), (xc, y + 4 * BAR_H - 4), WHITE, 2)
    cv2.circle(img, (xc, y + 14), 4, WHITE, -1, AA)


def render(v, names, clips, wins):
    colors = {n: COLORS.get(n, (200, 200, 200)) for n in names}
    kp_all = np.load(KP_DIR / f'{v}_keypoints.npy')
    cap = cv2.VideoCapture(str(VIDEO_DIR / f'{v}.mp4'))
    if not cap.isOpened():
        raise FileNotFoundError(f'مش قادر أفتح {VIDEO_DIR / v}.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    VW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2 * 2      # libx264 عايز أبعاد زوجية
    VH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2 * 2
    dur = total / fps

    base = build_base(v, names, colors, clips, wins, dur, VW, VH)
    H, W = base.shape[:2]
    out = RES / f'annotated_{v}.mp4'
    _, write, close = open_writer(out, fps / FRAME_SKIP, W, H)
    print(f'🎬 {v}: {total // FRAME_SKIP} فريم @ {fps / FRAME_SKIP:.0f}fps، {W}x{H} → {out.name}')

    i = 0
    while True:
        if i % FRAME_SKIP:
            if not cap.grab():
                break
            i += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        img = base.copy()
        img[HEAD_H:HEAD_H + VH, :VW] = frame[:VH, :VW]
        draw_skeleton(img, kp_all[min(i // FRAME_SKIP, len(kp_all) - 1)], HEAD_H)
        draw_live(img, v, t, names, colors, clips, wins, dur, VW)
        write(img)
        i += 1
    cap.release()
    close()
    print(f'✅ {out}  ({out.stat().st_size / 1e6:.1f} MB)')
    return out


def main():
    videos = [a for a in sys.argv[1:] if a in HELDOUT] or HELDOUT
    names, clips, wins = load_all()
    for v in videos:
        render(v, names, clips, wins)


if __name__ == '__main__':
    main()
