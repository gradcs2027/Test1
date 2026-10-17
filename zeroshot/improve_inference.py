"""
تحسين الـ zero shot من غير إعادة تدريب — بيتقاس على فيديوهات التطوير بس.

    python zeroshot/improve_inference.py

الاختبار المخبّي (vidtest5/6) **مش** هنا. اتشاف بالعين وقت التحليل، فأي
اختيار نطلعه من هنا لازم يتظبط على الـ 4 فيديوهات، وبعدين يتقاس على المخبّي
مرة واحدة في الآخر (`final_heldout.py`).

التجارب (كلها تعديل على درجات الـ cosine بعد الموديل، مفيش تدريب):
  center / z   بنطرح متوسط درجات كل وصف على نوافذ فيديوهات تانية (وبنقسم على
               الانحراف في z). الفكرة: وصف زي brush_hair درجته عالية على أي
               حاجة فيها إيد عند الراس، فبتفوز على الصح. ده «hubness».
  csls         نفس الفكرة بس بمتوسط أعلى k نافذة (الطريقة المعروفة في الترجمة).
  crop         القصاصة الطويلة (16 ثانية) كانت بتتضغط في 30 فريم = سرعة غير
               اللي اتدرّب عليها الموديل (NTU حركات ثانيتين-تلاتة). بدل كده
               بنوسّط درجات نوافذ الـ 3 ثواني اللي جوّاها.
  smooth       متوسط درجات النوافذ المتجاورة قبل الاختيار (النوافذ المنزلقة بس).

⚠️ إحصائيات الـ center/z/csls بتتحسب من الـ 3 فيديوهات التانية (leave-one-video-
   out) عشان مانقيسش على الفيديو اللي منه الإحصائيات. مفيش لابل بيدخل فيها —
   النوافذ كلها من غير ما نبص على الإجابة.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')     # قبل أي import لـ sentence_transformers

import numpy as np
import torch

import _bootstrap  # noqa: F401
from ground_truth import EXCLUDED, STILL, VIDEOS, label_at, spans
from paths import load_keypoints

import descriptions as D
from predict_video import CKPT, windows
from zsl_model import SkeletonEncoder, embed

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

HERE = Path(__file__).resolve().parent
NAMES = list(D.OURS)
ALIASES = {'sitting': ['sitting', 'sit_down']}
GT_TO_NAME = {'sit_down': 'sitting'}
SEEN = np.array([k in D.SEEN for k in NAMES])
WIN, STEP, TOPK = 3.0, 0.5, 20


# ==============================================================================
# بيانات التطوير
# ==============================================================================

def dev_windows():
    """نوافذ كل فيديو تطوير: درجات (n، 18)، منتصف النافذة، لابل النافذة (-1 = مش متقيّمة)."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SkeletonEncoder().to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    txt = np.load(HERE / 'results' / 'text_embeddings.npz')
    assert list(txt['our_names']) == NAMES
    txt = txt['ours'].astype(np.float32)
    out = {}
    for v in VIDEOS:
        kp, fps = load_keypoints(v)
        X, t0, t1 = windows(kp[..., :2], fps, WIN, STEP)
        mid = (t0 + t1) / 2
        lab = np.array([NAMES.index(GT_TO_NAME.get(g, g)) if GT_TO_NAME.get(g, g) in NAMES else -1
                        for g in (label_at(v, t) for t in mid)])
        out[v] = dict(sim=embed(model, X, device) @ txt.T, mid=mid, lab=lab)
    return out


def dev_clips(W):
    """الـ 40 قصاصة بنفس ترتيب ours_scores.npz + نوافذ كل قصاصة."""
    z = np.load(HERE / 'results' / 'ours_scores.npz')
    meta = []
    for v in VIDEOS:
        kp, fps = load_keypoints(v)
        for k, name in enumerate(NAMES):
            for g in ALIASES.get(name, [name]):
                for s, e in spans(v, g):
                    lo, hi = int(round(s * fps)), int(round(e * fps))
                    if len(kp[max(0, lo):min(len(kp), hi)]) >= 4:
                        meta.append((v, k, s, e))
    assert [m[1] for m in meta] == list(z['y']) and [m[0] for m in meta] == list(z['vid']), \
        'ترتيب القصاصات مختلف عن ours_scores.npz'
    crop = []
    for v, k, s, e in meta:
        w = W[v]
        m = (w['mid'] >= s) & (w['mid'] < e)
        crop.append(w['sim'][m].mean(0) if m.sum() >= 2 else None)
    return z['sim'], np.array([m[1] for m in meta]), np.array([m[0] for m in meta]), crop


# ==============================================================================
# التعديلات
# ==============================================================================

def make_adjuster(method, ref):
    """ref = (n، 18) درجات نوافذ من فيديوهات تانية. بيرجّع دالة S → S معدّلة."""
    mu, sd = ref.mean(0), ref.std(0) + 1e-6
    r = np.sort(ref, 0)[-TOPK:].mean(0)
    return {'base': lambda S: S,
            'center': lambda S: S - mu,
            'z': lambda S: (S - mu) / sd,
            'csls': lambda S: S - 0.5 * r}[method]


def smooth(S, m):
    if m == 0:
        return S
    k = np.ones(2 * m + 1)
    num = np.stack([np.convolve(S[:, j], k, 'same') for j in range(S.shape[1])], 1)
    return num / np.convolve(np.ones(len(S)), k, 'same')[:, None]


def pct(x):
    return f'{100 * x:5.1f}%'


def main():
    W = dev_windows()
    S_whole, y, vid, S_crop_raw = dev_clips(W)
    S_crop = np.stack([c if c is not None else s for c, s in zip(S_crop_raw, S_whole)])
    print(f'قصاصات التطوير: {len(y)} ({SEEN[y].sum()} متشافة + {(~SEEN[y]).sum()} مش متشافة) | '
          f'{sum(c is not None for c in S_crop_raw)} قصاصة اتوسّطت من نوافذ | '
          f'نوافذ: {", ".join(f"{v} {len(W[v]["mid"])}" for v in VIDEOS)}')
    print(f'صدفة 1 من 18 = {100 / 18:.1f}%\n')

    lines = [f'  {"التعديل":<10}{"قصاصة كاملة":>13}{"متوسط نوافذ":>13}   '
             f'{"نوافذ":>7}{"+smooth2":>9}{"+smooth4":>9}   أكتر وصف بيتقال (قصاصات كاملة)']
    print(lines[0])
    res = {}
    for method in ['base', 'center', 'z', 'csls']:
        acc = {k: [] for k in ('whole', 'crop', 'win0', 'win2', 'win4')}
        preds = []
        for v in VIDEOS:
            adj = make_adjuster(method, np.concatenate([W[o]['sim'] for o in VIDEOS if o != v]))
            m = vid == v
            for key, S in (('whole', S_whole), ('crop', S_crop)):
                p = adj(S[m]).argmax(1)
                acc[key] += list(p == y[m])
                if key == 'whole':
                    preds += list(p)
            w = W[v]
            for key, mm in (('win0', 0), ('win2', 2), ('win4', 4)):
                p = smooth(adj(w['sim']), mm).argmax(1)
                ok = w['lab'] >= 0
                acc[key] += list(p[ok] == w['lab'][ok])
        a = {k: np.mean(x) for k, x in acc.items()}
        top = np.bincount(preds, minlength=18)
        res[method] = a
        print(f'  {method:<10}{pct(a["whole"]):>13}{pct(a["crop"]):>13}   {pct(a["win0"]):>7}'
              f'{pct(a["win2"]):>9}{pct(a["win4"]):>9}   {NAMES[top.argmax()]} ×{top.max()} من {len(y)}')

    print('\n  متشافة / مش متشافة (قصاصة كاملة):')
    for method in ['base', 'z']:
        p = np.zeros(len(y), int)
        for v in VIDEOS:
            adj = make_adjuster(method, np.concatenate([W[o]['sim'] for o in VIDEOS if o != v]))
            p[vid == v] = adj(S_whole[vid == v]).argmax(1)
        print(f'    {method:<8}متشافة {pct((p == y)[SEEN[y]].mean())} | '
              f'مش متشافة {pct((p == y)[~SEEN[y]].mean())}')


if __name__ == '__main__':
    main()
