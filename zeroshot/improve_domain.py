"""
تعديلات على فرق الدومين (NTU ← كاميرتنا) من غير إعادة تدريب — على فيديوهات التطوير بس.

    python zeroshot/improve_domain.py

  adabn   بنعيد حساب إحصائيات طبقة BatchNorm في رأس الموديل على نوافذ من
          فيديوهات تانية (من غير لابل). الموديل اتدرّب على إحصائيات NTU؛ ده
          بيخلّيها على إحصائيات كاميرتنا. نفس فكرة AdaBN.
  flip    توقّع الهيكل + توقّع مقلوبه (شمال/يمين) ونوسّطهم. الشخص عندنا ممكن
          يستخدم الإيد التانية (مكالمة بإيد شمال مثلاً).
  crop    (من improve_inference.py) متوسط نوافذ الـ 3 ثواني جوّه القصاصة.

⚠️ زي improve_inference.py: إحصائيات adabn من الـ 3 فيديوهات التانية
   (leave-one-video-out)، ومفيش لابل بيدخل. الاختبار المخبّي مالمسناهوش.
"""
import copy
import os
import sys

os.environ.setdefault('HF_HUB_OFFLINE', '1')

import numpy as np
import torch
import torch.nn as nn

import _bootstrap  # noqa: F401
from ground_truth import VIDEOS, label_at, spans
from paths import load_keypoints
from skeleton_norm import normalize_skeleton, resample_to

from improve_inference import (ALIASES, CKPT, GT_TO_NAME, HERE, NAMES, SEEN, STEP, WIN, pct,
                               smooth, windows)
from prepare_data import COCO_FLIP
from zsl_model import SkeletonEncoder, embed

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def flip(X):
    s = X.reshape(len(X), -1, 17, 2)[:, :, COCO_FLIP, :].copy()
    s[..., 0] *= -1
    return s.reshape(X.shape).astype(np.float32)


def adabn(model, X):
    """نسخة من الموديل إحصائيات الـ BatchNorm بتاعتها من X (الدروب-أوت مقفول)."""
    m = copy.deepcopy(model)
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm1d):
            mod.reset_running_stats()
            mod.momentum = None                    # متوسط تراكمي على كل الدفعات
    m.train()
    for mod in m.modules():
        if isinstance(mod, nn.Dropout):
            mod.eval()
    with torch.no_grad():
        for i in range(0, len(X), 256):
            m(torch.as_tensor(X[i:i + 256], device=DEVICE))
    return m.eval()


def emb(model, X, use_flip):
    e = embed(model, X, DEVICE)
    if use_flip:
        e = e + embed(model, flip(X), DEVICE)
        e /= np.linalg.norm(e, axis=1, keepdims=True)
    return e


def load_dev():
    """بيانات التطوير: نوافذ كل فيديو (Xw، منتصفها mid، لابلها lab) + القصاصات (Xc، y، vid، spn)."""
    Xw, mid, lab, Xc, y, vid, spn = {}, {}, {}, [], [], [], []
    for v in VIDEOS:
        kp, fps = load_keypoints(v)
        kp = kp[..., :2]
        X, t0, t1 = windows(kp, fps, WIN, STEP)
        Xw[v], mid[v] = X, (t0 + t1) / 2
        lab[v] = np.array([NAMES.index(GT_TO_NAME.get(g, g)) if GT_TO_NAME.get(g, g) in NAMES else -1
                           for g in (label_at(v, t) for t in mid[v])])
        for k, name in enumerate(NAMES):
            for g in ALIASES.get(name, [name]):
                for s, e in spans(v, g):
                    clip = kp[max(0, int(round(s * fps))):min(len(kp), int(round(e * fps)))]
                    if len(clip) >= 4:
                        Xc.append(resample_to(normalize_skeleton(clip)))
                        y.append(k); vid.append(v); spn.append((s, e))
    return Xw, mid, lab, np.array(Xc, dtype=np.float32), np.array(y), np.array(vid), spn


def main():
    base = SkeletonEncoder().to(DEVICE)
    base.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    base.eval()
    txt = np.load(HERE / 'results' / 'text_embeddings.npz')['ours'].astype(np.float32)
    Xw, mid, lab, Xc, y, vid, spn = load_dev()
    print(f'قصاصات التطوير {len(y)} | نوافذ {sum(len(x) for x in Xw.values())} | صدفة {100 / 18:.1f}%\n')
    print(f'  {"الطريقة":<16}{"قصاصة كاملة":>13}{"متوسط نوافذ":>13}   {"نوافذ":>7}{"+smooth4":>9}   متشافة / مش متشافة')

    for name, use_ada, use_flip in [('base', 0, 0), ('flip', 0, 1), ('adabn', 1, 0), ('adabn+flip', 1, 1)]:
        ok_whole, ok_crop, ok_w0, ok_w4 = [], [], [], []
        seen_ok = []
        for v in VIDEOS:
            model = adabn(base, np.concatenate([Xw[o] for o in VIDEOS if o != v])) if use_ada else base
            m = vid == v
            S_c = emb(model, Xc[m], use_flip) @ txt.T
            S_w = emb(model, Xw[v], use_flip) @ txt.T
            ok_whole += list(S_c.argmax(1) == y[m])
            seen_ok += list(SEEN[y[m]])
            for (s, e), row, yy in zip([sp for sp, mm in zip(spn, m) if mm], S_c, y[m]):
                sel = (mid[v] >= s) & (mid[v] < e)
                ok_crop.append((S_w[sel].mean(0) if sel.sum() >= 2 else row).argmax() == yy)
            ok = lab[v] >= 0
            ok_w0 += list(S_w.argmax(1)[ok] == lab[v][ok])
            ok_w4 += list(smooth(S_w, 4).argmax(1)[ok] == lab[v][ok])
        seen = np.array(seen_ok)
        w = np.array(ok_whole)
        print(f'  {name:<16}{pct(w.mean()):>13}{pct(np.mean(ok_crop)):>13}   {pct(np.mean(ok_w0)):>7}'
              f'{pct(np.mean(ok_w4)):>9}   {pct(w[seen].mean())} / {pct(w[~seen].mean())}')


if __name__ == '__main__':
    main()
