"""
توقّع أعمى على فيديو كامل (من غير GT): نافذة بتتحرّك على الفيديو، وكل نافذة
بتتحوّل لكود بالـ BiLSTM المتدرّب وبتتقارن بكود كل وصف (MiniLM).

الأوصاف = الـ 18 بتوعنا + EXTRA (حركات جديدة مكتوبة نص بس، الموديل ماشافهاش
خالص — ده بالظبط الـ zero shot: نزوّد حركة بجملة من غير تدريب).

من غير تصحيح gamma: الخطوة 6 وضّحت إنه بيبوّظ على فيديوهاتنا (فرق الدومين).

    python zeroshot/predict_video.py vidtest5 vidtest6 [--win 3] [--step 0.5]

الناتج: results/blind_{video}.txt (الخط الزمني) و blind_{video}.npz (الدرجات
لكل نافذة، عشان لما الـ GT ييجي نقيس من غير ما نعيد التشغيل).
"""
import os
import sys
from pathlib import Path

# MiniLM متنزّل من الخطوة 3. من غير ده بيحاول يكلّم Hugging Face، ولو مفيش نت
# بيعيد المحاولة ~دقيقتين قبل النسخة المحلية. لازم قبل أي import لـ
# sentence_transformers (huggingface_hub بيقرا المتغيّر ده وقت الـ import).
os.environ.setdefault('HF_HUB_OFFLINE', '1')

import numpy as np
import torch

import _bootstrap  # noqa: F401
from paths import load_keypoints
from skeleton_norm import normalize_skeleton, resample_to

import descriptions as D
from text_embed import MODEL
from zsl_model import SkeletonEncoder, embed

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

HERE = Path(__file__).resolve().parent
CKPT = HERE.parent / '_scratch' / 'zeroshot' / 'model.pt'

EXTRA = {
    'ironing': 'a person stands at an ironing board and moves an iron back and forth over clothes',
}


def arg(name, default):
    return float(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def text_codes():
    t = np.load(HERE / 'results' / 'text_embeddings.npz')
    names = list(t['our_names'])
    codes = [t['ours'].astype(np.float32)]
    if EXTRA:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(MODEL, device='cpu')
        codes.append(st.encode(list(EXTRA.values()), normalize_embeddings=True).astype(np.float32))
        names += list(EXTRA)
    return names, np.concatenate(codes)


def windows(kp, fps, win, step):
    n, w, s = len(kp), max(4, int(round(win * fps))), max(1, int(round(step * fps)))
    starts = range(0, max(1, n - w + 1), s)
    X = np.stack([resample_to(normalize_skeleton(kp[a:a + w])) for a in starts])
    t0 = np.array([a / fps for a in starts])
    return X.astype(np.float32), t0, t0 + w / fps


def segments(t0, t1, pred, sim):
    """نوافذ ورا بعض بنفس التوقّع → مقطع واحد. الوقت = نص النافذة."""
    mid = (t0 + t1) / 2
    half = (t0[1] - t0[0]) / 2 if len(t0) > 1 else (t1[0] - t0[0]) / 2
    out, i = [], 0
    while i < len(pred):
        j = i
        while j + 1 < len(pred) and pred[j + 1] == pred[i]:
            j += 1
        out.append((max(0.0, mid[i] - half), mid[j] + half, int(pred[i]),
                    float(sim[i:j + 1, pred[i]].mean())))
        i = j + 1
    return out


def main():
    videos = [a for a in sys.argv[1:] if a.startswith('vidtest')]
    win, step = arg('--win', 3.0), arg('--step', 0.5)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SkeletonEncoder().to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    names, txt = text_codes()
    tag = {k: '' if k in D.SEEN else ' (مش متشافة)' for k in D.OURS}
    tag.update({k: ' (جديدة — نص بس)' for k in EXTRA})

    for v in videos:
        kp, fps = load_keypoints(v)
        X, t0, t1 = windows(kp[..., :2], fps, win, step)
        sim = embed(model, X, device) @ txt.T
        pred = sim.argmax(1)
        srt = np.sort(sim, 1)
        margin = srt[:, -1] - srt[:, -2]

        lines = [f'🎬 {v}: {len(kp) / fps:.1f} ثانية، {len(X)} نافذة ({win:g}s كل {step:g}s)، '
                 f'{len(names)} وصف ({len(EXTRA)} جديد) — توقّع أعمى، مفيش GT',
                 f'   cosine أعلى وصف: متوسط {srt[:, -1].mean():.2f} | '
                 f'الفرق عن التاني: متوسط {margin.mean():.3f}',
                 '', f'  {"من":>6} {"لحد":>6}  {"الحركة":<16}{"cosine":>7}']
        for a, b, k, s in segments(t0, t1, pred, sim):
            lines.append(f'  {a:6.1f} {b:6.1f}  {names[k]:<16}{s:7.2f}{tag[names[k]]}')
        cnt = np.bincount(pred, minlength=len(names))
        lines += ['', '  نسبة الوقت لكل حركة:'] + [
            f'    {names[k]:<16}{100 * cnt[k] / len(pred):5.1f}%{tag[names[k]]}'
            for k in np.argsort(-cnt) if cnt[k]]
        print('\n'.join(lines) + '\n')
        (HERE / 'results' / f'blind_{v}.txt').write_text('\n'.join(lines), encoding='utf-8')
        np.savez(HERE / 'results' / f'blind_{v}.npz', sim=sim, t0=t0, t1=t1,
                 names=np.array(names), win=win, step=step)


if __name__ == '__main__':
    main()
