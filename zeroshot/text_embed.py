"""
الخطوة 3: نحوّل وصف الحركات لأكواد (embeddings) بموديل MiniLM.

الموديل بيتنزّل مرة واحدة من Hugging Face (~90MB) وبيتخزّن في الكاش.
الناتج: results/text_embeddings.npz (كل كود متطبّع، فحاصل الضرب = cosine).
بيطبع كمان فحص سريع: كل حركة عندنا أقرب لمين في المعنى؟
"""
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

import descriptions as D

MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
OUT = Path(__file__).resolve().parent / 'results'


def main():
    OUT.mkdir(exist_ok=True)
    model = SentenceTransformer(MODEL, device='cpu')
    our_names = list(D.OURS)
    ntu = model.encode([d for _, d in D.NTU49], normalize_embeddings=True)
    ours = model.encode([D.OURS[k] for k in our_names], normalize_embeddings=True)
    np.savez(OUT / 'text_embeddings.npz', ntu=ntu, ours=ours,
             our_names=np.array(our_names), ntu_names=np.array([n for n, _ in D.NTU49]))
    print(f'✅ {MODEL}: كود من {ntu.shape[1]} رقم — {len(ntu)} حركة NTU + {len(ours)} حركة عندنا')

    sim_o = ours @ ours.T
    sim_n = ours @ ntu.T
    print(f'\n{"الحركة":<15}{"أقرب حركتين عندنا":<38}أقرب حركة NTU (غير نفس النص)')
    for i, k in enumerate(our_names):
        near = [j for j in np.argsort(-sim_o[i]) if j != i][:2]
        near_txt = '  '.join(f'{our_names[j]} {sim_o[i, j]:.2f}' for j in near)
        n = [j for j in np.argsort(-sim_n[i]) if sim_n[i, j] < 0.999][0]
        tag = '' if k in D.SEEN else '  (مش متشافة)'
        print(f'{k:<15}{near_txt:<38}{D.NTU49[n][0]} {sim_n[i, n]:.2f}{tag}')


if __name__ == '__main__':
    main()
