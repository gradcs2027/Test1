"""
موديل الـ zero shot: BiLSTM بيحوّل هيكل الفيديو لكود 384 رقم (نفس حجم كود MiniLM).

نفس طبقات lstm/lstm_ntu60.ipynb بالظبط (BiLSTM 256 → 128 → 128 → Dense 256)،
الفرق في الآخر بس:
  النوتبوك:  Dense(عدد الحركات) + softmax  → بيختار من حركات ثابتة
  هنا:       Dense(384) + تطبيع             → كود بنقارنه بكود أي وصف

الدرجة لكل حركة = cosine(كود الفيديو، كود الوصف) × scale.
الـ scale بيتعلّم (زي CLIP) عشان الـ softmax مايبقاش مسطّح.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SkeletonEncoder(nn.Module):
    def __init__(self, in_dim=34, emb_dim=384):
        super().__init__()
        self.lstm1 = nn.LSTM(in_dim, 256, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(512, 128, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(256, 128, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.4),
            nn.Linear(256, emb_dim),
        )
        # log(1/0.07) ≈ 2.66 — نفس بداية CLIP
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))

    def forward(self, x):
        x = self.drop(self.lstm1(x)[0])
        x = self.drop(self.lstm2(x)[0])
        _, (h, _) = self.lstm3(x)
        x = self.drop(torch.cat([h[-2], h[-1]], dim=1))   # آخر حالة من الاتجاهين
        return F.normalize(self.head(x), dim=1)

    def logits(self, x, text):
        """text: (عدد الحركات، 384) متطبّع. الناتج: (batch، عدد الحركات)."""
        scale = self.logit_scale.clamp(max=np.log(100)).exp()
        return scale * self(x) @ text.T


@torch.no_grad()
def embed(model, X, device, batch=1024):
    model.eval()
    out = [model(torch.as_tensor(X[i:i + batch], dtype=torch.float32, device=device)).cpu()
           for i in range(0, len(X), batch)]
    return torch.cat(out).numpy()


def predict(video_emb, text_emb, penalty=0.0):
    """أقرب وصف لكل فيديو. الاتنين متطبّعين، فحاصل الضرب = cosine.
    penalty: بيتطرح من درجة كل وصف قبل الاختيار (تصحيح الحركات المتشافة)."""
    sim = video_emb @ text_emb.T
    return (sim - penalty).argmax(1), sim
