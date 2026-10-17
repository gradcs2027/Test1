"""
تدريب zero shot بـ augmentation أقوى وقت التدريب (على الـ GPU) — الهدف: تصغير فرق NTU ← كاميرتنا.

    python zeroshot/train_aug.py --aug none   --seed 0     نفس train.py بالظبط (خط الأساس)
    python zeroshot/train_aug.py --aug strong --seed 0

التشخيص (results/train.txt): NTU متشافة 83% لكن فيديوهاتنا المتشافة 42% — الموديل
بيحفظ Kinect. البيانات المخزّنة فيها 3 نسخ augmentation ثابتة بس (دوران ±12°،
مقياس، رعشة). الـ strong بيضيف كل batch جديد:
  yaw     x × U(0.55, 1) — الشخص متلفّف عن الكاميرا (NTU 3 كاميرات ثابتة بس)
  shear   x += k·y       — ميلان الكاميرا
  دوران   ±17° + مقياس 0.8–1.2
  ضوضاء   لكل مفصل في كل فريم + انجراف بطيء لكل مفصل (رعشة YOLO)
  إخفاء   مفاصل عشوائية 8% + نص سفلي (الرجلين) 15%
  سرعة    قص زمني 50–100% من الحركة وتمديده على 30 فريم
اختيار أحسن epoch = أقل val loss على NTU بس (زي train.py). فيديوهاتنا بتتقاس
في الآخر ومابتدخلش الاختيار. الاختبار المخبّي (vidtest5/6) مش هنا.
"""
import argparse
import os
import sys
import time

os.environ.setdefault('HF_HUB_OFFLINE', '1')

import numpy as np
import torch
import torch.nn.functional as F

import _bootstrap  # noqa: F401
import descriptions as D
from improve_domain import load_dev
from prepare_data import COCO_FLIP
from train import CKPT, HERE, calibrate, evaluate, fmt, load
from zsl_model import SkeletonEncoder, embed

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

FLIP = torch.as_tensor(COCO_FLIP)
LOWER = torch.arange(11, 17)


def strong_aug(x):
    """x: (B، 30، 34) على الـ GPU، متطبّع بالفعل (مركزه الحوض). بيرجّع نفس الشكل."""
    B, T, _ = x.shape
    dev = x.device
    s = x.view(B, T, 17, 2)

    # سرعة: قص زمني بنسبة عشوائية وتمديد
    span = torch.empty(B, device=dev).uniform_(0.5, 1.0)
    lo = torch.rand(B, device=dev) * (1 - span)
    pos = (lo[:, None] + span[:, None] * torch.linspace(0, 1, T, device=dev)) * (T - 1)
    i0 = pos.floor().long().clamp(0, T - 2)
    w = (pos - i0)[..., None, None]
    g = lambda i: torch.gather(s, 1, i[..., None, None].expand(-1, -1, 17, 2))
    s = g(i0) * (1 - w) + g(i0 + 1) * w
    keep_time = torch.rand(B, device=dev) < 0.3               # 30% من العينات من غير قص
    s = torch.where(keep_time[:, None, None, None], x.view(B, T, 17, 2), s)

    # قلب
    flip = torch.rand(B, device=dev) < 0.5
    sf = s[:, :, FLIP.to(dev), :] * torch.tensor([-1.0, 1.0], device=dev)
    s = torch.where(flip[:, None, None, None], sf, s)

    # yaw + shear + دوران + مقياس (كلها حول الحوض = نقطة الأصل)
    yaw = torch.where(torch.rand(B, device=dev) < 0.6,
                      torch.empty(B, device=dev).uniform_(0.55, 1.0), torch.ones(B, device=dev))
    shear = torch.empty(B, device=dev).uniform_(-0.25, 0.25) * (torch.rand(B, device=dev) < 0.5)
    th = torch.empty(B, device=dev).uniform_(-0.3, 0.3)
    sc = torch.empty(B, device=dev).uniform_(0.8, 1.2)
    c, sn = th[:, None, None].cos(), th[:, None, None].sin()
    px, py = s[..., 0] * yaw[:, None, None], s[..., 1]
    px = px + shear[:, None, None] * py
    s = torch.stack([px * c - py * sn, px * sn + py * c], -1) * sc[:, None, None, None]

    # ضوضاء: لكل فريم + انجراف بطيء لكل مفصل
    sig = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.0, 0.05)
    s = s + torch.randn_like(s) * sig
    drift = torch.randn(B, 4, 17, 2, device=dev) * 0.04
    drift = F.interpolate(drift.permute(0, 2, 3, 1).reshape(B, 34, 4), size=T, mode='linear',
                          align_corners=True).reshape(B, 17, 2, T).permute(0, 3, 1, 2)
    s = s + drift

    # إخفاء: مفاصل عشوائية + الرجلين
    dead = torch.rand(B, 1, 17, 1, device=dev) < 0.08
    legs = (torch.rand(B, 1, 1, 1, device=dev) < 0.15) & torch.isin(torch.arange(17, device=dev), LOWER.to(dev))[None, None, :, None]
    mean = s.mean(1, keepdim=True)
    s = torch.where(dead | legs, mean, s)
    return s.reshape(B, T, 34)


def dev_eval(model, dev, our_txt, device):
    """قياس على التطوير: قصاصة كاملة (40) + نوافذ 3 ثواني (528، اللي عليها لابل)."""
    Xw, mid, lab, Xc, y, vid, spn = dev
    seen = np.array([k in D.SEEN for k in D.OURS])
    p = (embed(model, Xc, device) @ our_txt.T).argmax(1)
    ok_clip = p == y
    ok_win = []
    for v in Xw:
        pw = (embed(model, Xw[v], device) @ our_txt.T).argmax(1)
        m = lab[v] >= 0
        ok_win += list(pw[m] == lab[v][m])
    return dict(clip=ok_clip.mean() * 100, seen=ok_clip[seen[y]].mean() * 100,
                unseen=ok_clip[~seen[y]].mean() * 100, win=np.mean(ok_win) * 100,
                top=np.bincount(p, minlength=18).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--aug', choices=['none', 'strong'], default='strong')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=8)
    a = ap.parse_args()
    tag = f'{a.aug}_s{a.seed}'
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    d, ntu_txt, our_txt, our_names = load()
    dev = load_dev()

    out_ids = set(D.HIDDEN.values()) | set(D.CALIB.values())
    seen_ids = sorted(set(range(len(D.NTU49))) - out_ids)
    remap = -np.ones(len(D.NTU49), dtype=np.int64)
    remap[seen_ids] = np.arange(len(seen_ids))
    y_all = d['y_train'].astype(np.int64)
    assert not np.isin(y_all, list(D.HIDDEN.values())).any(), 'فيه حركة مخبّية في التدريب'
    keep = remap[y_all] >= 0
    X = torch.as_tensor(d['X_train'][keep].astype(np.float32), device=device)
    y = torch.as_tensor(remap[y_all[keep]], device=device)
    txt_seen = torch.as_tensor(ntu_txt[seen_ids], device=device)
    yv_all = d['y_val_seen'].astype(np.int64)
    vkeep = remap[yv_all] >= 0
    Xv = d['X_val_seen'][vkeep].astype(np.float32)
    yv = torch.as_tensor(remap[yv_all[vkeep]], device=device)

    model = SkeletonEncoder().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    lines = [f'🧠 {tag} — تدريب {len(X)} عينة، {len(seen_ids)} حركة على {device}']
    print(lines[0])

    best, best_state, bad = np.inf, None, 0
    for ep in range(1, a.epochs + 1):
        t = time.time()
        model.train()
        perm = torch.randperm(len(X), device=device)
        tot = 0.0
        for i in range(0, len(X), 256):
            b = perm[i:i + 256]
            xb = strong_aug(X[b]) if a.aug == 'strong' else X[b]
            loss = F.cross_entropy(model.logits(xb, txt_seen), y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            vl = sum(F.cross_entropy(model.logits(torch.as_tensor(Xv[i:i + 1024], device=device), txt_seen),
                                     yv[i:i + 1024], reduction='sum').item()
                     for i in range(0, len(Xv), 1024)) / len(Xv)
        sched.step(vl)
        r = evaluate(model, d, ntu_txt, our_txt, our_names, device, seen_ids)
        line = (f'  epoch {ep:>2}  loss {tot / len(X):.3f}  val {vl:.3f}  ({time.time() - t:.0f}s)  '
                f'NTU متشافة {r["val_seen"]:.1f}%  عندنا {r["ours_all"]:.1f}% (متشافة {r["ours_seen"]:.1f}، '
                f'مش متشافة {r["ours_unseen"]:.1f})')
        print(line)
        lines.append(line)
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                print('  ⏹️ إيقاف بدري')
                break

    model.load_state_dict(best_state)
    r = evaluate(model, d, ntu_txt, our_txt, our_names, device, seen_ids)
    e = dev_eval(model, dev, our_txt, device)
    final = [f'\n✅ {tag} أحسن epoch (أقل val loss NTU): {fmt(r)}',
             f'   التطوير: قصاصة {e["clip"]:.1f}% (متشافة {e["seen"]:.1f}، مش متشافة {e["unseen"]:.1f}) | '
             f'نوافذ {e["win"]:.1f}% | أكتر وصف اتقال {e["top"]} من 40']
    for line in final:
        print(line)
    lines += final
    torch.save(model.state_dict(), CKPT / f'model_{tag}.pt')
    (HERE / 'results' / f'train_{tag}.txt').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
