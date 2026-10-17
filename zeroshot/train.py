"""
الخطوة 5/6: تدريب موديل الـ zero shot وقياسه.

    python zeroshot/train.py --quick     تجربة قصيرة (20 ألف عينة، 3 epochs)
    python zeroshot/train.py             تدريب كامل مع إيقاف بدري

التدريب: لكل فيديو من الـ 43 حركة المتشافة، الموديل بيطلّع كود، ودرجة كل حركة
= cosine مع كود وصفها من MiniLM. الـ loss = cross entropy على الـ 43 وصف.
أكواد MiniLM ثابتة ومابتتدرّبش. برّه التدريب: الـ 3 المخبّية (HIDDEN) للاختبار،
والـ 3 بتوع المعايرة (CALIB) عشان نظبط gamma بس.

القياس:
  NTU مراجعة     اختيار من الـ 43 (حركات متشافة)
  NTU مخبّية     zsl: اختيار من الـ 3 المخبّية بس | gzsl: اختيار من الـ 49 كلهم
                 + H = المتوسط التوافقي لدقة المتشافة ودقة المخبّية في الـ 49
  فيديوهاتنا     اختيار من الـ 18 وصف (زي FastDTW) + تقسيم متشافة/مش متشافة
                 + zsl: القصاصات المش متشافة، اختيار من الـ 13 المش متشافة بس

gamma (calibrated stacking): رقم بيتطرح من درجة كل حركة متشافة عشان الموديل
مايبلعش الحركات الجديدة. بيتظبط على NTU بس (المتشافة + CALIB) وبعدين بيتطبّق
زي ما هو على المخبّية وعلى فيديوهاتنا.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import descriptions as D
from zsl_model import SkeletonEncoder, embed, predict

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / '_scratch' / 'zeroshot' / 'data.npz'
CKPT = HERE.parent / '_scratch' / 'zeroshot'
SEED = 0
GAMMAS = np.round(np.arange(0, 1.0001, 0.005), 3)


def load():
    d = np.load(DATA)
    t = np.load(HERE / 'results' / 'text_embeddings.npz')
    return d, t['ntu'].astype(np.float32), t['ours'].astype(np.float32), list(t['our_names'])


def acc(pred, y):
    return 100.0 * (pred == y).mean() if len(y) else float('nan')


def hmean(s, u):
    return 2 * s * u / (s + u) if s + u else 0.0


def evaluate(model, d, ntu_txt, our_txt, our_names, device, seen_ids, gamma=0.0):
    """gamma: بيتطرح من درجة كل حركة متشافة (0 = من غير تصحيح)."""
    hidden_ids = sorted(D.HIDDEN.values())
    ntu_seen = np.isin(np.arange(len(ntu_txt)), seen_ids)

    # NTU مراجعة: اختيار من الـ 43، ونفس الفيديوهات في gzsl (اختيار من الـ 49)
    y = d['y_val_seen'].astype(np.int64)
    m = np.isin(y, seen_ids)
    e = embed(model, d['X_val_seen'][m].astype(np.float32), device)
    y = y[m]
    p, _ = predict(e, ntu_txt[seen_ids])
    r = {'val_seen': acc(np.array(seen_ids)[p], y)}
    r['gzsl_seen'] = acc(predict(e, ntu_txt, ntu_seen * gamma)[0], y)

    # NTU مخبّية
    e = embed(model, d['X_val_hidden'].astype(np.float32), device)
    y = d['y_val_hidden']
    p, _ = predict(e, ntu_txt[hidden_ids])
    r['ntu_zsl'] = acc(np.array(hidden_ids)[p], y)
    r['ntu_gzsl'] = acc(predict(e, ntu_txt, ntu_seen * gamma)[0], y)
    r['ntu_h'] = hmean(r['gzsl_seen'], r['ntu_gzsl'])

    # فيديوهاتنا
    e = embed(model, d['X_ours'], device)
    y = d['y_ours']
    our_seen = np.array([k in D.SEEN for k in our_names])
    p, sim = predict(e, our_txt, our_seen * gamma)
    unseen = ~our_seen[y]
    r['ours_all'] = acc(p, y)
    r['ours_seen'] = acc(p[~unseen], y[~unseen])
    r['ours_unseen'] = acc(p[unseen], y[unseen])
    un_ids = [our_names.index(k) for k in D.UNSEEN]
    pz, _ = predict(e[unseen], our_txt[un_ids])
    r['ours_zsl'] = acc(np.array(un_ids)[pz], y[unseen])
    r['_ours_pred'], r['_ours_sim'] = p, sim
    return r


def calibrate(model, d, ntu_txt, device, seen_ids):
    """
    بيدوّر على أحسن gamma على NTU بس: فيديوهات المراجعة للـ 43 متشافة + الـ 3
    بتوع المعايرة، والاختيار من الـ 46 وصف دول — من غير الحركات المخبّية ولا
    فيديوهاتنا. أحسن gamma = أعلى H. بيرجّع (H، gamma، متشافة، مش متشافة)
    للأحسن ولـ gamma = 0.
    """
    cand = np.array(sorted(seen_ids + list(D.CALIB.values())))
    y = d['y_val_seen'].astype(np.int64)
    e = embed(model, d['X_val_seen'].astype(np.float32), device)
    seen_c, seen_y = np.isin(cand, seen_ids), np.isin(y, seen_ids)
    rows = []
    for g in GAMMAS:
        p = cand[predict(e, ntu_txt[cand], seen_c * g)[0]]
        s, u = acc(p[seen_y], y[seen_y]), acc(p[~seen_y], y[~seen_y])
        rows.append((hmean(s, u), g, s, u))
    return max(rows), rows[0]


def fmt(r):
    return (f'NTU متشافة {r["val_seen"]:.1f}% | NTU مخبّية zsl {r["ntu_zsl"]:.1f}% '
            f'gzsl {r["ntu_gzsl"]:.1f}% (متشافة {r["gzsl_seen"]:.1f}، H {r["ntu_h"]:.1f}) | '
            f'عندنا الكل {r["ours_all"]:.1f}% '
            f'(متشافة {r["ours_seen"]:.1f}، مش متشافة {r["ours_unseen"]:.1f}، zsl {r["ours_zsl"]:.1f})')


def main():
    quick = '--quick' in sys.argv
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    d, ntu_txt, our_txt, our_names = load()

    out_ids = set(D.HIDDEN.values()) | set(D.CALIB.values())
    seen_ids = sorted(set(range(len(D.NTU49))) - out_ids)
    remap = -np.ones(len(D.NTU49), dtype=np.int64)
    remap[seen_ids] = np.arange(len(seen_ids))
    y_all = d['y_train'].astype(np.int64)
    assert not np.isin(y_all, list(D.HIDDEN.values())).any(), 'فيه حركة مخبّية في التدريب'
    keep = remap[y_all] >= 0                  # بيشيل حركات المعايرة
    X, y = d['X_train'][keep], remap[y_all[keep]]

    if quick:
        idx = np.random.default_rng(SEED).choice(len(X), 20000, replace=False)
        X, y = X[idx], y[idx]
        epochs, patience = 3, 99
    else:
        epochs, patience = 60, 8

    model = SkeletonEncoder().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    txt_seen = torch.as_tensor(ntu_txt[seen_ids], device=device)
    yv_all = d['y_val_seen'].astype(np.int64)
    vkeep = remap[yv_all] >= 0
    Xv = d['X_val_seen'][vkeep].astype(np.float32)
    yv = torch.as_tensor(remap[yv_all[vkeep]], device=device)

    lines = [f'🧠 zero shot — {"تجربة قصيرة" if quick else "تدريب كامل"} على {device} | '
             f'تدريب {len(X)} عينة، {len(seen_ids)} حركة | مخبّية: {", ".join(D.HIDDEN)} | '
             f'معايرة: {", ".join(D.CALIB)}',
             '   الصدفة: عندنا الكل 1/18 = 5.6% | zsl عندنا 1/13 = 7.7% | NTU zsl 1/3 = 33.3%']
    for line in lines:
        print(line)

    best, best_state, bad = np.inf, None, 0
    for ep in range(1, epochs + 1):
        t = time.time()
        model.train()
        perm = np.random.permutation(len(X))
        tot = 0.0
        for i in range(0, len(X), 256):
            b = perm[i:i + 256]
            xb = torch.as_tensor(X[b], dtype=torch.float32, device=device)
            yb = torch.as_tensor(y[b], device=device)
            loss = F.cross_entropy(model.logits(xb, txt_seen), yb)
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
        line = (f'  epoch {ep:>2}  loss {tot / len(X):.3f}  val {vl:.3f}  '
                f'scale {model.logit_scale.exp().item():.1f}  ({time.time() - t:.0f}s)\n      {fmt(r)}')
        print(line)
        lines.append(line)
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print('  ⏹️ إيقاف بدري')
                break

    model.load_state_dict(best_state)
    (h, gamma, s, u), (h0, _, s0, u0) = calibrate(model, d, ntu_txt, device, seen_ids)
    r0 = evaluate(model, d, ntu_txt, our_txt, our_names, device, seen_ids)
    rg = evaluate(model, d, ntu_txt, our_txt, our_names, device, seen_ids, gamma)
    out = [f'\n✅ أحسن epoch (أقل val loss): {fmt(r0)}',
           f'\n⚖️ المعايرة على NTU بس (المتشافة + {", ".join(D.CALIB)}):',
           f'   من غير تصحيح:    متشافة {s0:.1f}%  معايرة {u0:.1f}%  H {h0:.1f}',
           f'   gamma = {gamma:.3f}:  متشافة {s:.1f}%  معايرة {u:.1f}%  H {h:.1f}',
           f'\n🧪 نفس gamma على الاختبار: {fmt(rg)}',
           '\n  كل حركة عندنا اتوقّعت كام مرة:      [من غير تصحيح]   [بعد التصحيح]']
    yo, p0, pg = d['y_ours'], r0['_ours_pred'], rg['_ours_pred']
    for i, k in enumerate(our_names):
        n = (yo == i).sum()
        if n or (p0 == i).any() or (pg == i).any():
            out.append(f'    {k:<15} العدد {n:>2}'
                       f'   صح {((yo == i) & (p0 == i)).sum():>2}  اتقالت {(p0 == i).sum():>2}'
                       f'   صح {((yo == i) & (pg == i)).sum():>2}  اتقالت {(pg == i).sum():>2}'
                       f'{"" if k in D.SEEN else "  (مش متشافة)"}')
    for line in out:
        print(line)
    lines += out
    tag = '_quick' if quick else ''
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CKPT / f'model{tag}.pt')
    np.savez(HERE / 'results' / f'ours_scores{tag}.npz', sim=r0['_ours_sim'], y=yo,
             vid=d['vid_ours'], names=np.array(our_names), gamma=gamma,
             seen=np.array([k in D.SEEN for k in our_names]))
    (HERE / 'results' / f'train{tag}.txt').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
