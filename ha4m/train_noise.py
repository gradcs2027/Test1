"""
HA4M: تدريب بضجيج يحاكي YOLO — هل بيقلّل الفجوة Kinect ← YOLO؟

    python -I ha4m/train_noise.py

1) بنقيس فرق YOLO عن Kinect على V001 (النص الأول من الفريمات بس): لكل مفصل
   انحراف معياري sigma (بوحدة طول الجذع)، ارتباط بين فريمين متتاليين rho، انحياز ثابت bias، ونسبة ضياع المفصل.
2) بنضيف نفس النوع من الضجيج على نوافذ التدريب (Kinect) وقت التدريب بس.
3) بنقارن موديل بدون ضجيج مقابل بضجيج، على نفس تقسيم الأشخاص (results/split.json):
     - test Kinect (8 أشخاص): لازم ما يقعش
     - V001 النص التاني من الفيديو، بهيكل Kinect وبهيكل YOLO (النص الأول اتقاس منه الضجيج، فمش بيتختبر عليه)

⚠️ الضجيج متقاس من شخص واحد (الشخص 1 اللي هو كمان في الـ test)؛ ده معناه إن النص التاني من V001 هو
   الاختبار الوحيد النضيف، وهو شخص واحد، فالنتيجة إشارة مش دليل.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_att  # noqa: E402
from prepare import OUT  # noqa: E402
from train_att import CFG, predict, train  # noqa: E402
from train_stream import NONE, N, aug as base_aug, frame_pred, load_raw, norm17, smooth, windows  # noqa: E402

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

DEV = train_att.DEV
STEP_RHO = 1.5      # خطوة النافذة المتطبّعة (3 فريمات @30) = 1.5 خطوة YOLO (فريم كل 2)


def measure(kin, yolo, conf):
    """بيرجّع sigma[17]، rho[17]، bias[17،2]، dropout[17] من فرق YOLO - Kinect (بوحدة طول الجذع)."""
    torso = np.linalg.norm((kin[:, 5] + kin[:, 6]) / 2 - (kin[:, 11] + kin[:, 12]) / 2, axis=-1)[:, None, None]
    e = (yolo - kin) / torso
    ok = conf > 0.5
    sig, rho, bias, drop = np.zeros(17), np.zeros(17), np.zeros((17, 2)), 1 - ok.mean(0)
    for j in range(17):
        ej = e[ok[:, j], j]
        if len(ej) < 30:
            sig[j], rho[j] = 0.15, 0.5
            continue
        bias[j] = np.median(ej, 0)
        r = ej - bias[j]
        sig[j] = 1.4826 * np.median(np.abs(r - np.median(r, 0))) + 1e-4
        both = ok[1:, j] & ok[:-1, j]
        a, b = e[1:][both, j] - bias[j], e[:-1][both, j] - bias[j]
        rho[j] = np.clip((a * b).sum() / ((b * b).sum() + 1e-9), 0, 0.95)
    return sig, rho, bias, drop


def make_noisy_aug(sig, rho, bias, drop, step=STEP_RHO):
    """step = خطوة النافذة المتطبّعة بوحدة خطوة YOLO (1.5 للنافذة 3 ث/30 خطوة، 2 للنافذة 8 ث/60 خطوة)."""
    sig_t = torch.as_tensor(sig, dtype=torch.float32, device=DEV)
    rho_t = torch.as_tensor(rho ** step, dtype=torch.float32, device=DEV)
    bias_t = torch.as_tensor(bias, dtype=torch.float32, device=DEV)
    drop_t = torch.as_tensor(drop, dtype=torch.float32, device=DEV)

    def noisy(x):
        x = base_aug(x)
        B, T = x.shape[:2]
        s = x.view(B, T, 17, 2)
        use = (torch.rand(B, 1, 1, 1, device=x.device) < 0.8).float()          # 20% نضيفة
        lvl = torch.empty(B, 1, 1, 1, device=x.device).uniform_(0.5, 1.5)
        eps = torch.randn(B, T, 17, 2, device=x.device)
        z = torch.zeros_like(eps)
        z[:, 0] = eps[:, 0]
        for t in range(1, T):          # AR(1) بنفس rho لكل مفصل، تباين ثابت
            z[:, t] = rho_t[None, :, None] * z[:, t - 1] + torch.sqrt(1 - rho_t[None, :, None] ** 2) * eps[:, t]
        noise = z * sig_t[None, None, :, None] + bias_t[None, None] * torch.rand(B, 1, 1, 1, device=x.device) * 1.2
        s = s + use * lvl * noise
        dead = torch.rand(B, T, 17, 1, device=x.device) < drop_t[None, None, :, None] * use
        s = torch.where(dead, s.mean(1, keepdim=True), s)                       # مفصل ضايع = متوسطه في النافذة
        return s.reshape(B, T, 34)
    return noisy


def v001_eval(m, prep, kin, yolo, gt, lo):
    n = len(gt)
    st = np.arange(0, n - 45 + 1, 8)
    res = {}
    for name, src in (('Kinect', kin), ('YOLO', yolo)):
        P = predict(m, prep, np.stack([norm17(src[s:s + 45]) for s in st]))
        p = frame_pred(smooth(P, 5), st + 22, n).argmax(1)
        res[name] = (p[lo:] == gt[lo:]).mean() * 100
    return res


def main():
    raw = load_raw()
    split = json.loads((HERE / 'results' / 'split.json').read_text())
    subj = {v: int(v[3:6]) for v in raw}
    W = {v: windows(*raw[v]) for v in raw}

    def pick(S, pur=0.8):
        L = [(W[v][0][W[v][2] >= pur], W[v][1][W[v][2] >= pur]) for v in raw if subj[v] in S]
        return np.concatenate([a for a, _ in L]), np.concatenate([b for _, b in L])

    Xtr, ytr = pick(split['train']); Xva, yva = pick(split['val'])
    d = np.load(OUT.parent / 'ha4m_kp' / 'IDU001V001_kp2.npz')
    yolo, conf, act = d['new'][:, :, :2], d['new'][:, :, 2], d['act']
    n = len(act)
    gt = np.array([NONE if a <= 0 else a - 1 for a in act])
    kin = raw['IDU001V001'][0][:2 * n:2]
    half = n // 2
    sig, rho, bias, drop = measure(kin[:half], yolo[:half], conf[:half])
    NM = ['nose', 'eyeL', 'eyeR', 'earL', 'earR', 'shL', 'shR', 'elbL', 'elbR', 'wriL', 'wriR', 'hipL', 'hipR', 'knL', 'knR', 'anL', 'anR']
    out = ['ضجيج YOLO مقابل Kinect (النص الأول من V001، بوحدة طول الجذع): sigma / rho / ضياع',
           ' | '.join(f'{NM[j]} {sig[j]:.2f}/{rho[j]:.2f}/{drop[j]*100:.0f}%' for j in range(17)), '']
    print('\n'.join(out), flush=True)

    noisy = make_noisy_aug(sig, rho, bias, drop)
    tests = [(v, W[v]) for v in raw if subj[v] in split['test']]
    final = {}
    for mname in ('A: LSTM', 'C: spatial att'):
        for tag, augf in (('بدون ضجيج', base_aug), ('بضجيج YOLO', noisy)):
            train_att.aug = augf
            r = {'ktest': [], 'v_kin': [], 'v_yolo': []}
            for seed in range(3):
                t0 = time.time()
                m, prep, bep = train(CFG[mname], Xtr, ytr, Xva, yva, seed)
                fa = []
                for v, (Xw, yw, _, cen) in tests:
                    y = raw[v][1]
                    p = frame_pred(smooth(predict(m, prep, Xw), 5), cen, len(y)).argmax(1)
                    fa.append(p == y)
                r['ktest'].append(np.concatenate(fa).mean() * 100)
                v1 = v001_eval(m, prep, kin, yolo, gt, half)
                r['v_kin'].append(v1['Kinect']); r['v_yolo'].append(v1['YOLO'])
                print(f'{mname} | {tag} | seed {seed}: test Kinect {r["ktest"][-1]:.1f}% | V001 نص تاني: Kinect {v1["Kinect"]:.1f}% YOLO {v1["YOLO"]:.1f}% ({time.time()-t0:.0f}s)', flush=True)
            final[(mname, tag)] = r
    f = lambda a: f'{np.mean(a):.1f}% ({", ".join(f"{x:.1f}" for x in a)})'
    out += [f'{"الموديل":<16}{"التدريب":<14}{"test Kinect (8 أشخاص)":<32}{"V001 نص تاني Kinect":<30}{"V001 نص تاني YOLO"}']
    for (mn, tag), r in final.items():
        out.append(f'{mn:<16}{tag:<14}{f(r["ktest"]):<32}{f(r["v_kin"]):<30}{f(r["v_yolo"])}')
    print('\n'.join(out[2:]))
    (HERE / 'results' / 'train_noise.txt').write_text('\n'.join(out), encoding='utf-8')


if __name__ == '__main__':
    main()
