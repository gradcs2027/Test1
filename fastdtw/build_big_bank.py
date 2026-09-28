"""
بنك قوالب أكبر ومتنوّع: لحد BANK_CAP قصاصة لكل حركة (افتراضي 10) بدل 2.

ليه؟
────
البنك القديم (35 قصاصة، build_external_templates.py) كان فيه مشكلتين:
1. قليل: قصاصتين لكل حركة، وspray_perfume قصاصة واحدة.
2. مكرّر: القصاصتين بتوع hugging وhand_shake وbrush_hair وwalking جايين من
   **نفس الفيديو الأصلي** (أول اتنين بالترتيب الأبجدي) — يعني الحركة كان
   عندها مثال حقيقي واحد مش اتنين.

هنا كل حركة بتاخد قصاصات من **فيديوهات أصلية مختلفة** (ناس/مشاهد مختلفة):
بنجمّع المرشّحين بالفيديو الأصلي (group) وناخد واحد من كل group بالدور،
وترتيب الـ groups عشوائي بـ seed ثابت (عشان مانخدش أول 10 بالأبجدية وهمّا
كلهم من نفس المصدر الفرعي).

المصادر
───────
  محلي:   USER   فيديوهاتك (external_local — متسجّلة في git)
          NTU60  ntu60_2d.pkl — سكيلتون COCO-17 جاهز من غير YOLO، 12 حركة
                 من حركاتنا، ~940 قصاصة لكل واحدة من 40 شخص. group = الشخص
          CCTV   الفيديوهات متنزّلة كاملة في datasets/ — 200 قصاصة لكل حركة
                 من 100+ فيديو أصلي
  Kaggle: HMDB51 / Charades / UCF101 (مش متنزّلين كاملين على الجهاز)

كل مصدر بيكتب part_<source>.npy جنب قصاصاته. بعدين --merge بيجمعهم: لكل
حركة بياخد بالدور من كل مصدر (USER، NTU60، CCTV، HMDB51، ...) لحد BANK_CAP
→ manifest.npy — نفس الشكل اللي experiment_frame_scales بيقراه.

الاستخدام:
    python build_big_bank.py --sources user,ntu60,cctv         # محلياً
    python build_big_bank.py --sources hmdb51,charades,ucf101  # على Kaggle
    python build_big_bank.py --merge [--cap 5] [--into <فولدر>]
    BANK_DIR=<الفولدر> python experiment_ensemble.py --gt-only

متغيرات: BANK_OUT (افتراضي shared/keypoints/external_big)، BANK_CAP (10)
"""
import csv
import os
import pickle
import re
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
import build_external_templates as B
from paths import KP_OUT, ROOT

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

BANK_OUT = Path(os.environ.get('BANK_OUT', KP_OUT / 'external_big'))
CAP = int(os.environ.get('BANK_CAP', 10))
SEED = 0
MIN_FRAMES = 8
MIN_VALID = 0.6     # أقل نسبة فريمات لازم يكون فيها شخص متلاقي

# ترتيب الدور في الدمج: فيديوهاتك الأول، وبعدين الأنضف فالأقل نضافة
SOURCE_ORDER = ('USER', 'NTU60', 'CCTV', 'HMDB51', 'CHARADES', 'UCF101')

# index صفر-أساس = رقم الـ A في اسم القصاصة ناقص 1 (A001 → 0). اتأكدنا
# إن كل frame_dir في الفئة بينتهي بـ A رقمها — ونفس التأكيد جوّه src_ntu60
NTU60_TO_LABEL = {
    0: 'drink_water',    # A001 drink water
    3: 'brush_hair',     # A004 brush hair
    7: 'sitting',        # A008 sitting down
    8: 'stand_up',       # A009 standing up
    9: 'clapping',       # A010 clapping
    17: 'wear_glasses',  # A018 wear on glasses
    22: 'wave',          # A023 hand waving
    27: 'phone_call',    # A028 make a phone call
    43: 'touch_head',    # A044 touch head (headache)
    54: 'hugging',       # A055 hugging other person
    57: 'hand_shake',    # A058 handshaking
    58: 'walking',       # A059 walking towards each other
}
NTU60_URL = 'https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ntu60_2d.pkl'
NTU60_CACHE = ROOT / '_scratch' / 'ntu60_2d.pkl'

# HMDB51 عنده كمان sit / stand / walk / run — مصدر تاني للحركات دي غير CCTV
HMDB_TO_LABEL = {**B.HMDB_TO_LABEL,
                 'sit': 'sitting', 'stand': 'stand_up', 'walk': 'walking', 'run': 'running'}

# Charades بالاسم مش بالكود — الكود بيتحل من Charades_v1_classes.txt ويتطبع
CHARADES_BY_NAME = {
    'talking on a phone': 'phone_call',
    'turning on a light': 'turn_on_light',
    'awakening': 'wake_up',
    'drinking from': 'drink_water',
    'standing up': 'stand_up',
    'sitting down': 'sitting',
}
CHARADES_MIN_SEC, CHARADES_MAX_SEC = 1.5, 20.0


# ─────────────────── اختيار متنوّع ───────────────────

def diverse_order(cands, group):
    """أول مرشّح من كل فيديو أصلي، بعدين التاني من كل واحد... بترتيب ثابت."""
    rng = np.random.default_rng(SEED)
    groups = {}
    for c in cands:
        groups.setdefault(group(c), []).append(c)
    keys = sorted(groups)
    keys = [keys[i] for i in rng.permutation(len(keys))]
    for k in keys:
        rng.shuffle(groups[k])
    out = []
    for r in range(max(map(len, groups.values()), default=0)):
        out += [groups[k][r] for k in keys if r < len(groups[k])]
    return out


def good(kp):
    if len(kp) < MIN_FRAMES:
        return False
    found = np.abs(kp).reshape(len(kp), -1).sum(axis=1) > 0
    return found.mean() >= MIN_VALID


def collect(source, label, cands, group, load, desc):
    """بيمشي على المرشّحين بالترتيب المتنوّع ويستخرج لحد CAP قصاصة سليمة."""
    n_groups = len({group(c) for c in cands})
    print(f'\n🎬 [{source}] {label}: {len(cands)} مرشّح من {n_groups} فيديو أصلي')
    picked, rejected = [], 0
    for c in diverse_order(cands, group):
        if len(picked) >= CAP:
            break
        t0 = time.perf_counter()
        kp = np.asarray(load(c), dtype=np.float32)
        if not good(kp):
            rejected += 1
            print(f'   ✗ {desc(c)}: {len(kp)} فريم، الشخص مش ظاهر كفاية — اتخطّى')
            continue
        f = f'{label}__{source.lower()}{len(picked)}.npy'
        np.save(BANK_OUT / f, kp)
        picked.append({'label': label, 'source': source, 'group': str(group(c)),
                       'clip': desc(c), 'file': f, 'n_frames': int(len(kp))})
        print(f'   ✓ {desc(c)}: {len(kp)} فريم ({time.perf_counter() - t0:.1f}s)')
    groups = len({p['group'] for p in picked})
    print(f'   → {len(picked)}/{CAP} ({groups} فيديو أصلي مختلف، {rejected} اترفض)')
    return picked


# ─────────────────── المصادر ───────────────────

def src_user(model):
    manifest = np.load(B.LOCAL_EXT_DIR / 'manifest_local.npy', allow_pickle=True)
    out = []
    for m in manifest:
        if m['source'] != 'USER':
            continue
        f = f"{m['label']}__user0.npy"
        shutil.copy(B.LOCAL_EXT_DIR / m['file'], BANK_OUT / f)
        out.append({'label': m['label'], 'source': 'USER', 'group': m['clip'],
                    'clip': m['clip'], 'file': f, 'n_frames': int(m['n_frames'])})
    print(f'\n📦 [USER] {len(out)} قصاصة من فيديوهاتك')
    return out


def src_ntu60(model):
    if not NTU60_CACHE.exists():
        NTU60_CACHE.parent.mkdir(parents=True, exist_ok=True)
        print('📥 تحميل ntu60_2d.pkl (~700MB)...')
        urllib.request.urlretrieve(NTU60_URL, NTU60_CACHE)
    with open(NTU60_CACHE, 'rb') as f:
        data = pickle.load(f)
    by_label = {}
    for a in data['annotations']:
        if a['label'] in NTU60_TO_LABEL:
            assert a['frame_dir'].endswith(f"A{a['label'] + 1:03d}"), a['frame_dir']
            by_label.setdefault(NTU60_TO_LABEL[a['label']], []).append(a)
    out = []
    for label, cands in sorted(by_label.items()):
        out += collect('NTU60', label, cands,
                       group=lambda a: re.search(r'P\d+', a['frame_dir']).group(),
                       load=lambda a: a['keypoint'][0],       # أول شخص: (frames, 17, 2)
                       desc=lambda a: a['frame_dir'])
    return out


def src_cctv(model):
    found = B._find_clips_by_suffix(
        B.CCTV_TO_LABEL, r'_\d+', 'jonathannield/cctv-action-recognition-dataset')
    out = []
    for key, label in B.CCTV_TO_LABEL.items():
        # NTU_fight0027_sit_1.mp4 → الفيديو الأصلي NTU_fight0027
        out += collect('CCTV', label, found[key],
                       group=lambda p, k=key: re.sub(rf'_{k}_\d+$', '', p.stem, flags=re.I),
                       load=lambda p: B._extract_from_video(p, model),
                       desc=lambda p: p.name)
    return out


def src_hmdb51(model):
    root = B._find_rawframes_root('clap', 'jizeyong/hmdb51')
    print(f'\n📁 HMDB51: {root}')
    out = []
    for cls, label in HMDB_TO_LABEL.items():
        d = root / cls
        if not d.is_dir():
            print(f'  ⚠️ مالقتش فولدر {cls} — اتخطّى')
            continue
        cands = sorted(p for p in d.iterdir() if p.is_dir())
        # April_09_brush_hair_u_nm_np1_ba_goo_0 → الفيديو الأصلي April_09
        out += collect('HMDB51', label, cands,
                       group=lambda p, c=cls: p.name.rsplit(f'_{c}_', 1)[0],
                       load=lambda p: B._extract_from_frames(p, model),
                       desc=lambda p: p.name)
    return out


def charades_codes(csv_path):
    """كود Charades → حركتنا، بالاسم من Charades_v1_classes.txt لو موجود."""
    root = B._dataset_dir('jizeyong/charades')
    cls_file = next((p for d in (csv_path.parent, csv_path.parent.parent, root)
                     for p in d.glob('Charades_v1_classes.txt')), None)
    if cls_file is None:
        cls_file = next(root.glob('*/Charades_v1_classes.txt'), None)
    if cls_file is None:
        print('⚠️ مالقتش Charades_v1_classes.txt — هستخدم الأكواد المعروفة بس')
        return dict(B.CHARADES_TO_LABEL)
    codes = {}
    print(f'\n📋 أكواد Charades من {cls_file.name}:')
    for line in cls_file.read_text(encoding='utf-8').splitlines():
        code, _, name = line.strip().partition(' ')
        for pat, label in CHARADES_BY_NAME.items():
            if pat in name.lower():
                codes[code] = label
                print(f'   {code} "{name}" → {label}')
    for code, label in B.CHARADES_TO_LABEL.items():
        if codes.get(code) != label:
            raise RuntimeError(f'الكود {code} المعروف ({label}) مش متطابق مع ملف الفئات')
    return codes


def src_charades(model):
    csv_path, rgb_start = B._find_charades_root()
    codes = charades_codes(csv_path)
    cands = {}
    with open(csv_path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            for trip in (row.get('actions') or '').split(';'):
                parts = trip.split()
                if len(parts) != 3 or parts[0] not in codes:
                    continue
                s, e = float(parts[1]), float(parts[2])
                if CHARADES_MIN_SEC <= e - s <= CHARADES_MAX_SEC:
                    # group = الشخص اللي بيمثّل (عمود subject)، مش الفيديو بس
                    cands.setdefault(codes[parts[0]], []).append(
                        (parts[0], row['id'], row.get('subject') or row['id'], s, e))
    sample_ids = [c[1] for cs in cands.values() for c in cs[:3]]
    rgb_dir = B._resolve_frames_dir(rgb_start, sample_ids)
    out = []
    for label, cs in sorted(cands.items()):
        out += collect('CHARADES', label, cs,
                       group=lambda c: c[2],
                       load=lambda c: B._extract_charades_clip(rgb_dir, c[1], c[3], c[4], model),
                       desc=lambda c: f'{c[1]}_{c[0]}_{c[3]:.1f}-{c[4]:.1f}')
    return out


def src_ucf101(model):
    found = B._find_clips_by_suffix(
        B.UCF101_TO_LABEL, r'_g\d+_c\d+', 'matthewjansen/ucf101-action-recognition')
    out = []
    for key, label in B.UCF101_TO_LABEL.items():
        # نفس الملف ممكن يتكرر في train/val/test — بنشيل التكرار بالاسم
        uniq = list({p.name: p for p in found[key]}.values())
        out += collect('UCF101', label, uniq,
                       group=lambda p: re.search(r'_g(\d+)_c', p.name).group(1),
                       load=lambda p: B._extract_from_video(p, model),
                       desc=lambda p: p.name)
    return out


SOURCES = {'user': src_user, 'ntu60': src_ntu60, 'cctv': src_cctv,
           'hmdb51': src_hmdb51, 'charades': src_charades, 'ucf101': src_ucf101}
NEEDS_YOLO = {'cctv', 'hmdb51', 'charades', 'ucf101'}


# ─────────────────── الدمج ───────────────────

def merge(cap, into):
    by_label = {}
    for f in sorted(BANK_OUT.glob('part_*.npy')):
        for m in np.load(f, allow_pickle=True):
            by_label.setdefault(m['label'], {}).setdefault(m['source'], []).append(dict(m))
    if not by_label:
        raise FileNotFoundError(f'مفيش part_*.npy في {BANK_OUT} — شغّل --sources الأول')

    into.mkdir(parents=True, exist_ok=True)
    manifest = []
    print(f'\n{"الحركة":<16}{"العدد":>6}   المصادر')
    for label in sorted(by_label):
        srcs = [s for s in SOURCE_ORDER if s in by_label[label]]
        chosen = []
        for r in range(max(len(by_label[label][s]) for s in srcs)):
            for s in srcs:
                if len(chosen) < cap and r < len(by_label[label][s]):
                    chosen.append(by_label[label][s][r])
        for m in chosen:
            if into != BANK_OUT:
                shutil.copy(BANK_OUT / m['file'], into / m['file'])
        manifest += chosen
        counts = {s: sum(m['source'] == s for m in chosen) for s in srcs}
        mix = '  '.join(f'{s}×{c}' for s, c in counts.items() if c)
        flag = '  ⚠️' if len(chosen) < cap else ''
        print(f'{label:<16}{len(chosen):>6}   {mix}{flag}')

    np.save(into / 'manifest.npy', manifest, allow_pickle=True)
    print(f'\n✅ {len(manifest)} قصاصة، {len(by_label)} حركة → {into / "manifest.npy"}')


def main():
    args = sys.argv[1:]
    if '--merge' in args:
        cap = int(args[args.index('--cap') + 1]) if '--cap' in args else CAP
        into = Path(args[args.index('--into') + 1]) if '--into' in args else BANK_OUT
        merge(cap, into)
        return

    names = args[args.index('--sources') + 1].split(',') if '--sources' in args else list(SOURCES)
    unknown = set(names) - set(SOURCES)
    assert not unknown, f'مصادر مش معروفة: {unknown} — المتاح: {list(SOURCES)}'
    BANK_OUT.mkdir(parents=True, exist_ok=True)
    print(f'🏦 بنك كبير: لحد {CAP} قصاصة لكل حركة من كل مصدر → {BANK_OUT}')

    model = None
    if NEEDS_YOLO & set(names):
        from ultralytics import YOLO
        local_w = Path(__file__).with_name('yolov8n-pose.pt')
        model = YOLO(str(local_w) if local_w.exists() else 'yolov8n-pose.pt')

    t0 = time.perf_counter()
    for name in names:
        part = SOURCES[name](model)
        np.save(BANK_OUT / f'part_{name}.npy', part, allow_pickle=True)
        labels = sorted({m['label'] for m in part})
        print(f'\n💾 part_{name}.npy: {len(part)} قصاصة، {len(labels)} حركة '
              f'({time.perf_counter() - t0:.0f}s)')


if __name__ == '__main__':
    main()
