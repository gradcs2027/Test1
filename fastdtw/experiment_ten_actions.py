"""
عشر حركات من HMDB51: قصاصتين تدريب + قصاصات اختبار لكل حركة.

ليه الملف ده موجود؟
────────────────────
طلب صريح من صاحب المشروع (2026-09-19): بلاش فيديوهاتنا خالص (لا
vidtest1-4 ولا الفيديوهات المصوّرة في البيت) — التجربة تبقى **جوّه
الداتاسِت** من أولها لآخرها. نختار عشر حركات سهلة يقدر يعملها في البيت،
نمرّن على قصاصتين لكل حركة، ونختبر على قصاصات تانية من نفس الداتاسِت،
ونشوف بعينينا الخوارزمية شغّالة ولا لأ.

ليه ده أنضف من اللي قبله؟
- **صفر تسريب**: قصاصات التدريب وقصاصات الاختبار فيديوهات مختلفة
  بالكامل، بناس مختلفة وكاميرات مختلفة.
- **صفر فجوة مجال مصطنعة**: التدريب والاختبار من نفس الداتاسِت. لو
  الرقم طلع واطي هنا، يبقى المشكلة في الخوارزمية نفسها مش في اختلاف
  الكاميرا — وده اللي عايزين نعرفه.
- **داتاسِت واحد بس** (jizeyong/hmdb51): فورمات موحّد، قارئ واحد
  مجرّب، و~100 قصاصة متاحة لكل حركة.

التمثيل المستخدم وليه
──────────────────────
اتقاس محلياً على 156 نافذة في 2026-09-19، والفرق مش بسيط:

    الجزء العلوي + موضع (اللي هنا)       26.3%
    كل الـ17 مفصل + موضع                 21.2%
    أي مفاصل + **سرعة** (اللي كان شغّال)  3.8%  ← تحت الصدفة (16.7%)

⚠️ `to_features(mode='vel')` — اللي experiment_frame_scales.py لسه
   مستخدمه — بيطلّع 3.8% وبيتنبّأ بنفس الحركة لكل نافذة في الداتا. عشان
   كده الملف ده بيبني التمثيل بنفسه بدل ما ينادي `to_features`.

`normalize_window` بيعمل تلات حاجات مهمة لوحده: بيملا الفريمات الضايعة،
بيطرح مرساة حوض واحدة للنافذة كلها (فبيحتفظ بحركة الجسم رأسياً — ده
الفرق بين القعود والوقوف)، وبيقسم على طول الجذع (فبعد الكاميرا مايفرقش).
اللي زايد هنا: أخد المفاصل العلوية بس، وإعادة أخذ العيّنات لعدد فريمات
ثابت.

الاستخدام (على Kaggle، بعد ما jizeyong/hmdb51 يتوصّل بالنوتبوك):
    python fastdtw/experiment_ten_actions.py

أول تشغيلة بتستخرج الـ pose وبتاخد وقت (~70 قصاصة). النتيجة بتتخزّن في
KP_OUT/ten_actions/ فأي تشغيلة بعد كده بتبقى في ثواني — يعني تقدر تعدّل
في التمثيل وتعيد التجربة من غير ما تستنى الاستخراج تاني.
"""
import sys
import time

import numpy as np

import _bootstrap  # noqa: F401
from build_external_templates import _extract_from_frames, _find_rawframes_root
from classifier import norm_distance, normalize_window, resample_linear
from paths import KP_OUT, out_dir

if hasattr(sys.stdout, 'reconfigure'):
    # من غير line_buffering الاستخراج بيفضل شكله واقف عشر دقايق في نوتبوك
    # Kaggle لأن المخرجات رايحة لأنبوبة مش لشاشة. حصل فعلاً 2026-09-18.
    sys.stdout.reconfigure(encoding='utf-8', errors='replace',
                           line_buffering=True)

# فئة HMDB51 → اسمها بالعربي. العشرة دول كلهم يتعملوا في البيت قدام
# كاميرا عادية من غير أي أدوات، وكل واحد فيهم شكله الحركي مختلف عن التاني.
TEN_ACTIONS = {
    'clap':       'تصفيق',
    'wave':       'تلويح',
    'drink':      'شرب',
    'brush_hair': 'تسريح الشعر',
    'sit':        'قعود',
    'stand':      'وقوف',
    'walk':       'مشي',
    'pushup':     'تمرين ضغط',
    'situp':      'تمرين بطن',
    'pick':       'التقاط حاجة من الأرض',
}

N_TRAIN = 2      # اللي المستخدم طلبه بالظبط: قصاصتين تدريب لكل حركة
N_TEST = 5       # أكتر من واحدة عشان الرقم يبقى ليه معنى — 50 اختبار مش 10
N_FRAMES = 30    # كل القصاصات بترجع للطول ده قبل المقارنة
RADIUS = 1

# الأنف + الكتفين + الكوعين + الرسغين (ترقيم COCO-17). الرجلين مطرودين
# بقصد: اتقاسوا وبيوقّعوا الدقة (21.2% بدل 26.3%) لأن كشف الرجلين أضعف
# بكتير من كشف الجزء العلوي، فبيدخّلوا ضوضاء أكتر من إشارة.
UPPER = [0, 5, 6, 7, 8, 9, 10]

CACHE_DIR = KP_OUT / 'ten_actions'
OUT_DIR = out_dir(__file__)


# ==============================================================================
# التمثيل
# ==============================================================================

def pose_features(kp, n_frames=N_FRAMES):
    """(frames, 17, 2) خام -> (n_frames, 14) جاهزة للمقارنة بالـ DTW."""
    seq = normalize_window(kp)                       # (frames, 34)
    seq = resample_linear(seq, n=n_frames)
    pts = seq.reshape(len(seq), 17, 2)[:, UPPER, :]
    return np.ascontiguousarray(pts.reshape(len(seq), -1), dtype=np.float64)


def detect_rate(kp):
    """نسبة الفريمات اللي اتكشف فيها شخص فعلاً.

    ⚠️ الدالة دي مش رفاهية: `_extract_frame_files` بيحطّ صفر مكان أي فريم
       مافيهوش شخص، **من غير أي تحذير**. قصاصة اتكشف فيها 20% بس بتبقى
       80% أصفار وشكلها قصاصة سليمة تماماً في الملف.
    """
    if len(kp) == 0:
        return 0.0
    return float(np.mean(np.any(kp != 0, axis=(1, 2))))


# ==============================================================================
# الاستخراج (مرة واحدة، بيتخزّن)
# ==============================================================================

def _cache_path(action, idx):
    return CACHE_DIR / f'{action}_{idx:02d}.npy'


def _source_video(clip_name, action):
    """عنوان الفيديو الأصلي اللي القصاصة دي اتقصّت منه.

    أسامي HMDB51 شكلها ثابت:
        <عنوان الفيديو>_<الحركة>_<f|u>_<nm|cm>_<np1|np2>_<اتجاه>_<جودة>_<رقم>
    فاللي قبل `_<الحركة>_` هو الفيديو الأصلي.
    """
    marker = f'_{action}_'
    return clip_name.rsplit(marker, 1)[0] if marker in clip_name else clip_name


def pick_diverse_clips(class_dir, action, n):
    """n قصاصة من **n فيديو أصلي مختلف** — قصاصة واحدة بالكتير من كل فيديو.

    ⚠️ ليه مش `_pick_clips` العادية (أول n بالترتيب الأبجدي)؟ لأن HMDB51
       بيقصّ أكتر من قصاصة من نفس الفيديو الأصلي، وبتبقى ورا بعض بالظبط
       في الترتيب الأبجدي:
           April_09_brush_hair_u_nm_np1_ba_goo_0
           April_09_brush_hair_u_nm_np1_ba_goo_1
       يعني أول 7 قصاصات ممكن يطلعوا كلهم نفس الشخص في نفس الفيديو —
       فالقصاصتين بتوع التدريب والخمسة بتوع الاختبار يبقوا نفس الشخص
       ونفس الكاميرا. ده تسريب بيرفع الدقة بشكل كداب تماماً: اتقاس
       محلياً 2026-09-19 إن تصنيف قطعة بقطعة من نفس الفيديو بيدّي
       **90.5%**، وبفيديوهات مختلفة **26.3%** — بنفس الكود بالظبط.
    """
    by_source = {}
    for p in sorted(d for d in class_dir.iterdir() if d.is_dir()):
        by_source.setdefault(_source_video(p.name, action), p)
    return list(by_source.values())[:n]


def extract_clips():
    """
    بيرجّع {فئة: [(اسم القصاصة, keypoints), ...]} لكل العشر حركات.

    اللي اتخزّن قبل كده بيتقرا من الملف على طول — الاستخراج بيتعمل بس
    للقصاصات الجديدة. يعني تعديل التمثيل وإعادة التجربة بتاخد ثواني.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    n_want = N_TRAIN + N_TEST

    missing = {a for a in TEN_ACTIONS
               if any(not _cache_path(a, i).exists() for i in range(n_want))}
    model = None
    root = None
    if missing:
        from ultralytics import YOLO
        print(f'📥 تحميل YOLOv8n-pose ({len(missing)} حركة محتاجة استخراج)...')
        model = YOLO('yolov8n-pose.pt')
        root = _find_rawframes_root('clap', 'jizeyong/hmdb51')
        print(f'📁 فولدرات حركات HMDB51 في: {root}')
    else:
        print('📦 كل القصاصات متخزّنة من تشغيلة قبل كده — مفيش استخراج')

    # اسم الفيديو الأصلي لكل قصاصة متخزّنة — عشان نقدر نطبع إثبات إن
    # التدريب والاختبار فعلاً من فيديوهات مختلفة، مش كلام.
    sources_file = CACHE_DIR / 'sources.npy'
    sources = (np.load(sources_file, allow_pickle=True).item()
               if sources_file.exists() else {})

    data = {}
    for action, arabic in TEN_ACTIONS.items():
        clips = []
        src_dirs = None
        for i in range(n_want):
            path = _cache_path(action, i)
            if path.exists():
                clips.append((path.stem, np.load(path)))
                continue

            if src_dirs is None:
                class_dir = root / action
                if not class_dir.is_dir():
                    raise FileNotFoundError(
                        f'مالقتش فولدر الفئة "{action}" جوّه {root} — '
                        f'اتأكد إن الاسم ده فعلاً موجود في HMDB51')
                src_dirs = pick_diverse_clips(class_dir, action, n_want)
                if len(src_dirs) < n_want:
                    raise RuntimeError(
                        f'الفئة "{action}" فيها {len(src_dirs)} فيديو أصلي '
                        f'مختلف بس، والتجربة محتاجة {n_want} ({N_TRAIN} تدريب '
                        f'+ {N_TEST} اختبار من فيديوهات مختلفة)')

            t0 = time.perf_counter()
            kp = _extract_from_frames(src_dirs[i], model)
            np.save(path, kp)
            sources[path.stem] = _source_video(src_dirs[i].name, action)
            np.save(sources_file, sources, allow_pickle=True)
            print(f'  {action:<11} [{i + 1}/{n_want}] {src_dirs[i].name[:40]:<42}'
                  f' {len(kp):>4} فريم  كشف {detect_rate(kp) * 100:5.1f}%'
                  f'  ({time.perf_counter() - t0:.0f}s)')
            clips.append((path.stem, kp))

        data[action] = clips
        n_src = len({sources.get(c[0], c[0]) for c in clips})
        print(f'✅ {action:<11} ({arabic}) — {len(clips)} قصاصة '
              f'من {n_src} فيديو أصلي مختلف')

    return data, sources


# ==============================================================================
# التصنيف
# ==============================================================================

def classify(feat, templates):
    """أقرب حركة + ثقة الهامش. كل حركة عندها نفس عدد القصاصات بالظبط،
    فمفيش انحياز لحركة عشان قصاصاتها أكتر."""
    per_label = {}
    for label, tfeat in templates:
        d = norm_distance(feat, tfeat, radius=RADIUS)
        if label not in per_label or d < per_label[label]:
            per_label[label] = d

    ranked = sorted(per_label.items(), key=lambda kv: kv[1])
    best, best_d = ranked[0]
    second_d = ranked[1][1] if len(ranked) > 1 else 0.0
    conf = (second_d - best_d) / second_d if second_d > 1e-9 else 1.0
    return best, best_d, conf, ranked


def main():
    print('=' * 74)
    print('  🎬 عشر حركات من HMDB51 — قصاصتين تدريب واختبار على قصاصات تانية')
    print('=' * 74)

    data, sources = extract_clips()

    # ── إثبات عدم التسريب: الفيديو الأصلي لكل قصاصة تدريب واختبار ──
    print(f'\n{"=" * 74}')
    print('  🔍 الفيديوهات الأصلية — التدريب والاختبار لازم يبقوا مختلفين')
    print(f'{"=" * 74}')
    leaks = []
    for action, clips in data.items():
        names = [sources.get(stem, stem) for stem, _ in clips]
        train, test = names[:N_TRAIN], names[N_TRAIN:]
        overlap = set(train) & set(test)
        leaks += [(action, o) for o in overlap]
        print(f'  {"⚠️" if overlap else "✓"} {action:<11} '
              f'تدريب: {", ".join(n[:22] for n in train)}')
        print(f'     {"":<11} اختبار: {", ".join(n[:22] for n in test)}')
    if leaks:
        raise RuntimeError(
            f'تسريب: نفس الفيديو الأصلي في التدريب والاختبار — {leaks}')

    # ── القصاصتين الأولانيين لكل حركة = تدريب، والباقي = اختبار ──
    templates, tests, weak = [], [], []
    for action, clips in data.items():
        for i, (name, kp) in enumerate(clips):
            if detect_rate(kp) < 0.5:
                weak.append((name, detect_rate(kp), 'تدريب' if i < N_TRAIN else 'اختبار'))
            if len(kp) < 4:
                raise RuntimeError(f'القصاصة {name} فيها {len(kp)} فريم بس')
            feat = pose_features(kp)
            if i < N_TRAIN:
                templates.append((action, feat))
            else:
                tests.append((action, name, feat))

    print(f'\n📚 قصاصات التدريب: {len(templates)}  '
          f'({N_TRAIN} لكل حركة × {len(TEN_ACTIONS)} حركة)')
    print(f'🧪 قصاصات الاختبار: {len(tests)}  (الصدفة {100 / len(TEN_ACTIONS):.0f}%)')
    if weak:
        print(f'\n⚠️ قصاصات الكشف فيها ضعيف (أقل من 50% من الفريمات فيها شخص) — '
              f'دي أصفار مش داتا:')
        for name, rate, role in weak:
            print(f'     {name:<20} {rate * 100:5.1f}%  ({role})')

    # ── اللي المستخدم طلبه بالظبط: قصاصة اختبار واحدة لكل حركة ──
    print(f'\n{"=" * 74}')
    print(f'  1️⃣  قصاصة اختبار واحدة لكل حركة ({len(TEN_ACTIONS)} نتيجة)')
    print(f'{"=" * 74}')
    first = {}
    for action, name, feat in tests:
        first.setdefault(action, (name, feat))

    n_ok = 0
    for action, arabic in TEN_ACTIONS.items():
        name, feat = first[action]
        pred, dist, conf, ranked = classify(feat, templates)
        ok = pred == action
        n_ok += ok
        note = '' if ok else f'  ← قالها {TEN_ACTIONS[pred]}'
        print(f'  {"✅" if ok else "❌"} {arabic:<22} مسافة {dist:.3f}  '
              f'ثقة {conf:.2f}{note}')
    print(f'\n  النتيجة: {n_ok}/{len(TEN_ACTIONS)}')

    # ── النتيجة الكاملة على كل قصاصات الاختبار ──
    print(f'\n{"=" * 74}')
    print(f'  2️⃣  كل قصاصات الاختبار ({len(tests)} قصاصة)')
    print(f'{"=" * 74}')
    per = {a: [0, 0] for a in TEN_ACTIONS}
    confusion = {a: {} for a in TEN_ACTIONS}
    confs = {True: [], False: []}
    for action, name, feat in tests:
        pred, dist, conf, _ = classify(feat, templates)
        per[action][1] += 1
        per[action][0] += pred == action
        confusion[action][pred] = confusion[action].get(pred, 0) + 1
        confs[pred == action].append(conf)

    correct = sum(v[0] for v in per.values())
    total = sum(v[1] for v in per.values())
    print(f'  {"الحركة":<24}{"الدقة":<14}{"بيتلخبط مع"}')
    print(f'  {"-" * 68}')
    for action, arabic in TEN_ACTIONS.items():
        ok, n = per[action]
        wrong = sorted(((c, TEN_ACTIONS[p]) for p, c in confusion[action].items()
                        if p != action), reverse=True)
        print(f'  {arabic:<24}{ok}/{n} = {ok / n * 100:5.1f}%   '
              + ', '.join(f'{lab} ({c})' for c, lab in wrong[:3]))

    print(f'\n  🎯 الدقة الإجمالية: {correct}/{total} = {correct / total * 100:.1f}%'
          f'   (الصدفة {100 / len(TEN_ACTIONS):.0f}%)')
    for ok_flag, label in ((True, 'لما التصنيف يصح '), (False, 'لما التصنيف يغلط')):
        vals = confs[ok_flag]
        if vals:
            print(f'  متوسط الثقة {label}: {np.mean(vals):.3f}')

    np.save(OUT_DIR / 'ten_actions_results.npy',
            {'per_action': per, 'confusion': confusion,
             'accuracy': correct / total, 'n_train': N_TRAIN, 'n_test': N_TEST},
            allow_pickle=True)
    print(f'\n✅ النتيجة اتحفظت في {OUT_DIR / "ten_actions_results.npy"}')
    print(f'📦 الـ keypoints متخزّنة في {CACHE_DIR} — '
          f'أي تشغيلة تانية هتبقى في ثواني')


if __name__ == '__main__':
    main()
