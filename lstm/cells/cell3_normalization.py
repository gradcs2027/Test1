import numpy as np

NUM_FRAMES = 30  # عدد فريمات موحد لكل عينة

# ترتيب نقاط COCO-17: 5/6 كتف شمال/يمين، 11/12 ورك شمال/يمين
L_SHO, R_SHO, L_HIP, R_HIP = 5, 6, 11, 12


def fill_missing_frames(kp):
    """
    الفريمات اللي مفيهاش أي نقطة (كلها أصفار) بنملاها بأقرب فريم صالح.

    من غير كده بتفضل صفوف أصفار في نص السيكوينس، واللي كانت بتخلي الـ mask
    متقطع زي [T,T,F,T,T] — و cuDNN LSTM بيرفض ده (_assert_valid_mask).
    وبرضه أصلاً فريم فاضي مدخل ضايع للموديل مش معلومة.
    """
    valid = np.any(kp != 0, axis=(1, 2))  # (frames,)
    if valid.all() or not valid.any():
        return kp
    idx = np.arange(len(kp))
    valid_idx = idx[valid]
    nearest = valid_idx[np.abs(idx[:, None] - valid_idx[None, :]).argmin(axis=1)]
    return kp[nearest]


def normalize_skeleton(kp):
    """
    kp: (frames, 17, 2) -> (frames, 34)

    بنركّز على نقطة نص الحوض بدل الأنف — الأنف بيهتز مع كل حركة راس وبيعمل
    noise على كل النقاط التانية. الحوض تقريباً ثابت بالنسبة لباقي الجسم.

    وبنقسم على طول الجذع (نص الكتف -> نص الحوض) بدل أكبر قيمة مطلقة، عشان
    المقياس يبقى مستقل عن بُعد الشخص عن الكاميرا وعن أي نقطة شاذة.
    """
    # بيانات NTU مخزّنة float16 — القسمة على scale فيها بتعمل underflow لصفر
    kp = np.asarray(kp, dtype=np.float32)
    kp = fill_missing_frames(kp)

    mid_hip = (kp[:, L_HIP:L_HIP + 1, :] + kp[:, R_HIP:R_HIP + 1, :]) / 2.0
    mid_sho = (kp[:, L_SHO:L_SHO + 1, :] + kp[:, R_SHO:R_SHO + 1, :]) / 2.0

    # لو الحوض مش متكتشف في فريم (0,0) نستخدم متوسط النقاط الموجودة بدله
    missing = np.all(mid_hip == 0, axis=-1)[:, 0]
    if missing.any():
        for f in np.where(missing)[0]:
            pts = kp[f][np.any(kp[f] != 0, axis=-1)]
            if len(pts):
                mid_hip[f, 0] = pts.mean(axis=0)

    centered = kp - mid_hip

    # طول الجذع لكل فريم، وناخد الوسيط عبر السيكوينس عشان مايتأثرش بفريم وحش
    torso = np.linalg.norm((mid_sho - mid_hip)[:, 0, :], axis=-1)
    torso = torso[torso > 1e-3]
    scale = np.median(torso) if torso.size else 0.0
    if scale < 1e-3:  # fallback لو الجذع مش مكتشف خالص
        scale = np.abs(centered).max() + 1e-6

    return (centered / scale).reshape(kp.shape[0], -1).astype(np.float32)


def process_skeleton_sample(ann, num_frames=NUM_FRAMES):
    """يستخرج الـ keypoints بشكل موحد من كل عينة"""
    kp = ann['keypoint'][0]  # أول شخص بس: (frames, 17, 2)
    indices = np.linspace(0, kp.shape[0] - 1, num_frames, dtype=int)
    return normalize_skeleton(kp[indices])  # (num_frames, 34)


# ----------------------------------------------------------------------
# Augmentation — بيتطبّق على الـ train بس (في خلية 4، بعد ما نعرف القسمة)
# ----------------------------------------------------------------------
# الهدف مش تكتير الداتا، الهدف تضييق فجوة الدومين. كل تحويل هنا بيقلّد
# فرق *متقاس* بين NTU وبين اللي بيطلع من YOLO على فيديو حقيقي.

# مقابل كل مفصل في COCO-17 عند القلب الأفقي (شمال <-> يمين)
COCO_FLIP = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15])


def sample_indices(n, rng, mode):
    """
    فهارس الفريمات لعينة تدريب واحدة — دي أهم augmentation عندنا.

    NTU كليبات *مقصوصة*: الحركة بتبدأ مع أول فريم وبتخلص مع آخر واحد.
    نوافذ الاستدلال عندنا 3 ثواني ثابتة من فيديو متواصل، فبتقع في نص
    الحركة، أو فيها الحركة + وقفة، أو فيها انتقال. الموديل اتدرب على
    النوع الأول بس وشاف التاني لأول مرة على الفيديو.

      full : الكليب كامل — زي التدريب القديم
      crop : جزء من الحركة (60-100%) — نافذة وقعت في النص
      pad  : الحركة بتاخد جزء من النافذة والباقي وقفة قبلها/بعدها
    """
    if mode == 'full':
        return np.linspace(0, n - 1, NUM_FRAMES, dtype=int)
    if mode == 'crop':
        span = max(2, int(n * rng.uniform(0.6, 1.0)))
        lo = int(rng.integers(0, n - span + 1))
        return np.linspace(lo, lo + span - 1, NUM_FRAMES, dtype=int)
    # pad — تكرار أول/آخر فريم بيقلّد الوقفة قبل وبعد الحركة
    k = max(2, int(NUM_FRAMES * rng.uniform(0.5, 0.9)))
    before = int(rng.integers(0, NUM_FRAMES - k + 1))
    return np.concatenate([
        np.zeros(before, dtype=int),
        np.linspace(0, n - 1, k, dtype=int),
        np.full(NUM_FRAMES - k - before, n - 1, dtype=int),
    ])


def augment_geom(seq, rng):
    """
    تشويهات هندسية على سيكوينس متطبّع (NUM_FRAMES, 34).

    الأرقام معايرة على تشخيص فجوة الدومين اللي عملناه:
      - الرعشة: قِسنا تسارع نقاط YOLO 1.2x وسيط و 2.3x عند p90 مقارنة
        بـ NTU. sigma عشوائي في [0, 0.045] بوحدة طول الجذع بيغطي المدى ده،
        وبيخلي الموديل يشوف عينات نضيفة ومهزوزة مع بعض.
      - القلب: كاميرا المستخدم ممكن تبقى مواجهة أو من الجنب — NTU بتلات
        زوايا ثابتة بس.
      - إخفاء مفصل: YOLO بيطلّع إحداثيات مخمّنة للمفاصل المحجوبة، وغالباً
        بتبقى شبه ثابتة. بنقلّد ده بتثبيت المفصل على متوسطه.
    """
    s = seq.reshape(NUM_FRAMES, 17, 2).copy()

    if rng.random() < 0.5:
        s = s[:, COCO_FLIP, :]
        s[..., 0] *= -1

    th = rng.uniform(-0.21, 0.21)          # ±12 درجة
    c, sn = np.cos(th), np.sin(th)
    s = s @ np.array([[c, sn], [-sn, c]], dtype=np.float32)

    s *= rng.uniform(0.85, 1.15)
    s += rng.normal(0, rng.uniform(0.0, 0.045), s.shape).astype(np.float32)

    dead = rng.random(17) < 0.05
    if dead.any():
        s[:, dead, :] = s[:, dead, :].mean(axis=0, keepdims=True)

    return s.reshape(NUM_FRAMES, -1).astype(np.float32)


X_skeleton = []
y_skeleton = []
fd_skeleton = []   # frame_dir لكل عينة — محتاجينه في خلية 4 عشان الـ X-Sub split
ann_skeleton = []  # الـ annotation نفسه — خلية 4 محتاجة الفريمات الخام للـ augmentation

for ann in filtered_annotations:
    try:
        processed = process_skeleton_sample(ann, NUM_FRAMES)
        X_skeleton.append(processed)
        y_skeleton.append(ann['label'])
        fd_skeleton.append(ann['frame_dir'])
        ann_skeleton.append(ann)
    except Exception as e:
        continue

X_skeleton = np.array(X_skeleton, dtype=np.float32)
y_skeleton = np.array(y_skeleton)
fd_skeleton = np.array(fd_skeleton)
# لازم يفضلوا متطابقين في الطول — العينات اللي فشلت اتشالت من الكل مع بعض
assert len(X_skeleton) == len(y_skeleton) == len(fd_skeleton) == len(ann_skeleton)

print(f"✅ شكل X_skeleton: {X_skeleton.shape}")  # (samples, 30, 34)
print(f"✅ شكل y_skeleton: {y_skeleton.shape}")
print(f"✅ عدد العينات الناجحة: {len(X_skeleton)} من {len(filtered_annotations)}")

# تأكيد إن مفيش فريمات أصفار فاضلة — دي كانت سبب فشل الـ cuDNN LSTM
zero_frames = int((~np.any(X_skeleton != 0, axis=2)).sum())
print(f"✅ فريمات كلها أصفار متبقية: {zero_frames} (المفروض 0)")
