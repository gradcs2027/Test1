import cv2
import numpy as np

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"FPS: {fps} | Total frames: {total_frames} | Duration: {total_frames/fps:.1f}s")

# كان مكتوب 2 على طول، وده كان مظبوط للفيديو 60fps بس. على فيديو 30fps
# كان هيدّينا 15fps فعلي — نص الدقة الزمنية. بنحسبه من الـ fps عشان
# effective_fps تفضل ~30 مهما كان مصدر الفيديو.
TARGET_FPS = 30.0
FRAME_SKIP = max(1, int(round(fps / TARGET_FPS)))
KP_CONF_MIN = 0.30    # مفصل تحت الثقة دي = مش مرصود، بيتحسب بالاستيفاء
print(f"FRAME_SKIP = {FRAME_SKIP} -> fps فعلي {fps/FRAME_SKIP:.1f}")

# ----------------------------------------------------------------------
# تلات مشاكل كانت هنا وبتتصلح دلوقتي
# ----------------------------------------------------------------------
# 1) الفريمات اللي YOLO فشل فيها كانت بتتشال خالص، والباقي بيتلزق ورا بعضه.
#    يعني 90 صف في المصفوفة ممكن يكونوا 4 ثواني حقيقية مش 3، والحركة
#    بتبان أسرع مما هي. دلوقتي بنسيبها NaN وبنستوفيها زمنياً — الشبكة
#    الزمنية بقت منتظمة تماماً.
#
# 2) كنا بناخد keypoints.xy[0] — أول شخص في ترتيب YOLO، والترتيب ده
#    بيتغيّر بين الفريمات. دلوقتي بنتبّع أقرب حوض للفريم اللي قبله.
#
# 3) YOLO بيطلّع إحداثيات لكل الـ 17 مفصل حتى لو مش شايفهم — بيخمّنهم.
#    التخمين ده كان داخل في الـ normalization كأنه قياس. دلوقتي المفصل
#    اللي ثقته تحت KP_CONF_MIN بيتشال ويتحسب بالاستيفاء من الفريمات
#    اللي حواليه.
sampled = np.arange(0, total_frames, FRAME_SKIP)
T = len(sampled)
raw_kp = np.full((T, 17, 2), np.nan, dtype=np.float32)
raw_cf = np.full((T, 17), np.nan, dtype=np.float32)

n_detected = 0
prev_hip = None
slot = 0
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx % FRAME_SKIP == 0 and slot < T:
        results = yolo_pose(frame, verbose=False)
        k = results[0].keypoints
        xy = k.xy.cpu().numpy() if k is not None and len(k.xy) else np.zeros((0, 17, 2))
        if xy.shape[0] and xy.shape[1] == 17:
            cf = (k.conf.cpu().numpy() if k.conf is not None
                  else np.ones((xy.shape[0], 17), dtype=np.float32))
            hips = (xy[:, L_HIP] + xy[:, R_HIP]) / 2.0
            if prev_hip is None:
                # أول فريم: بناخد أطول شخص في الكادر كتقريب لـ "الشخص الرئيسي"
                p = int(np.argmax(xy[:, :, 1].max(1) - xy[:, :, 1].min(1)))
            else:
                p = int(np.argmin(np.linalg.norm(hips - prev_hip, axis=-1)))
            raw_kp[slot] = xy[p]
            raw_cf[slot] = cf[p]
            if np.all(hips[p] != 0):
                prev_hip = hips[p]
            n_detected += 1
        slot += 1
    frame_idx += 1

cap.release()


def interp_time(a):
    """
    a: (T, 17, 2) فيها NaN -> نفس الشكل من غير NaN.

    استيفاء خطي على محور الزمن لكل مفصل/إحداثي لوحده. المفصل اللي مش
    مرصود في أي فريم بيبقى صفر (زي NTU لما المفصل مش متكتشف).
    """
    out = a.copy()
    t = np.arange(len(out))
    for j in range(out.shape[1]):
        for c in range(out.shape[2]):
            v = out[:, j, c]
            m = np.isfinite(v)
            if not m.any():
                v[:] = 0.0
            elif not m.all():
                v[~m] = np.interp(t[~m], t[m], v[m])
    return out


missing_frames = int(np.isnan(raw_kp[:, 0, 0]).sum())
low_conf_mask = np.isfinite(raw_cf) & (raw_cf < KP_CONF_MIN)
low_conf_rate = float(low_conf_mask.mean())
raw_kp[low_conf_mask] = np.nan

all_keypoints = interp_time(raw_kp)
frame_indices = sampled          # شبكة منتظمة — مفيش فجوات تاني

print(f"\n✅ فريمات متعيّنة: {T} | نجح فيها الاكتشاف: {n_detected} "
      f"({n_detected/T*100:.1f}%)")
print(f"✅ فريمات مالهاش اكتشاف خالص واتحسبت بالاستيفاء: {missing_frames} "
      f"({missing_frames/T*100:.1f}%)")
print(f"✅ مفاصل تحت ثقة {KP_CONF_MIN} واتحسبت بالاستيفاء: {low_conf_rate*100:.1f}% "
      f"من كل (فريم × مفصل)")
print(f"✅ شكل البيانات: {all_keypoints.shape} | الشبكة الزمنية منتظمة: "
      f"{bool(np.all(np.diff(frame_indices) == FRAME_SKIP))}")
