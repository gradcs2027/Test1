"""
تطبيع الهيكل العظمي — نسخة **مستقلة** قابلة للاستيراد.

ليه الملف ده موجود؟
  `cell3_normalization.py` فيه كود بيشتغل على مستوى الموديول (اللوب على
  `filtered_annotations` في آخر الملف). يعني مجرد ما تعمل
  `from cell3_normalization import normalize_skeleton` الملف بيتنفّذ كله
  وبيقع بـ NameError لأن `filtered_annotations` مش موجودة.

  عشان كده الدوال النقية متنسوخة هنا **حرفياً** من cell3 عشان مسار الـ
  one-shot يشتغل لوحده من غير NTU ولا خلايا النوتبوك.

⚠️ لو عدّلت التطبيع في cell3_normalization.py، عدّله هنا كمان — الاتنين
   لازم يفضلوا متطابقين، وإلا الـ templates والنوافذ هيتطبّعوا بطريقتين
   مختلفتين والنتيجة تبقى ملهاش معنى.
"""

import numpy as np

NUM_FRAMES = 30

# ترتيب نقاط COCO-17: 5/6 كتف شمال/يمين، 11/12 ورك شمال/يمين
L_SHO, R_SHO, L_HIP, R_HIP = 5, 6, 11, 12


def fill_missing_frames(kp):
    """الفريمات اللي كلها أصفار بتتملي بأقرب فريم صالح."""
    valid = np.any(kp != 0, axis=(1, 2))
    if valid.all() or not valid.any():
        return kp
    idx = np.arange(len(kp))
    valid_idx = idx[valid]
    nearest = valid_idx[np.abs(idx[:, None] - valid_idx[None, :]).argmin(axis=1)]
    return kp[nearest]


def normalize_skeleton(kp):
    """
    kp: (frames, 17, 2) -> (frames, 34)

    توسيط على نص الحوض، وقسمة على وسيط طول الجذع.
    """
    kp = np.asarray(kp, dtype=np.float32)
    kp = fill_missing_frames(kp)

    mid_hip = (kp[:, L_HIP:L_HIP + 1, :] + kp[:, R_HIP:R_HIP + 1, :]) / 2.0
    mid_sho = (kp[:, L_SHO:L_SHO + 1, :] + kp[:, R_SHO:R_SHO + 1, :]) / 2.0

    missing = np.all(mid_hip == 0, axis=-1)[:, 0]
    if missing.any():
        for f in np.where(missing)[0]:
            pts = kp[f][np.any(kp[f] != 0, axis=-1)]
            if len(pts):
                mid_hip[f, 0] = pts.mean(axis=0)

    centered = kp - mid_hip

    torso = np.linalg.norm((mid_sho - mid_hip)[:, 0, :], axis=-1)
    torso = torso[torso > 1e-3]
    scale = np.median(torso) if torso.size else 0.0
    if scale < 1e-3:
        scale = np.abs(centered).max() + 1e-6

    return (centered / scale).reshape(kp.shape[0], -1).astype(np.float32)


def resample_to(kp_norm, n=NUM_FRAMES):
    """ضغط/مد سيكوينس متطبّع لـ n فريم — نفس طريقة التدريب (linspace)."""
    if len(kp_norm) == n:
        return kp_norm
    idx = np.linspace(0, len(kp_norm) - 1, n, dtype=int)
    return kp_norm[idx]
