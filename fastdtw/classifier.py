"""
Few-Shot FastDTW — المكتبة الأساسية (من غير أي داتاسِت)

كل المكوّنات هنا مبنية على تشخيص فشل النسخة الأولى (real recall = 0.0%).
كل واحد منهم بيعالج سبب فشل **متقاس**، مش تخمين:

  ١. تمثيل الفروق (velocity)
     المشكلة: الـ DTW على المواضع بتقيس الوضعية مش الحركة. قِسنا إن
     template وقفة ساكنة (rub_hands) طلع أقرب حاجة لنوافذ القعود
     الحقيقية (62.0 مقابل 81.8 للـ template الصح).
     الحل: نقارن على الفرق بين الفريمات. اتقاس: stand_up من 0/5 لـ 4/5.

  ٢. بنك templates متعدد (few-shot)
     المشكلة: مثال واحد لكل حركة = هشاشة شديدة.
     الحل: vidtest1 فيه مثالين لكل من sit_down/stand_up، و vidtest2 فيه
     مثال كمان. يبقى 3 أمثلة لكل حركة — **من غير أي داتاسِت**.

  ٣. نوافذ متعددة المقاسات
     المشكلة: النافذة 3s والحركة 1.0-1.5s، فنص النافذة سكون بيلوّث المقارنة.
     الحل: نجرّب 1.5s/2.0s/3.0s وناخد أحسن تطابق.

  ٤. تطبيع المسافة بطول المسار
     المشكلة: مسافة الـ DTW بتكبر مع طول المسار، فالمقارنة بين templates
     بأطوال مختلفة مش عادلة.
     الحل: نقسم على عدد خطوات المسار.

  ٥. بوابة طاقة الحركة
     المشكلة: مفيش تمييز بين "الشخص واقف ساكن" و"الشخص بيعمل حركة".
     الحل: نافذة طاقتها واطية => 'other' فوراً من غير DTW. ده كمان
     ضروري قبل تطبيع الشكل (خطوة ٦) لأن تطبيع نافذة ساكنة بيكبّر الضوضاء.

  ٦. تطبيع الشكل (اختياري)
     نقسم على الـ RMS عشان نقارن *شكل* الحركة مش قوّتها — عشان القعدة
     السريعة والبطيئة يبقوا متشابهين.

⚠️ ملاحظة على التنعيم:
   مابنستخدمش `baselines_common.moving_average_predictions` — فيها بج:
   بتاخد متوسط حسابي لأرقام الفئات، فـ [other, other, stand_up, other,
   other] بتطلع [rub_hands ×5] — بتخترع لابل ماحدش توقّعه.
   بنستخدم تصويت الأغلبية (majority vote) بدلها، وده الصح للفئات.
"""

import numpy as np

from fastdtw_core import fastdtw
from skeleton_norm import (L_HIP, L_SHO, NUM_FRAMES, R_HIP, R_SHO,
                           fill_missing_frames)


# ==============================================================================
# التطبيع — نسخة خاصة بمسار الـ few-shot
# ==============================================================================

def normalize_window(kp):
    """
    kp: (frames, 17, 2) -> (frames, 34)، متوسّطة على **مرساة واحدة للنافذة**.

    🐛 ليه مش `skeleton_norm.normalize_skeleton`؟ ده كان السبب الجذري
       للفشل التاني، وهو أخطر من الأول:

         mid_hip = (kp[:, L_HIP] + kp[:, R_HIP]) / 2   # لكل فريم لوحده
         centered = kp - mid_hip

       الحوض بيتطرح من **كل فريم بمركزه هو**. يعني نص الحوض قيمته (0,0)
       في كل فريم، وسرعته **صفر بالظبط دايماً**. فحركة الجسم لفوق أو
       لتحت بتتشال بالكامل من الإشارة قبل ما الـ DTW يشوفها.

       والقعود والوقوف بيختلفوا أساساً في **اتجاه** حركة الحوض رأسياً.
       اتقاس في vidtest3: النظام لقى الحركة في مكانها الصح بالظبط
       (63.6-66.9 مقابل الحقيقي 65.0-66.5) بس سمّاها sit_down بدل
       stand_up — الكشف شغّال والاتجاه مفقود.

       هنا بنطرح **مرساة واحدة للنافذة كلها** (متوسط الحوض عبر النافذة)
       بدل مركز كل فريم. كده بنشيل الموضع المطلق في الكادر — وده اللي
       عايزين نشيله فعلاً — ونحتفظ بحركة الجسم **جوه** النافذة.

    ⚠️ مسار الـ LSTM لازم يفضل على `normalize_skeleton` عشان يطابق
       التدريب على NTU. المسار ده مالوش علاقة بالـ NTU فمش مقيّد بيها.
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

    anchor = mid_hip.mean(axis=0, keepdims=True)      # ← مرساة واحدة
    centered = kp - anchor

    torso = np.linalg.norm((mid_sho - mid_hip)[:, 0, :], axis=-1)
    torso = torso[torso > 1e-3]
    scale = np.median(torso) if torso.size else 0.0
    if scale < 1e-3:
        scale = np.abs(centered).max() + 1e-6

    return (centered / scale).reshape(kp.shape[0], -1).astype(np.float32)


# ==============================================================================
# التمثيل
# ==============================================================================

def to_features(seq_norm, mode='vel', shape_norm=True):
    """
    (frames, 34) متطبّعة -> تمثيل المقارنة.

    mode='vel'  : الفروق بين الفريمات — بتقيس الحركة مش الوضعية
    mode='pos'  : المواضع زي ما هي (اتقاس إنه بيفشل: real recall 0%)
    shape_norm  : قسمة على الـ RMS — يقارن شكل الحركة بغض النظر عن قوّتها

    ⚠️ shape_norm على نافذة ساكنة بيكبّر الضوضاء — لازم بوابة الطاقة
       تشتغل الأول.
    """
    x = np.diff(seq_norm, axis=0) if mode == 'vel' else np.asarray(seq_norm)

    if shape_norm:
        rms = float(np.sqrt((x ** 2).sum(axis=1).mean()))
        if rms > 1e-6:
            x = x / rms

    return np.ascontiguousarray(x, dtype=np.float64)


def resample_linear(seq, n=NUM_FRAMES):
    """
    إعادة أخذ عيّنات بالاستيفاء الخطي — مش بأقرب فهرس.

    ⚠️ ليه مش `skeleton_norm.resample_to`؟
       دي بتستخدم `linspace(dtype=int)`، يعني بتكرّر فريمات لما تمدّ.
       vidtest3 معدله 15 فريم/ث، فنافذة 1.5s = 22 فريم بتتمدّ لـ 30 →
       8 فريمات مكررة → **8 خطوات سرعتها صفر بالظبط** من أصل 29.
       وده بيخلي شكل إشارة السرعة في vidtest3 مختلف جوهرياً عن
       vidtest1/2 (30 فريم/ث) — واتقاس فعلاً: وسيط أقرب مسافة 0.800
       في vidtest3 مقابل 1.005 و 1.111 في التانيين، يعني العتبة
       المعايرة على فيديو مابتنفعش على التاني.

       الاستيفاء الخطي بيشيل الأثر ده تماماً.

       مسار الـ LSTM بيفضل على `resample_to` الأصلية عشان يطابق التدريب —
       المسار ده مالوش علاقة بالـ NTU فمش مقيّد بيها.
    """
    seq = np.asarray(seq, dtype=np.float64)
    if len(seq) == n:
        return seq
    src = np.linspace(0, len(seq) - 1, n)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, len(seq) - 1)
    w = (src - lo)[:, None]
    return seq[lo] * (1 - w) + seq[hi] * w


def motion_energy(seq_norm_raw, eff_fps):
    """
    "سرعة" الجسم — بوحدة (طول الجذع / ثانية).

    ⚠️ لازم تتحسب من الفريمات **الخام** بمعدلها الأصلي، مش من النسخة
       المعاد أخذ عيّناتها لـ 30 فريم. لو حسبناها بعد إعادة العيّنات،
       الرقم بيتأثر بمعدل الفيديو الأصلي وبيبقى مش قابل للمقارنة بين
       الفيديوهات — واتقاس: طاقة vidtest2 كانت 3.7× طاقة vidtest1
       رغم إن الاتنين نفس الشخص ونفس الكاميرا.

    كده الوحدة فيزيائية حقيقية: عند نص المعدل، الإزاحة لكل فريم بتتضاعف
    والمعدل بينص، فالحاصل ثابت.
    """
    d = np.diff(np.asarray(seq_norm_raw), axis=0)
    if len(d) == 0:
        return 0.0
    per_frame = np.linalg.norm(d.reshape(len(d), 17, 2), axis=2).mean()
    return float(per_frame * eff_fps)


# ==============================================================================
# المسافة
# ==============================================================================

def norm_distance(a, b, radius=1):
    """
    مسافة FastDTW مقسومة على طول المسار.

    من غير القسمة، الـ templates الأطول بتاخد مسافات أكبر تلقائياً
    والمقارنة بينها مش عادلة.
    """
    dist, path = fastdtw(a, b, radius=radius)
    return dist / max(1, len(path))


# ==============================================================================
# بنك الـ templates
# ==============================================================================

def cut_templates(kp, eff_fps, clips, source='', mode='vel', shape_norm=True,
                  scales=(3.0,), min_frames=4):
    """
    بيقص قصاصات من فيديو ويحوّلها templates.

    clips: list من (بداية_ثانية, نهاية_ثانية, اسم_الحركة)

    ⚠️ القصاصات الطويلة بتتقسّم:
       'wave' في vidtest1 طولها 6.6s و 'phone_call' 5.4s — دي حركات
       **مستمرة ومتكررة**، مش انتقالات زي القعود. لو ضغطنا 6.6s على 30
       فريم وقارنّاها بنافذة 2s، بنقارن تلويحة بطيئة بتلويحة سريعة والـ
       DTW هيشوفهم مختلفين رغم إنهم نفس الحركة.
       فبنقصّها لقصاصات فرعية بطول النوافذ نفسها. ده كمان بيزوّد عدد
       الأمثلة من غير أي بيانات جديدة.

    القصاصات القصيرة (الانتقالات) بتتاخد زي ما هي — مالهاش داعي تتقسّم.
    """
    out = []
    max_scale = max(scales)

    for t0, t1, label in clips:
        dur = t1 - t0
        # قصاصة طويلة -> قصاصات فرعية بكل مقاس، بخطوة نص المقاس
        if dur > max_scale * 1.3:
            spans = []
            for sec in scales:
                s = t0
                while s + sec <= t1 + 1e-6:
                    spans.append((s, s + sec))
                    s += sec / 2.0
        else:
            spans = [(t0, t1)]

        for s0, s1 in spans:
            lo, hi = int(round(s0 * eff_fps)), int(round(s1 * eff_fps))
            clip = kp[max(0, lo):min(len(kp), hi)]
            if len(clip) < min_frames:
                continue
            seq = resample_linear(normalize_window(clip), NUM_FRAMES)
            out.append({
                'label': label,
                'feat': to_features(seq, mode=mode, shape_norm=shape_norm),
                'span': (s0, s1),
                'source': source,
                'raw_frames': len(clip),
            })
    return out


def balance_templates(templates, max_per_label, seed=0):
    """
    بيحدّد عدد الـ templates لكل حركة.

    ⚠️ ليه ده ضروري؟
       تقسيم القصاصات الطويلة بيطلّع 15 template لـ 'wave' مقابل 3 بس
       لـ 'sit_down'. وبما إن التصنيف بياخد **أقل** مسافة، الكلاس اللي
       عنده templates أكتر بياخد فرص أكتر إنه يكسب بالصدفة. ده بالظبط
       نفس فخ "rub_hands بيبلع كل النوافذ" اللي فشلت بيه النسخة الأولى،
       راجع من باب تاني.

    بناخد عيّنة **متباعدة بانتظام** (مش عشوائية) عشان نغطّي كل الحركة
    من أولها لآخرها، والنتيجة تبقى قابلة للتكرار.
    """
    by_label = {}
    for t in templates:
        by_label.setdefault(t['label'], []).append(t)

    out = []
    for label, group in by_label.items():
        if len(group) <= max_per_label:
            out += group
        else:
            idx = np.linspace(0, len(group) - 1, max_per_label, dtype=int)
            out += [group[i] for i in idx]

    return out


# ==============================================================================
# النوافذ
# ==============================================================================

def build_multiscale_windows(kp, eff_fps, centers, scales,
                             mode='vel', shape_norm=True):
    """
    لكل مركز، بيبني نافذة بكل مقاس من scales.

    بيرجّع:
      feats  : dict {scale: array (n_centers, n_feat, 34)}
      energy : array (n_centers, n_scales) — طاقة الحركة لكل نافذة
    """
    T = len(kp)
    feats = {}
    energy = np.zeros((len(centers), len(scales)))

    for si, sec in enumerate(scales):
        wf = max(4, int(round(sec * eff_fps)))
        half = wf // 2
        n_feat = NUM_FRAMES - 1 if mode == 'vel' else NUM_FRAMES
        arr = np.empty((len(centers), n_feat, 34), dtype=np.float64)

        for i, c in enumerate(centers):
            lo, hi = max(0, c - half), min(T, c + half + 1)
            raw = normalize_window(kp[lo:hi])        # بالمعدل الأصلي
            energy[i, si] = motion_energy(raw, eff_fps)
            arr[i] = to_features(resample_linear(raw, NUM_FRAMES),
                                 mode=mode, shape_norm=shape_norm)

        feats[sec] = arr

    return feats, energy


def window_edges(centers, eff_fps, n_frames):
    """حدود النوافذ بالثواني — للتقييم والرسم."""
    e = np.empty(len(centers) + 1)
    e[0] = 0.0
    e[1:-1] = (centers[:-1] + centers[1:]) / 2.0 / eff_fps
    e[-1] = (n_frames - 1) / eff_fps
    return e


# ==============================================================================
# التصنيف
# ==============================================================================

def distance_cache(feats, templates, scales, radius=1, progress=None):
    """
    بيحسب كل مسافات الـ DTW **مرة واحدة** -> مصفوفة (n_centers, n_scales, n_templates).

    ده أهم قرار في التصميم: البحث عن أحسن (بوابة، عتبة) محتاج يجرّب مئات
    التركيبات. من غير التخزين ده، كل تركيبة كانت هتعيد الـ DTW كله (دقايق
    لكل تركيبة). بالتخزين، الحساب بيحصل مرة والبحث بيبقى numpy خالص.
    """
    n = len(feats[scales[0]])
    D = np.empty((n, len(scales), len(templates)))

    for si, sec in enumerate(scales):
        arr = feats[sec]
        for ti, t in enumerate(templates):
            tf = t['feat']
            for i in range(n):
                D[i, si, ti] = norm_distance(arr[i], tf, radius=radius)
        if progress:
            progress(si + 1, len(scales))

    return D


def classify(D, energy, labels_of_templates, gate, threshold):
    """
    تصنيف من المسافات المخزّنة. الترتيب مهم:

      1. بوابة الطاقة — النافذة الساكنة => 'other' من غير النظر للمسافة
         (ضروري: تطبيع الشكل بيحوّل ضوضاء النافذة الساكنة لنمط وحدوي
          ممكن يطابق أي حاجة بالصدفة)
      2. المسافة — أبعد من العتبة => 'other'
      3. غير كده => لابل أقرب template

    بيرجّع (labels, best_dist, best_scale_index)
    """
    n = D.shape[0]
    masked = np.where(energy[:, :, None] >= gate, D, np.inf)

    flat = masked.reshape(n, -1)
    best_flat = flat.argmin(axis=1)
    best_d = flat[np.arange(n), best_flat]

    n_t = D.shape[2]
    best_si = best_flat // n_t
    best_ti = best_flat % n_t

    out = []
    for i in range(n):
        if not np.isfinite(best_d[i]) or best_d[i] > threshold:
            out.append('other')
        else:
            out.append(labels_of_templates[best_ti[i]])

    return out, best_d, best_si


# ==============================================================================
# التنعيم — تصويت الأغلبية (بديل الـ moving average المكسور)
# ==============================================================================

def majority_smooth(labels, k=3):
    """
    كل نافذة بتاخد اللابل الأكتر تكراراً في جيرانها.

    ده الصح للفئات — على عكس متوسط أرقام الفئات اللي ممكن يطلّع لابل
    ماحدش توقّعه أصلاً.
    """
    if k <= 1:
        return list(labels)
    out = []
    pad = k // 2
    n = len(labels)
    for i in range(n):
        lo, hi = max(0, i - pad), min(n, i + pad + 1)
        window = labels[lo:hi]
        vals, counts = np.unique(window, return_counts=True)
        # عند التعادل، نفضّل اللابل الحالي عشان ما نمسحش كشف حقيقي
        top = counts.max()
        winners = set(vals[counts == top])
        out.append(labels[i] if labels[i] in winners else vals[counts.argmax()])
    return out


def drop_short_segments(labels, min_len, protect='other'):
    """
    بيشيل القطع الأقصر من min_len — بس بيحمي 'other'.

    (نسخة محلية من enforce_min_duration عشان الملف ده يفضل مستقل)
    """
    labels = list(labels)
    i = 0
    while i < len(labels):
        j = i
        while j < len(labels) and labels[j] == labels[i]:
            j += 1
        if j - i < min_len and labels[i] != protect:
            fill = labels[i - 1] if i > 0 else protect
            labels[i:j] = [fill] * (j - i)
        i = j
    return labels


# ==============================================================================
# التقييم
# ==============================================================================

def frame_metrics(labels, edges, gt, step=0.1):
    """تقييم على مستوى الزمن — عيّنة كل step ثانية."""
    r = {'total': 0, 'hit': 0, 'other_total': 0, 'other_hit': 0,
         'real_total': 0, 'real_hit': 0, 'false_alarm': 0}

    for t in np.arange(edges[0], edges[-1], step):
        truth = None
        for s, e, lab in gt:
            if s <= t < e:
                truth = lab
                break
        if truth is None or truth == '?':
            continue

        w = int(np.searchsorted(edges, t, side='right') - 1)
        if not (0 <= w < len(labels)):
            continue

        target = 'other' if truth.endswith('*') else truth
        pred = labels[w]

        r['total'] += 1
        r['hit'] += (pred == target)

        if target == 'other':
            r['other_total'] += 1
            r['other_hit'] += (pred == target)
            r['false_alarm'] += (pred != 'other')
        else:
            r['real_total'] += 1
            r['real_hit'] += (pred == target)

    return r


def event_metrics(labels, edges, gt, tolerance=1.0):
    """
    تقييم على مستوى الحدث — ده الأهم للحركات النادرة القصيرة.

    حدث متكشّف = فيه نافذة واحدة على الأقل، متداخلة معاه (± tolerance)،
    توقّعت اللابل الصح.

    إنذار كاذب = قطعة متوقّعة بلابل حقيقي مش متداخلة مع أي حدث حقيقي.
    """
    events = [(s, e, lab) for s, e, lab in gt
              if lab != '?' and not lab.endswith('*')]

    detected = []
    for s, e, lab in events:
        hit = False
        for w in range(len(labels)):
            w0, w1 = edges[w], edges[w + 1]
            if w1 >= s - tolerance and w0 <= e + tolerance:
                if labels[w] == lab:
                    hit = True
                    break
        detected.append({'span': (s, e), 'label': lab, 'detected': hit})

    # المناطق المستثناة ('?') — مثلاً اللقطات القريبة في أول وآخر vidtest2،
    # مافيهاش جسم كامل فمينفعش نحاسب عليها لا صح ولا غلط
    invalid = [(s, e) for s, e, lab in gt if lab == '?']

    # الإنذارات الكاذبة — قطع متصلة بلابل حقيقي بعيدة عن أي حدث
    false_alarms = []
    i = 0
    while i < len(labels):
        j = i
        while j < len(labels) and labels[j] == labels[i]:
            j += 1
        if labels[i] != 'other':
            seg0, seg1 = edges[i], edges[j]
            overlaps = any(seg1 >= s - tolerance and seg0 <= e + tolerance
                           for s, e, lab in events if lab == labels[i])
            in_invalid = any(seg1 > s and seg0 < e for s, e in invalid)
            if not overlaps and not in_invalid:
                false_alarms.append((seg0, seg1, labels[i]))
        i = j

    n_det = sum(d['detected'] for d in detected)
    return {
        'events': detected,
        'n_events': len(events),
        'n_detected': n_det,
        'recall': n_det / max(1, len(events)),
        'false_alarms': false_alarms,
        'n_false_alarms': len(false_alarms),
    }
