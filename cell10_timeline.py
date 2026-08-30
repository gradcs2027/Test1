import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(16, 5))

unique_actions = list(set(predicted_actions_sk))
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_actions)))
action_color_map = dict(zip(unique_actions, colors))

for i in range(len(window_times_sk) - 1):
    ax.axvspan(window_times_sk[i], window_times_sk[i+1],
               color=action_color_map[predicted_actions_sk[i]], alpha=0.6)

# Ground Truth لفيديو vidtest1.mp4 (37.48s، 60fps، 576x1024)
#
# اصطلاح اللابلز:
#   'اسم_كلاس'  = الإجابة الصح، بتتحسب في real recall
#   'اسم*'      = الحركة مش من ضمن الكلاسات المختارة، فالإجابة الصح 'other'
#   'اسم?'      = مش متأكدين، بتتستبعد من القياس خالص
#   أي وقت مش مذكور هنا أصلاً بيتستبعد تلقائياً
#
# ⚠️ القاعدة: ground truth زمني ما يتكتبش من عيّنة أوسع من 0.5s.
# الجدول ده اتكتب من contact sheets فعلية: overview كل 1.0s + zoom كل
# 0.3s على الأربع انتقالات. مش من الذاكرة ولا من وصف مكتوب قبل كده.
#
# الفيديو ده الأنضف عندنا: كاميرا ثابتة، الجسم كامل من الراس للرجل طول
# الوقت، الشخص مواجه الكاميرا، والحركات ممثّلة بوضوح.
#
# اتكتب من جديد لمفردات الـ 10 كلاسات (contact sheets S1-S6 كل 0.3-0.5s).
# قبل كده كان مكتوب بمفردات 3 كلاسات، فالدعك والتلويح كانوا 'other*' —
# دلوقتي بقوا إجابات صح، وده بيقلب الفيديو من ~86% other لـ ~35% other.
#
# ⚠️ مفيش phone_call ولا drink_water ولا nod_head في الفيديو ده. التلات
# كلاسات دول يقدروا يطلّعوا false positives بس. رن 26 أثبت إن أي كلاس
# مالوش إصابة صحيحة بيتحوّل لسلة مهملات (phone_call أخد 24 ثانية @97%).
ground_truth_segments = [
    (0.3,   4.4, 'rub_hands'),   # إيدين متلاصقين قدام الصدر وبيتدعكوا
    (4.4,   4.8, '?'),           # الإيد اليمين بتطلع — انتقال
    (4.8,  11.4, 'wave'),        # الإيد اليمين مرفوعة جنب الراس وبتلوّح
    (11.4, 12.2, 'other*'),      # الإيد نزلت، واقف ثابت
    (12.2, 14.3, 'sit_down'),    # نزول متصل من الوقوف للقعود على البوف
    (14.3, 19.5, 'other*'),      # قاعد ثابت وإيديه على ركبه
    (19.5, 20.4, 'stand_up'),    # قيام لوقوف كامل
    (20.4, 21.2, 'other*'),      # واقف ثابت
    (21.2, 27.1, '?'),           # دراع بيتقوّس فوق الراس وبعدين الساعد
                                 # متطبّق على الصدر تحت الدقن ~4 ثواني.
                                 # ملتبس بين touch_head و cross_hands
                                 # (NTU cross_hands دراعين مش دراع) —
                                 # مستبعد بدل ما نحكم بمسطرة مشكوك فيها
    (27.1, 28.4, 'other*'),      # واقف ثابت
    (28.4, 29.7, 'sit_down'),    # نزول تاني للقعود على البوف
    (29.7, 32.4, 'other*'),      # قاعد ثابت وإيديه على ركبه
    (32.4, 33.4, 'stand_up'),    # قيام تاني لوقوف كامل
    (33.4, 33.7, '?'),           # انتقال
    (33.7, 37.4, 'wave'),        # تلويح بالإيد اليمين لآخر الفيديو
]
for start, end, label in ground_truth_segments:
    ax.text((start+end)/2, 1.05, label, ha='center', fontsize=9,
            fontweight='bold', rotation=30, transform=ax.get_xaxis_transform())
    ax.axvline(start, color='k', lw=0.6, ls='--', alpha=0.4)

handles = [plt.Rectangle((0,0),1,1, color=action_color_map[a]) for a in unique_actions]
ax.legend(handles, unique_actions, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)

ax.set_xlabel('Time (s)')
ax.set_yticks([])
ax.set_title(f'{VIDEO_PATH.split("/")[-1]} - {num_classes_skeleton} classes  '
             '(* = no matching class, unlisted spans = excluded)', fontsize=13)
plt.tight_layout()
plt.savefig('/kaggle/working/final_timeline.png', dpi=120, bbox_inches='tight')
plt.show()

# ----------------------------------------------------------------------
# قياس رقمي بدل التفرج بالعين
# ----------------------------------------------------------------------
# ⚠️ الـ "دقة إجمالية" لوحدها مضللة: أغلب الـ ground truth إجابته 'other'،
# فموديل بيقول 'other' على الفيديو كله بياخد 66-71% — أعلى من رنات كتير
# عملناها. الرقم ده بيقيس أساساً "بترفض قد إيه" مش "بتفهم قد إيه".
# عشان كده بنطبع خط الأساس جنبه دايماً.
#
# فبنطبع تلاتة أرقام منفصلة، وخط الأساس جنبهم:
#   1) other recall  — كام % من اللي مالوش كلاس اتكتب other صح
#   2) real recall   — كام % من الحركات اللي *ليها* كلاس اتعرفت صح  <-- ده المهم
#   3) الدقة الإجمالية مع خط الأساس عشان ما تتقريش لوحدها
step = 0.1


def score_labels(actions):
    """يقيس قايمة توقعات (واحد لكل نافذة) مقابل الـ ground truth"""
    r = dict(total=0, hit=0, other_total=0, other_hit=0,
             real_total=0, real_hit=0, other_false=0, skip=0)
    for t in np.arange(edges_sk[0], edges_sk[-1], step):
        gt = next((l for s, e, l in ground_truth_segments if s <= t < e), None)
        w = int(np.searchsorted(edges_sk, t, side='right') - 1)
        if gt is None or gt.endswith('?') or not (0 <= w < len(actions)):
            r['skip'] += 1
            continue
        gt = 'other' if gt.endswith('*') else gt
        pred = actions[w]
        r['total'] += 1
        r['hit'] += (pred == gt)
        if gt == 'other':
            r['other_total'] += 1
            r['other_hit'] += (pred == gt)
        else:
            r['real_total'] += 1
            r['real_hit'] += (pred == gt)
            r['other_false'] += (pred == 'other')
    return r


cur = score_labels(predicted_actions_sk)
base = score_labels(['other'] * len(predicted_actions_sk))

print(f"\n📏 مقارنة بالـ ground truth (عينة كل {step}s، {cur['skip']} عينة مستبعدة):")
print(f"   {'دقة إجمالية':28} {cur['hit']:3d}/{cur['total']:<3d} = {cur['hit']/max(1,cur['total'])*100:5.1f}%")
print(f"   {'خط الأساس (other للكل)':28} {base['hit']:3d}/{base['total']:<3d} = "
      f"{base['hit']/max(1,base['total'])*100:5.1f}%  <-- لازم نعدّيه")
print(f"   {'other recall':28} {cur['other_hit']:3d}/{cur['other_total']:<3d} = "
      f"{cur['other_hit']/max(1,cur['other_total'])*100:5.1f}%")
print(f"   {'real recall (الأهم)':28} {cur['real_hit']:3d}/{cur['real_total']:<3d} = "
      f"{cur['real_hit']/max(1,cur['real_total'])*100:5.1f}%")
print(f"   {'other غلط على حركة حقيقية':28} {cur['other_false']:3d}/{cur['real_total']:<3d} = "
      f"{cur['other_false']/max(1,cur['real_total'])*100:5.1f}%")

# ----------------------------------------------------------------------
# sweep على العتبة مقابل الـ ground truth
# ----------------------------------------------------------------------
# ده بيلغي الحاجة إننا نحرق رن كامل كل مرة عشان نجرب عتبة. التوقعات
# اتحسبت خلاص، إحنا بس بنعيد تطبيق العتبة عليها.
print(f"\n🎚️ العتبة مقابل الـ ground truth (خط الأساس "
      f"{base['hit']/max(1,base['total'])*100:.1f}%):")
print(f"   {'عتبة':>5} {'إجمالي':>8} {'other':>8} {'real':>8}  {'other%':>7}")
for th in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.01):
    lab = argmax_sk.copy()
    lab[max_conf < th] = OTHER_IDX
    lab = enforce_min_duration(lab, MIN_SEGMENT, protect=OTHER_IDX)
    acts = [class_labels_list[c] for c in lab]
    s = score_labels(acts)
    mark = "  <-- المستخدمة" if abs(th - CONF_REJECT) < 1e-9 else ""
    print(f"   {th:5.2f} {s['hit']/max(1,s['total'])*100:7.1f}% "
          f"{s['other_hit']/max(1,s['other_total'])*100:7.1f}% "
          f"{s['real_hit']/max(1,s['real_total'])*100:7.1f}% "
          f"{np.mean([a == 'other' for a in acts])*100:6.0f}%{mark}")
