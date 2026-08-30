from collections import Counter

print("🔑 محتويات split:")
for k in data['split'].keys():
    print(f"   {k}: {len(data['split'][k])} عينة")

# ----------------------------------------------------------------------
# الـ 10 كلاسات الأصلية — مختارة على مقاس فيديوهات الاختبار
# ----------------------------------------------------------------------
# الكلاسات 49-59 حركات شخصين وإحنا بناخد ann['keypoint'][0] يعني هيكل
# واحد بس — مستبعدة من الأول.
#
# ⚠️⚠️ القايمة دي اترجّعت زي ما كانت في النوتبوك الأصلي. جرّبنا نستبدلها
# بقايمة مبنية على F1 على NTU (12 -> 4 -> 3 كلاس) عبر رنات 16-26 وده كان
# **غلط منهجي**:
#
#   القايمة اللي بنيتها بـ F1 شالت 6 من الـ 10 دول — wave, clap,
#   rub_hands, nod_head, touch_head, drink_water — وحطّت مكانهم 8 حركات
#   (falling, jump_up, staggering, hopping, pickup, throw, kick_something,
#   cheer_up) مالهاش وجود في أي فيديو اختبار عندنا.
#
#   النتيجة: الموديل مالوش كلاس صح يحط فيه التلويح والدعك، فكان بيرمي في
#   أقرب حاجة — staggering طلع 8 مرات غلط، throw 3 مرات، وفي رن 26 كل
#   التلويح في vidtest1 اتحط في phone_call بثقة 97%.
#
#   الدرس: F1 على NTU بيقيس "الموديل بيتعلمها كويس؟". اللي احنا محتاجينه
#   "الحركة دي موجودة في الفيديو؟". تحسين الأولانية على حساب التانية
#   بيدّي أرقام NTU أحسن ونتيجة فيديو أسوأ.
#
# الـ 10 دول بيغطّوا فعلياً اللي في vidtest1/2/3: تلويح، دعك إيدين،
# قعود، وقوف، مكالمة، إيد على الراس.
action_names = {
    0:  'drink_water',
    7:  'sit_down',
    8:  'stand_up',
    9:  'clap',
    22: 'wave',
    27: 'phone_call',
    33: 'rub_hands',
    34: 'nod_head',
    39: 'cross_hands',
    43: 'touch_head',
}
selected_labels = sorted(action_names)

# ملحوظة معروفة: clap/rub_hands زوج متخالط على NTU، وكذلك
# nod_head/touch_head. سايبينهم لأن التغطية أهم من نضافة الـ F1 —
# الخلط بين اتنين موجودين في الفيديو أهون من غياب الاتنين.
#
# المشي مش موجود في NTU-60 أصلاً، فالمشي هيفضل 'other'.
#
# ⛔ جرّبنا كلاس 'other' *مدرّب* من الكلاسات المرمية (رن 18/19): أخد
# F1 0.93 على NTU وقال 'other' صفر مرة من 77 نافذة على الفيديو. الرفض
# المتعلّم من مجموعة مقفولة مابيعمّمش على مدخل جديد. الرفض دلوقتي بعتبة
# ثقة في خلية 9. التفاصيل في .wolf/buglog.json
filtered_annotations = [ann for ann in data['annotations']
                        if ann['label'] in selected_labels]

label_counts = Counter([ann['label'] for ann in filtered_annotations])
counts = [label_counts[l] for l in selected_labels]
print(f"\n✅ كلاسات: {len(selected_labels)} | عينات: {len(filtered_annotations)}")
print(f"✅ عينات لكل كلاس: أقل {min(counts)} | أكبر {max(counts)} "
      f"| متوسط {sum(counts)//len(counts)}")
