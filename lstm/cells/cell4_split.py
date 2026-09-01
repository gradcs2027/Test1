from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

le_skeleton = LabelEncoder()
y_encoded_skeleton = le_skeleton.fit_transform(y_skeleton)
num_classes_skeleton = len(le_skeleton.classes_)

# action_names جاي من خلية 2 — مش بنعيد تعريفه هنا عشان ما يحصلش
# اختلاف بين النسختين لو عدّلنا واحدة ونسينا التانية

print(f"✅ عدد الكلاسات: {num_classes_skeleton}")

y_onehot_skeleton = to_categorical(y_encoded_skeleton, num_classes=num_classes_skeleton)

# ----------------------------------------------------------------------
# Cross-Subject split — مش عشوائي
# ----------------------------------------------------------------------
# NTU متصوّر بـ 40 شخص × 3 كاميرات × تكرارات. train_test_split العشوائي
# كان بيحط نفس الشخص ونفس الحركة من كاميرا تانية في train و test مع بعض،
# فالموديل كان بيحفظ الأشخاص مش الحركات — ودي كانت الـ 90% الوهمية.
#
# xsub_train/xsub_val جايين جاهزين في data['split'] وبيقسموا بالأشخاص:
# مفيش شخص واحد بيظهر في الاتنين. ده الـ benchmark المعتمد لـ NTU.
xsub_train = set(data['split']['xsub_train'])
xsub_val = set(data['split']['xsub_val'])

is_train = np.array([fd in xsub_train for fd in fd_skeleton])
is_val = np.array([fd in xsub_val for fd in fd_skeleton])
unknown = int((~is_train & ~is_val).sum())
assert unknown == 0, f"{unknown} عينة مش في أي split — الـ frame_dir مش متطابق"

# الـ val الرسمي بنقسمه نصين: نص للـ validation أثناء التدريب ونص للتقييم
# النهائي. القسمة دي عشوائية بس مفيهاش تسريب — الأشخاص أصلاً متفصلين.
val_idx = np.where(is_val)[0]
val_half, test_half = train_test_split(
    val_idx, test_size=0.5, random_state=42,
    stratify=y_encoded_skeleton[val_idx]
)

X_train_sk, y_train_sk = X_skeleton[is_train], y_onehot_skeleton[is_train]
X_val_sk, y_val_sk = X_skeleton[val_half], y_onehot_skeleton[val_half]
X_test_sk, y_test_sk = X_skeleton[test_half], y_onehot_skeleton[test_half]

print("\n📌 Cross-Subject split (مفيش شخص مشترك بين train و test):")
print("Training:", X_train_sk.shape)
print("Validation:", X_val_sk.shape)
print("Test:", X_test_sk.shape)

# ----------------------------------------------------------------------
# Augmentation — على الـ train بس
# ----------------------------------------------------------------------
# val و test بيفضلوا NTU نضيف من غير أي تشويه. ده مقصود: عايزين الرقم
# اللي بنقيسه يفضل مقارن بالرنات القديمة، ولو زوّدنا الضوضاء في التقييم
# كمان مش هنعرف التحسن جه منين.
#
# بنولّد النسخ مرة واحدة هنا بدل generator كل إيبوك. أقل تنوّع، بس
# بيخلي الرن قابل للتكرار بالظبط ومابيلمسش خلية الموديل.
N_AUG = 3
AUG_MODES = ('full', 'crop', 'pad')
AUG_P = (0.25, 0.40, 0.35)

rng_aug = np.random.default_rng(0)
aug_X, aug_y = [], []
for i in np.where(is_train)[0]:
    kp_raw = np.asarray(ann_skeleton[i]['keypoint'][0], dtype=np.float32)
    n = kp_raw.shape[0]
    for _ in range(N_AUG):
        # الكليبات القصيرة جداً مفيش فيها مساحة للقص — بناخدها كاملة
        mode = 'full' if n < 4 else rng_aug.choice(AUG_MODES, p=AUG_P)
        seq = normalize_skeleton(kp_raw[sample_indices(n, rng_aug, mode)])
        aug_X.append(augment_geom(seq, rng_aug))
        aug_y.append(y_encoded_skeleton[i])

X_train_sk = np.concatenate([X_train_sk, np.array(aug_X, dtype=np.float32)])
y_train_sk = np.concatenate([y_train_sk, to_categorical(aug_y, num_classes_skeleton)])
y_train_enc = np.concatenate([y_encoded_skeleton[is_train], np.array(aug_y)])
del aug_X, aug_y

print(f"\n🔀 Augmentation: {N_AUG} نسخة لكل عينة تدريب "
      f"(قص زمني + قلب + دوران + مقياس + رعشة + إخفاء مفصل)")
print("Training بعد الـ augmentation:", X_train_sk.shape)
print("⚠️ الدقة المطبوعة بعدين على NTU نضيف — الـ augmentation ممكن ينزّلها"
      " شوية وده مقبول، المكسب المستهدف على الفيديو")

# الأوزان بتتحسب على الـ train بس — لو حسبناها على الكل بنسرّب توزيع
# الـ test في التدريب
class_weights_arr_sk = compute_class_weight('balanced', classes=np.unique(y_train_enc), y=y_train_enc)
class_weights_sk = dict(enumerate(class_weights_arr_sk))
print("Class weights:", {i: round(w, 3) for i, w in class_weights_sk.items()})
