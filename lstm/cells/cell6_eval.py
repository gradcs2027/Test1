import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# مش بنعتمد على أسماء المقاييس اللي evaluate بيرجّعها — في Keras 3 الترتيب
# والأسماء دول بيتغيروا حسب المقاييس المكتوبة في compile، وده كسر الـ run
# مرتين. الدقة بنحسبها من التوقعات اللي إحنا محتاجينها أصلاً، والـ loss
# مفتاحه 'loss' وده الثابت الوحيد.
eval_sk = model_skeleton.evaluate(X_test_sk, y_test_sk, verbose=1, return_dict=True)
print("مقاييس التقييم:", {k: round(float(v), 4) for k, v in eval_sk.items()})
test_loss_sk = float(eval_sk['loss'])

y_pred_sk = model_skeleton.predict(X_test_sk)
y_pred_classes_sk = np.argmax(y_pred_sk, axis=1)
y_true_classes_sk = np.argmax(y_test_sk, axis=1)

test_acc_sk = float((y_pred_classes_sk == y_true_classes_sk).mean())
print(f"Test Accuracy: {test_acc_sk:.4f} ({test_acc_sk*100:.1f}%)")
print(f"Test Loss: {test_loss_sk:.4f}")

class_labels = [action_names[le_skeleton.classes_[i]] for i in range(num_classes_skeleton)]

# مع 49 كلاس الـ top-1 لوحده مضلل: كتير من الأخطاء بتبقى بين كلاسين
# متشابهين فعلاً (wear_jacket / takeoff_jacket)، والـ top-5 بيوضح ده
for k in (3, 5):
    acc_k = top_k_accuracy_score(y_true_classes_sk, y_pred_sk,
                                 k=k, labels=np.arange(num_classes_skeleton))
    print(f"Top-{k} Accuracy: {acc_k:.4f} ({acc_k*100:.1f}%)")

print("\nClassification Report:")
print(classification_report(y_true_classes_sk, y_pred_classes_sk, target_names=class_labels))

cm = confusion_matrix(y_true_classes_sk, y_pred_classes_sk)
# مع عدد كلاسات قليل الخانات بتبقى مقروءة فبنكتب الأرقام جواها.
# كانت متقفلة أيام الـ 49x49 لأنها كانت بتبقى عجينة.
readable = num_classes_skeleton <= 20
plt.figure(figsize=(16, 14) if not readable else (11, 9))
sns.heatmap(cm, annot=readable, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels,
            cbar_kws={'shrink': 0.6})
plt.title(f'Confusion Matrix - Skeleton LSTM ({num_classes_skeleton} Classes)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('/kaggle/working/confusion_matrix_skeleton.png', dpi=100, bbox_inches='tight')
plt.show()

# أوضح 15 خلط — دي اللي هتقولنا الكلاسات دي تستاهل تتدمج ولا لأ
cm_off = cm.copy()
np.fill_diagonal(cm_off, 0)
pairs = np.dstack(np.unravel_index(np.argsort(cm_off, axis=None)[::-1], cm_off.shape))[0][:15]
print("\n🔀 أكتر 15 خلط:")
for t, p in pairs:
    if cm_off[t, p] == 0:
        break
    print(f"   {class_labels[t]:18s} -> {class_labels[p]:18s} : {cm_off[t, p]:3d} "
          f"({cm_off[t, p]/cm[t].sum()*100:.0f}% من الكلاس)")
