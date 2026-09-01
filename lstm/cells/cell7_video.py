!pip install ultralytics -q

from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

# تحميل موديل YOLOv8-Pose (خفيف وسريع)
yolo_pose = YOLO('yolov8n-pose.pt')

# المسار متعرّف هنا بس، والخلايا اللي بعده بتستخدم المتغير ده —
# قبل كده كان مكتوب بالإيد في 3 خلايا وأي تغيير لازم يتكرر 3 مرات
#
# فيديوهات الـ dataset (الأسماء اتصلّحت — قبل كده كان vidtest1 و vid مقلوبين
# على Kaggle وده خلّانا نقيس على فيديو ونفتكره التاني):
#   vidtest1.mp4   37.5s  60fps  576x1024  كاميرا ثابتة، جسم كامل، مواجه
#                                          الكاميرا. حركات ممثّلة واضحة:
#                                          تلويح، قعود، وقوف، إيد مرفوعة.
#   vidtest2.mp4   28.5s  60fps  576x1024  كاميرا ثابتة، جسم كامل 2.2-24.9s
#                                          (البداية والنهاية لقطة لاصقة).
#                                          ضهره للكاميرا 2.2-9s بس.
#   vidtest3.mp4  126.5s  30fps  852x480   كاميرا ثابتة، جسم كامل، لقطة واسعة
#
# ⚠️ الأوصاف دي اتصلّحت بعد ما اتفرجنا على contact sheets فعلاً. الوصف
# القديم كان بيقول vidtest1 "لقطة قريبة مش صالحة" و vidtest2 "فيشآي
# هندهيلد" — الاتنين غلط، والغلط ده كان بيبرّر نتايج ضعيفة بسبب خطأ.
VIDEO_PATH = "/kaggle/input/datasets/abdallahhsamir/testvid/vidtest1.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, 300)  # فريم من حوالي ثانية 5 (الأول فيه إيد على العدسة)
ret, frame = cap.read()
cap.release()

results = yolo_pose(frame, verbose=False)

# نرسم النتيجة
annotated = results[0].plot()
annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8, 8))
plt.imshow(annotated_rgb)
plt.axis('off')
plt.title("YOLOv8-Pose Detection Test")
plt.show()

# نشوف شكل الـ keypoints
if len(results[0].keypoints.xy) > 0:
    kps = results[0].keypoints.xy[0].cpu().numpy()
    print(f"✅ عدد النقاط المكتشفة: {kps.shape}")
    print(f"✅ عينة من النقاط:\n{kps[:5]}")
else:
    print("❌ لم يتم اكتشاف أي شخص")
