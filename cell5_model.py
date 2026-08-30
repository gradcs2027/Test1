import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def build_skeleton_lstm(input_shape, num_classes):
    """
    موديل أكبر لـ 49 كلاس.

    الدافع: النسخة الصغيرة (128/64) طلعت top-1 87.2% بس top-5 98.8% —
    يعني الإشارة موجودة في البيانات والموديل شايفها، بس مش قادر يرتّب
    أول اختيار. ده نقص سعة مش نقص معلومة، فبنزوّد السعة.
    """
    inputs = Input(shape=input_shape)

    # شيلنا Masking(mask_value=0.0):
    # إحنا بنعمل uniform sampling لـ 30 فريم حقيقي، يعني مفيش padding أصلاً
    # فالطبقة دي كانت بلا فايدة. وكمان لما بيطلع فريم أصفار في نص السيكوينس
    # الـ mask بيبقى متقطع و cuDNN LSTM بيرفضه (_assert_valid_mask).
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(128))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)   # زوّدناه: الموديل أكبر يبقى overfitting أسهل

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5')]
    )
    return model


model_skeleton = build_skeleton_lstm(input_shape=(30, 34),
                                     num_classes=num_classes_skeleton)
model_skeleton.summary()

callbacks_sk = [
    # patience أطول: الموديل الأكبر بياخد وقت أطول قبل ما يستقر
    EarlyStopping(monitor='val_loss', patience=14, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-6, verbose=1)
]

history_skeleton = model_skeleton.fit(
    X_train_sk, y_train_sk,
    validation_data=(X_val_sk, y_val_sk),
    epochs=70,
    batch_size=64,   # الداتا بقت ~5 أضعاف (46k بدل 9k) — batch أكبر يقلل الوقت للنص
    class_weight=class_weights_sk,
    callbacks=callbacks_sk,
    verbose=1
)

# الفرق بين التدريب والـ validation بيقولنا الموديل كبر زيادة ولا لسه
tr = history_skeleton.history['accuracy'][-1]
va = history_skeleton.history['val_accuracy'][-1]
print(f"\n📐 آخر إيبوك — train: {tr*100:.1f}% | val: {va*100:.1f}% | الفجوة: {(tr-va)*100:+.1f} نقطة")

# التدريب بياخد ~10 دقايق. أي خطأ في خلية بعد كده كان بيضيّعه كله ويخلينا
# نعيد من الأول — فبنحفظه دلوقتي.
model_skeleton.save('/kaggle/working/skeleton_lstm.keras')
print("✅ الموديل اتحفظ: /kaggle/working/skeleton_lstm.keras")
