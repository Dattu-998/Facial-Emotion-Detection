import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

BASE_DIR    = r"E:\New folder\htdocs\emotion-ai"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR   = os.path.join(BASE_DIR, "model")
INPUT_MODEL = os.path.join(MODEL_DIR, "emotion_model.h5")  # your 60.95% model
OUTPUT      = os.path.join(MODEL_DIR, "emotion_model.h5")
IMG_SIZE    = 112
BATCH       = 16

print("="*55)
print("  Resume Training from 60.95%")
print(f"  TensorFlow: {tf.__version__}")
print("="*55)

# ── Load the 60.95% model directly ────────────────────────
print("\nLoading 60.95% model...")
model = load_model(INPUT_MODEL)
print(f"✅ Loaded! Params: {model.count_params():,}")

# ── Data generators ────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.75, 1.25],
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, 'train'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH, class_mode='categorical', shuffle=True
)
test_gen = test_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, 'test'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH, class_mode='categorical', shuffle=False
)

# ── Verify current accuracy ────────────────────────────────
print("\nVerifying current accuracy...")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
loss, acc = model.evaluate(test_gen, verbose=0)
print(f"✅ Confirmed accuracy: {acc*100:.2f}%")

# ── Class weights ──────────────────────────────────────────
labels = train_gen.classes
cw = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = {i: min(float(w), 2.5) for i, w in enumerate(cw)}

# ── Unfreeze more layers this time — top 60 ────────────────
print("\nUnfreezing top 60 layers for deeper fine-tuning...")
for layer in model.layers:
    layer.trainable = False
for layer in model.layers[-60:]:
    layer.trainable = True

trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"Trainable params: {trainable:,}")

# ── Recompile with fresh higher LR ────────────────────────
# Key fix: start with higher LR so ReduceLROnPlateau has room to drop
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=['accuracy']
)

print("\n" + "="*55)
print("  ROUND 2: Deeper fine-tuning")
print("  Starting from: 60.95%")
print("  Target: 78-84%")
print("  Each epoch: ~3-4 mins | ~2 hours total")
print("  LEAVE OVERNIGHT!")
print("="*55 + "\n")

callbacks = [
    ModelCheckpoint(
        OUTPUT,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,           # More patience this time
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,            # Gentler reduction
        patience=4,            # More patience before reducing
        min_lr=1e-7,
        verbose=1
    )
]

history = model.fit(
    train_gen,
    epochs=40,                 # More epochs
    validation_data=test_gen,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# ── Final results ──────────────────────────────────────────
model.save(OUTPUT)
loss, acc = model.evaluate(test_gen, verbose=0)
best = max(history.history['val_accuracy'])

print("\n" + "="*55)
print("  TRAINING COMPLETE")
print("="*55)
print(f"  ✅ Final Accuracy : {acc*100:.2f}%")
print(f"  ✅ Best achieved  : {best*100:.2f}%")
print(f"  💾 Saved to      : {OUTPUT}")
print("="*55)