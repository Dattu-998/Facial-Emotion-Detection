import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D,
                                     Dropout, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

BASE_DIR    = r"E:\New folder\htdocs\emotion-ai"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR   = os.path.join(BASE_DIR, "model")
WEIGHTS_SRC = os.path.join(MODEL_DIR, "best_model.h5")
OUTPUT      = os.path.join(MODEL_DIR, "emotion_model.h5")
IMG_SIZE    = 112
BATCH       = 16

print("="*55)
print("  Stage 2 Fine-tuning — TF Version Fix")
print(f"  TensorFlow: {tf.__version__}")
print("="*55)

# ── Rebuild exact same architecture ────────────────────────
print("\nRebuilding model architecture...")
base = MobileNetV2(weights='imagenet', include_top=False,
                   input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

x   = base.output
x   = GlobalAveragePooling2D()(x)
x   = BatchNormalization()(x)
x   = Dense(512, activation='relu')(x)
x   = BatchNormalization()(x)
x   = Dropout(0.5)(x)
x   = Dense(256, activation='relu')(x)
x   = BatchNormalization()(x)
x   = Dropout(0.3)(x)
out = Dense(7, activation='softmax')(x)
model = Model(inputs=base.input, outputs=out)
print(f"✅ Architecture rebuilt: {model.count_params():,} params")

# ── Load weights only ──────────────────────────────────────
print(f"\nLoading weights from: {WEIGHTS_SRC}")
try:
    model.load_weights(WEIGHTS_SRC)
    print("✅ Weights loaded successfully!")
except Exception as e:
    print(f"⚠️  Direct load failed: {e}")
    print("Trying by_name=True...")
    try:
        model.load_weights(WEIGHTS_SRC, by_name=True, skip_mismatch=True)
        print("✅ Weights loaded with by_name=True")
    except Exception as e2:
        print(f"❌ Weight load failed: {e2}")
        print("Starting from ImageNet weights only...")

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

# ── Save class indices ─────────────────────────────────────
CLASS_NAMES = {'1':'surprise','2':'fear','3':'disgust',
               '4':'happy','5':'sad','6':'angry','7':'neutral'}
emap = {CLASS_NAMES.get(k,k): v for k,v in train_gen.class_indices.items()}
with open(os.path.join(MODEL_DIR,'class_indices.json'),'w') as f:
    json.dump(emap, f, indent=2)
print(f"\n✅ Emotion map: {emap}")

# ── Evaluate current state ─────────────────────────────────
print("\nEvaluating current model...")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
loss, acc = model.evaluate(test_gen, verbose=0)
print(f"✅ Starting accuracy: {acc*100:.2f}%")

# ── Class weights ──────────────────────────────────────────
labels = train_gen.classes
cw = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = {i: min(float(w), 2.5) for i, w in enumerate(cw)}

# ── STAGE 1: Warm up top layers ────────────────────────────
print("\n" + "="*55)
print("  STAGE 1: Warm-up (5 epochs, base frozen)")
print("="*55)
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
h0 = model.fit(
    train_gen, epochs=5,
    validation_data=test_gen,
    class_weight=class_weights,
    verbose=1
)
print(f"✅ Warm-up best: {max(h0.history['val_accuracy'])*100:.2f}%")

# ── STAGE 2: Fine-tune top 40 layers ──────────────────────
print("\n" + "="*55)
print("  STAGE 2: Fine-tuning top 40 layers")
print("  ~20 min/epoch on CPU | Leave overnight!")
print("="*55)

base.trainable = True
for layer in base.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=0.00005),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

callbacks = [
    ModelCheckpoint(OUTPUT, monitor='val_accuracy',
                    save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=8,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                      patience=3, min_lr=1e-8, verbose=1)
]

history = model.fit(
    train_gen, epochs=30,
    validation_data=test_gen,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# ── Save & report ──────────────────────────────────────────
model.save(OUTPUT)
loss, acc = model.evaluate(test_gen, verbose=0)
print("\n" + "="*55)
print("  FINE-TUNING COMPLETE")
print("="*55)
print(f"  ✅ Final Accuracy : {acc*100:.2f}%")
print(f"  ✅ Best achieved  : {max(history.history['val_accuracy'])*100:.2f}%")
print(f"  💾 Saved to      : {OUTPUT}")
print("="*55)