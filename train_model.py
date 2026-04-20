import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D,
                                      Dropout, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                         ReduceLROnPlateau)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

# ── Configuration ──────────────────────────────────────────
BASE_DIR    = r"E:\New folder\htdocs\emotion-ai"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR   = os.path.join(BASE_DIR, "model")
MODEL_PATH  = os.path.join(MODEL_DIR, "emotion_model.h5")
CHECKPOINT  = os.path.join(MODEL_DIR, "best_model.h5")

IMG_SIZE    = 96
BATCH_SIZE  = 32
NUM_CLASSES = 7

# RAF-DB official class mapping
CLASS_NAMES = {
    '1': 'surprise',
    '2': 'fear',
    '3': 'disgust',
    '4': 'happy',
    '5': 'sad',
    '6': 'angry',
    '7': 'neutral'
}

print("="*55)
print("  RAF-DB Emotion Recognition Training")
print("="*55)
print(f"  TensorFlow : {tf.__version__}")
print(f"  Device     : {'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'}")
print(f"  Image size : {IMG_SIZE}x{IMG_SIZE} RGB")
print(f"  Batch size : {BATCH_SIZE}")
print("="*55)

# ── Data Generators ────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.75, 1.25],
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, 'train'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, 'test'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='categorical',
    shuffle=False
)

print(f"\n  Train samples : {train_generator.samples}")
print(f"  Test samples  : {test_generator.samples}")
print(f"  Class indices : {train_generator.class_indices}")

# ── Handle Class Imbalance ─────────────────────────────────
labels = train_generator.classes
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(class_weights_array))
print(f"\n  Class weights (imbalance correction):")
for idx, weight in class_weights.items():
    print(f"    Class {idx}: {weight:.3f}")

# ── Save Class Mapping ─────────────────────────────────────
os.makedirs(MODEL_DIR, exist_ok=True)
indices_path = os.path.join(MODEL_DIR, "class_indices.json")
# Save with emotion names for use in Flask app
emotion_map = {}
for folder_num, idx in train_generator.class_indices.items():
    emotion_name = CLASS_NAMES.get(folder_num, folder_num)
    emotion_map[emotion_name] = idx
with open(indices_path, 'w') as f:
    json.dump(emotion_map, f, indent=2)
print(f"\n  Class indices saved: {indices_path}")

# ── Build Model ────────────────────────────────────────────
print("\n  Building EfficientNetB0 model...")

base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

total     = model.count_params()
trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"  Total params     : {total:,}")
print(f"  Trainable params : {trainable:,}")

# ── Stage 1: Frozen Base ───────────────────────────────────
print("\n" + "="*55)
print("  STAGE 1: Training top layers (base frozen)")
print("  Expected: ~1.5 hours | Target: 65-70%")
print("="*55)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_s1 = [
    ModelCheckpoint(
        CHECKPOINT,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    )
]

history1 = model.fit(
    train_generator,
    epochs=25,
    validation_data=test_generator,
    callbacks=callbacks_s1,
    class_weight=class_weights,
    verbose=1
)

stage1_best = max(history1.history['val_accuracy'])
print(f"\n  ✅ Stage 1 Complete")
print(f"  Best val_accuracy: {stage1_best*100:.2f}%")

# ── Stage 2: Fine-tune Top 50 Layers ──────────────────────
print("\n" + "="*55)
print("  STAGE 2: Fine-tuning top 50 layers")
print("  Expected: ~3-4 hours | Target: 78-84%")
print("="*55)

base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

trainable2 = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"  Trainable params now: {trainable2:,}")

model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_s2 = [
    ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=4,
        min_lr=1e-8,
        verbose=1
    )
]

initial_epoch = len(history1.history['val_accuracy'])

history2 = model.fit(
    train_generator,
    epochs=initial_epoch + 50,
    initial_epoch=initial_epoch,
    validation_data=test_generator,
    callbacks=callbacks_s2,
    class_weight=class_weights,
    verbose=1
)

# ── Final Results ──────────────────────────────────────────
stage2_best = max(history2.history['val_accuracy'])
overall_best = max(stage1_best, stage2_best)

model.save(MODEL_PATH)

print("\n" + "="*55)
print("  TRAINING COMPLETE")
print("="*55)

loss, accuracy = model.evaluate(test_generator, verbose=0)
print(f"  ✅ Final Test Accuracy : {accuracy*100:.2f}%")
print(f"  ✅ Final Test Loss     : {loss:.4f}")
print(f"  📊 Stage 1 best       : {stage1_best*100:.2f}%")
print(f"  📊 Stage 2 best       : {stage2_best*100:.2f}%")
print(f"  📊 Overall best       : {overall_best*100:.2f}%")
print(f"  💾 Model saved        : {MODEL_PATH}")
print("="*55)