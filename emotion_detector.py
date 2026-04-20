import cv2
import numpy as np
import json
import os
from collections import deque
import mediapipe as mp

# ── Emotion Labels ─────────────────────────────────────────
EMOTION_LABELS = {
    0: 'surprise',
    1: 'fear',
    2: 'disgust',
    3: 'happy',
    4: 'sad',
    5: 'angry',
    6: 'neutral'
}

EMOTION_COLORS = {
    'happy':    (0, 255, 0),
    'neutral':  (255, 255, 0),
    'surprise': (0, 165, 255),
    'angry':    (0, 0, 255),
    'sad':      (255, 0, 0),
    'fear':     (128, 0, 128),
    'disgust':  (0, 128, 0)
}

EMOTION_EMOJI = {
    'happy':    '😊',
    'neutral':  '😐',
    'surprise': '😲',
    'angry':    '😠',
    'sad':      '😢',
    'fear':     '😨',
    'disgust':  '🤢'
}

MODEL_PATH  = os.path.join(os.path.dirname(__file__), 'model', 'emotion_model.keras')
INDEX_PATH  = os.path.join(os.path.dirname(__file__), 'model', 'class_indices.json')
IMG_SIZE    = 112

# ── Load Model ─────────────────────────────────────────────
model = None
class_indices = None
def load_model():
    global model, class_indices
    try:
        import tensorflow as tf
        print("Loading emotion model...")

        base = tf.keras.applications.MobileNetV2(
            input_shape=(112, 112, 3),
            include_top=False,
            weights=None
        )
        inputs = base.input
        x = base.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001)(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001)(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001)(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(7, activation='softmax')(x)
        model = tf.keras.Model(inputs, outputs)

        weights_path = os.path.join(os.path.dirname(__file__), 'model', 'emotion_model.h5')
        model.load_weights(weights_path)
        print(f"✅ Model weights loaded!")

        if os.path.exists(INDEX_PATH):
            with open(INDEX_PATH, 'r') as f:
                class_indices = json.load(f)
            global EMOTION_LABELS
            EMOTION_LABELS = {v: k for k, v in class_indices.items()}
            print(f"✅ Class indices loaded: {EMOTION_LABELS}")
        return True
    except Exception as e:
        print(f"❌ Model load error: {e}")
        return False

# ── MediaPipe Face Detection ────────────────────────────────
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

# ── Frame Smoothing Buffer ──────────────────────────────────
prediction_buffer = deque(maxlen=10)

def preprocess_image(img_array):
    """Preprocess image for model prediction"""
    img = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_emotion(img_array):
    """Predict emotion from image array"""
    if model is None:
        return None, 0.0
    try:
        processed = preprocess_image(img_array)
        predictions = model.predict(processed, verbose=0)
        predicted_idx = int(np.argmax(predictions[0]))
        confidence    = float(np.max(predictions[0]))
        emotion       = EMOTION_LABELS.get(predicted_idx, 'unknown')
        all_scores    = {
            EMOTION_LABELS.get(i, str(i)): float(predictions[0][i])
            for i in range(len(predictions[0]))
        }
        return emotion, confidence, all_scores
    except Exception as e:
        print(f"Prediction error: {e}")
        return 'unknown', 0.0, {}

def detect_faces_mediapipe(frame):
    """Detect faces using MediaPipe"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb)
    faces = []
    if results.detections:
        h, w = frame.shape[:2]
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x = max(0, int(bbox.xmin * w))
            y = max(0, int(bbox.ymin * h))
            fw = min(int(bbox.width * w), w - x)
            fh = min(int(bbox.height * h), h - y)
            if fw > 20 and fh > 20:
                faces.append((x, y, fw, fh))
    return faces

def get_smoothed_emotion(emotion):
    """Apply majority voting across last 10 frames"""
    prediction_buffer.append(emotion)
    if len(prediction_buffer) >= 3:
        from collections import Counter
        counter = Counter(prediction_buffer)
        return counter.most_common(1)[0][0]
    return emotion

def process_uploaded_image(image_path):
    """Process an uploaded image file"""
    if model is None:
        return {'error': 'Model not loaded'}
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {'error': 'Could not read image'}

        faces = detect_faces_mediapipe(img)

        if not faces:
            # No face detected — predict on full image
            emotion, confidence, all_scores = predict_emotion(img)
            return {
                'emotion': emotion,
                'confidence': round(confidence * 100, 2),
                'all_scores': {k: round(v*100, 2) for k, v in all_scores.items()},
                'faces_detected': 0,
                'emoji': EMOTION_EMOJI.get(emotion, ''),
                'color': EMOTION_COLORS.get(emotion, (255,255,255))
            }

        # Use first detected face
        x, y, fw, fh = faces[0]
        face_roi = img[y:y+fh, x:x+fw]
        emotion, confidence, all_scores = predict_emotion(face_roi)

        return {
            'emotion': emotion,
            'confidence': round(confidence * 100, 2),
            'all_scores': {k: round(v*100, 2) for k, v in all_scores.items()},
            'faces_detected': len(faces),
            'face_box': [x, y, fw, fh],
            'emoji': EMOTION_EMOJI.get(emotion, ''),
            'color': EMOTION_COLORS.get(emotion, (255,255,255))
        }
    except Exception as e:
        return {'error': str(e)}

def process_webcam_frame(frame_array):
    """Process a single webcam frame"""
    if model is None:
        return {'error': 'Model not loaded'}
    try:
        faces = detect_faces_mediapipe(frame_array)

        if not faces:
            return {
                'emotion': 'No face detected',
                'confidence': 0,
                'faces_detected': 0
            }

        x, y, fw, fh = faces[0]
        face_roi = frame_array[y:y+fh, x:x+fw]
        emotion, confidence, all_scores = predict_emotion(face_roi)
        smoothed = get_smoothed_emotion(emotion)

        return {
            'emotion': smoothed,
            'raw_emotion': emotion,
            'confidence': round(confidence * 100, 2),
            'all_scores': {k: round(v*100, 2) for k, v in all_scores.items()},
            'faces_detected': len(faces),
            'face_box': [x, y, fw, fh],
            'emoji': EMOTION_EMOJI.get(smoothed, '')
        }
    except Exception as e:
        return {'error': str(e)}