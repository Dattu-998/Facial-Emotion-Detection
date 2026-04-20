import os
import uuid
import base64
import numpy as np
import cv2
from flask import (Flask, request, jsonify, render_template, 
                   send_from_directory)
from flask_cors import CORS
from werkzeug.utils import secure_filename
from database import (save_prediction, save_webcam_log,
                      get_prediction_history, get_emotion_stats,
                      get_recent_webcam_stats)
from emotion_detector import load_model, process_uploaded_image, process_webcam_frame

# ── App Setup ──────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER  = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXT    = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_CONTENT_MB = 16

app.config['UPLOAD_FOLDER']    = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_MB * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Load Model on Startup ──────────────────────────────────
print("Initializing Emotion AI System...")
model_loaded = load_model()
if model_loaded:
    print("✅ System ready!")
else:
    print("⚠️  Model not found — place emotion_model.h5 in model/ folder")

# ── Helper ─────────────────────────────────────────────────
def allowed_file(filename):
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT)

# ── Page Routes ────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload_page():
    return render_template('upload.html')

@app.route('/webcam')
def webcam_page():
    return render_template('webcam.html')

@app.route('/dashboard')
def dashboard_page():
    return render_template('dashboard.html')

# ── API: Upload Image Prediction ───────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        filename  = secure_filename(f"{uuid.uuid4()}_{file.filename}")
        filepath  = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result = process_uploaded_image(filepath)

        if 'error' not in result:
            save_prediction(
                image_path=f"uploads/{filename}",
                emotion=result['emotion'],
                confidence=result['confidence'],
                source='upload'
            )
            result['image_path'] = f"/uploads/{filename}"

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── API: Webcam Frame Prediction ───────────────────────────
@app.route('/predict-live', methods=['POST'])
def predict_live():
    try:
        data = request.get_json()
        if not data or 'frame' not in data:
            return jsonify({'error': 'No frame data'}), 400

        frame_data = data['frame']
        session_id = data.get('session_id', str(uuid.uuid4()))

        # Decode base64 frame
        if ',' in frame_data:
            frame_data = frame_data.split(',')[1]

        frame_bytes  = base64.b64decode(frame_data)
        frame_array  = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame        = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Could not decode frame'}), 400

        result = process_webcam_frame(frame)

        # Save to DB every 5th frame (don't flood the database)
        if ('error' not in result and
                result.get('faces_detected', 0) > 0 and
                result.get('confidence', 0) > 40):
            save_webcam_log(
                session_id=session_id,
                emotion=result['emotion'],
                confidence=result['confidence']
            )

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── API: History ───────────────────────────────────────────
@app.route('/history', methods=['GET'])
def history():
    try:
        limit   = int(request.args.get('limit', 50))
        records = get_prediction_history(limit)
        for r in records:
            if r.get('created_at'):
                r['created_at'] = str(r['created_at'])
        return jsonify({'history': records, 'count': len(records)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── API: Stats ─────────────────────────────────────────────
@app.route('/stats', methods=['GET'])
def stats():
    try:
        upload_stats = get_emotion_stats()
        webcam_stats = get_recent_webcam_stats()
        return jsonify({
            'upload_stats': upload_stats,
            'webcam_stats': webcam_stats
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Serve Uploaded Images ──────────────────────────────────
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ── Run ────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
