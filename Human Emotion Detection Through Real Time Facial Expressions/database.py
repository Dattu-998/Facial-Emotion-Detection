import mysql.connector
from datetime import datetime

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'emotion_ai'
}

def get_connection():
    return mysql.connector.connect(**DB_CONFIG)

def save_prediction(image_path, emotion, confidence, source='upload'):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO emotion_predictions 
            (image_path, emotion, confidence, source, created_at)
            VALUES (%s, %s, %s, %s, %s)
        """, (image_path, emotion, float(confidence), source, datetime.now()))
        conn.commit()
        prediction_id = cursor.lastrowid
        cursor.close()
        conn.close()
        return prediction_id
    except Exception as e:
        print(f"DB Error (save_prediction): {e}")
        return None

def save_webcam_log(session_id, emotion, confidence):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO webcam_logs 
            (session_id, emotion, confidence, timestamp)
            VALUES (%s, %s, %s, %s)
        """, (session_id, emotion, float(confidence), datetime.now()))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"DB Error (save_webcam_log): {e}")

def get_prediction_history(limit=50):
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT id, image_path, emotion, confidence, source, created_at
            FROM emotion_predictions
            ORDER BY created_at DESC
            LIMIT %s
        """, (limit,))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return rows
    except Exception as e:
        print(f"DB Error (get_history): {e}")
        return []

def get_emotion_stats():
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT emotion, COUNT(*) as count,
                   AVG(confidence) as avg_confidence
            FROM emotion_predictions
            GROUP BY emotion
            ORDER BY count DESC
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return rows
    except Exception as e:
        print(f"DB Error (get_stats): {e}")
        return []

def get_recent_webcam_stats():
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT emotion, COUNT(*) as count
            FROM webcam_logs
            GROUP BY emotion
            ORDER BY count DESC
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return rows
    except Exception as e:
        print(f"DB Error (webcam_stats): {e}")
        return []