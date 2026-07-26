"""
app.py - Dashboard Pembelajaran Mahasiswa dengan Deteksi Emosi YOLOv13
======================================================================
Flask + SocketIO server untuk real-time emotion detection dashboard.
Model: YOLOv13 Master Final 4-Class (engaged, confused, bored, frustrated) + Neutral trick
"""

import os
import sys
import json
import time
import base64
import threading
from pathlib import Path
from collections import deque, Counter

import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_socketio import SocketIO, emit

# ─── Add parent directory to path so we can resolve model weights ───
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ultralytics import YOLO

# ─── App Configuration ──────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = 'skripsi-engagement-dashboard-2026'
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ─── Model Configuration ────────────────────────────────────────────
MODEL_V2_PATH = BASE_DIR / 'runs' / 'yolov13_master_combined_v2' / 'weights' / 'best.pt'
MODEL_V1_PATH = BASE_DIR / 'runs' / 'yolov13_master_combined' / 'weights' / 'best.pt'
MODEL_PATH = str(MODEL_V2_PATH if MODEL_V2_PATH.exists() else MODEL_V1_PATH)
MODULES_PATH = Path(__file__).resolve().parent / 'modules.json'
HISTORY_PATH = Path(__file__).resolve().parent / 'study_history.json'

TARGET_CLASSES = {0: 'engaged', 1: 'confused', 2: 'bored', 3: 'frustrated'}
ALL_EMOTIONS = ['engaged', 'confused', 'bored', 'frustrated', 'neutral']

# Inference parameters
IMGSZ = 640
CONF_THRESHOLD = 0.25
MIN_VOTE_RATIO = 0.50
MIN_AVG_CONFIDENCE = 0.65
WINDOW_SIZE = 30

# ─── Global State ────────────────────────────────────────────────────
model = None
camera = None
camera_active = False
camera_thread = None
camera_lock = threading.Lock()

# Per-session emotion tracking
emotion_sessions = {}  # session_id -> { 'history': [...], 'window': deque, 'start_time': ... }


def load_model():
    """Load the YOLOv13 model."""
    global model
    if model is None:
        print(f"[INFO] Loading YOLOv13 model from: {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            print(f"[ERROR] Model not found at: {MODEL_PATH}")
            print(f"[INFO] Falling back to simulation mode (no model)")
            return False
        model = YOLO(MODEL_PATH)
        print("[INFO] Model loaded successfully!")
    return True


def load_modules():
    """Load learning modules from JSON."""
    with open(MODULES_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_history():
    """Load study session history from JSON."""
    if not HISTORY_PATH.exists():
        return []
    try:
        with open(HISTORY_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load history: {e}")
        return []


def save_history(history_data):
    """Save study session history to JSON."""
    try:
        with open(HISTORY_PATH, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save history: {e}")
        return False


def get_device():
    """Get the best available device for inference."""
    try:
        import torch
        if torch.cuda.is_available():
            return '0'
    except ImportError:
        pass
    return 'cpu'


def process_frame(frame, window):
    """
    Run YOLOv13 inference on a frame and return emotion prediction.
    Uses smoothing window + neutral trick from the original script.
    """
    global model

    if model is None:
        # Simulation mode - return random emotions for testing
        import random
        emotions = ALL_EMOTIONS
        emotion = random.choice(emotions)
        confidence = random.uniform(0.4, 0.95)
        return emotion, confidence, {}

    frame_h, frame_w = frame.shape[:2]
    frame_area = frame_w * frame_h

    device = get_device()
    results = model.predict(frame, imgsz=IMGSZ, conf=CONF_THRESHOLD,
                            device=device, verbose=False)
    det = results[0].boxes

    stable_label = "neutral"
    raw_conf = 0.0
    vote_ratio = 0.0
    avg_conf = 0.0
    bbox = None

    if len(det) > 0:
        # Find largest detection (closest face)
        largest_area = 0
        best_det = None

        for i in range(len(det)):
            xyxy = det.xyxy[i].cpu().numpy()
            w = xyxy[2] - xyxy[0]
            h = xyxy[3] - xyxy[1]
            area = w * h
            if (area / frame_area) >= 0.02 and area > largest_area:
                largest_area = area
                best_det = (int(det.cls[i].item()), float(det.conf[i].item()), xyxy)

        if best_det is not None:
            cls_id, conf, bbox_coords = best_det
            raw_label = TARGET_CLASSES.get(cls_id, "unknown")
            raw_conf = conf
            bbox = bbox_coords

            window.append({'class_id': cls_id, 'conf': conf})

            if len(window) >= 8:
                counts = Counter([w['class_id'] for w in window])
                dom_id, dom_count = counts.most_common(1)[0]
                dom_label = TARGET_CLASSES.get(dom_id, "unknown")
                vote_ratio = dom_count / len(window)
                dom_confs = [w['conf'] for w in window if w['class_id'] == dom_id]
                avg_conf = sum(dom_confs) / len(dom_confs)

                if vote_ratio >= MIN_VOTE_RATIO and avg_conf >= MIN_AVG_CONFIDENCE:
                    stable_label = dom_label
                else:
                    stable_label = "neutral"
            else:
                stable_label = "neutral"
        else:
            window.clear()
    else:
        window.clear()

    # Draw bounding box on frame
    if bbox is not None:
        colors = {
            'engaged': (0, 255, 0),
            'confused': (255, 165, 0),
            'bored': (0, 165, 255),
            'frustrated': (0, 0, 255),
            'neutral': (200, 200, 200),
        }
        color = colors.get(stable_label, (255, 255, 255))
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label_text = f"{stable_label} ({avg_conf:.0%})"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
        text_color = (255, 255, 255) if stable_label != 'neutral' else (0, 0, 0)
        cv2.putText(frame, label_text, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

    confidence_info = {
        'vote_ratio': vote_ratio,
        'avg_confidence': avg_conf,
        'raw_confidence': raw_conf
    }

    return stable_label, avg_conf, confidence_info


def camera_stream():
    """Background thread for camera capture + inference + streaming."""
    global camera, camera_active

    print("[INFO] Camera stream thread started")

    while camera_active:
        if camera is None or not camera.isOpened():
            time.sleep(0.1)
            continue

        ret, frame = camera.read()
        if not ret:
            time.sleep(0.1)
            continue

        # Process all active emotion sessions
        for sid, session_data in list(emotion_sessions.items()):
            if not session_data.get('active', False):
                continue

            # Run inference
            frame_copy = frame.copy()
            emotion, confidence, info = process_frame(frame_copy, session_data['window'])

            # Record to history
            timestamp = time.time() - session_data['start_time']
            session_data['history'].append({
                'emotion': emotion,
                'confidence': confidence,
                'timestamp': timestamp
            })

            # Encode frame to base64 JPEG
            _, buffer = cv2.imencode('.jpg', frame_copy, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')

            # Calculate emotion distribution from history
            emotion_counts = Counter([h['emotion'] for h in session_data['history']])
            total = len(session_data['history'])
            distribution = {}
            for em in ALL_EMOTIONS:
                distribution[em] = round((emotion_counts.get(em, 0) / total) * 100, 1) if total > 0 else 0

            # Emit to specific client
            socketio.emit('camera_frame', {
                'frame': frame_b64,
                'emotion': emotion,
                'confidence': round(confidence, 2),
                'distribution': distribution,
                'duration': round(timestamp, 0),
                'info': info
            }, room=sid)

        # Control frame rate (~15 FPS for balance between smoothness and performance)
        time.sleep(0.066)

    print("[INFO] Camera stream thread stopped")


# ─── Routes ──────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Login page."""
    return render_template('index.html')


@app.route('/login', methods=['POST'])
def login():
    """Handle login form submission."""
    name = request.form.get('name', '').strip()
    nim = request.form.get('nim', '').strip()

    if not name or not nim:
        return redirect(url_for('index'))

    session['name'] = name
    session['nim'] = nim
    return redirect(url_for('dashboard'))


@app.route('/dashboard')
def dashboard():
    """Main dashboard page."""
    if 'name' not in session:
        return redirect(url_for('index'))

    modules = load_modules()
    return render_template('dashboard.html',
                           name=session['name'],
                           nim=session['nim'],
                           modules=modules)


@app.route('/api/modules')
def api_modules():
    """API to get module data."""
    modules = load_modules()
    return jsonify(modules)


@app.route('/api/check-answer', methods=['POST'])
def check_answer():
    """API to check quiz answers."""
    data = request.json
    module_id = data.get('module_id')
    answers = data.get('answers', {})

    modules = load_modules()
    module = next((m for m in modules if m['id'] == module_id), None)

    if not module:
        return jsonify({'error': 'Module not found'}), 404

    correct = 0
    total = len(module['questions'])
    results = []

    for q in module['questions']:
        user_answer = answers.get(q['id'])
        is_correct = user_answer == q['correct']
        if is_correct:
            correct += 1
        results.append({
            'question_id': q['id'],
            'correct_answer': q['correct'],
            'user_answer': user_answer,
            'is_correct': is_correct
        })

    score = round((correct / total) * 100) if total > 0 else 0

    return jsonify({
        'correct': correct,
        'total': total,
        'score': score,
        'results': results
    })


@app.route('/api/history', methods=['GET'])
def get_history():
    """API to get study session history for the logged-in student."""
    if 'name' not in session or 'nim' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    user_nim = session['nim']
    history = load_history()

    # Filter history for current logged-in user
    user_history = [item for item in history if item.get('nim') == user_nim]

    # Calculate overall statistics
    total_sessions = len(user_history)
    avg_score = round(sum(item.get('score', 0) for item in user_history) / total_sessions) if total_sessions > 0 else 0
    total_duration_sec = sum(item.get('duration_seconds', 0) for item in user_history)

    # Calculate overall dominant emotion
    emotion_counts = Counter([item.get('dominant_emotion') for item in user_history if item.get('dominant_emotion')])
    overall_dominant = emotion_counts.most_common(1)[0][0] if emotion_counts else 'neutral'

    return jsonify({
        'history': user_history,
        'stats': {
            'total_sessions': total_sessions,
            'avg_score': avg_score,
            'total_duration_sec': total_duration_sec,
            'overall_dominant': overall_dominant
        }
    })


@app.route('/api/save-session', methods=['POST'])
def save_session():
    """API to record a completed study session into history."""
    if 'name' not in session or 'nim' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.json or {}
    if not data.get('module_id'):
        return jsonify({'error': 'Invalid session data'}), 400

    from datetime import datetime
    import uuid

    history = load_history()

    new_record = {
        'id': str(uuid.uuid4())[:8],
        'nim': session['nim'],
        'name': session['name'],
        'module_id': data.get('module_id'),
        'module_title': data.get('module_title', 'Modul Pembelajaran'),
        'completed_at': datetime.now().strftime("%d %b %Y, %H:%M"),
        'score': data.get('score', 0),
        'correct_answers': data.get('correct_answers', 0),
        'total_questions': data.get('total_questions', 0),
        'duration_seconds': data.get('duration_seconds', 0),
        'dominant_emotion': data.get('dominant_emotion', 'neutral'),
        'emotion_distribution': data.get('emotion_distribution', {})
    }

    history.insert(0, new_record)  # Insert newest at top
    saved = save_history(history)

    if saved:
        return jsonify({'message': 'Session saved successfully', 'record': new_record}), 201
    else:
        return jsonify({'error': 'Failed to save session'}), 500


@app.route('/logout')
def logout():
    """Handle logout."""
    session.clear()
    return redirect(url_for('index'))


# ─── SocketIO Events ────────────────────────────────────────────────

@socketio.on('connect')
def handle_connect():
    print(f"[INFO] Client connected: {request.sid}")


@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    print(f"[INFO] Client disconnected: {sid}")
    # Cleanup session
    if sid in emotion_sessions:
        emotion_sessions[sid]['active'] = False
        del emotion_sessions[sid]

    # Stop camera if no active sessions
    if not any(s.get('active', False) for s in emotion_sessions.values()):
        stop_camera()


@socketio.on('start_camera')
def handle_start_camera():
    """Start camera and emotion tracking for this client."""
    global camera, camera_active, camera_thread

    sid = request.sid
    print(f"[INFO] Starting camera for session: {sid}")

    # Initialize emotion session
    emotion_sessions[sid] = {
        'window': deque(maxlen=WINDOW_SIZE),
        'history': [],
        'start_time': time.time(),
        'active': True
    }

    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                emit('camera_error', {'message': 'Cannot open camera. Please check your webcam connection.'})
                return

            # Set camera properties
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not camera_active:
            camera_active = True
            camera_thread = threading.Thread(target=camera_stream, daemon=True)
            camera_thread.start()

    emit('camera_started', {'message': 'Camera started successfully'})


@socketio.on('stop_camera')
def handle_stop_camera():
    """Stop camera for this client."""
    sid = request.sid
    if sid in emotion_sessions:
        emotion_sessions[sid]['active'] = False

    # Stop camera if no active sessions
    if not any(s.get('active', False) for s in emotion_sessions.values()):
        stop_camera()

    emit('camera_stopped', {'message': 'Camera stopped'})


@socketio.on('get_emotion_report')
def handle_emotion_report():
    """Get the final emotion report for this session."""
    sid = request.sid
    session_data = emotion_sessions.get(sid, {})
    history = session_data.get('history', [])

    if not history:
        emit('emotion_report', {
            'distribution': {em: 0 for em in ALL_EMOTIONS},
            'dominant': 'neutral',
            'total_frames': 0,
            'duration': 0,
            'timeline': []
        })
        return

    # Calculate distribution
    emotion_counts = Counter([h['emotion'] for h in history])
    total = len(history)
    distribution = {}
    for em in ALL_EMOTIONS:
        distribution[em] = round((emotion_counts.get(em, 0) / total) * 100, 1)

    # Find dominant emotion
    dominant = max(distribution, key=distribution.get)

    # Create timeline (sample every 5 seconds)
    duration = history[-1]['timestamp'] if history else 0
    timeline = []
    step = 5  # seconds
    for t in range(0, int(duration) + 1, step):
        nearby = [h for h in history if abs(h['timestamp'] - t) < step / 2]
        if nearby:
            most_common = Counter([h['emotion'] for h in nearby]).most_common(1)[0][0]
            timeline.append({'time': t, 'emotion': most_common})

    emit('emotion_report', {
        'distribution': distribution,
        'dominant': dominant,
        'total_frames': total,
        'duration': round(duration, 0),
        'timeline': timeline
    })


def stop_camera():
    """Stop the camera and release resources."""
    global camera, camera_active

    camera_active = False
    time.sleep(0.2)  # Wait for camera thread to finish

    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None

    print("[INFO] Camera released")


# ─── Main ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("  Dashboard Pembelajaran Mahasiswa")
    print("  Deteksi Emosi Real-time dengan YOLOv13")
    print("=" * 60)

    # Load model at startup
    has_model = load_model()
    if not has_model:
        print("[WARNING] Running in simulation mode (no model loaded)")
        print("[WARNING] Emotion detection will use random values")
    
    device = get_device()
    print(f"[INFO] Inference device: {device}")
    print(f"[INFO] Starting server on http://localhost:5000")
    print("=" * 60)

    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
