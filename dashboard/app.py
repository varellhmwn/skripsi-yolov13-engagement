"""
app.py - Dashboard Pembelajaran Mahasiswa dengan Deteksi Emosi YOLOv13
======================================================================
Flask + SocketIO server untuk real-time emotion detection dashboard.
Model: YOLOv13 Master Final 4-Class (engaged, confused, bored, frustrated) + Neutral trick
"""

from ultralytics import YOLO
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


# ─── App Configuration ──────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = 'skripsi-engagement-dashboard-2026'
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ─── Model Configuration ────────────────────────────────────────────
MODEL_PATH = str(BASE_DIR / 'runs' /
                 'yolov13_master_combined_v3' / 'weights' / 'best.pt')
MODULES_PATH = Path(__file__).resolve().parent / 'modules.json'
HISTORY_PATH = Path(__file__).resolve().parent / 'study_history.json'

TARGET_CLASSES = {0: 'engaged', 1: 'confused', 2: 'bored', 3: 'frustrated'}
ALL_EMOTIONS = ['engaged', 'confused', 'bored', 'frustrated']

# Inference parameters
IMGSZ = 640
CONF_THRESHOLD = 0.20
MIN_VOTE_RATIO = 0.40
MIN_AVG_CONFIDENCE = 0.40
WINDOW_SIZE = 30

# ─── Global State ────────────────────────────────────────────────────
model = None
camera = None
camera_active = False
camera_thread = None
camera_lock = threading.Lock()

# Per-session emotion tracking
# session_id -> { 'history': [...], 'window': deque, 'start_time': ... }
emotion_sessions = {}

# ─── Shared Frame Buffer (dual-thread architecture) ──────────────────
# Inference thread writes here; render thread reads and emits to browser
# Structure per session_id: { 'frame': np.ndarray, 'emotion': str, 'confidence': float,
#                             'info': dict, 'distribution': dict, 'duration': float }
_latest_results: dict = {}  # sid -> last inference result
_latest_frame: dict = {}    # sid -> last annotated frame (numpy array)
_frame_lock = threading.Lock()

# Render thread target: ~30 FPS display regardless of inference speed
RENDER_FPS = 30
INFER_FPS = 30   # YOLO inference target (GPU can handle this easily)


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
                best_det = (int(det.cls[i].item()),
                            float(det.conf[i].item()), xyxy)

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
                dom_confs = [w['conf']
                             for w in window if w['class_id'] == dom_id]
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
        (tw, th), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
        text_color = (
            255, 255, 255) if stable_label != 'neutral' else (0, 0, 0)
        cv2.putText(frame, label_text, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

    confidence_info = {
        'vote_ratio': vote_ratio,
        'avg_confidence': avg_conf,
        'raw_confidence': raw_conf
    }

    return stable_label, avg_conf, confidence_info


def _inference_thread():
    """Thread A — YOLO inference at full GPU speed.
    Reads raw frames from camera, runs inference, writes annotated frame
    and latest results into shared buffer (_latest_frame, _latest_results).
    Does NOT encode or transmit — that is handled by the render thread.
    """
    global camera, camera_active

    infer_interval = 1.0 / INFER_FPS
    print("[INFO] Inference thread started")

    while camera_active:
        if camera is None or not camera.isOpened():
            time.sleep(0.05)
            continue

        t0 = time.time()

        with camera_lock:
            ret, frame = camera.read()
        if not ret:
            time.sleep(0.02)
            continue

        active_sessions = [(sid, sd) for sid, sd in list(emotion_sessions.items())
                           if sd.get('active', False)]
        if not active_sessions:
            time.sleep(0.02)
            continue

        # Run YOLO inference once per frame (shared across sessions)
        with camera_lock:
            frame_copy = frame.copy()

        # Use the first active session's window for inference
        # (all active sessions share the same camera feed)
        primary_sid, primary_sd = active_sessions[0]
        emotion, confidence, info = process_frame(
            frame_copy, primary_sd['window'])
        timestamp = time.time() - primary_sd['start_time']

        # Update history and distribution for ALL active sessions
        for sid, session_data in active_sessions:
            session_data['history'].append({
                'emotion': emotion,
                'confidence': confidence,
                'timestamp': timestamp
            })

            emotion_counts = Counter([h['emotion']
                                     for h in session_data['history']])
            total = len(session_data['history'])
            distribution = {
                em: round((emotion_counts.get(em, 0) / total)
                          * 100, 1) if total > 0 else 0
                for em in ALL_EMOTIONS
            }

            with _frame_lock:
                _latest_results[sid] = {
                    'emotion': emotion,
                    'confidence': round(confidence, 2),
                    'distribution': distribution,
                    'duration': round(timestamp, 0),
                    'info': info
                }
                _latest_frame[sid] = frame_copy.copy()

        # Pace inference to target rate (GPU can handle 30 FPS easily)
        elapsed = time.time() - t0
        sleep_t = infer_interval - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

    print("[INFO] Inference thread stopped")


def _render_thread():
    """Thread B — Encode & emit frames to browser at RENDER_FPS.
    Reads the latest annotated frame + results from shared buffer and
    sends them to connected clients. Completely independent of inference speed.
    """
    global camera_active

    render_interval = 1.0 / RENDER_FPS
    print("[INFO] Render thread started")

    while camera_active:
        t0 = time.time()

        with _frame_lock:
            sessions_to_emit = {
                sid: (frame.copy(), dict(result))
                for sid, frame in _latest_frame.items()
                if sid in _latest_results
                for result in [_latest_results[sid]]
            }

        for sid, (frame, result) in sessions_to_emit.items():
            try:
                _, buffer = cv2.imencode(
                    '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                socketio.emit('camera_frame', {
                    'frame': frame_b64,
                    **result
                }, room=sid)
            except Exception as e:
                print(f"[WARN] Render emit error for {sid}: {e}")

        elapsed = time.time() - t0
        sleep_t = render_interval - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

    print("[INFO] Render thread stopped")


def camera_stream():
    """Launch dual-thread camera pipeline:
    - Thread A: YOLO inference at GPU speed
    - Thread B: Frame encoding & emit at display rate
    """
    t_infer = threading.Thread(target=_inference_thread, daemon=True)
    t_render = threading.Thread(target=_render_thread, daemon=True)
    t_infer.start()
    t_render.start()
    t_infer.join()
    t_render.join()
    print("[INFO] Camera stream (dual-thread) stopped")


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


def generate_academic_interpretation(score, concept_correct, problem_solving_correct, emotion_dist):
    """
    Rule-based non-clinical academic emotion interpretation based on learning score
    and facial expression-based emotion detection.
    """
    eng = emotion_dist.get('engaged', 0)
    conf = emotion_dist.get('confused', 0)
    frust = emotion_dist.get('frustrated', 0)

    # Priority Rules:
    # Rule 1: High score AND High Engagement
    if score >= 75 and eng >= 40.0:
        return "Mahasiswa menunjukkan keterlibatan belajar yang baik."

    # Rule 2: Low Problem Solving score AND High Confusion
    if problem_solving_correct < 3 and conf >= 25.0:
        return "Mahasiswa menunjukkan indikasi kesulitan memahami materi."

    # Rule 3: High/Increasing Frustration
    if frust >= 20.0:
        return "Terdapat peningkatan indikasi kesulitan selama proses pembelajaran."

    # Default Rule:
    if score >= 70:
        return "Mahasiswa menunjukkan indikasi emosi belajar yang relatif stabil dengan hasil pemahaman yang baik."
    else:
        return "Mahasiswa menunjukkan indikasi emosi belajar yang relatif stabil selama sesi pembelajaran."


@app.route('/api/modules')
def api_modules():
    """API to get module data."""
    modules = load_modules()
    return jsonify(modules)


@app.route('/api/check-answer', methods=['POST'])
def check_answer():
    """API to check quiz or coding exercise answers."""
    data = request.json or {}
    module_id = data.get('module_id')
    answers = data.get('answers', {})
    is_coding_mode = data.get('is_coding_mode', False)
    emotion_dist = data.get('emotion_distribution', {})

    modules = load_modules()
    module = next((m for m in modules if m['id'] == module_id), None)

    if not module:
        return jsonify({'error': 'Module not found'}), 404

    correct = 0
    results = []

    if is_coding_mode:
        coding_exercises = module.get('coding_exercises', [])
        total = len(coding_exercises)
        for ex in coding_exercises:
            user_val = str(answers.get(ex['id'], '')).strip()
            expected_val = str(ex.get('expected_answer', '')).strip()
            is_correct = user_val.lower() == expected_val.lower()
            if is_correct:
                correct += 1

            results.append({
                'question_id': ex['id'],
                'category': 'Coding Exercise',
                'correct_answer': expected_val,
                'user_answer': user_val,
                'is_correct': is_correct
            })

        score = round((correct / total) * 100) if total > 0 else 0
        concept_correct = None
        concept_total = None
        concept_score_pct = None
        problem_solving_correct = correct
        problem_solving_total = total
        problem_solving_score_pct = score

    else:
        questions = module.get('questions', [])
        total = len(questions)
        concept_correct = 0
        concept_total = 0
        problem_solving_correct = 0
        problem_solving_total = 0

        letter_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        rev_letter_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

        for q in questions:
            cat = q.get('category', 'Concept')
            if cat == 'Concept':
                concept_total += 1
            elif cat == 'Problem Solving':
                problem_solving_total += 1

            user_answer = answers.get(q['id'])
            correct_target = q.get('correct_answer')

            is_correct = False
            if correct_target is not None:
                if isinstance(user_answer, str) and user_answer.upper() == str(correct_target).upper():
                    is_correct = True
                elif isinstance(user_answer, int) and rev_letter_map.get(user_answer) == str(correct_target).upper():
                    is_correct = True
            elif 'correct' in q:
                if user_answer == q['correct'] or (isinstance(user_answer, str) and letter_map.get(user_answer.upper()) == q['correct']):
                    is_correct = True

            if is_correct:
                correct += 1
                if cat == 'Concept':
                    concept_correct += 1
                elif cat == 'Problem Solving':
                    problem_solving_correct += 1

            results.append({
                'question_id': q['id'],
                'category': cat,
                'correct_answer': q.get('correct_answer', rev_letter_map.get(q.get('correct', 0))),
                'user_answer': user_answer,
                'is_correct': is_correct
            })

        score = round((correct / total) * 100) if total > 0 else 0
        concept_score_pct = round(
            (concept_correct / concept_total) * 100) if concept_total > 0 else 0
        problem_solving_score_pct = round(
            (problem_solving_correct / problem_solving_total) * 100) if problem_solving_total > 0 else 0

    interpretation = generate_academic_interpretation(
        score, concept_correct, problem_solving_correct, emotion_dist
    )

    return jsonify({
        'is_coding_mode': is_coding_mode,
        'correct': correct,
        'total': total,
        'score': score,
        'concept_correct': concept_correct,
        'concept_total': concept_total,
        'concept_score_pct': concept_score_pct,
        'problem_solving_correct': problem_solving_correct,
        'problem_solving_total': problem_solving_total,
        'problem_solving_score_pct': problem_solving_score_pct,
        'interpretation': interpretation,
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
    avg_score = round(sum(item.get('score', 0) for item in user_history) /
                      total_sessions) if total_sessions > 0 else 0
    total_duration_sec = sum(item.get('duration_seconds', 0)
                             for item in user_history)

    # Calculate overall dominant emotion
    emotion_counts = Counter([item.get('dominant_emotion')
                             for item in user_history if item.get('dominant_emotion')])
    overall_dominant = emotion_counts.most_common(
        1)[0][0] if emotion_counts else 'neutral'

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

    score = data.get('score', 0)
    concept_correct = data.get('concept_correct', 0)
    problem_solving_correct = data.get('problem_solving_correct', 0)
    emotion_dist = data.get('emotion_distribution', {})

    interpretation = data.get('interpretation') or generate_academic_interpretation(
        score, concept_correct, problem_solving_correct, emotion_dist
    )

    new_record = {
        'id': str(uuid.uuid4())[:8],
        'nim': session['nim'],
        'name': session['name'],
        'module_id': data.get('module_id'),
        'module_title': data.get('module_title', 'Modul Pembelajaran'),
        'completed_at': datetime.now().strftime("%d %b %Y, %H:%M"),
        'score': score,
        'correct_answers': data.get('correct_answers', 0),
        'total_questions': data.get('total_questions', 10),
        'concept_correct': concept_correct,
        'concept_total': data.get('concept_total', 5),
        'problem_solving_correct': problem_solving_correct,
        'problem_solving_total': data.get('problem_solving_total', 5),
        'duration_seconds': data.get('duration_seconds', 0),
        'dominant_emotion': data.get('dominant_emotion', 'neutral'),
        'emotion_distribution': emotion_dist,
        'timeline': data.get('timeline', []),
        'question_tracking': data.get('question_tracking', []),
        'interpretation': interpretation
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

    # Cleanup shared frame buffers
    with _frame_lock:
        _latest_frame.pop(sid, None)
        _latest_results.pop(sid, None)

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
                emit('camera_error', {
                     'message': 'Cannot open camera. Please check your webcam connection.'})
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

    # Create timeline with narrow window sampling & transition detection
    duration = history[-1]['timestamp'] if history else 0
    timeline = []

    if history:
        # Select base step interval based on total duration
        if duration <= 180:
            step = 10
        elif duration <= 600:
            step = 20
        else:
            step = 30

        raw_points = []
        for t in range(0, int(duration) + 1, step):
            nearby = [h for h in history if abs(h['timestamp'] - t) <= 4.0]
            if nearby:
                # Active Cognitive Emotion Prioritization:
                # If any active cognitive emotion (engaged, confused, bored, frustrated) is present,
                # prioritize it over neutral to highlight learning state changes.
                active_emotions = [h['emotion']
                                   for h in nearby if h['emotion'] != 'neutral']
                if active_emotions:
                    best_em = Counter(active_emotions).most_common(1)[0][0]
                else:
                    best_em = 'neutral'
                raw_points.append({'time': t, 'emotion': best_em})
            else:
                closest = min(history, key=lambda h: abs(h['timestamp'] - t))
                raw_points.append({'time': t, 'emotion': closest['emotion']})

        # Detect transition points where emotion changes
        smoothed_history = []
        win_size = 5
        for i in range(len(history)):
            sub = history[max(0, i - win_size // 2)
                              : min(len(history), i + win_size // 2 + 1)]
            top_em = Counter([h['emotion'] for h in sub]).most_common(1)[0][0]
            smoothed_history.append((history[i]['timestamp'], top_em))

        transition_points = []
        last_em = None
        for ts, em in smoothed_history:
            if last_em is not None and em != last_em:
                transition_points.append({'time': round(ts), 'emotion': em})
            last_em = em

        # Combine & deduplicate points
        combined = {round(pt['time']): pt['emotion']
                    for pt in raw_points + transition_points}
        sorted_times = sorted(combined.keys())

        # Build initial timeline
        timeline = [{'time': t, 'emotion': combined[t]} for t in sorted_times]

        # Filter out neutral completely from timeline
        suppressed_timeline = [
            pt for pt in timeline if pt['emotion'] != 'neutral']

        timeline = suppressed_timeline

        # Guarantee every active emotion with >= 2.0% distribution is represented in timeline
        present_emotions = {pt['emotion'] for pt in timeline}
        for em, pct in distribution.items():
            if em != 'neutral' and pct >= 2.0 and em not in present_emotions:
                em_frames = [h for h in history if h['emotion'] == em]
                if em_frames:
                    best_frame = max(
                        em_frames, key=lambda h: h.get('confidence', 0))
                    t_val = round(best_frame['timestamp'])
                    timeline.append({'time': t_val, 'emotion': em})

        # Sort timeline chronologically
        timeline.sort(key=lambda x: x['time'])

        # Cap max points to 12 if too dense
        if len(timeline) > 12:
            step_idx = (len(timeline) - 1) / 11.0
            sampled = [timeline[int(round(i * step_idx))] for i in range(12)]
            unique_em_pts = []
            seen_em = set()
            for pt in timeline:
                if pt['emotion'] not in seen_em:
                    unique_em_pts.append(pt)
                    seen_em.add(pt['emotion'])

            combined_pts = {pt['time']: pt['emotion']
                            for pt in sampled + unique_em_pts}
            timeline = [{'time': t, 'emotion': combined_pts[t]}
                        for t in sorted(combined_pts.keys())]

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

    socketio.run(app, host='0.0.0.0', port=5000,
                 debug=False, allow_unsafe_werkzeug=True)
