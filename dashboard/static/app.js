/**
 * app.js — Engagement Learning Dashboard Frontend Logic
 * =====================================================
 * Handles: SocketIO communication, step flow, camera stream,
 * emotion visualization, quiz logic, and timer management.
 */

// ─── SocketIO Connection ────────────────────────────────────
const socket = io();

// ─── State ──────────────────────────────────────────────────
let currentStep = 'modules';    // modules | camera | read | quiz | finish
let selectedModule = null;
let selectedModuleData = null;
let userAnswers = {};
let quizTimer = null;
let quizTimeLeft = 300;          // 5 minutes in seconds
let readTimer = null;
let readTimeSeconds = 0;
let sessionStartTime = null;
let cameraActive = false;
let currentDistribution = {};

// Emotion emoji mapping
const EMOTION_EMOJI = {
    engaged: '😊',
    confused: '😕',
    bored: '😴',
    frustrated: '😤',
    neutral: '😐'
};

const EMOTION_COLORS = {
    engaged: '#22c55e',
    confused: '#f59e0b',
    bored: '#f97316',
    frustrated: '#ef4444',
    neutral: '#94a3b8'
};

// ─── Step Navigation ────────────────────────────────────────
function showStep(stepName) {
    document.querySelectorAll('.step-view').forEach(el => el.classList.remove('active'));
    const stepEl = document.getElementById('step' + stepName.charAt(0).toUpperCase() + stepName.slice(1));
    if (stepEl) {
        stepEl.classList.add('active');
    }
    currentStep = stepName;
}

// ─── Module Selection ───────────────────────────────────────
function selectModule(moduleId) {
    selectedModule = moduleId;
    selectedModuleData = MODULES_DATA.find(m => m.id === moduleId);

    if (!selectedModuleData) {
        console.error('Module not found:', moduleId);
        return;
    }

    // Update sidebar
    document.getElementById('sidebarTotalQ').textContent = selectedModuleData.questions.length;

    // Go to camera activation step
    showStep('camera');
}

// ─── Camera Activation ─────────────────────────────────────
function activateCamera() {
    const statusEl = document.getElementById('cameraStatus');
    statusEl.innerHTML = '<span class="status-dot"></span><span>Menghubungkan ke kamera...</span>';

    socket.emit('start_camera');
}

socket.on('camera_started', (data) => {
    cameraActive = true;
    const statusEl = document.getElementById('cameraStatus');
    statusEl.classList.add('active');
    statusEl.innerHTML = '<span class="status-dot"></span><span>Kamera aktif! Memulai deteksi emosi...</span>';

    // Show live badge
    document.getElementById('liveBadge').classList.add('active');
    // Hide camera placeholder
    document.getElementById('cameraPlaceholder').classList.add('hidden');

    // Session start
    sessionStartTime = Date.now();

    // Auto-proceed to read module after 1.5s
    setTimeout(() => {
        startReading();
    }, 1500);
});

socket.on('camera_error', (data) => {
    const statusEl = document.getElementById('cameraStatus');
    statusEl.innerHTML = `<span class="status-dot"></span><span style="color: #ef4444;">❌ ${data.message}</span>`;
});

// ─── Camera Frame Handling ──────────────────────────────────
const cameraCanvas = document.getElementById('cameraCanvas');
const cameraCtx = cameraCanvas ? cameraCanvas.getContext('2d') : null;

socket.on('camera_frame', (data) => {
    // Draw frame on canvas
    if (cameraCtx && data.frame) {
        const img = new Image();
        img.onload = () => {
            cameraCanvas.width = img.width;
            cameraCanvas.height = img.height;
            cameraCtx.drawImage(img, 0, 0);
        };
        img.src = 'data:image/jpeg;base64,' + data.frame;
    }

    // Update current emotion
    updateCurrentEmotion(data.emotion, data.confidence);

    // Update emotion bars
    if (data.distribution) {
        currentDistribution = data.distribution;
        updateEmotionBars(data.distribution);
    }

    // Update sidebar duration
    if (data.duration !== undefined) {
        updateDuration(data.duration);
    }
});

// ─── Emotion Display ────────────────────────────────────────
function updateCurrentEmotion(emotion, confidence) {
    const emojiEl = document.getElementById('currentEmotionEmoji');
    const labelEl = document.getElementById('currentEmotionLabel');
    const confEl = document.getElementById('currentEmotionConf');

    emojiEl.textContent = EMOTION_EMOJI[emotion] || '—';
    labelEl.textContent = emotion || 'Menunggu...';
    labelEl.style.color = EMOTION_COLORS[emotion] || '#94a3b8';
    confEl.textContent = confidence ? `${Math.round(confidence * 100)}%` : '—';
}

function updateEmotionBars(distribution) {
    const emotions = ['engaged', 'confused', 'bored', 'frustrated', 'neutral'];
    emotions.forEach(em => {
        const row = document.querySelector(`.emotion-bar-row[data-emotion="${em}"]`);
        if (row) {
            const fill = row.querySelector('.emotion-bar-fill');
            const value = row.querySelector('.emotion-bar-value');
            const pct = distribution[em] || 0;
            fill.style.width = pct + '%';
            value.textContent = pct.toFixed(1) + '%';
        }
    });

    // Update detail emotion bars if panel is visible
    updateDetailBars(distribution);
}

function updateDuration(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const formatted = `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    document.getElementById('sidebarDuration').textContent = formatted;
}

// ─── Detail Emotion Toggle ──────────────────────────────────
function toggleDetailEmotion() {
    const panel = document.getElementById('detailEmotionPanel');
    const btn = document.querySelector('.btn-detail-emotion');

    if (panel.style.display === 'none') {
        panel.style.display = 'block';
        btn.classList.add('expanded');
        updateDetailBars(currentDistribution);
    } else {
        panel.style.display = 'none';
        btn.classList.remove('expanded');
    }
}

function updateDetailBars(distribution) {
    const emotions = ['engaged', 'confused', 'bored', 'frustrated', 'neutral'];
    emotions.forEach(em => {
        const bar = document.getElementById(`detailBar-${em}`);
        const val = document.getElementById(`detailVal-${em}`);
        if (bar && val) {
            const pct = distribution[em] || 0;
            bar.style.width = pct + '%';
            val.textContent = pct.toFixed(1) + '%';
        }
    });
}

// ─── Reading Module ─────────────────────────────────────────
function startReading() {
    if (!selectedModuleData) return;

    showStep('read');

    // Set title
    document.getElementById('readModuleTitle').textContent = selectedModuleData.title;

    // Render content sections
    const contentEl = document.getElementById('readingContent');
    contentEl.innerHTML = '';

    selectedModuleData.content.forEach((section, i) => {
        const sectionDiv = document.createElement('div');
        sectionDiv.className = 'reading-section';
        sectionDiv.innerHTML = `
            <h2>${section.section}</h2>
            <div class="section-text">${formatText(section.text)}</div>
        `;
        contentEl.appendChild(sectionDiv);
    });

    // Start read timer
    readTimeSeconds = 0;
    const btnNext = document.getElementById('btnNextToQuiz');
    const noteEl = document.getElementById('minTimeNote');
    btnNext.disabled = true;
    noteEl.style.display = 'inline';

    readTimer = setInterval(() => {
        readTimeSeconds++;
        const mins = Math.floor(readTimeSeconds / 60);
        const secs = readTimeSeconds % 60;
        document.getElementById('readTimer').textContent =
            `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;

        // Enable next button after 1 minute
        if (readTimeSeconds >= 60) {
            btnNext.disabled = false;
            noteEl.style.display = 'none';
        }
    }, 1000);
}

function formatText(text) {
    // Convert **bold** to <strong>
    return text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
}

// ─── Quiz ───────────────────────────────────────────────────
function goToQuiz() {
    if (readTimer) {
        clearInterval(readTimer);
        readTimer = null;
    }

    showStep('quiz');
    userAnswers = {};
    renderQuizQuestions();
    startQuizTimer();
}

function renderQuizQuestions() {
    if (!selectedModuleData) return;

    const container = document.getElementById('quizQuestions');
    container.innerHTML = '';

    selectedModuleData.questions.forEach((q, idx) => {
        const card = document.createElement('div');
        card.className = 'quiz-question-card';
        card.id = `question-${q.id}`;

        let optionsHTML = '';
        q.options.forEach((opt, optIdx) => {
            optionsHTML += `
                <div class="quiz-option" data-question="${q.id}" data-option="${optIdx}" onclick="selectAnswer('${q.id}', ${optIdx}, this)">
                    <div class="option-radio"></div>
                    <span>${opt}</span>
                </div>
            `;
        });

        card.innerHTML = `
            <span class="q-number">Soal ${idx + 1}</span>
            <div class="q-text">${q.question}</div>
            <div class="quiz-options">${optionsHTML}</div>
        `;

        container.appendChild(card);
    });

    updateQuizProgress();
}

function selectAnswer(questionId, optionIdx, element) {
    // Deselect siblings
    const parent = element.parentElement;
    parent.querySelectorAll('.quiz-option').forEach(el => el.classList.remove('selected'));

    // Select this
    element.classList.add('selected');
    userAnswers[questionId] = optionIdx;

    // Mark card as answered
    document.getElementById(`question-${questionId}`).classList.add('answered');

    updateQuizProgress();
}

function updateQuizProgress() {
    const total = selectedModuleData ? selectedModuleData.questions.length : 0;
    const answered = Object.keys(userAnswers).length;
    document.getElementById('quizProgress').textContent = `${answered} / ${total} dijawab`;
}

function startQuizTimer() {
    quizTimeLeft = 300; // 5 minutes
    updateQuizTimerDisplay();

    quizTimer = setInterval(() => {
        quizTimeLeft--;
        updateQuizTimerDisplay();

        if (quizTimeLeft <= 0) {
            clearInterval(quizTimer);
            quizTimer = null;
            submitQuiz();
        }
    }, 1000);
}

function updateQuizTimerDisplay() {
    const mins = Math.floor(quizTimeLeft / 60);
    const secs = quizTimeLeft % 60;
    const display = `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    document.getElementById('quizTimerValue').textContent = display;

    // Timer fill
    const pct = (quizTimeLeft / 300) * 100;
    document.getElementById('quizTimerFill').style.width = pct + '%';

    // Warning/danger states
    const bar = document.getElementById('quizTimerBar');
    bar.classList.remove('warning', 'danger');
    if (quizTimeLeft <= 30) {
        bar.classList.add('danger');
    } else if (quizTimeLeft <= 60) {
        bar.classList.add('warning');
    }
}

async function submitQuiz() {
    if (quizTimer) {
        clearInterval(quizTimer);
        quizTimer = null;
    }

    // Send answers to server
    try {
        const response = await fetch('/api/check-answer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                module_id: selectedModule,
                answers: userAnswers
            })
        });

        const result = await response.json();

        // Request emotion report
        socket.emit('get_emotion_report');

        // Show finish with result
        showFinish(result);

    } catch (error) {
        console.error('Error submitting quiz:', error);
        showFinish({ score: 0, correct: 0, total: 5, results: [] });
    }
}

// ─── Finish / Results ───────────────────────────────────────
function showFinish(quizResult) {
    showStep('finish');

    // Stop camera
    socket.emit('stop_camera');
    cameraActive = false;

    // Calculate session duration
    const duration = sessionStartTime ? Math.floor((Date.now() - sessionStartTime) / 1000) : 0;
    const durationMins = Math.floor(duration / 60);
    const durationSecs = duration % 60;

    // Update result cards
    document.getElementById('finishModuleName').textContent =
        `Modul: ${selectedModuleData ? selectedModuleData.title : '-'}`;
    document.getElementById('finalScore').textContent = quizResult.score;
    document.getElementById('finalCorrect').textContent = quizResult.correct;
    document.getElementById('finalTotal').textContent = quizResult.total;
    document.getElementById('finalDuration').textContent =
        `${durationMins}:${String(durationSecs).padStart(2, '0')}`;

    // Update sidebar
    document.getElementById('sidebarScore').textContent = quizResult.score;

    // Find dominant emotion
    const dominantEmotion = Object.entries(currentDistribution)
        .sort(([,a], [,b]) => b - a)[0];

    if (dominantEmotion) {
        document.getElementById('finalEmotionEmoji').textContent = EMOTION_EMOJI[dominantEmotion[0]] || '😐';
        document.getElementById('finalEmotionName').textContent = dominantEmotion[0];
    }

    // Draw finish emotion bars (use small delay to ensure DOM is ready)
    setTimeout(() => drawFinishEmotionBars(), 100);
}

function drawFinishEmotionBars() {
    console.log('[Finish] Drawing bars, currentDistribution:', JSON.stringify(currentDistribution));
    const emotions = ['engaged', 'confused', 'bored', 'frustrated', 'neutral'];
    emotions.forEach(em => {
        const bar = document.getElementById(`finishBar-${em}`);
        const pct = document.getElementById(`finishPct-${em}`);
        if (bar && pct) {
            const value = currentDistribution[em] || 0;
            console.log(`[Finish] ${em}: ${value}%`);
            bar.style.width = value + '%';
            pct.textContent = value.toFixed(1) + '%';
        } else {
            console.warn(`[Finish] Element not found: finishBar-${em} or finishPct-${em}`);
        }
    });
}

// ─── Emotion Report Handler ─────────────────────────────────
socket.on('emotion_report', (data) => {
    if (data.distribution) {
        currentDistribution = data.distribution;
    }

    // Update finish page if we're on it
    if (currentStep === 'finish') {
        if (data.dominant) {
            document.getElementById('finalEmotionEmoji').textContent = EMOTION_EMOJI[data.dominant] || '😐';
            document.getElementById('finalEmotionName').textContent = data.dominant;
        }
        drawFinishEmotionBars();
    }
});

// ─── Emotion Modal ──────────────────────────────────────────
function closeEmotionModal() {
    document.getElementById('emotionModal').style.display = 'none';
}

// ─── Back to Modules ────────────────────────────────────────
function backToModules() {
    // Reset state
    selectedModule = null;
    selectedModuleData = null;
    userAnswers = {};
    currentDistribution = {};
    sessionStartTime = null;

    // Reset sidebar
    document.getElementById('sidebarDuration').textContent = '00:00';
    document.getElementById('sidebarTotalQ').textContent = '—';
    document.getElementById('sidebarScore').textContent = '—';
    document.getElementById('liveBadge').classList.remove('active');
    document.getElementById('cameraPlaceholder').classList.remove('hidden');

    // Reset emotion bars
    document.querySelectorAll('.emotion-bar-fill').forEach(el => el.style.width = '0%');
    document.querySelectorAll('.emotion-bar-value').forEach(el => el.textContent = '0%');

    // Reset current emotion
    document.getElementById('currentEmotionEmoji').textContent = '—';
    document.getElementById('currentEmotionLabel').textContent = 'Menunggu...';
    document.getElementById('currentEmotionLabel').style.color = '';
    document.getElementById('currentEmotionConf').textContent = '—';

    // Hide detail panel
    document.getElementById('detailEmotionPanel').style.display = 'none';
    document.querySelector('.btn-detail-emotion').classList.remove('expanded');

    // Reset finish bars
    ['engaged', 'confused', 'bored', 'frustrated', 'neutral'].forEach(em => {
        const fb = document.getElementById(`finishBar-${em}`);
        const fp = document.getElementById(`finishPct-${em}`);
        const db = document.getElementById(`detailBar-${em}`);
        const dv = document.getElementById(`detailVal-${em}`);
        if (fb) fb.style.width = '0%';
        if (fp) fp.textContent = '0%';
        if (db) db.style.width = '0%';
        if (dv) dv.textContent = '0%';
    });

    showStep('modules');
}

// ─── SocketIO Connection Status ─────────────────────────────
socket.on('connect', () => {
    console.log('[SocketIO] Connected');
});

socket.on('disconnect', () => {
    console.log('[SocketIO] Disconnected');
});

// ─── Initialize ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    showStep('modules');
    console.log('[Dashboard] Initialized');
    console.log('[Dashboard] Modules loaded:', MODULES_DATA.length);
});
