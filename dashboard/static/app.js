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

function skipReading() {
    if (readTimer) {
        clearInterval(readTimer);
        readTimer = null;
    }
    const btnNext = document.getElementById('btnNextToQuiz');
    if (btnNext) btnNext.disabled = false;

    goToModeChoice();
}

function formatText(text) {
    // Convert **bold** to <strong>
    return text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
}

// ─── Evaluation Mode Choice & Coding Exercises ───────────────
let codingAnswers = {};
let currentEvaluationMode = 'quiz';

function goToModeChoice() {
    if (readTimer) {
        clearInterval(readTimer);
        readTimer = null;
    }
    showStep('choice');
}

function selectEvaluationMode(mode) {
    currentEvaluationMode = mode;
    if (mode === 'quiz') {
        showStep('quiz');
        userAnswers = {};
        renderQuizQuestions();
        startQuizTimer();
    } else if (mode === 'coding') {
        showStep('coding');
        codingAnswers = {};
        renderCodingExercises();
        startCodingTimer();
    }
}

function renderCodingExercises() {
    if (!selectedModuleData || !selectedModuleData.coding_exercises) return;

    const container = document.getElementById('codingQuestions');
    if (!container) return;

    container.innerHTML = '';
    questionStartTimes = {};
    questionTrackingData = [];

    selectedModuleData.coding_exercises.forEach((ex, idx) => {
        const card = document.createElement('div');
        card.className = 'coding-card-box';
        card.id = `coding-${ex.id}`;
        questionStartTimes[ex.id] = Date.now();

        // Convert template code: replace ___ with input element
        let formattedCode = escapeHtml(ex.code_template);
        const blankReplacement = `<input type="text" class="code-blank-input" id="blank-${ex.id}" placeholder="..." oninput="onCodingInput('${ex.id}', this)">`;
        
        // Replace ___ with input field
        const htmlCode = formattedCode.replace(/___/g, blankReplacement);

        card.innerHTML = `
            <div class="coding-card-title">
                <strong>Soal ${idx + 1}:</strong> ${escapeHtml(ex.instruction)}
            </div>
            <div class="w3-code-container">
                <div class="code-line">${htmlCode}</div>
                <div class="w3-code-actions">
                    <button type="button" class="btn-show-answer" onclick="toggleShowAnswer('${ex.id}')">💡 Show Answer / Hint</button>
                </div>
            </div>
            <div class="coding-hint-box" id="hint-${ex.id}">
                🔑 <strong>Petunjuk / Jawaban:</strong> ${escapeHtml(ex.hint || 'Jawaban: ' + ex.expected_answer)}
            </div>
        `;

        container.appendChild(card);
    });

    updateCodingProgress();
}

function onCodingInput(exerciseId, inputEl) {
    const val = inputEl.value.trim();
    codingAnswers[exerciseId] = val;

    const startTime = questionStartTimes[exerciseId] || Date.now();
    const timeSpent = Math.max(1, Math.round((Date.now() - startTime) / 1000));

    const existingIdx = questionTrackingData.findIndex(t => t.question_id === exerciseId);
    const trackingItem = {
        question_id: exerciseId,
        question_category: 'Coding Exercise',
        answer: val,
        time_spent: timeSpent,
        timestamp: new Date().toLocaleTimeString()
    };

    if (existingIdx >= 0) {
        questionTrackingData[existingIdx] = trackingItem;
    } else {
        questionTrackingData.push(trackingItem);
    }

    updateCodingProgress();
}

function toggleShowAnswer(exerciseId) {
    const hintBox = document.getElementById(`hint-${exerciseId}`);
    if (hintBox) {
        hintBox.style.display = hintBox.style.display === 'block' ? 'none' : 'block';
    }
}

function updateCodingProgress() {
    const total = (selectedModuleData && selectedModuleData.coding_exercises) ? selectedModuleData.coding_exercises.length : 0;
    const answered = Object.values(codingAnswers).filter(val => val.length > 0).length;
    const progressEl = document.getElementById('codingProgress');
    if (progressEl) {
        progressEl.textContent = `${answered} / ${total} dijawab`;
    }
}

function startCodingTimer() {
    quizTimeLeft = 300;
    updateCodingTimerDisplay();

    if (quizTimer) clearInterval(quizTimer);
    quizTimer = setInterval(() => {
        quizTimeLeft--;
        updateCodingTimerDisplay();

        if (quizTimeLeft <= 0) {
            clearInterval(quizTimer);
            quizTimer = null;
            submitCodingExercises();
        }
    }, 1000);
}

function updateCodingTimerDisplay() {
    const mins = Math.floor(quizTimeLeft / 60);
    const secs = quizTimeLeft % 60;
    const display = `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    const timerVal = document.getElementById('codingTimerValue');
    if (timerVal) timerVal.textContent = display;

    const pct = (quizTimeLeft / 300) * 100;
    const fillEl = document.getElementById('codingTimerFill');
    if (fillEl) fillEl.style.width = pct + '%';
}

async function submitCodingExercises() {
    if (quizTimer) {
        clearInterval(quizTimer);
        quizTimer = null;
    }

    try {
        const response = await fetch('/api/check-answer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                module_id: selectedModule,
                is_coding_mode: true,
                answers: codingAnswers,
                emotion_distribution: currentDistribution
            })
        });

        const result = await response.json();
        socket.emit('get_emotion_report');
        showFinish(result);
    } catch (err) {
        console.error('[Coding] Error submitting exercises:', err);
        showFinish({
            is_coding_mode: true,
            score: 0,
            correct: 0,
            total: 5,
            concept_correct: null,
            concept_total: null,
            concept_score_pct: null,
            problem_solving_correct: 0,
            problem_solving_total: 5,
            problem_solving_score_pct: 0,
            interpretation: "Data pengerjaan koding telah dicatat.",
            results: []
        });
    }
}

// ─── Quiz ───────────────────────────────────────────────────
function goToQuiz() {
    goToModeChoice();
}

let questionStartTimes = {};
let questionTrackingData = [];
let timelineChart = null;

function renderQuizQuestions() {
    if (!selectedModuleData) return;

    const container = document.getElementById('quizQuestions');
    container.innerHTML = '';
    questionStartTimes = {};
    questionTrackingData = [];

    selectedModuleData.questions.forEach((q, idx) => {
        const card = document.createElement('div');
        card.className = 'quiz-question-card';
        card.id = `question-${q.id}`;
        questionStartTimes[q.id] = Date.now();

        const category = q.category || 'Concept';
        const categoryClass = category.toLowerCase().replace(/\s+/g, '-');
        const categoryIcon = category === 'Problem Solving' ? '🧩' : '💡';

        // Options rendering (support option_a..d or options array)
        let optionsList = [];
        if (q.option_a) {
            optionsList = [
                { key: 'A', text: q.option_a, val: 0 },
                { key: 'B', text: q.option_b, val: 1 },
                { key: 'C', text: q.option_c, val: 2 },
                { key: 'D', text: q.option_d, val: 3 }
            ];
        } else if (q.options) {
            const letters = ['A', 'B', 'C', 'D'];
            optionsList = q.options.map((opt, i) => ({ key: letters[i], text: opt, val: i }));
        }

        let optionsHTML = '';
        optionsList.forEach(optObj => {
            optionsHTML += `
                <div class="quiz-option" data-question="${q.id}" data-option="${optObj.key}" onclick="selectAnswer('${q.id}', '${optObj.key}', ${optObj.val}, this)">
                    <div class="option-radio"></div>
                    <span><strong>${optObj.key}.</strong> ${formatText(optObj.text)}</span>
                </div>
            `;
        });

        // Format question code snippet if present
        let formattedQuestion = formatQuestionText(q.question);

        card.innerHTML = `
            <div class="q-header-row">
                <span class="q-number">Soal ${idx + 1}</span>
                <span class="q-category-badge ${categoryClass}">${categoryIcon} ${category}</span>
            </div>
            <div class="q-text">${formattedQuestion}</div>
            <div class="quiz-options">${optionsHTML}</div>
        `;

        container.appendChild(card);
    });

    updateQuizProgress();
}

function formatQuestionText(text) {
    if (!text) return '';
    
    // Check if text contains code blocks
    if (text.includes('Kode:') || text.includes('Perhatikan kode') || text.includes('print(') || text.includes('if ') || text.includes('for ')) {
        const parts = text.split('\n\n');
        let html = '';
        parts.forEach(part => {
            if (part.includes(' = ') || part.includes('print(') || part.includes('if ') || part.includes('for ') || part.includes('while ')) {
                html += `<pre><code>${escapeHtml(part)}</code></pre>`;
            } else {
                html += `<p>${formatText(escapeHtml(part))}</p>`;
            }
        });
        return html;
    }
    return formatText(escapeHtml(text));
}

function escapeHtml(str) {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function selectAnswer(questionId, optionKey, optionIdx, element) {
    const parent = element.parentElement;
    parent.querySelectorAll('.quiz-option').forEach(el => el.classList.remove('selected'));

    element.classList.add('selected');
    userAnswers[questionId] = optionKey;

    // Track question answering metrics
    const startTime = questionStartTimes[questionId] || Date.now();
    const timeSpent = Math.max(1, Math.round((Date.now() - startTime) / 1000));
    
    // Find question category
    const qObj = selectedModuleData ? selectedModuleData.questions.find(q => q.id === questionId) : null;
    const category = qObj ? (qObj.category || 'Concept') : 'Concept';

    const existingIdx = questionTrackingData.findIndex(t => t.question_id === questionId);
    const trackingItem = {
        question_id: questionId,
        question_category: category,
        answer: optionKey,
        time_spent: timeSpent,
        timestamp: new Date().toLocaleTimeString()
    };

    if (existingIdx >= 0) {
        questionTrackingData[existingIdx] = trackingItem;
    } else {
        questionTrackingData.push(trackingItem);
    }

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

    const pct = (quizTimeLeft / 300) * 100;
    document.getElementById('quizTimerFill').style.width = pct + '%';

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

    try {
        const response = await fetch('/api/check-answer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                module_id: selectedModule,
                answers: userAnswers,
                emotion_distribution: currentDistribution
            })
        });

        const result = await response.json();

        // Request final emotion report from socket
        socket.emit('get_emotion_report');

        // Show finish with result
        showFinish(result);

    } catch (error) {
        console.error('Error submitting quiz:', error);
        showFinish({
            score: 0,
            correct: 0,
            total: 10,
            concept_correct: 0,
            concept_total: 5,
            concept_score_pct: 0,
            problem_solving_correct: 0,
            problem_solving_total: 5,
            problem_solving_score_pct: 0,
            interpretation: "Data pengerjaan kuis telah dicatat.",
            results: []
        });
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
    document.getElementById('finalScore').textContent = quizResult.score || 0;
    
    // Concept & Problem Solving Score Cards handling
    const cardConcept = document.getElementById('cardConceptScore');
    const labelPS = document.getElementById('labelProblemSolvingScore');

    const isCodingMode = (currentEvaluationMode === 'coding') || quizResult.is_coding_mode || (quizResult.concept_correct === null) || (quizResult.concept_correct === undefined);

    if (isCodingMode) {
        if (cardConcept) cardConcept.style.setProperty('display', 'none', 'important');
        if (labelPS) labelPS.textContent = 'Coding Score';
    } else {
        if (cardConcept) cardConcept.style.setProperty('display', 'block');
        if (labelPS) labelPS.textContent = 'Problem Solving Score';

        document.getElementById('conceptScore').textContent = 
            `${quizResult.concept_correct || 0}/${quizResult.concept_total || 5}`;
        document.getElementById('conceptPct').textContent = 
            `${quizResult.concept_score_pct || 0}%`;
    }

    document.getElementById('problemSolvingScore').textContent = 
        `${quizResult.problem_solving_correct || 0}/${quizResult.problem_solving_total || 5}`;
    document.getElementById('problemSolvingPct').textContent = 
        `${quizResult.problem_solving_score_pct || 0}%`;

    document.getElementById('finalDuration').textContent =
        `${durationMins}:${String(durationSecs).padStart(2, '0')}`;

    // Academic Interpretation Box
    if (document.getElementById('finalInterpretation')) {
        document.getElementById('finalInterpretation').textContent = 
            quizResult.interpretation || "Mahasiswa menunjukkan keterlibatan belajar yang baik.";
    }

    // Find dominant emotion
    const dominantEmotion = Object.entries(currentDistribution)
        .sort(([,a], [,b]) => b - a)[0];

    if (dominantEmotion) {
        document.getElementById('finalEmotionEmoji').textContent = EMOTION_EMOJI[dominantEmotion[0]] || '😐';
        document.getElementById('finalEmotionName').textContent = dominantEmotion[0];
    }

    // Draw finish emotion bars & timeline chart
    setTimeout(() => {
        drawFinishEmotionBars();
        renderTimelineChart();
    }, 100);

    // Auto-save session to backend history
    const sessionPayload = {
        module_id: selectedModuleData ? selectedModuleData.id : 1,
        module_title: selectedModuleData ? selectedModuleData.title : 'Modul Pembelajaran',
        score: quizResult.score || 0,
        correct_answers: quizResult.correct || 0,
        total_questions: quizResult.total || 10,
        concept_correct: quizResult.concept_correct || 0,
        concept_total: quizResult.concept_total || 5,
        problem_solving_correct: quizResult.problem_solving_correct || 0,
        problem_solving_total: quizResult.problem_solving_total || 5,
        duration_seconds: duration,
        dominant_emotion: dominantEmotion ? dominantEmotion[0] : 'neutral',
        emotion_distribution: currentDistribution,
        interpretation: quizResult.interpretation || "Sesi belajar selesai.",
        question_tracking: questionTrackingData
    };

    fetch('/api/save-session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(sessionPayload)
    })
    .then(res => res.json())
    .then(data => console.log('[History] Session auto-saved:', data))
    .catch(err => console.error('[History] Failed to auto-save session:', err));
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

let lastReceivedTimeline = null;

function renderTimelineChart(timelineData) {
    const canvas = document.getElementById('timelineChartCanvas');
    if (!canvas) return;

    if (timelineChart) {
        timelineChart.destroy();
        timelineChart = null;
    }

    if (!timelineData || timelineData.length === 0) {
        if (lastReceivedTimeline && lastReceivedTimeline.length > 0) {
            timelineData = lastReceivedTimeline;
        } else {
            const totalDuration = sessionStartTime ? Math.max(10, Math.floor((Date.now() - sessionStartTime) / 1000)) : 180;
            const activeEmotions = Object.entries(currentDistribution)
                .filter(([, val]) => val > 0)
                .sort(([, a], [, b]) => b - a)
                .map(([em]) => em);

            if (activeEmotions.length === 0) activeEmotions.push('engaged');

            const numSteps = Math.max(4, Math.min(8, activeEmotions.length * 2));
            timelineData = [];
            for (let i = 0; i < numSteps; i++) {
                const t = Math.round((i / (numSteps - 1)) * totalDuration);
                const em = activeEmotions[i % activeEmotions.length];
                timelineData.push({ time: t, emotion: em });
            }
        }
    }

    const emotionMap = { 'engaged': 4, 'confused': 3, 'frustrated': 2, 'bored': 1, 'neutral': 0 };
    const emotionLabels = ['Neutral', 'Bored', 'Frustrated', 'Confused', 'Engaged'];
    const emotionColors = {
        'engaged': '#10b981',
        'confused': '#f59e0b',
        'frustrated': '#ef4444',
        'bored': '#8b5cf6',
        'neutral': '#6b7280'
    };

    const labels = timelineData.map(item => {
        const mins = Math.floor(item.time / 60);
        const secs = Math.floor(item.time % 60);
        return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    });

    const values = timelineData.map(item => emotionMap[item.emotion] !== undefined ? emotionMap[item.emotion] : 0);
    const pointColors = timelineData.map(item => emotionColors[item.emotion] || '#3b82f6');

    try {
        const ctx = canvas.getContext('2d');
        timelineChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Emosi Belajar',
                    data: values,
                    borderColor: '#8b5cf6',
                    borderWidth: 3,
                    pointBackgroundColor: pointColors,
                    pointBorderColor: '#ffffff',
                    pointRadius: 6,
                    pointHoverRadius: 8,
                    stepped: true,
                    fill: false,
                    tension: 0.2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        min: -0.3,
                        max: 4.3,
                        ticks: {
                            stepSize: 1,
                            callback: function(val) {
                                if (Number.isInteger(val) && val >= 0 && val <= 4) {
                                    return emotionLabels[val] || '';
                                }
                                return '';
                            },
                            color: '#94a3b8',
                            font: { family: 'Inter', weight: '600' }
                        },
                        grid: { color: 'rgba(255,255,255,0.05)' }
                    },
                    x: {
                        ticks: { color: '#94a3b8', font: { family: 'Inter' } },
                        grid: { color: 'rgba(255,255,255,0.05)' }
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const val = context.parsed.y;
                                return ` Emosi: ${emotionLabels[val] || 'Neutral'}`;
                            }
                        }
                    }
                }
            }
        });
    } catch (e) {
        console.error('[Timeline] Error rendering chart:', e);
    }

    const eventsList = document.getElementById('timelineEventsList');
    if (eventsList) {
        let eventsHTML = '';
        timelineData.forEach(item => {
            const mins = Math.floor(item.time / 60);
            const secs = Math.floor(item.time % 60);
            const timeStr = `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
            const em = item.emotion || 'neutral';
            const emoji = EMOTION_EMOJI[em] || '😐';
            eventsHTML += `
                <div class="timeline-event-item ${em}">
                    <span class="timeline-event-time">${timeStr}</span>
                    <span class="timeline-event-badge">${emoji} ${em}</span>
                </div>
            `;
        });
        eventsList.innerHTML = eventsHTML;
    }
}

// ─── Emotion Report Handler ─────────────────────────────────
socket.on('emotion_report', (data) => {
    if (data.distribution) {
        currentDistribution = data.distribution;
    }
    if (data.timeline && data.timeline.length > 0) {
        lastReceivedTimeline = data.timeline;
    }

    // Update finish page if we're on it
    if (currentStep === 'finish') {
        if (data.dominant) {
            document.getElementById('finalEmotionEmoji').textContent = EMOTION_EMOJI[data.dominant] || '😐';
            document.getElementById('finalEmotionName').textContent = data.dominant;
        }
        drawFinishEmotionBars();
        renderTimelineChart(data.timeline || lastReceivedTimeline);
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

// ─── History Modal Functions ──────────────────────────────────
function openHistoryModal() {
    const overlay = document.getElementById('historyModalOverlay');
    overlay.classList.add('active');

    // Fetch history from backend
    fetch('/api/history')
        .then(res => res.json())
        .then(data => {
            if (data.history && data.stats) {
                renderHistory(data.history, data.stats);
            }
        })
        .catch(err => {
            console.error('[History] Error fetching history:', err);
        });
}

function closeHistoryModal() {
    document.getElementById('historyModalOverlay').classList.remove('active');
}

function closeHistoryModalOnOverlay(event) {
    if (event.target.id === 'historyModalOverlay') {
        closeHistoryModal();
    }
}

function renderHistory(items, stats) {
    // Stats Summary
    document.getElementById('histStatTotal').textContent = stats.total_sessions || 0;
    document.getElementById('histStatAvgScore').textContent = (stats.avg_score || 0) + '%';

    const totalMins = Math.round((stats.total_duration_sec || 0) / 60);
    document.getElementById('histStatTotalTime').textContent = totalMins + ' mnt';

    const domEmoji = EMOTION_EMOJI[stats.overall_dominant] || '😐';
    document.getElementById('histStatDominant').textContent = `${domEmoji} ${stats.overall_dominant || '—'}`;

    // Items List
    const container = document.getElementById('historyItemsContainer');
    if (!items || items.length === 0) {
        container.innerHTML = `
            <div class="history-empty">
                <div class="empty-icon">📭</div>
                <h3>Belum Ada Riwayat Sesi Belajar</h3>
                <p>Selesaikan modul pembelajaran untuk mencatat sesi belajar pertamamu!</p>
            </div>
        `;
        return;
    }

    let html = '';
    items.forEach(item => {
        const durationMins = Math.floor((item.duration_seconds || 0) / 60);
        const durationSecs = (item.duration_seconds || 0) % 60;
        const durationStr = `${durationMins}m ${durationSecs}s`;
        const domEm = item.dominant_emotion || 'neutral';
        const emoji = EMOTION_EMOJI[domEm] || '😐';

        let distBars = '';
        const dist = item.emotion_distribution || {};
        ['engaged', 'confused', 'bored', 'frustrated', 'neutral'].forEach(em => {
            const val = dist[em] || 0;
            distBars += `
                <div class="hist-dist-item">
                    <span class="hist-dist-label">${EMOTION_EMOJI[em]} ${em}</span>
                    <div class="hist-dist-track"><div class="hist-dist-fill ${em}" style="width: ${val}%"></div></div>
                    <span class="hist-dist-val">${val}%</span>
                </div>
            `;
        });

        html += `
            <div class="history-item-card">
                <div class="history-item-header">
                    <div class="history-item-title-group">
                        <span class="history-item-badge">Modul ${item.module_id || 1}</span>
                        <h3>${item.module_title}</h3>
                    </div>
                    <span class="history-item-time">🕒 ${item.completed_at}</span>
                </div>

                <div class="history-item-body">
                    <div class="history-meta-box">
                        <div class="meta-box-item">
                            <span class="meta-box-label">Skor Kuis</span>
                            <span class="meta-box-val score">${item.score}/100 <small>(${item.correct_answers}/${item.total_questions} benar)</small></span>
                        </div>
                        <div class="meta-box-item">
                            <span class="meta-box-label">Durasi Sesi</span>
                            <span class="meta-box-val">${durationStr}</span>
                        </div>
                        <div class="meta-box-item">
                            <span class="meta-box-label">Emosi Dominan</span>
                            <span class="meta-box-val emotion ${domEm}">${emoji} ${domEm}</span>
                        </div>
                    </div>

                    <div class="history-dist-wrapper">
                        <div class="history-dist-title">Distribusi Emosi Sesi</div>
                        <div class="history-dist-grid">
                            ${distBars}
                        </div>
                    </div>
                </div>
            </div>
        `;
    });

    container.innerHTML = html;
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
