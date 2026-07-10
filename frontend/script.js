// ================= STATE CONFIGURATION =================
const CONFIG = {
    apiBase: '/api',
    frameIntervalMs: 120, // Frequency of sending frames to backend
    minConfidenceThreshold: 0.70, // Min score to print prediction
    stableWordFramesRequired: 4, // Consecutive frames required to lock a word
    maxHistoryLogs: 50
};

const state = {
    isStreaming: false,
    stream: null,
    animationFrameId: null,
    lastPredictionTime: 0,
    systemStatus: 'OFFLINE',
    
    // Sentence buffer
    sentenceWords: [],
    lastDetectedWord: null,
    wordCounter: 0,
    isProcessingFrame: false,
    
    // Loaded list of database gestures
    activeGestures: []
};

// Hand skeletal connections mapping (indices 0-20)
const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8],       // Index Finger
    [0, 9], [9, 10], [10, 11], [11, 12],  // Middle Finger
    [0, 13], [13, 14], [14, 15], [15, 16], // Ring Finger
    [0, 17], [17, 18], [18, 19], [19, 20], // Pinky Finger
    [5, 9], [9, 13], [13, 17]             // Palm/Knuckles
];

// DOM Elements
const video = document.getElementById('webcam-elem');
const overlayCanvas = document.getElementById('overlay-canvas');
const overlayCtx = overlayCanvas.getContext('2d');
const hiddenCanvas = document.getElementById('hidden-capture-canvas');
const hiddenCtx = hiddenCanvas.getContext('2d');

const startStreamBtn = document.getElementById('start-stream-btn');
const stopStreamBtn = document.getElementById('stop-stream-btn');
const quickStartCamBtn = document.getElementById('quick-start-cam-btn');
const camPlaceholder = document.getElementById('cam-placeholder');
const recIndicator = document.getElementById('rec-indicator');

const predictedGestureLabel = document.getElementById('predicted-gesture-label');
const confidencePercentage = document.getElementById('confidence-percentage');
const confidenceProgressBar = document.getElementById('confidence-progress-bar');
const sentenceBufferDisplay = document.getElementById('sentence-buffer-display');
const speakCurrentGestureBtn = document.getElementById('speak-current-gesture-btn');
const voiceSelect = document.getElementById('voice-select');

const speechToggle = document.getElementById('speech-toggle');
const serverSpeechToggle = document.getElementById('server-speech-toggle');
const speakSentenceBtn = document.getElementById('speak-sentence-btn');
const clearSentenceBtn = document.getElementById('clear-sentence-btn');

const addGestureForm = document.getElementById('add-gesture-form');
const gesturesGalleryContainer = document.getElementById('gestures-gallery-container');
const gesturesCountBadge = document.getElementById('gestures-count-badge');

const historyTableBody = document.getElementById('history-table-body');
const clearHistoryBtn = document.getElementById('clear-history-btn');

const systemStatusText = document.getElementById('system-status-text');
const globalAlertContainer = document.getElementById('global-alert-container');
const globalAlertMessage = document.getElementById('global-alert-message');
const alertSpinner = document.getElementById('alert-spinner');
const alertSuccessIcon = document.getElementById('alert-success-icon');

// ================= SPA NAVIGATION ROUTER =================
function initRouter() {
    const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('data-target');
            navigateToSection(targetId);
            
            // Update active state in nav links
            navLinks.forEach(l => l.classList.remove('active'));
            this.classList.add('active');
            
            // Close mobile menu if open
            const navbarCollapse = document.getElementById('navbarNav');
            if (navbarCollapse.classList.contains('show')) {
                bootstrap.Collapse.getInstance(navbarCollapse).hide();
            }
        });
    });

    // Handle hash routing if page reloads
    const currentHash = window.location.hash;
    if (currentHash) {
        const matchingLink = document.querySelector(`.navbar-nav .nav-link[href="${currentHash}"]`);
        if (matchingLink) {
            matchingLink.click();
        }
    }
}

function navigateToSection(sectionId) {
    // Hide all sections
    const sections = document.querySelectorAll('.spa-section');
    sections.forEach(sec => {
        sec.classList.remove('active-section');
    });

    // Show target section
    const targetSection = document.getElementById(sectionId);
    if (targetSection) {
        targetSection.classList.add('active-section');
        
        // Trigger specific section loaders
        if (sectionId === 'gestures-section') {
            loadGesturesList();
        } else if (sectionId === 'history-section') {
            loadHistoryLogs();
        } else if (sectionId === 'recognition-section') {
            resizeOverlayCanvas();
        }
    }
}

// ================= CAMERA ENGINE =================
async function startWebcam() {
    try {
        state.stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            },
            audio: false
        });
        
        video.srcObject = state.stream;
        camPlaceholder.classList.add('d-none');
        recIndicator.classList.remove('d-none');
        
        startStreamBtn.disabled = true;
        stopStreamBtn.disabled = false;
        state.isStreaming = true;
        
        document.querySelector('.webcam-viewport-wrapper').classList.add('rec-active-scan');
        
        // Resize canvas overlays to match video dimensions
        video.onloadedmetadata = () => {
            resizeOverlayCanvas();
            // Start the frame looping process
            requestAnimationFrame(processFrameLoop);
        };
    } catch (err) {
        console.error("Webcam initiation failed:", err);
        alert("Camera error: Could not access video capture devices. Verify permissions.");
    }
}

function stopWebcam() {
    if (state.stream) {
        state.stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
    }
    
    if (state.animationFrameId) {
        cancelAnimationFrame(state.animationFrameId);
    }
    
    camPlaceholder.classList.remove('d-none');
    recIndicator.classList.add('d-none');
    startStreamBtn.disabled = false;
    stopStreamBtn.disabled = true;
    state.isStreaming = false;
    
    document.querySelector('.webcam-viewport-wrapper').classList.remove('rec-active-scan');
    
    // Clear prediction overlays
    clearPredictionDisplay();
}

function resizeOverlayCanvas() {
    if (video.videoWidth > 0) {
        overlayCanvas.width = video.videoWidth;
        overlayCanvas.height = video.videoHeight;
        hiddenCanvas.width = video.videoWidth;
        hiddenCanvas.height = video.videoHeight;
    } else {
        overlayCanvas.width = 640;
        overlayCanvas.height = 480;
        hiddenCanvas.width = 640;
        hiddenCanvas.height = 480;
    }
}

// ================= AI FRAME RECOGNITION LOOP =================
async function processFrameLoop(timestamp) {
    if (!state.isStreaming) return;
    
    // Limit frame sending rate to conserve CPU cycles
    if (timestamp - state.lastPredictionTime >= CONFIG.frameIntervalMs) {
        state.lastPredictionTime = timestamp;
        
        if (!state.isProcessingFrame) {
            state.isProcessingFrame = true;
            
            // Capture webcam frame onto hidden canvas
            hiddenCtx.drawImage(video, 0, 0, hiddenCanvas.width, hiddenCanvas.height);
            // Get base64 string
            const base64Frame = hiddenCanvas.toDataURL('image/jpeg', 0.6);
            
            await sendFrameToBackend(base64Frame);
            state.isProcessingFrame = false;
        }
    }
    
    state.animationFrameId = requestAnimationFrame(processFrameLoop);
}

async function sendFrameToBackend(base64Frame) {
    try {
        const response = await fetch(`${CONFIG.apiBase}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: base64Frame })
        });
        
        if (!response.ok) throw new Error("Backend response error");
        
        const data = await response.json();
        
        // Clear overlay canvas prior to drawing new frames
        overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
        
        if (data.detected && data.landmarks) {
            // Draw skeleton joints and lines
            drawHandLandmarks(data.landmarks);
            
            // Update metrics
            updatePredictionDisplay(data.gesture, data.confidence);
            
            // Compile sentence words buffer
            processSentenceBuilder(data.gesture, data.confidence, data.history_id);
        } else {
            clearPredictionDisplay();
            // Reset consecutive word counter if hand exits viewport
            state.lastDetectedWord = null;
            state.wordCounter = 0;
        }
    } catch (err) {
        console.error("Frame inference error:", err);
    }
}

// Draw hand mesh structure
function drawHandLandmarks(landmarks) {
    const width = overlayCanvas.width;
    const height = overlayCanvas.height;
    
    // Draw connection paths (skeleton lines)
    overlayCtx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
    overlayCtx.lineWidth = 3;
    overlayCtx.shadowBlur = 0;
    
    for (const conn of HAND_CONNECTIONS) {
        const p1 = landmarks[conn[0]];
        const p2 = landmarks[conn[1]];
        
        if (p1 && p2) {
            overlayCtx.beginPath();
            overlayCtx.moveTo(p1.x * width, p1.y * height);
            overlayCtx.lineTo(p2.x * width, p2.y * height);
            overlayCtx.stroke();
        }
    }
    
    // Draw joints (points) with futuristic glowing dots
    for (let i = 0; i < landmarks.length; i++) {
        const pt = landmarks[i];
        
        overlayCtx.beginPath();
        overlayCtx.arc(pt.x * width, pt.y * height, 6, 0, 2 * Math.PI);
        
        // Color code nodes based on finger chains for visual appeal
        if (i === 0) overlayCtx.fillStyle = '#ff007f'; // Wrist - Pink
        else if (i <= 4) overlayCtx.fillStyle = '#0072ff'; // Thumb - Blue
        else if (i <= 8) overlayCtx.fillStyle = '#00f0ff'; // Index - Cyan
        else if (i <= 12) overlayCtx.fillStyle = '#39ff14'; // Middle - Green
        else if (i <= 16) overlayCtx.fillStyle = '#ffcc00'; // Ring - Yellow
        else overlayCtx.fillStyle = '#bd00ff'; // Pinky - Purple
        
        // Add neon outer shadow glows to points
        overlayCtx.shadowColor = overlayCtx.fillStyle;
        overlayCtx.shadowBlur = 8;
        overlayCtx.fill();
    }
}

// Update metrics panel
function updatePredictionDisplay(gesture, confidence) {
    if (gesture === 'Uncertain') {
        predictedGestureLabel.innerText = '?';
        predictedGestureLabel.className = 'prediction-large text-muted font-orbitron my-0';
    } else {
        predictedGestureLabel.innerText = gesture;
        predictedGestureLabel.className = 'prediction-large text-glow-purple font-orbitron my-0';
    }
    
    const percentage = Math.round(confidence * 100);
    confidencePercentage.innerText = `${percentage}%`;
    confidenceProgressBar.style.width = `${percentage}%`;
    
    // Enable Speak button if valid gesture
    if (speakCurrentGestureBtn) {
        if (gesture && gesture !== 'Uncertain' && gesture !== 'No hand detected' && gesture !== 'Model Not Loaded') {
            speakCurrentGestureBtn.disabled = false;
        } else {
            speakCurrentGestureBtn.disabled = true;
        }
    }
}

function clearPredictionDisplay() {
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    predictedGestureLabel.innerText = '--';
    predictedGestureLabel.className = 'prediction-large text-muted font-orbitron my-0';
    confidencePercentage.innerText = '0%';
    confidenceProgressBar.style.width = '0%';
    
    if (speakCurrentGestureBtn) {
        speakCurrentGestureBtn.disabled = true;
    }
}

// ================= SENTENCE BUFFER & SPEECH SYNTHESIS =================
function processSentenceBuilder(word, confidence, historyId) {
    // If prediction is uncertain or fails the confidence threshold, skip
    if (word === 'Uncertain' || word === 'Model Not Loaded' || confidence < CONFIG.minConfidenceThreshold) {
        return;
    }
    
    // Check if the current word matches the last detected frame
    if (word === state.lastDetectedWord) {
        state.wordCounter++;
        
        // Trigger append when word stabilizes for consecutive frames
        if (state.wordCounter === CONFIG.stableWordFramesRequired) {
            appendWordToSentence(word, historyId);
        }
    } else {
        // Reset counter if word changes
        state.lastDetectedWord = word;
        state.wordCounter = 1;
    }
}

function appendWordToSentence(word, historyId) {
    const len = state.sentenceWords.length;
    // Prevent repeating the same word consecutively to keep sentence clean
    if (len === 0 || state.sentenceWords[len - 1] !== word) {
        state.sentenceWords.push(word);
        updateSentenceBufferUI();
        
        // Trigger Text-to-Speech synthesis
        triggerSpeech(word, historyId);
    }
}

function updateSentenceBufferUI() {
    if (state.sentenceWords.length === 0) {
        sentenceBufferDisplay.innerText = "Ready to transcribe... Perform gestures to begin.";
        sentenceBufferDisplay.classList.add('text-muted');
        speakSentenceBtn.disabled = true;
    } else {
        sentenceBufferDisplay.innerText = state.sentenceWords.join(' ');
        sentenceBufferDisplay.classList.remove('text-muted');
        speakSentenceBtn.disabled = false;
    }
}

function triggerSpeech(text, historyId = null) {
    const isBrowserToggled = speechToggle.checked;
    const isServerToggled = serverSpeechToggle.checked;
    
    if (isBrowserToggled) {
        speakWebSpeech(text);
    }
    
    if (isServerToggled) {
        speakServerSpeech(text, historyId);
    }
}

function populateVoiceList() {
    if (!('speechSynthesis' in window)) return;
    const voices = window.speechSynthesis.getVoices();
    if (!voiceSelect) return;
    
    const currentSelection = voiceSelect.value;
    voiceSelect.innerHTML = '<option value="">Default System Voice</option>';
    
    voices.forEach(voice => {
        const option = document.createElement('option');
        option.value = voice.name;
        option.textContent = `${voice.name} (${voice.lang})${voice.default ? ' [Default]' : ''}`;
        if (voice.name === currentSelection) {
            option.selected = true;
        }
        voiceSelect.appendChild(option);
    });
}

// 1. Client Side Speech Synthesis (Web Speech API)
function speakWebSpeech(text) {
    if ('speechSynthesis' in window) {
        try {
            // Cancel ongoing speakings to prevent overlaps
            window.speechSynthesis.cancel();
            
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1.0;
            utterance.pitch = 1.1; // Futuristic slightly higher pitch
            
            // Read selected voice from select element
            const selectedVoiceName = voiceSelect ? voiceSelect.value : '';
            const voices = window.speechSynthesis.getVoices();
            
            if (selectedVoiceName) {
                const voice = voices.find(v => v.name === selectedVoiceName);
                if (voice) {
                    utterance.voice = voice;
                }
            } else {
                // Try selecting a clean English voice if available
                const englishVoice = voices.find(voice => voice.lang.startsWith('en-'));
                if (englishVoice) {
                    utterance.voice = englishVoice;
                }
            }
            
            utterance.onerror = (event) => {
                console.error("Web Speech API Synthesis error:", event.error);
            };
            
            window.speechSynthesis.speak(utterance);
        } catch (err) {
            console.error("Web Speech invocation failed:", err);
        }
    } else {
        console.warn("Web Speech API is not supported in this browser.");
    }
}

// 2. Server Side Speech Synthesis (pyttsx3 REST API call)
async function speakServerSpeech(text, historyId = null) {
    try {
        await fetch(`${CONFIG.apiBase}/speak`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: text,
                history_id: historyId
            })
        });
    } catch (err) {
        console.error("Failed to trigger server TTS:", err);
    }
}

// ================= DICTIONARY LIBRARY MANAGEMENT =================
async function loadGesturesList() {
    try {
        const response = await fetch(`${CONFIG.apiBase}/gestures`);
        if (!response.ok) throw new Error("Could not load gestures");
        
        state.activeGestures = await response.json();
        renderGesturesGallery(state.activeGestures);
        
        // Update home stats badge
        gesturesCountBadge.innerText = `${state.activeGestures.length} Gestures`;
    } catch (err) {
        console.error("Dictionary load error:", err);
        gesturesGalleryContainer.innerHTML = `
            <div class="col-12 text-center py-5 text-danger">
                <i class="bi bi-exclamation-triangle-fill fs-3 mb-2"></i>
                <p>Failed to sync dictionary library. Please check server connections.</p>
            </div>
        `;
    }
}

function renderGesturesGallery(gestures) {
    gesturesGalleryContainer.innerHTML = '';
    
    if (gestures.length === 0) {
        gesturesGalleryContainer.innerHTML = `
            <div class="col-12 text-center py-5 text-muted">
                <i class="bi bi-inbox-fill fs-2 mb-2 d-block"></i>
                <p>Gesture library empty. Create your first gesture calibration!</p>
            </div>
        `;
        return;
    }
    
    gestures.forEach(g => {
        const col = document.createElement('div');
        col.className = 'col-sm-6 col-lg-4';
        
        col.innerHTML = `
            <div class="gesture-dictionary-card p-3 h-100 d-flex flex-column justify-content-between">
                <div>
                    <div class="d-flex justify-content-between align-items-start mb-2">
                        <h5 class="font-orbitron text-white mb-0">${g.name}</h5>
                        <button class="btn btn-link p-0 btn-delete-gesture" title="Delete Gesture" onclick="deleteGesture('${g.name}')">
                            <i class="bi bi-trash3-fill"></i>
                        </button>
                    </div>
                    <p class="text-muted font-size-xs mb-0">${g.description || 'No calibration notes.'}</p>
                </div>
                <div class="mt-3 d-flex justify-content-between align-items-center border-top border-secondary pt-2">
                    <small class="text-purple font-size-xs font-orbitron"><i class="bi bi-cpu"></i> Calibrated</small>
                    <small class="text-muted font-size-xs">${new Date(g.created_at).toLocaleDateString()}</small>
                </div>
            </div>
        `;
        gesturesGalleryContainer.appendChild(col);
    });
}

async function addCustomGesture(name, description) {
    const submitBtn = document.getElementById('add-gesture-submit-btn');
    const btnSpinner = document.getElementById('add-btn-spinner');
    const btnText = document.getElementById('add-btn-text');
    
    submitBtn.disabled = true;
    btnSpinner.classList.remove('d-none');
    btnText.innerText = " Optimizing Classifier...";
    
    showRetrainingAlert("Optimizing model coordinates... Please hold while the AI network retrains.");
    
    try {
        const response = await fetch(`${CONFIG.apiBase}/gestures`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, description })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || "Failed to add gesture");
        }
        
        // Reset Form
        addGestureForm.reset();
        
        // Reload List
        await loadGesturesList();
        
        showRetrainingSuccess(`Gesture '${name}' calibrated and TensorFlow model retrained successfully!`);
    } catch (err) {
        console.error("Error adding gesture:", err);
        alert(err.message || "Failed to configure custom gesture.");
        hideRetrainingAlert();
    } finally {
        submitBtn.disabled = false;
        btnSpinner.classList.add('d-none');
        btnText.innerHTML = `<i class="bi bi-check-lg me-1"></i> Calibrate Gesture`;
    }
}

async function deleteGesture(name) {
    if (!confirm(`Are you sure you want to delete gesture '${name}'? This will trigger model retraining.`)) {
        return;
    }
    
    showRetrainingAlert(`Removing '${name}' from AI classifications. Retraining model...`);
    
    try {
        const response = await fetch(`${CONFIG.apiBase}/gestures/${encodeURIComponent(name)}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) throw new Error("Could not delete gesture");
        
        await loadGesturesList();
        showRetrainingSuccess(`Gesture '${name}' removed and network weights optimized.`);
    } catch (err) {
        console.error("Error deleting gesture:", err);
        alert("Failed to delete gesture.");
        hideRetrainingAlert();
    }
}

// Banner alert handlers
function showRetrainingAlert(message) {
    globalAlertContainer.classList.remove('d-none');
    globalAlertMessage.innerText = message;
    alertSpinner.classList.remove('d-none');
    alertSuccessIcon.classList.add('d-none');
}

function showRetrainingSuccess(message) {
    globalAlertContainer.classList.remove('d-none');
    globalAlertMessage.innerText = message;
    alertSpinner.classList.add('d-none');
    alertSuccessIcon.classList.remove('d-none');
    
    // Auto-hide alert after 5 seconds
    setTimeout(() => {
        globalAlertContainer.classList.add('d-none');
    }, 5000);
}

function hideRetrainingAlert() {
    globalAlertContainer.classList.add('d-none');
}

// ================= HISTORY LOGS PERSISTENCE =================
async function loadHistoryLogs() {
    try {
        const response = await fetch(`${CONFIG.apiBase}/history`);
        if (!response.ok) throw new Error("Could not load history");
        
        const history = await response.json();
        renderHistoryLogs(history);
    } catch (err) {
        console.error("History loading failed:", err);
        historyTableBody.innerHTML = `
            <tr>
                <td colspan="6" class="text-center py-5 text-danger">
                    <i class="bi bi-exclamation-triangle-fill fs-3 mb-2"></i>
                    <p>Failed to load recognition history. Sync error.</p>
                </td>
            </tr>
        `;
    }
}

function renderHistoryLogs(logs) {
    historyTableBody.innerHTML = '';
    
    if (logs.length === 0) {
        historyTableBody.innerHTML = `
            <tr>
                <td colspan="6" class="text-center py-5 text-muted">
                    <i class="bi bi-chat-square-text fs-3 mb-2 d-block"></i>
                    <p>Translation log history is clear.</p>
                </td>
            </tr>
        `;
        return;
    }
    
    logs.forEach(log => {
        const tr = document.createElement('tr');
        const formattedDate = new Date(log.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) + ' ' + new Date(log.timestamp).toLocaleDateString();
        const confPercent = Math.round(log.confidence * 100);
        
        // CSS confidence styles
        let confColor = 'text-danger';
        if (log.confidence >= 0.85) confColor = 'text-success';
        else if (log.confidence >= 0.70) confColor = 'text-warning';
        
        tr.innerHTML = `
            <td class="font-orbitron text-muted">${log.id}</td>
            <td><strong class="text-white">${log.gesture_name}</strong></td>
            <td><span class="${confColor} font-orbitron font-weight-bold">${confPercent}%</span></td>
            <td class="text-muted font-size-xs">${formattedDate}</td>
            <td>
                <span class="badge ${log.spoken === 1 ? 'badge-spoken' : 'badge-unspoken'} font-size-xs">
                    ${log.spoken === 1 ? '<i class="bi bi-check2-all me-1"></i> Spoken' : '<i class="bi bi-volume-mute me-1"></i> Unspoken'}
                </span>
            </td>
            <td class="text-center">
                <button class="btn btn-play-audio" title="Speak Word" onclick="playHistoryWord('${log.gesture_name}', ${log.id})">
                    <i class="bi bi-volume-up-fill"></i>
                </button>
            </td>
        `;
        
        historyTableBody.appendChild(tr);
    });
}

async function playHistoryWord(word, logId) {
    // Re-speak word using Web Speech API locally
    speakWebSpeech(word);
    
    // Mark as spoken on backend server
    try {
        await fetch(`${CONFIG.apiBase}/speak`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: word,
                history_id: logId
            })
        });
        
        // Reload list to update spoken badge
        loadHistoryLogs();
    } catch (err) {
        console.error("Playback speak logging failed:", err);
    }
}

async function clearAllHistoryLogs() {
    if (!confirm("Are you sure you want to clear all translation history?")) {
        return;
    }
    
    try {
        const response = await fetch(`${CONFIG.apiBase}/history`, {
            method: 'DELETE'
        });
        
        if (!response.ok) throw new Error("Could not clear history");
        
        loadHistoryLogs();
    } catch (err) {
        console.error("Error wiping history:", err);
        alert("Wipe operation failed.");
    }
}

// ================= SYSTEM INITIALIZATION =================
async function checkSystemStatus() {
    try {
        const response = await fetch(`${CONFIG.apiBase}/status`);
        if (!response.ok) throw new Error("Server offline");
        
        const data = await response.json();
        state.systemStatus = 'ONLINE';
        systemStatusText.innerText = 'ONLINE';
        systemStatusText.className = 'text-cyan';
        
        console.log("Sign Speak AI system status verified. Classifier active.");
    } catch (err) {
        state.systemStatus = 'OFFLINE';
        systemStatusText.innerText = 'OFFLINE (SERVER ERROR)';
        systemStatusText.className = 'text-danger';
        console.error("System connection check failed:", err);
    }
}

// Event Listeners setup
function initEventListeners() {
    // Camera trigger buttons
    startStreamBtn.addEventListener('click', startWebcam);
    stopStreamBtn.addEventListener('click', stopWebcam);
    quickStartCamBtn.addEventListener('click', startWebcam);
    
    // Sentence buffer triggers
    clearSentenceBtn.addEventListener('click', () => {
        state.sentenceWords = [];
        updateSentenceBufferUI();
    });
    
    speakSentenceBtn.addEventListener('click', () => {
        if (state.sentenceWords.length > 0) {
            const fullSentence = state.sentenceWords.join(' ');
            triggerSpeech(fullSentence);
        }
    });
    
    // Add custom gesture submit trigger
    addGestureForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const name = document.getElementById('gesture-input-name').value.trim();
        const desc = document.getElementById('gesture-input-desc').value.trim();
        
        if (name) {
            addCustomGesture(name, desc);
        }
    });
    
    // Add Speak Current Gesture click handler
    if (speakCurrentGestureBtn) {
        speakCurrentGestureBtn.addEventListener('click', () => {
            const text = predictedGestureLabel.innerText;
            if (text && text !== '--' && text !== '?') {
                triggerSpeech(text);
            }
        });
    }
    
    // Clear history logs trigger
    clearHistoryBtn.addEventListener('click', clearAllHistoryLogs);
    
    // Window resize event to adjust viewport elements
    window.addEventListener('resize', resizeOverlayCanvas);
    
    // Make sure voice list is loaded on browser startup (async load support)
    if ('speechSynthesis' in window) {
        window.speechSynthesis.onvoiceschanged = populateVoiceList;
    }
}

// Boot up everything
window.addEventListener('DOMContentLoaded', async () => {
    initRouter();
    initEventListeners();
    await checkSystemStatus();
    
    // Populate voice dropdown
    populateVoiceList();
    
    // Fetch initial list of gestures
    loadGesturesList();
});
