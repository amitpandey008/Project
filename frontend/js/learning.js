/**
 * NeuroAdapt - Learning Page Module (Combined CNN Emotion + Online Posture)
 *
 * LAYOUT:
 *  - #youtube-iframe  → LEFT section (ML video & lesson content)
 *  - #webcam-video    → RIGHT panel  (compact single stream detection monitor)
 *
 * 4 SUPPORTED EMOTIONS:
 *  - engaged
 *  - bored
 *  - confused
 *  - frustrated
 */

// ─────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────

const LessonConfig = {
    processingInterval:  1500,   // ms between prediction checks
    videoWarmupDelay:    1000,   // ms to wait after camera opens before first capture
    maxVideoWaitMs:      8000,   // max ms to wait for video ready
    historyWindowSize:   3,      // lightweight rolling window size for temporal smoothing
};

// ─────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────

const LearningState = {
    isCameraActive:       false,
    isProcessing:         false,
    isProcessingRequest:  false,   // prevent concurrent in-flight requests
    videoStream:          null,
    processingIntervalId: null,
    predictionHistory:    [],      // rolling window of { emotion, posture, learning_state, confidence }
    frameCount:           0,
};

// ─────────────────────────────────────────────────────────────
// UI Display Constants (Strictly 4 Emotions & 6 Postures)
// ─────────────────────────────────────────────────────────────

const EMOTION_DISPLAY = {
    engaged:          { icon: "🎯", label: "Engaged" },
    bored:            { icon: "😑", label: "Bored" },
    confused:         { icon: "🤔", label: "Confused" },
    frustrated:       { icon: "😤", label: "Frustrated" },
};

const POSTURE_RAW_LABELS = {
    upright:          "Upright",
    leaning_forward:  "Leaning Forward",
    leaning_backward: "Leaning Backward",
    leaning_left:     "Leaning Left",
    leaning_right:    "Leaning Right",
    slouched:         "Slouched",
};

const POSTURE_ICONS_BY_NAME = {
    upright:          "🧍",
    leaning_forward:  "📐",
    leaning_backward: "🪑",
    leaning_left:     "↙️",
    leaning_right:    "↘️",
    slouched:         "🛋️",
};

/** Manual emotion override state (null = auto detection active) */
let manualEmotionOverride = null;
let currentInferenceMode = "auto"; // "auto" | "local"

// ─────────────────────────────────────────────────────────────
// Theme Adaptation (Whole-Page Background)
// ─────────────────────────────────────────────────────────────

/**
 * Applies the emotion theme to the entire page background.
 * Sets body[data-emotion] which is styled via CSS.
 */
function applyEmotionTheme(autoEmotion) {
    const rawEmotion = manualEmotionOverride || autoEmotion || "none";
    const emotion = rawEmotion.toLowerCase();

    // Ensure only valid emotion states are applied; unknown -> "none"
    const validForTheme = ["engaged", "bored", "confused", "frustrated"];
    const validEmotion = validForTheme.includes(emotion) ? emotion : "none";
    document.body.setAttribute("data-emotion", validEmotion);
    updateEmotionButtonIndicator(validEmotion, manualEmotionOverride !== null);
}

/**
 * Updates the navbar Emotion button label and icon.
 */
function updateEmotionButtonIndicator(emotion, isManual) {
    const btn      = document.getElementById("emotion-toggle-btn");
    const iconEl   = document.getElementById("emotion-btn-icon");
    const labelEl  = document.getElementById("emotion-btn-label");
    if (!btn) return;

    const info = EMOTION_DISPLAY[emotion];
    if (info) {
        if (iconEl)  iconEl.textContent  = info.icon;
        if (labelEl) labelEl.textContent = info.label;
        if (isManual) {
            btn.classList.add("has-manual");
            btn.title = `Manual override: ${info.label}`;
        } else {
            btn.classList.remove("has-manual");
            btn.title = `Detected emotion: ${info.label}`;
        }
    } else {
        if (iconEl)  iconEl.textContent  = "🎭";
        if (labelEl) labelEl.textContent = "Emotion";
        btn.classList.remove("has-manual");
        btn.title = "Emotion selector";
    }
}

// ─────────────────────────────────────────────────────────────
// Initialization
// ─────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
    setupCameraControls();
    setupEmotionDropdown();
    setupYouTubePlayer();
    setupFocusMode();
    console.log("[NeuroAdapt] Dashboard initialized with CNN Emotion + Posture pipeline.");
});

// ─────────────────────────────────────────────────────────────
// Camera Controls & Lifecycle
// ─────────────────────────────────────────────────────────────

function setupCameraControls() {
    const startAutoBtn = document.getElementById("start-auto");
    const stopAutoBtn  = document.getElementById("stop-auto");
    const modeEl       = document.getElementById("mode-indicator");

    if (startAutoBtn) startAutoBtn.addEventListener("click", startAutoDetection);
    if (stopAutoBtn)  stopAutoBtn.addEventListener("click",  stopAutoDetection);

    if (modeEl) {
        modeEl.style.cursor = "pointer";
        modeEl.title = "Click to toggle inference engine (AUTO / LOCAL CNN)";
        modeEl.addEventListener("click", () => {
            currentInferenceMode = currentInferenceMode === "auto" ? "local" : "auto";
            modeEl.textContent = currentInferenceMode === "local" ? "LOCAL CNN" : "AUTO";
            setStatus(`Inference engine set to: ${currentInferenceMode.toUpperCase()}`);
        });
    }
}

/**
 * Starts the webcam stream and auto detection loop.
 */
async function startAutoDetection() {
    if (LearningState.isProcessing) return;

    setStatus("Starting camera...");
    const started = await startCamera();
    if (!started) return;

    setStatus("Warming up camera sensor...");
    const ready = await waitForVideoReady();
    if (!ready) {
        setStatus("Camera stream ready timeout. Please restart.");
        console.warn("[NeuroAdapt] Video stream ready timeout.");
        return;
    }

    LearningState.isProcessing = true;
    setElDisplay("start-auto", false);
    setElDisplay("stop-auto",  true);

    setStatus("Active — analyzing emotion & posture...");
    console.log(`[NeuroAdapt] Auto detection active. Interval: ${LessonConfig.processingInterval}ms`);

    startProcessingLoop();
}

/**
 * Stops auto detection and terminates the webcam stream.
 */
function stopAutoDetection() {
    LearningState.isProcessing = false;

    if (LearningState.processingIntervalId) {
        clearInterval(LearningState.processingIntervalId);
        LearningState.processingIntervalId = null;
    }

    stopCameraStream();

    setElDisplay("start-auto", true);
    setElDisplay("stop-auto",  false);

    resetDisplay();
    setStatus("");
    console.log("[NeuroAdapt] Auto detection stopped and webcam stream closed.");
}

/**
 * Opens webcam stream and attaches to the video element.
 */
async function startCamera() {
    if (LearningState.isCameraActive) return true;

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: "user"
            },
            audio: false,
        });

        LearningState.videoStream    = stream;
        LearningState.isCameraActive = true;

        const webcamVideo = document.getElementById("webcam-video");
        const placeholder = document.getElementById("camera-placeholder");
        const badge       = document.getElementById("camera-badge");

        if (webcamVideo) {
            webcamVideo.srcObject = stream;
            webcamVideo.style.display = "block";
            try {
                await webcamVideo.play();
            } catch (playErr) {
                console.warn("[NeuroAdapt] video.play() warning:", playErr);
            }
        }
        if (placeholder) placeholder.style.display = "none";
        if (badge)       badge.style.display        = "inline-block";

        console.log("[NeuroAdapt] Camera stream acquired.");
        return true;
    } catch (err) {
        console.error("[NeuroAdapt] Camera access error:", err);
        setStatus("Camera permission denied or camera not found.");
        alert("Camera permission is required for emotion & posture detection. Please allow camera access in your browser settings.");
        return false;
    }
}

/**
 * Stops all tracks in the active MediaStream.
 */
function stopCameraStream() {
    if (LearningState.videoStream) {
        try {
            LearningState.videoStream.getTracks().forEach((track) => track.stop());
        } catch (e) {
            console.warn("[NeuroAdapt] Error stopping tracks:", e);
        }
        LearningState.videoStream = null;
    }
    LearningState.isCameraActive = false;
    LearningState.frameCount     = 0;

    const webcamVideo = document.getElementById("webcam-video");
    const placeholder = document.getElementById("camera-placeholder");
    const badge       = document.getElementById("camera-badge");

    if (webcamVideo) {
        webcamVideo.srcObject = null;
        webcamVideo.style.display = "none";
    }
    if (placeholder) placeholder.style.display = "flex";
    if (badge)       badge.style.display        = "none";
}

/**
 * Polls until the video element has non-zero dimensions.
 */
function waitForVideoReady() {
    return new Promise((resolve) => {
        const video = document.getElementById("webcam-video");
        if (!video) return resolve(false);
        if (video.videoWidth > 0) return resolve(true);

        const deadline = Date.now() + LessonConfig.maxVideoWaitMs;
        const poll = setInterval(() => {
            if (video.videoWidth > 0) {
                clearInterval(poll);
                resolve(true);
            } else if (Date.now() > deadline) {
                clearInterval(poll);
                resolve(false);
            }
        }, 100);
    });
}

// ─────────────────────────────────────────────────────────────
// Frame Capture & Processing Loop
// ─────────────────────────────────────────────────────────────

function startProcessingLoop() {
    setTimeout(() => {
        if (!LearningState.isProcessing) return;
        processNextFrame();

        LearningState.processingIntervalId = setInterval(
            processNextFrame,
            LessonConfig.processingInterval
        );
    }, LessonConfig.videoWarmupDelay);
}

async function processNextFrame() {
    if (!LearningState.isProcessing || LearningState.isProcessingRequest) return;

    const video = document.getElementById("webcam-video");
    if (!video || !video.srcObject || video.videoWidth === 0) return;

    LearningState.isProcessingRequest = true;
    LearningState.frameCount++;
    const frameNum = LearningState.frameCount;

    try {
        const canvas = document.getElementById("capture-canvas");
        if (!canvas) return;

        canvas.width  = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        ctx.drawImage(video, 0, 0);

        const base64Image = canvas.toDataURL("image/jpeg", 0.80);
        await sendFrameToBackend(base64Image, frameNum);
    } catch (err) {
        console.error(`[NeuroAdapt] Frame #${frameNum} processing error:`, err);
    } finally {
        LearningState.isProcessingRequest = false;
    }
}

async function sendFrameToBackend(base64Image, frameNum) {
    const t0 = performance.now();

    try {
        const response = await fetch("/api/emotion/predict-base64", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ image: base64Image, mode: currentInferenceMode }),
        });

        const elapsed = Math.round(performance.now() - t0);

        if (!response.ok) {
            const errText = await response.text();
            console.error(`[NeuroAdapt] HTTP ${response.status}: ${errText}`);
            setStatus(`Backend error (HTTP ${response.status})`);
            return;
        }

        const result = await response.json();
        updateDetectionUI(result);
        setStatus(`Frame #${frameNum} evaluated (${elapsed}ms)`);
    } catch (err) {
        console.error(`[NeuroAdapt] Network error on frame #${frameNum}:`, err);
        setStatus("Network error connecting to backend");
    }
}

// ─────────────────────────────────────────────────────────────
// Temporal Smoothing & UI Update
// ─────────────────────────────────────────────────────────────

function updateDetectionUI(result) {
    const rawEmotion = (result.emotion || result.final_emotion || "engaged").toLowerCase();
    const rawPosture = (result.posture || "upright").toLowerCase();
    const confidence = Number(result.confidence || 0);
    const postureConf = Number(result.posture_confidence || 0);
    const avgConf    = postureConf > 0 ? (confidence + postureConf) / 2 : confidence;

    // Push into rolling window
    LearningState.predictionHistory.push({
        emotion: rawEmotion,
        posture: rawPosture,
        confidence: avgConf,
        ts: Date.now(),
    });

    if (LearningState.predictionHistory.length > LessonConfig.historyWindowSize) {
        LearningState.predictionHistory = LearningState.predictionHistory.slice(-LessonConfig.historyWindowSize);
    }

    // Get smoothed stable prediction from backend history
    const stable = getStablePrediction();

    // Backend ALWAYS overrides manual selection when auto detection is running.
    if (manualEmotionOverride !== null) {
        manualEmotionOverride = null;
        document.querySelectorAll(".emotion-option").forEach(b => b.classList.remove("active"));
    }

    const activeEmotion = stable.emotion;

    // Update Monitor UI with backend source of truth
    updateStateDisplay(activeEmotion, stable.posture, confidence, postureConf, result.deepface_emotion, result.source);

    // Apply whole-page theme derived from active emotion
    applyEmotionTheme(activeEmotion);

    // Display adaptive message and posture tip from backend response
    showAdaptationBanner(result.message, result.posture_tip, activeEmotion);
}

/**
 * Calculates majority emotion & posture from rolling history.
 */
function getStablePrediction() {
    const hist = LearningState.predictionHistory;
    if (!hist.length) return { emotion: "engaged", posture: "upright" };

    const emotionCounts = {};
    const postureCounts = {};

    hist.forEach((item) => {
        // Give genuine frustrated/confused predictions double weight so they aren't suppressed by smoothing
        const weight = (item.emotion === "frustrated" || item.emotion === "confused") && item.confidence >= 0.5 ? 2 : 1;
        emotionCounts[item.emotion] = (emotionCounts[item.emotion] || 0) + weight;
        postureCounts[item.posture] = (postureCounts[item.posture] || 0) + 1;
    });

    const stableEmotion = Object.entries(emotionCounts).sort((a, b) => b[1] - a[1])[0][0];
    const stablePosture = Object.entries(postureCounts).sort((a, b) => b[1] - a[1])[0][0];

    return { emotion: stableEmotion, posture: stablePosture };
}

function updateStateDisplay(emotion, posture, emotionConf, postureConf, deepfaceEmotion = null, source = null) {
    const stateEl        = document.getElementById("current-state");
    const confEl         = document.getElementById("confidence-value");
    const rawLabelEl     = document.getElementById("raw-posture-label");
    const modeEl         = document.getElementById("mode-indicator");
    const postureTextEl  = document.getElementById("posture-text");
    const postureBadgeEl = document.getElementById("posture-badge");
    const postureIconEl  = document.getElementById("posture-icon");
    const panel          = document.getElementById("emotion-panel");

    const emotionInfo = EMOTION_DISPLAY[emotion] || EMOTION_DISPLAY.engaged;
    const formattedPosture = POSTURE_RAW_LABELS[posture] || posture.replace(/_/g, " ");
    const postureIcon = POSTURE_ICONS_BY_NAME[posture] || "🧍";

    if (stateEl) {
        stateEl.textContent = `${emotionInfo.icon} ${emotionInfo.label}`;
    }
    if (rawLabelEl) rawLabelEl.textContent = formattedPosture;

    if (confEl) {
        confEl.textContent = `${Math.round(emotionConf * 100)}%`;
        confEl.style.color = emotionConf >= 0.5 ? "#22c55e" : "#f59e0b";
    }

    if (postureTextEl) postureTextEl.textContent = formattedPosture;
    if (postureBadgeEl) {
        postureBadgeEl.textContent = posture.toUpperCase().replace(/_/g, " ");
        postureBadgeEl.className = `posture-badge badge-${posture.replace(/_/g, "-")}`;
    }
    if (postureIconEl) postureIconEl.textContent = postureIcon;

    if (modeEl) {
        if (manualEmotionOverride) {
            modeEl.textContent = "PREVIEW";
            modeEl.className   = "mode-indicator manual";
        } else if (currentInferenceMode === "local" || (source && source.includes("local_cnn"))) {
            modeEl.textContent = "LOCAL CNN";
            modeEl.className   = "mode-indicator auto";
        } else {
            modeEl.textContent = "AUTO";
            modeEl.className   = "mode-indicator auto";
        }
    }

    if (panel) {
        panel.classList.remove("state-engaged", "state-bored", "state-confused", "state-frustrated");
        const panelClass = ["engaged", "bored", "confused", "frustrated"].includes(emotion)
            ? `state-${emotion}`
            : "state-engaged";
        panel.classList.add(panelClass);
    }
}

function showAdaptationBanner(message, postureTip, emotion) {
    const banner = document.getElementById("adaptation-banner");
    const msgEl  = document.getElementById("adaptation-message");
    const tipEl  = document.getElementById("adaptation-tip");
    const iconEl = document.getElementById("adaptation-icon");

    if (!banner || !msgEl || !tipEl) return;

    const info = EMOTION_DISPLAY[emotion] || EMOTION_DISPLAY.engaged;
    if (iconEl) iconEl.textContent = info.icon;
    msgEl.textContent = message || "Maintaining focus.";
    tipEl.textContent = postureTip ? `Tip: ${postureTip}` : "";
    banner.style.display = "block";
    banner.className = `adaptation-banner banner-${emotion}`;
}

function resetDisplay() {
    const stateEl        = document.getElementById("current-state");
    const confEl         = document.getElementById("confidence-value");
    const rawLabelEl     = document.getElementById("raw-posture-label");
    const postureTextEl  = document.getElementById("posture-text");
    const postureBadgeEl = document.getElementById("posture-badge");
    const postureIconEl  = document.getElementById("posture-icon");
    const banner         = document.getElementById("adaptation-banner");
    const panel          = document.getElementById("emotion-panel");

    if (stateEl) stateEl.textContent = "—";
    if (confEl)  confEl.textContent  = "—";
    if (rawLabelEl) rawLabelEl.textContent = "";
    if (postureTextEl) postureTextEl.textContent = "—";
    if (postureBadgeEl) {
        postureBadgeEl.textContent = "INACTIVE";
        postureBadgeEl.className = "posture-badge";
    }
    if (postureIconEl) postureIconEl.textContent = "🧍";
    if (banner) banner.style.display = "none";
    if (panel) panel.classList.remove("state-engaged", "state-bored", "state-confused", "state-frustrated");

    if (!manualEmotionOverride) {
        document.body.setAttribute("data-emotion", "none");
        updateEmotionButtonIndicator("none", false);
    }

    LearningState.predictionHistory = [];
}

// ─────────────────────────────────────────────────────────────
// Navbar Emotion Dropdown (Strictly 4 Emotions for Testing)
// ─────────────────────────────────────────────────────────────

function setupEmotionDropdown() {
    const toggleBtn  = document.getElementById("emotion-toggle-btn");
    const dropdown   = document.getElementById("emotion-dropdown");
    const resetBtn   = document.getElementById("btn-reset-emotion");
    const optionBtns = document.querySelectorAll(".emotion-option[data-emotion]");

    if (!toggleBtn || !dropdown) return;

    toggleBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        const isOpen = !dropdown.hidden;
        dropdown.hidden = isOpen;
        toggleBtn.setAttribute("aria-expanded", String(!isOpen));
    });

    document.addEventListener("click", (e) => {
        if (!dropdown.hidden && !dropdown.contains(e.target) && !toggleBtn.contains(e.target)) {
            dropdown.hidden = true;
            toggleBtn.setAttribute("aria-expanded", "false");
        }
    });

    optionBtns.forEach((btn) => {
        btn.addEventListener("click", () => {
            const rawVal = btn.dataset.emotion.toLowerCase(); // engaged | bored | confused | frustrated

            // Set temporary UI preview
            manualEmotionOverride = rawVal;
            applyEmotionTheme(rawVal);

            optionBtns.forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");

            dropdown.hidden = true;
            toggleBtn.setAttribute("aria-expanded", "false");

            const modeEl = document.getElementById("mode-indicator");
            if (modeEl) {
                modeEl.textContent = "PREVIEW";
                modeEl.className = "mode-indicator manual";
            }

            const stateEl = document.getElementById("current-state");
            const confEl  = document.getElementById("confidence-value");
            const info = EMOTION_DISPLAY[rawVal];
            if (stateEl && info) stateEl.textContent = `${info.icon} ${info.label}`;
            if (confEl) {
                confEl.textContent = "Preview";
                confEl.style.color = "#818cf8";
            }

            // Immediately apply test adaptation message for user preview
            fetch("/api/adaptation/evaluate", {
                method:  "POST",
                headers: { "Content-Type": "application/json" },
                body:    JSON.stringify({ emotion: rawVal, posture: "upright", confidence: 1.0, posture_confidence: 1.0 }),
            })
            .then((r) => r.json())
            .then((data) => {
                // Apply if still in preview for this emotion
                if (manualEmotionOverride === rawVal) {
                    showAdaptationBanner(data.message, data.posture_tip, rawVal);
                }
            })
            .catch(() => {
                if (manualEmotionOverride === rawVal) {
                    const fallbackMsgs = {
                        engaged: "Great engagement! Content paced normally.",
                        bored: "Stimulation boost applied. Interactive elements enabled.",
                        confused: "Clarification mode active. Additional guidance provided.",
                        frustrated: "Calming mode active. Take a steady breath and continue.",
                    };
                    showAdaptationBanner(fallbackMsgs[rawVal] || "Testing emotion theme.", "Sit upright: back straight.", rawVal);
                }
            });

            console.log(`[NeuroAdapt] Manual preview applied: ${rawVal} (live CNN detection will auto-apply next backend result)`);
        });
    });

    if (resetBtn) {
        resetBtn.addEventListener("click", () => {
            manualEmotionOverride = null;
            optionBtns.forEach((b) => b.classList.remove("active"));
            dropdown.hidden = true;
            toggleBtn.setAttribute("aria-expanded", "false");

            const modeEl = document.getElementById("mode-indicator");
            if (modeEl) {
                modeEl.textContent = "AUTO";
                modeEl.className = "mode-indicator auto";
            }

            if (!LearningState.isProcessing) {
                document.body.setAttribute("data-emotion", "none");
                updateEmotionButtonIndicator("none", false);
                resetDisplay();
            } else {
                const stable = getStablePrediction();
                applyEmotionTheme(stable.emotion);
            }

            console.log("[NeuroAdapt] Manual preview cleared — live Auto detection active.");
        });
    }
}

// ─────────────────────────────────────────────────────────────
// YouTube Video Loader
// ─────────────────────────────────────────────────────────────

function setupYouTubePlayer() {
    const input   = document.getElementById("youtube-url-input");
    const loadBtn = document.getElementById("btn-load-video");
    const iframe  = document.getElementById("youtube-iframe");

    if (!input || !loadBtn || !iframe) return;

    function extractYouTubeId(urlOrId) {
        if (!urlOrId) return null;
        const trimmed = urlOrId.trim();
        if (/^[a-zA-Z0-9_-]{11}$/.test(trimmed)) {
            return trimmed;
        }
        const match = trimmed.match(/(?:youtu\.be\/|youtube\.com\/(?:embed\/|v\/|watch\?v=|watch\?.+&v=|live\/|shorts\/))([\w-]{11})/i);
        return match ? match[1] : null;
    }

    function loadVideoFromInput() {
        const url = input.value.trim();
        const videoId = extractYouTubeId(url);
        if (!videoId) {
            alert("Please enter a valid YouTube Video URL (e.g., https://www.youtube.com/watch?v=HcqpanDadyQ)");
            return;
        }
        iframe.src = `https://www.youtube.com/embed/${videoId}?autoplay=1&rel=0&showinfo=0&enablejsapi=1`;
        console.log(`[NeuroAdapt] Loaded YouTube Video ID: ${videoId}`);
    }

    loadBtn.addEventListener("click", loadVideoFromInput);
    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            e.preventDefault();
            loadVideoFromInput();
        }
    });
}

// ─────────────────────────────────────────────────────────────
// Focus Mode Toggle
// ─────────────────────────────────────────────────────────────

function setupFocusMode() {
    const focusBtn = document.getElementById("btn-focus-mode");
    if (!focusBtn) return;

    focusBtn.addEventListener("click", () => {
        document.body.classList.toggle("focus-mode");
        const isFocus = document.body.classList.contains("focus-mode");
        focusBtn.innerHTML = isFocus ? "⬛ Exit Focus" : "🌑 Focus Mode";
        focusBtn.classList.toggle("active", isFocus);
        focusBtn.setAttribute("aria-pressed", String(isFocus));
        console.log(`[NeuroAdapt] Focus mode: ${isFocus ? "ON" : "OFF"}`);
    });
}

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────

function setElDisplay(id, visible) {
    const el = document.getElementById(id);
    if (el) el.style.display = visible ? "inline-block" : "none";
}

function setStatus(msg) {
    const el = document.getElementById("camera-status");
    if (el) {
        el.textContent = msg;
        el.style.display = msg ? "block" : "none";
    }
}