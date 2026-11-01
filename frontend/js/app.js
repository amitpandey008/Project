/**
 * NeuroAdapt - Application Controller
 * 
 * Main application initialization and global state management.
 */
import { TEAM_MEMBERS } from "./team.js";

// Application state
const AppState = {
    currentMode: "auto", // cloud, local, auto, manual
    isProcessing: false,
    lastPrediction: null,
    emotionHistory: [],
    processingInterval: 1500,
    lastPredictionTime: 0,
};

// Initialize app when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
    renderTeamMembers();
    setupEventListeners();
    updateModeDisplay();
});

/**
 * Render team members in the home page
 */
function renderTeamMembers() {
    const teamContainer = document.querySelector(".team-members");
    if (!teamContainer) return;

    teamContainer.innerHTML = "";

    TEAM_MEMBERS.forEach((member, index) => {
        const element = document.createElement("div");
        element.className = "team-member";
        element.innerHTML = `
            <span>${member.name}</span>
            <span>/ ${member.role}</span>
        `;
        teamContainer.appendChild(element);
    });
}

/**
 * Setup all event listeners
 */
function setupEventListeners() {
    // Navigation
    setupNavigation();

    // Mode toggle listeners
    const modeRadios = document.querySelectorAll('input[name="emotion-mode"]');
    modeRadios.forEach((radio) => {
        radio.addEventListener("change", (e) => {
            const mode = e.target.value;
            setManualMode(mode);
        });
    });

    // Camera controls
    const startCameraBtn = document.getElementById("start-camera");
    const stopCameraBtn = document.getElementById("stop-camera");
    if (startCameraBtn) {
        startCameraBtn.addEventListener("click", startCamera);
    }
    if (stopCameraBtn) {
        stopCameraBtn.addEventListener("click", stopCamera);
    }

    // Processing controls
    const startProcessingBtn = document.getElementById("start-processing");
    const stopProcessingBtn = document.getElementById("stop-processing");
    const startAutoBtn = document.getElementById("start-auto");
    const stopAutoBtn = document.getElementById("stop-auto");

    if (startProcessingBtn) {
        startProcessingBtn.addEventListener("click", startAutoDetection);
    }
    if (stopProcessingBtn) {
        stopProcessingBtn.addEventListener("click", stopAutoDetection);
    }
    if (startAutoBtn) {
        startAutoBtn.addEventListener("click", startAutoDetection);
    }
    if (stopAutoBtn) {
        stopAutoBtn.addEventListener("click", stopAutoDetection);
    }
}

/**
 * Setup navigation between pages
 */
function setupNavigation() {
    const navLinks = document.querySelectorAll(".nav-link");
    navLinks.forEach((link) => {
        link.addEventListener("click", (e) => {
            navLinks.forEach((l) => l.classList.remove("active"));
            link.classList.add("active");
        });
    });
}

/**
 * Set manual emotion mode
 * @param {string} state - Emotion state (calm, stressed, happy, focused, tired, confused)
 */
function setManualMode(state) {
    // Update UI
    document.querySelectorAll('input[name="emotion-mode"]').forEach((radio) => {
        radio.disabled = false;
    });

    // Send to backend
    fetch("/api/emotion/mode", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ mode: "manual", state: state }),
    })
        .then((res) => res.json())
        .then((data) => {
            updateModeDisplay();
            showManualModeMessage();
        })
        .catch((err) => {
            console.error("Error setting manual mode:", err);
        });
}

/**
 * Start auto emotion detection
 */
function startAutoDetection() {
    AppState.currentMode = "auto";
    AppState.isProcessing = true;
    updateModeDisplay();

    // Disable manual radios
    document.querySelectorAll('input[name="emotion-mode"]').forEach((radio) => {
        radio.disabled = true;
    });

    // Request camera access and start processing
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices
            .getUserMedia({ video: { facingMode: "environment" } })
            .then((stream) => {
                // Display camera stream
                const videoElement = document.querySelector(".webcam-frame video");
                if (videoElement) {
                    videoElement.srcObject = stream;
                }

                // Start the processing loop
                startProcessingLoop();
            })
            .catch((err) => {
                console.error("Camera access denied:", err);
                showError("Camera permission is required for automatic detection.");
                AppState.isProcessing = false;
                updateModeDisplay();
            });
    } else {
        showError("Camera API not supported in this browser.");
        AppState.isProcessing = false;
        updateModeDisplay();
    }
}

/**
 * Stop auto emotion detection
 */
function stopAutoDetection() {
    AppState.isProcessing = false;
    AppState.currentMode = "manual";
    updateModeDisplay();

    // Re-enable manual radios
    document.querySelectorAll('input[name="emotion-mode"]').forEach((radio) => {
        radio.disabled = false;
    });

    // Stop all tracks
    const tracks = document.querySelectorAll(".webcam-frame video");
    tracks.forEach((video) => {
        if (video.srcObject) {
            video.srcObject.getTracks().forEach((track) => track.stop());
        }
        video.srcObject = null;
    });

    // Send mode update to backend
    fetch("/api/emotion/mode", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ mode: "manual" }),
    })
        .then((res) => res.json())
        .catch((err) => console.error("Error updating mode:", err));
}

/**
 * Start the processing loop for emotion detection
 */
function startProcessingLoop() {
    if (!AppState.isProcessing) return;

    processNextFrame();

    // Set interval for subsequent frames
    AppState.processingInterval = 1500; // 1-2 predictions per second
    AppState.intervalId = setInterval(processNextFrame, AppState.processingInterval);
}

/**
 * Process a single frame from webcam
 */
async function processNextFrame() {
    if (!AppState.isProcessing) return;

    const videoElement = document.querySelector(".webcam-frame video");
    if (!videoElement || !videoElement.srcObject) return;

    // Capture frame using canvas
    const canvas = document.createElement("canvas");
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(videoElement, 0, 0);

    // Convert to JPEG and send to backend
    const base64Image = canvas.toDataURL("image/jpeg", 0.7);

    // Send to backend for emotion prediction
    sendFrameToBackend(base64Image);
}

/**
 * Send frame to backend for emotion prediction
 * @param {string} base64Image - Base64 encoded JPEG image
 */
async function sendFrameToBackend(base64Image) {
    try {
        const response = await fetch("/api/emotion/predict-base64", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                image: base64Image,
                mode: AppState.currentMode,
            }),
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const result = await response.json();

        // Update UI with prediction
        updateUIWithPrediction(result);

        // Log the prediction
        console.log("Emotion prediction:", result);
    } catch (err) {
        console.error("Error sending frame to backend:", err);
        // If cloud API fails and mode is auto, try local model
        if (AppState.currentMode === "auto") {
            // The backend handles fallback, just show error UI
            showError("Emotion service temporarily unavailable.");
        }
    }
}

/**
 * Update UI with emotion prediction result
 * @param {object} result - Emotion prediction response from backend
 */
function updateUIWithPrediction(result) {
    const emotion = result.emotion || "neutral";
    const confidence = result.confidence || 0;
    const learningState = result.learning_state || "CALM";
    const source = result.source || "none";
    const processingTime = result.processing_time_ms || 0;

    // Update temporal smoothing
    updateEmotionHistory(emotion, confidence);

    // Get stable emotion after smoothing
    const stableEmotion = getStableEmotion();

    // Update current state display
    updateCurrentStateDisplay(stableEmotion, confidence, source, processingTime);

    // Apply adaptation changes
    applyAdaptationChanges(stableEmotion, confidence);

    // Update mode indicator
    updateModeIndicator();
}

/**
 * Update emotion history for temporal smoothing
 * @param {string} emotion - Detected emotion
 * @param {number} confidence - Prediction confidence
 */
function updateEmotionHistory(emotion, confidence) {
    AppState.emotionHistory.push({
        emotion: emotion,
        confidence: confidence,
        timestamp: Date.now(),
    });

    // Keep only recent predictions (window size = 5)
    const windowSize = 5;
    if (AppState.emotionHistory.length > windowSize) {
        AppState.emotionHistory = AppState.emotionHistory.slice(-windowSize);
    }
}

/**
 * Get the stable emotion using temporal smoothing
 * @returns {string} The stable/dominant emotion state
 */
function getStableEmotion() {
    if (AppState.emotionHistory.length === 0) {
        return "neutral";
    }

    // Filter predictions with minimum confidence
    const minConfidence = 0.55;
    const filtered = AppState.emotionHistory.filter(
        (p) => p.confidence >= minConfidence
    );

    if (filtered.length === 0) {
        // If no predictions meet confidence threshold, return last one
        return AppState.emotionHistory[AppState.emotionHistory.length - 1].emotion;
    }

    // Count occurrences of each emotion
    const counts = {};
    filtered.forEach((p) => {
        counts[p.emotion] = (counts[p.emotion] || 0) + 1;
    });

    // Find the dominant emotion
    let dominantEmotion = null;
    let maxCount = 0;

    for (const [emotion, count] of Object.entries(counts)) {
        if (count > maxCount) {
            maxCount = count;
            dominantEmotion = emotion;
        }
    }

    // Require minimum stable predictions (optional)
    // If dominant emotion appears at least 3 times, use it
    // Otherwise, return the last prediction
    if (maxCount >= 3 && dominantEmotion) {
        return dominantEmotion;
    }

    // Fallback: return the most recent prediction
    return AppState.emotionHistory[AppState.emotionHistory.length - 1].emotion;
}

/**
 * Update current state display in the UI
 * @param {string} state - Learning state (ENGAGED, BORED, CONFUSED, FRUSTRATED)
 * @param {number} confidence - Prediction confidence
 * @param {string} source - Inference source (cloud, local, manual)
 * @param {number} processingTime - Processing time in ms
 */
function updateCurrentStateDisplay(state, confidence, source, processingTime) {
    const stateIndicator = document.getElementById("current-state");
    const confidenceValue = document.getElementById("confidence-value");
    const modeIndicator = document.getElementById("mode-indicator");

    if (!stateIndicator || !confidenceValue || !modeIndicator) return;

    // Map emotion to display icon (strictly 4 canonical states)
    const emotionIcons = {
        ENGAGED:    "🎯",
        BORED:      "😑",
        CONFUSED:   "🤔",
        FRUSTRATED: "😤",
    };

    const icon = emotionIcons[state] || "🎯";

    // Update state display (safe textContent to prevent DOM XSS)
    stateIndicator.textContent = `${icon} ${state}`;
    confidenceValue.textContent = `${Math.round(confidence * 100)}%`;

    // Update mode indicator
    modeIndicator.textContent = AppState.currentMode.toUpperCase();
    modeIndicator.className = "mode-indicator";
    if (AppState.currentMode === "manual") {
        modeIndicator.classList.add("manual");
    } else {
        modeIndicator.classList.add("auto");
    }

    // Show low confidence warning
    if (confidenceValue) {
        if (confidence < 0.55) {
            confidenceValue.style.color = "#e53e3e";
        } else {
            confidenceValue.style.color = "var(--primary-color, #22c55e)";
        }
    }
}

/**
 * Apply adaptation changes to the UI based on current emotion state
 * @param {string} state - Learning state (ENGAGED, BORED, CONFUSED, FRUSTRATED)
 * @param {number} confidence - Prediction confidence
 */
function applyAdaptationChanges(state, confidence) {
    const contentSection = document.querySelector(".learning-content");
    if (!contentSection) return;

    // Remove any existing state classes
    contentSection.classList.remove(
        "state-engaged",
        "state-bored",
        "state-confused",
        "state-frustrated"
    );

    // Add current state class (normalize to lowercase)
    if (state) {
        const normalized = state.toLowerCase();
        const valid = ["engaged", "bored", "confused", "frustrated"];
        contentSection.classList.add(`state-${valid.includes(normalized) ? normalized : "engaged"}`);
    }
}

/**
 * Show state-specific message (deprecated — messages now come from backend)
 * @param {string} state - Learning state
 * @param {number} confidence - Prediction confidence
 */
function showStateMessage(state, confidence) {
    // No-op: adaptation messages are now sourced from backend API response
    // and shown via the adaptation-banner in learning.js
}

/**
 * Update mode indicator display
 */
function updateModeIndicator() {
    // Update based on current state
    const modeIndicators = document.querySelectorAll(".mode-indicator");
    modeIndicators.forEach((indicator) => {
        indicator.textContent = AppState.currentMode.toUpperCase();
        indicator.className = "mode-indicator";
        if (AppState.currentMode === "manual") {
            indicator.classList.add("manual");
        } else {
            indicator.classList.add("auto");
        }
    });
}

/**
 * Show manual mode message
 */
function showManualModeMessage() {
    // Simple visual feedback — avoid blocking alert()
    console.log("[NeuroAdapt] Manual mode active.");
}

/**
 * Show error message to user
 * @param {string} message - Error message to display
 */
function showError(message) {
    // Create error banner
    const errorBanner = document.createElement("div");
    errorBanner.style.position = "fixed";
    errorBanner.style.top = "0";
    errorBanner.style.left = "0";
    errorBanner.style.right = "0";
    errorBanner.style.backgroundColor = "#fef2f2";
    errorBanner.style.color = "#e53e3e";
    errorBanner.style.padding = "1rem";
    errorBanner.style.borderBottom = "4px solid #e53e3e";
    errorBanner.style.zIndex = "1000";
    errorBanner.textContent = message;

    document.body.insertBefore(errorBanner, document.body.firstChild);

    setTimeout(() => {
        errorBanner.remove();
    }, 5000);
}