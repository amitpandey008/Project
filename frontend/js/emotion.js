/**
 * NeuroAdapt - Emotion Processing Module
 *
 * Handles emotion normalization to 4 core states + unknown.
 * Mapping per spec:
 *   happy/neutral/focused/calm -> engaged
 *   sad/tired                  -> bored
 *   angry/fear/stressed        -> frustrated
 *   disgust                    -> confused
 *   surprise/low-conf          -> unknown
 */

const EMOTION_MAP = {
    engaged:          "ENGAGED",
    bored:            "BORED",
    confused:         "CONFUSED",
    frustrated:       "FRUSTRATED",

    // DeepFace raw emotion mappings
    happy:    "ENGAGED",
    surprise: "ENGAGED",
    neutral:  "BORED",
    sad:      "BORED",
    fear:     "CONFUSED",
    disgust:  "FRUSTRATED",
    angry:    "FRUSTRATED",
};

export function normalizeEmotionResponse(response) {
    if (!response) {
        return getDefaultEmotion();
    }

    const rawEmotion = (response.emotion || "engaged").toLowerCase();
    const confidence = Number(response.confidence || 0);
    const source = response.source || "deepface+stats";
    const processingTime = Number(response.processing_time_ms || 0);
    const posture = response.posture || "upright";
    const postureConfidence = Number(response.posture_confidence || 0);

    const learningState = (response.learning_state || EMOTION_MAP[rawEmotion] || "ENGAGED").toUpperCase();

    return {
        emotion: rawEmotion,
        posture: posture,
        learning_state: learningState,
        confidence: confidence,
        posture_confidence: postureConfidence,
        theme: response.theme || "positive",
        background: response.background || rawEmotion,
        message: response.message || "",
        posture_tip: response.posture_tip || "",
        action: response.action || "maintain",
        assistance: response.assistance || "none",
        source: source,
        processing_time_ms: processingTime,
    };
}

export function getDefaultEmotion() {
    return {
        emotion: "engaged",
        posture: "upright",
        learning_state: "ENGAGED",
        confidence: 0.85,
        posture_confidence: 0.80,
        theme: "positive",
        background: "engaged",
        message: "Maintaining focus.",
        posture_tip: "Sit comfortably upright.",
        action: "maintain",
        assistance: "none",
        source: "default",
        processing_time_ms: 0,
    };
}

export async function predictEmotionFromBase64(base64Image) {
    try {
        const response = await fetch("/api/emotion/predict-base64", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                image: base64Image,
            }),
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const result = await response.json();
        return normalizeEmotionResponse(result);
    } catch (err) {
        console.error("Error predicting emotion:", err);
        return getDefaultEmotion();
    }
}