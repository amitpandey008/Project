/**
 * NeuroAdapt - Adaptation Module
 * 
 * Handles UI adaptation based strictly on the 4 supported emotion states:
 * ENGAGED, BORED, CONFUSED, FRUSTRATED
 */

const STATE_ADAPTATIONS = {
    ENGAGED: {
        message: "Great focus! You are actively engaged with the lesson.",
        theme: "positive",
        background: "engaged",
        density: "high",
        assistance: "none",
        action: "maintain",
    },
    BORED: {
        message: "Losing interest? Consider stretching or taking a short pause.",
        theme: "stimulate",
        background: "bored",
        density: "low",
        assistance: "medium",
        action: "re-engage",
    },
    CONFUSED: {
        message: "Tricky concept? Let's clarify this step-by-step.",
        theme: "help",
        background: "confused",
        density: "standard",
        assistance: "high",
        action: "clarify",
    },
    FRUSTRATED: {
        message: "Take a deep breath and review at your own comfortable pace.",
        theme: "calm",
        background: "frustrated",
        density: "low",
        assistance: "high",
        action: "suggest_break",
    },
};

export const AdaptationState = {
    stateHistory: [],

    addState: function (state, confidence) {
        this.stateHistory.push({
            state: (state || "ENGAGED").toUpperCase(),
            confidence: Number(confidence || 0),
            timestamp: Date.now(),
        });

        const windowSize = 5;
        if (this.stateHistory.length > windowSize) {
            this.stateHistory = this.stateHistory.slice(-windowSize);
        }
    },

    getCurrentAdaptation: function () {
        if (this.stateHistory.length === 0) {
            return this.getDefaultAdaptation();
        }

        const latest = this.stateHistory[this.stateHistory.length - 1];
        const state = latest.state;
        return STATE_ADAPTATIONS[state] || STATE_ADAPTATIONS.ENGAGED;
    },

    getSingleStateAdaptation: function (state) {
        const key = (state || "ENGAGED").toUpperCase();
        return STATE_ADAPTATIONS[key] || STATE_ADAPTATIONS.ENGAGED;
    },

    getDefaultAdaptation: function () {
        return STATE_ADAPTATIONS.ENGAGED;
    },
};

export function processAdaptation(state, confidence) {
    AdaptationState.addState(state, confidence);
    return AdaptationState.getCurrentAdaptation();
}

window.Config = {
    MIN_CONFIDENCE: 0.40,
};