from typing import Dict, Any, Tuple
from ..config import Config


# ==============================================================================
# Centralized 24-Combination Adaptation Matrix (4 Emotions × 6 Postures)
# ==============================================================================

ADAPTATION_MATRIX: Dict[Tuple[str, str], Dict[str, Any]] = {
    # --------------------------------------------------------------------------
    # 1. ENGAGED (Maintain focused, low-distraction study environment)
    # --------------------------------------------------------------------------
    ("engaged", "upright"): {
        "learning_state": "ENGAGED",
        "theme": "positive",
        "background": "engaged",
        "message": "Excellent focus! You are actively engaged with the material.",
        "posture_tip": "Great posture keeps your energy and concentration at peak levels.",
        "action": "maintain",
        "assistance": "none",
        "density": "high",
    },
    ("engaged", "leaning_forward"): {
        "learning_state": "ENGAGED",
        "theme": "positive",
        "background": "engaged",
        "message": "Deeply engaged in the lesson! Keep up the great work.",
        "posture_tip": "Be mindful not to lean too close to the screen to prevent neck strain.",
        "action": "maintain",
        "assistance": "none",
        "density": "high",
    },
    ("engaged", "leaning_backward"): {
        "learning_state": "ENGAGED",
        "theme": "positive",
        "background": "engaged",
        "message": "Engaged and reviewing comfortably.",
        "posture_tip": "Keep your back supported and spine aligned to sustain long-term focus.",
        "action": "maintain",
        "assistance": "none",
        "density": "standard",
    },
    ("engaged", "slouched"): {
        "learning_state": "ENGAGED",
        "theme": "positive",
        "background": "engaged",
        "message": "Great learning focus! Straighten up to prevent fatigue.",
        "posture_tip": "Sit upright and roll your shoulders back to maintain good airflow and energy.",
        "action": "posture_reminder",
        "assistance": "none",
        "density": "standard",
    },
    ("engaged", "leaning_left"): {
        "learning_state": "ENGAGED",
        "theme": "positive",
        "background": "engaged",
        "message": "Actively following the lesson.",
        "posture_tip": "Distribute your weight evenly on both hips to avoid side muscle fatigue.",
        "action": "maintain",
        "assistance": "none",
        "density": "high",
    },
    ("engaged", "leaning_right"): {
        "learning_state": "ENGAGED",
        "theme": "positive",
        "background": "engaged",
        "message": "Actively following the lesson.",
        "posture_tip": "Distribute your weight evenly on both hips to avoid side muscle fatigue.",
        "action": "maintain",
        "assistance": "none",
        "density": "high",
    },

    # --------------------------------------------------------------------------
    # 2. BORED (Provide mild stimulation & attention/re-engagement)
    # --------------------------------------------------------------------------
    ("bored", "upright"): {
        "learning_state": "BORED",
        "theme": "stimulate",
        "background": "bored",
        "message": "Feeling disengaged? Try an interactive challenge or review key points.",
        "posture_tip": "Good sitting posture! A quick stretch can help re-energize your mind.",
        "action": "re-engage",
        "assistance": "low",
        "density": "standard",
    },
    ("bored", "leaning_forward"): {
        "learning_state": "BORED",
        "theme": "stimulate",
        "background": "bored",
        "message": "Losing momentum? Let's highlight the most exciting concepts.",
        "posture_tip": "Relax your posture, sit back slightly, and take a fresh breath.",
        "action": "re-engage",
        "assistance": "medium",
        "density": "standard",
    },
    ("bored", "leaning_backward"): {
        "learning_state": "BORED",
        "theme": "stimulate",
        "background": "bored",
        "message": "Drifting away? Let's refresh the pace with a quick summary.",
        "posture_tip": "Sit upright and adjust your screen to actively re-engage your attention.",
        "action": "re-engage",
        "assistance": "medium",
        "density": "low",
    },
    ("bored", "slouched"): {
        "learning_state": "BORED",
        "theme": "stimulate",
        "background": "bored",
        "message": "Low energy detected. Consider standing up or stretching before continuing.",
        "posture_tip": "Slouching reduces oxygen intake. Sit up tall to wake up your body and brain.",
        "action": "suggest_break",
        "assistance": "medium",
        "density": "low",
    },
    ("bored", "leaning_left"): {
        "learning_state": "BORED",
        "theme": "stimulate",
        "background": "bored",
        "message": "Losing focus. Let's refocus on the current concept.",
        "posture_tip": "Center your seating position to boost physical alertness.",
        "action": "re-engage",
        "assistance": "medium",
        "density": "standard",
    },
    ("bored", "leaning_right"): {
        "learning_state": "BORED",
        "theme": "stimulate",
        "background": "bored",
        "message": "Losing focus. Let's refocus on the current concept.",
        "posture_tip": "Center your seating position to boost physical alertness.",
        "action": "re-engage",
        "assistance": "medium",
        "density": "standard",
    },

    # --------------------------------------------------------------------------
    # 3. CONFUSED (Provide calm clarification & step-by-step guidance)
    # --------------------------------------------------------------------------
    ("confused", "upright"): {
        "learning_state": "CONFUSED",
        "theme": "help",
        "background": "confused",
        "message": "Challenging concept? Let's break this down into simpler steps.",
        "posture_tip": "Alert and attentive — take your time reviewing explanations.",
        "action": "clarify",
        "assistance": "high",
        "density": "standard",
    },
    ("confused", "leaning_forward"): {
        "learning_state": "CONFUSED",
        "theme": "help",
        "background": "confused",
        "message": "Straining to understand? We can simplify the explanation for you.",
        "posture_tip": "Avoid leaning in too close; sit comfortably while reviewing key examples.",
        "action": "clarify",
        "assistance": "high",
        "density": "standard",
    },
    ("confused", "leaning_backward"): {
        "learning_state": "CONFUSED",
        "theme": "help",
        "background": "confused",
        "message": "Puzzled by this section? Step-by-step guidance is available.",
        "posture_tip": "Lean comfortably against the chair back while checking the summary.",
        "action": "clarify",
        "assistance": "high",
        "density": "standard",
    },
    ("confused", "slouched"): {
        "learning_state": "CONFUSED",
        "theme": "help",
        "background": "confused",
        "message": "Feeling stuck and tired? Let's take a calm pause and simplify the idea.",
        "posture_tip": "Reset your posture to clear mental fatigue before continuing.",
        "action": "clarify",
        "assistance": "high",
        "density": "low",
    },
    ("confused", "leaning_left"): {
        "learning_state": "CONFUSED",
        "theme": "help",
        "background": "confused",
        "message": "Working through a tricky topic? Take it one step at a time.",
        "posture_tip": "Align your posture evenly to relieve neck and shoulder tension.",
        "action": "clarify",
        "assistance": "high",
        "density": "standard",
    },
    ("confused", "leaning_right"): {
        "learning_state": "CONFUSED",
        "theme": "help",
        "background": "confused",
        "message": "Working through a tricky topic? Take it one step at a time.",
        "posture_tip": "Align your posture evenly to relieve neck and shoulder tension.",
        "action": "clarify",
        "assistance": "high",
        "density": "standard",
    },

    # --------------------------------------------------------------------------
    # 4. FRUSTRATED (Provide soothing environment, simplified view & break)
    # --------------------------------------------------------------------------
    ("frustrated", "upright"): {
        "learning_state": "FRUSTRATED",
        "theme": "calm",
        "background": "frustrated",
        "message": "Take a deep breath. Review at your own pace without pressure.",
        "posture_tip": "Release tension in your jaw and drop your shoulders away from your ears.",
        "action": "soothe",
        "assistance": "high",
        "density": "low",
    },
    ("frustrated", "leaning_forward"): {
        "learning_state": "FRUSTRATED",
        "theme": "calm",
        "background": "frustrated",
        "message": "Tension detected. Step back slightly, breathe, and take your time.",
        "posture_tip": "Lean back gently and rest your arms comfortably on the desk.",
        "action": "soothe",
        "assistance": "high",
        "density": "low",
    },
    ("frustrated", "leaning_backward"): {
        "learning_state": "FRUSTRATED",
        "theme": "calm",
        "background": "frustrated",
        "message": "Feeling overwhelmed? A short 2-minute breather will help reset your mind.",
        "posture_tip": "Rest your back fully, close your eyes for a moment, and relax.",
        "action": "suggest_break",
        "assistance": "high",
        "density": "low",
    },
    ("frustrated", "slouched"): {
        "learning_state": "FRUSTRATED",
        "theme": "calm",
        "background": "frustrated",
        "message": "Stress and fatigue detected. We recommend stepping away for a short break.",
        "posture_tip": "Stand up, gently stretch your neck and back, and hydrate.",
        "action": "suggest_break",
        "assistance": "high",
        "density": "low",
    },
    ("frustrated", "leaning_left"): {
        "learning_state": "FRUSTRATED",
        "theme": "calm",
        "background": "frustrated",
        "message": "High strain detected. Let's simplify the view and reduce cognitive load.",
        "posture_tip": "Center your posture to ease muscle strain across your back.",
        "action": "soothe",
        "assistance": "high",
        "density": "low",
    },
    ("frustrated", "leaning_right"): {
        "learning_state": "FRUSTRATED",
        "theme": "calm",
        "background": "frustrated",
        "message": "High strain detected. Let's simplify the view and reduce cognitive load.",
        "posture_tip": "Center your posture to ease muscle strain across your back.",
        "action": "soothe",
        "assistance": "high",
        "density": "low",
    },
}


class AdaptationService:
    """
    Centralized combination rule engine evaluating all 4 emotions
    against all posture classes deterministically.
    """

    @staticmethod
    def normalize_emotion(emotion: str) -> str:
        """Normalizes emotion to strictly one of the allowed classes."""
        if not emotion:
            return "engaged"
        raw = emotion.lower().strip()
        # Pass through valid states directly
        valid = ["engaged", "bored", "confused", "frustrated"]
        if raw in valid:
            return raw
        # Map DeepFace raw emotions / legacy values
        legacy_map = {
            "happy": "engaged",
            "focused": "engaged",
            "calm": "engaged",
            "neutral": "bored",
            "surprise": "engaged",
            "sad": "bored",
            "tired": "bored",
            "fear": "confused",
            "disgust": "frustrated",
            "angry": "frustrated",
            "stressed": "frustrated",
        }
        return legacy_map.get(raw, "engaged")

    @staticmethod
    def normalize_posture(posture: str) -> str:
        """Normalizes posture to strictly one of the allowed posture classes."""
        if not posture:
            return "upright"
        raw = posture.lower().strip().replace(" ", "_")
        if raw in Config.ALLOWED_POSTURES:
            return raw
        # Fallbacks for aliases
        aliases = {
            "straight": "upright",
            "normal": "upright",
            "lean_forward": "leaning_forward",
            "lean_backward": "leaning_backward",
            "lean_left": "leaning_left",
            "lean_right": "leaning_right",
            "slouch": "slouched",
            "hunched": "slouched",
        }
        return aliases.get(raw, "upright")

    @classmethod
    def evaluate(
        cls,
        emotion: str,
        posture: str = "upright",
        confidence: float = 1.0,
        posture_confidence: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Evaluates the combination of facial emotion and posture against
        the centralized 24-combination matrix. Returns complete adaptation dict.
        """
        norm_emotion = cls.normalize_emotion(emotion)
        norm_posture = cls.normalize_posture(posture)

        # Retrieve combination from matrix or generate dynamic fallback
        key = (norm_emotion, norm_posture)
        if key in ADAPTATION_MATRIX:
            entry = dict(ADAPTATION_MATRIX[key])
        else:
            # Dynamic fallback
            state = norm_emotion.upper()
            entry = {
                "learning_state": state,
                "theme": "positive" if norm_emotion == "engaged" else "help" if norm_emotion == "confused" else "stimulate" if norm_emotion == "bored" else "calm",
                "background": norm_emotion,
                "message": f"Learning state: {norm_emotion.capitalize()}.",
                "posture_tip": f"Posture: {norm_posture.replace('_', ' ').capitalize()}.",
                "action": "maintain",
                "assistance": "low",
                "density": "standard",
            }

        # Handle low-confidence predictions gracefully while preserving schema
        is_low_confidence = confidence < Config.MIN_CONFIDENCE and posture_confidence < Config.MIN_CONFIDENCE
        if is_low_confidence:
            entry["message"] = "Maintaining current state (gathering stable predictions)..."
            entry["action"] = "maintain"

        # Build comprehensive payload
        result = {
            "emotion": norm_emotion,
            "posture": norm_posture,
            "learning_state": entry["learning_state"],
            "theme": entry["theme"],
            "background": entry["background"],
            "message": entry["message"],
            "posture_tip": entry["posture_tip"],
            "action": entry["action"],
            "assistance": entry["assistance"],
            "density": entry.get("density", "standard"),
        }

        return result

    @classmethod
    def map_to_learning_state(cls, emotion: str, posture: str = "upright") -> str:
        """Helper to get learning state string."""
        norm_emotion = cls.normalize_emotion(emotion)
        return norm_emotion.upper()

    @classmethod
    def get_adaptation_ui_changes(cls, state: str, confidence: float, posture: str = "upright") -> Dict[str, Any]:
        """Backward compatibility helper for UI adaptation."""
        return cls.evaluate(state, posture, confidence)

    @staticmethod
    def handle_combinations(current_states: list) -> str:
        """Returns the primary state from current list."""
        if not current_states:
            return "ENGAGED"
        return current_states[0]