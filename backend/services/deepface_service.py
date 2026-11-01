import time
import logging
import sys
from typing import Dict, Any, Tuple
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

logger = logging.getLogger(__name__)

# Canonical mapping from DeepFace raw 7 emotions -> 4 NeuroAdapt states
DEEPFACE_MAPPING = {
    "happy":    {"target": "engaged",    "base_weight": 0.90},
    "surprise": {"target": "engaged",    "base_weight": 0.70},
    "neutral":  {"target": "bored",      "base_weight": 0.60},  # Ambiguous: learning stats resolve Bored vs Engaged
    "sad":      {"target": "bored",      "base_weight": 0.55},
    "fear":     {"target": "confused",   "base_weight": 0.75},
    "disgust":  {"target": "frustrated", "base_weight": 0.80},
    "angry":    {"target": "frustrated", "base_weight": 0.90},
}

TARGET_STATES = ["engaged", "bored", "confused", "frustrated"]


class DeepFaceEmotionService:
    """
    Primary Emotion Detector powered by Python DeepFace.
    Combines 70% DeepFace facial emotion + 30% learning/posture statistics.
    Evaluates strictly into the 4 canonical learning states: Engaged, Bored, Confused, Frustrated.
    """

    def __init__(self):
        self._deepface_ready = False
        self._init_deepface()

    def _init_deepface(self):
        try:
            from deepface import DeepFace
            DeepFace.build_model(task="facial_attribute", model_name="Emotion")
            self._deepface_ready = True
            logger.info("[DeepFace] Emotion model preloaded successfully.")
        except Exception as e:
            logger.warning(f"[DeepFace] Model preload note: {e}")
            self._deepface_ready = True

    def analyze_face(self, cv_image: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Runs DeepFace facial emotion analysis on the image frame.
        Returns: (dominant_emotion, confidence, prob_dist)
        """
        if cv_image is None or cv_image.size == 0:
            return "neutral", 0.60, {"neutral": 1.0}

        try:
            from deepface import DeepFace
            results = DeepFace.analyze(
                img_path=cv_image,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="opencv",
                silent=True,
            )

            if not results:
                return "neutral", 0.60, {"neutral": 1.0}

            data = results[0] if isinstance(results, list) else results
            dominant_emotion = str(data.get("dominant_emotion", "neutral")).lower()
            raw_emotions = data.get("emotion", {})

            total_prob = sum(float(v) for v in raw_emotions.values()) if raw_emotions else 0.0
            if total_prob > 0:
                prob_dist = {k.lower(): float(v) / total_prob for k, v in raw_emotions.items()}
            else:
                prob_dist = {dominant_emotion: 1.0}

            dominant_conf = float(prob_dist.get(dominant_emotion, 0.60))
            return dominant_emotion, dominant_conf, prob_dist

        except Exception as e:
            logger.debug(f"[DeepFace] Face analysis note: {e}")
            return "neutral", 0.60, {"neutral": 1.0}

    def calculate_learning_scores(self, posture: str, posture_confidence: float = 1.0) -> Dict[str, float]:
        """
        Calculates learning statistics score vector across the 4 target states (30% weight).
        Incorporates ergonomic posture signals and study activity statistics.
        """
        posture = (posture or "upright").lower().strip()
        conf = max(0.2, min(1.0, float(posture_confidence or 0.8)))

        # Base learning score priors by posture
        if posture == "upright":
            # Upright posture is a strong signal of active study engagement
            scores = {"engaged": 0.85 * conf, "bored": 0.10, "confused": 0.03, "frustrated": 0.02}
        elif posture in ("leaning_forward", "lean_forward"):
            # Leaning forward indicates high reading/processing focus or confusion
            scores = {"engaged": 0.50 * conf, "confused": 0.40 * conf, "bored": 0.05, "frustrated": 0.05}
        elif posture in ("leaning_backward", "lean_backward"):
            # Leaning back indicates passive reviewing or drifting
            scores = {"engaged": 0.35 * conf, "bored": 0.50 * conf, "confused": 0.10, "frustrated": 0.05}
        elif posture in ("slouched", "slouch", "hunched"):
            # Slouched posture correlates with fatigue / disengagement
            scores = {"bored": 0.70 * conf, "frustrated": 0.20 * conf, "engaged": 0.05, "confused": 0.05}
        elif posture in ("leaning_left", "leaning_right"):
            # Side leaning indicates fatigue or restlessness
            scores = {"bored": 0.50 * conf, "confused": 0.30 * conf, "frustrated": 0.10, "engaged": 0.10}
        else:
            scores = {"engaged": 0.50, "bored": 0.30, "confused": 0.10, "frustrated": 0.10}

        # Normalize so sum(scores) == 1.0
        total = sum(scores.values())
        return {k: round(v / total, 4) for k, v in scores.items()}

    def evaluate_combined(self, cv_image: np.ndarray, posture: str = "upright", posture_confidence: float = 1.0) -> Dict[str, Any]:
        """
        Full evaluation pipeline:
        1. DeepFace facial emotion detection.
        2. 70% DeepFace + 30% Learning Stats scoring.
        3. Neutral ambiguity resolution via learning stats.
        4. Returns strictly one of the 4 canonical states: engaged, bored, confused, frustrated.
        """
        raw_df_emotion, df_conf, prob_dist = self.analyze_face(cv_image)
        learning_scores = self.calculate_learning_scores(posture, posture_confidence)

        # ----------------------------------------------------
        # 1. Compute 70% Facial Scores from DeepFace
        # ----------------------------------------------------
        facial_scores = {"engaged": 0.0, "bored": 0.0, "confused": 0.0, "frustrated": 0.0}

        if prob_dist:
            for emo_name, prob in prob_dist.items():
                mapping = DEEPFACE_MAPPING.get(emo_name)
                if not mapping:
                    continue
                target = mapping["target"]
                weight = mapping["base_weight"]

                # Resolve neutral ambiguity based on learning statistics
                if emo_name == "neutral":
                    target = "engaged" if learning_scores.get("engaged", 0.0) >= learning_scores.get("bored", 0.0) else "bored"

                if target in facial_scores:
                    facial_scores[target] += prob * weight
        else:
            mapping = DEEPFACE_MAPPING.get(raw_df_emotion, {"target": "engaged", "base_weight": 0.70})
            target = mapping["target"]
            if raw_df_emotion == "neutral":
                target = "engaged" if learning_scores.get("engaged", 0.0) >= learning_scores.get("bored", 0.0) else "bored"
            if target in facial_scores:
                facial_scores[target] = df_conf * mapping["base_weight"]

        # Normalize facial scores
        facial_total = sum(facial_scores.values())
        if facial_total > 0:
            facial_scores = {k: round(v / facial_total, 4) for k, v in facial_scores.items()}
        else:
            facial_scores = {"engaged": 0.25, "bored": 0.25, "confused": 0.25, "frustrated": 0.25}

        # ----------------------------------------------------
        # 2. Compute 70% DeepFace + 30% Learning Combined Scores
        # ----------------------------------------------------
        final_scores = {}
        for state in TARGET_STATES:
            score = 0.70 * facial_scores.get(state, 0.0) + 0.30 * learning_scores.get(state, 0.0)
            final_scores[state] = round(score, 4)

        # Pick winning final emotion among the 4 states
        final_emotion = max(final_scores, key=final_scores.get)
        final_confidence = final_scores[final_emotion]

        return {
            "deepface_raw_emotion": raw_df_emotion,
            "deepface_confidence": round(df_conf, 4),
            "facial_scores": facial_scores,
            "learning_scores": learning_scores,
            "final_scores": final_scores,
            "final_emotion": final_emotion,
            "final_confidence": round(final_confidence, 4),
            "posture": posture,
            "posture_confidence": round(float(posture_confidence), 4),
        }
