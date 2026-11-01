import pickle
import numpy as np
import os
import logging
from typing import Dict, Any, Tuple
from .image_service import ImageService
from ..config import Config

logger = logging.getLogger(__name__)

class LocalEmotionService:
    """
    Local CNN Emotion Classifier powered by TensorFlow/Keras model (emotion_cnn.pkl).
    Classifies learner facial expressions into the 4 canonical states:
    engaged, bored, confused, frustrated.
    """

    def __init__(self):
        self.model = None
        self.classes = ["engaged", "bored", "confused", "frustrated"]
        self.face_cascade = None
        self._init_face_detector()
        self.load_model()

    def _init_face_detector(self):
        """Initializes OpenCV Haar cascade face detector for region cropping."""
        try:
            import cv2
            cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                logger.info("[LocalEmotionService] Haar cascade face detector loaded.")
        except Exception as e:
            logger.debug(f"[LocalEmotionService] Face detector note: {e}")

    def load_model(self):
        """Loads the local serialized emotion classification model."""
        try:
            model_path = Config.LOCAL_MODEL_PATH
            if os.path.exists(model_path):
                with open(model_path, "rb") as file:
                    self.model = pickle.load(file)
                logger.info(f"[LocalEmotionService] Local model loaded successfully from {model_path}")
            else:
                logger.warning(f"[LocalEmotionService] Local model file not found at {model_path}")
        except Exception as e:
            logger.error(f"[LocalEmotionService] Error loading local model: {e}")

    def extract_face(self, cv_image: np.ndarray) -> np.ndarray:
        """
        Detects and extracts the primary face region from the webcam frame.
        Adds padding to provide appropriate context for the CNN.
        Falls back to full image if no face is detected.
        """
        if cv_image is None or cv_image.size == 0:
            return cv_image

        try:
            import cv2
            if self.face_cascade is not None and not self.face_cascade.empty():
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(40, 40),
                )
                if len(faces) > 0:
                    # Select largest face in frame
                    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                    x, y, w, h = faces[0]
                    # Add 15% margin around face
                    pad_x = int(0.15 * w)
                    pad_y = int(0.15 * h)
                    x1 = max(0, x - pad_x)
                    y1 = max(0, y - pad_y)
                    x2 = min(cv_image.shape[1], x + w + pad_x)
                    y2 = min(cv_image.shape[0], y + h + pad_y)
                    return cv_image[y1:y2, x1:x2]
        except Exception as e:
            logger.debug(f"[LocalEmotionService] Face extraction note: {e}")

        return cv_image

    def predict_with_distribution(self, image: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Performs inference on the given image frame.
        Returns: (emotion, confidence, prob_dist)
        """
        if self.model is None:
            self.load_model()
            if self.model is None:
                return "engaged", 0.50, {"engaged": 0.50, "bored": 0.25, "confused": 0.15, "frustrated": 0.10}

        try:
            # Crop to face region if detected
            face_img = self.extract_face(image)

            # Preprocess for the 96x96 RGB CNN input
            preprocessed_img = ImageService.preprocess_for_local_model(face_img)
            if preprocessed_img is None:
                return "engaged", 0.50, {"engaged": 0.50, "bored": 0.25, "confused": 0.15, "frustrated": 0.10}

            input_data = np.expand_dims(preprocessed_img, axis=0)

            # Run inference
            raw_preds = self.model.predict(input_data, verbose=0)[0]

            # Normalize probabilities
            prob_sum = float(np.sum(raw_preds))
            if prob_sum > 0:
                prob_dist = {
                    self.classes[i]: round(float(raw_preds[i]) / prob_sum, 4)
                    for i in range(len(self.classes))
                }
            else:
                prob_dist = {c: 0.25 for c in self.classes}

            index = int(np.argmax(raw_preds))
            emotion = self.classes[index]
            confidence = prob_dist[emotion]

            return emotion, confidence, prob_dist

        except Exception as e:
            logger.error(f"[LocalEmotionService] Inference error: {e}")
            return "engaged", 0.50, {"engaged": 0.50, "bored": 0.25, "confused": 0.15, "frustrated": 0.10}

    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Backwards-compatible prediction method.
        Returns: (emotion, confidence)
        """
        emotion, confidence, _ = self.predict_with_distribution(image)
        return emotion, confidence

    def calculate_learning_scores(self, posture: str, posture_confidence: float = 1.0) -> Dict[str, float]:
        """
        Calculates learning statistics score vector across the 4 target states (30% weight)
        based on sitting posture and study engagement priors.
        """
        posture = (posture or "upright").lower().strip()
        conf = max(0.2, min(1.0, float(posture_confidence or 0.8)))

        if posture == "upright":
            scores = {"engaged": 0.85 * conf, "bored": 0.10, "confused": 0.03, "frustrated": 0.02}
        elif posture in ("leaning_forward", "lean_forward"):
            scores = {"engaged": 0.50 * conf, "confused": 0.40 * conf, "bored": 0.05, "frustrated": 0.05}
        elif posture in ("leaning_backward", "lean_backward"):
            scores = {"engaged": 0.35 * conf, "bored": 0.50 * conf, "confused": 0.10, "frustrated": 0.05}
        elif posture in ("slouched", "slouch", "hunched"):
            scores = {"bored": 0.70 * conf, "frustrated": 0.20 * conf, "engaged": 0.05, "confused": 0.05}
        elif posture in ("leaning_left", "leaning_right"):
            scores = {"bored": 0.50 * conf, "confused": 0.30 * conf, "frustrated": 0.10, "engaged": 0.10}
        else:
            scores = {"engaged": 0.50, "bored": 0.30, "confused": 0.10, "frustrated": 0.10}

        total = sum(scores.values())
        return {k: round(v / total, 4) for k, v in scores.items()}

    def evaluate_combined(
        self,
        cv_image: np.ndarray,
        posture: str = "upright",
        posture_confidence: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Full multimodal evaluation pipeline using Local CNN:
        1. 70% Local CNN facial emotion scoring.
        2. 30% Sitting posture & learning statistics prior scoring.
        3. Fused 4-state output: engaged, bored, confused, frustrated.
        """
        local_emotion, local_conf, facial_scores = self.predict_with_distribution(cv_image)
        learning_scores = self.calculate_learning_scores(posture, posture_confidence)

        final_scores = {}
        for state in self.classes:
            score = 0.70 * facial_scores.get(state, 0.0) + 0.30 * learning_scores.get(state, 0.0)
            final_scores[state] = round(score, 4)

        final_emotion = max(final_scores, key=final_scores.get)
        final_confidence = final_scores[final_emotion]

        return {
            "deepface_raw_emotion": local_emotion,
            "deepface_confidence": round(local_conf, 4),
            "facial_scores": facial_scores,
            "learning_scores": learning_scores,
            "final_scores": final_scores,
            "final_emotion": final_emotion,
            "final_confidence": round(final_confidence, 4),
            "posture": posture,
            "posture_confidence": round(float(posture_confidence), 4),
            "source": "local_cnn+stats",
        }
