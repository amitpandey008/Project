import time
import asyncio
import logging
from .cloud_emotion_service import CloudEmotionService
from .local_emotion_service import LocalEmotionService
from .deepface_service import DeepFaceEmotionService
from .image_service import ImageService
from .adaptation_service import AdaptationService
from ..config import Config

logger = logging.getLogger(__name__)

class EmotionService:
    def __init__(self):
        self.cloud_service = CloudEmotionService()
        self.local_service = LocalEmotionService()
        self.deepface_service = DeepFaceEmotionService()
        self.image_service = ImageService()

    async def predict(self, base64_image: str, mode: str = None):
        """
        Runs Emotion Detection (Local CNN or DeepFace) + Posture detector concurrently.
        Computes 70% Facial Emotion + 30% Learning Statistics combined scores.
        Supports modes:
          - "local": Runs the local CNN model (emotion_cnn.pkl)
          - "cloud": Runs Python DeepFace emotion analyzer
          - "auto" : Runs DeepFace with seamless fallback to Local CNN
        """
        cv_image = self.image_service.decode_base64_image(base64_image)

        start_time = time.time()

        # 1. Posture task (Cloud API)
        posture_task = self.cloud_service.predict(base64_image)

        # 2. Concurrently get cloud posture while decoding
        cloud_res = await posture_task

        # Parse cloud posture
        posture = "upright"
        posture_confidence = 0.80

        if cloud_res:
            cloud_data, _ = cloud_res
            if cloud_data and "error" not in cloud_data:
                p_label, p_conf = self.cloud_service.normalize_response(cloud_data)
                if p_label:
                    posture = AdaptationService.normalize_posture(p_label)
                    posture_confidence = float(p_conf)

        # Determine effective mode
        effective_mode = (mode or Config.EMOTION_MODE or "auto").lower().strip()

        # 3. Run Emotion Evaluation in worker thread
        def run_emotion_evaluation():
            if effective_mode == "local":
                res = self.local_service.evaluate_combined(
                    cv_image=cv_image,
                    posture=posture,
                    posture_confidence=posture_confidence,
                )
                res["source"] = "local_cnn+stats"
                return res

            elif effective_mode == "cloud":
                try:
                    res = self.deepface_service.evaluate_combined(
                        cv_image=cv_image,
                        posture=posture,
                        posture_confidence=posture_confidence,
                    )
                    if res.get("error"):
                        raise RuntimeError(res["error"])
                    res["source"] = "deepface+stats"
                    return res
                except Exception as e:
                    logger.warning(f"[EmotionService] Cloud emotion note ({e}), falling back to Local CNN")
                    res = self.local_service.evaluate_combined(
                        cv_image=cv_image,
                        posture=posture,
                        posture_confidence=posture_confidence,
                    )
                    res["source"] = "local_cnn+stats (fallback)"
                    return res

            else:  # "auto" mode
                try:
                    res = self.deepface_service.evaluate_combined(
                        cv_image=cv_image,
                        posture=posture,
                        posture_confidence=posture_confidence,
                    )
                    if res.get("error"):
                        raise RuntimeError(res["error"])
                    res["source"] = "deepface+stats"
                    return res
                except Exception as e:
                    logger.info(f"[EmotionService] DeepFace auto note ({e}), falling back to Local CNN")
                    res = self.local_service.evaluate_combined(
                        cv_image=cv_image,
                        posture=posture,
                        posture_confidence=posture_confidence,
                    )
                    res["source"] = "local_cnn+stats"
                    return res

        eval_result = await asyncio.to_thread(run_emotion_evaluation)

        eval_result["processing_time_ms"] = int((time.time() - start_time) * 1000)
        eval_result["error"] = None

        return eval_result
