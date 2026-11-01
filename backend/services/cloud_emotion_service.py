import asyncio
import httpx
import time
from ..config import Config


class CloudEmotionService:
    """
    Calls the Roboflow Sitting Posture Classification workflow.

    Workflow: sitting-posture-classification-6vwq1-kbzhk
    Classes : upright | leaning_forward | leaning_backward |
              leaning_left | leaning_right

    Always returns exactly (data_dict, processing_time_ms).
      - data_dict has {"emotion": str, "confidence": float} on success
      - data_dict has {"error": str} on failure
    """

    async def predict(self, base64_image: str):
        """
        Send image to Roboflow posture workflow and return the
        posture class with the highest confidence.

        Returns (data_dict, processing_time_ms).
        """
        if not Config.ROBOFLOW_API_KEY:
            return {"error": "Roboflow API key not configured"}, 0

        try:
            # ------------------------------------------------
            # Strip data URI header if present
            # (frontend sends  "data:image/jpeg;base64,<data>")
            # ------------------------------------------------
            if "," in base64_image:
                b64_value = base64_image.split(",")[1]
            else:
                b64_value = base64_image

            headers = {
                "Authorization": f"Bearer {Config.ROBOFLOW_API_KEY}",
                "Content-Type": "application/json",
            }

            payload = {
                "inputs": {
                    "image": {
                        "type": "base64",
                        "value": b64_value,
                    }
                }
            }

            start_time = time.time()

            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    Config.ROBOFLOW_API_URL,
                    json=payload,
                    headers=headers,
                )

            processing_time = int((time.time() - start_time) * 1000)

            if response.status_code != 200:
                return {
                    "error": f"Roboflow API error {response.status_code}: {response.text[:200]}"
                }, processing_time

            # ------------------------------------------------
            # Parse response
            # Structure:
            #   outputs[0].predictions.predictions
            #     = { class_name: {confidence, class_id}, ... }
            # ------------------------------------------------
            data = response.json()
            result = self._parse_posture_response(data)
            return result, processing_time

        except asyncio.CancelledError:
            # Re-raise so the server can handle graceful shutdown
            raise
        except Exception as e:
            return {"error": str(e)}, 0

    # ----------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------

    @staticmethod
    def _parse_posture_response(data: dict) -> dict:
        """
        Extract the posture class with max confidence from the
        Roboflow workflow response.

        Expected shape:
          data["outputs"][0]["predictions"]["predictions"]
            = { "upright": {"confidence": 0.57, ...}, ... }
        """
        try:
            preds_block = (
                data["outputs"][0]["predictions"]["predictions"]
            )

            if not preds_block:
                return {"error": "Empty predictions block"}

            # Find class with highest confidence
            top_class = max(
                preds_block,
                key=lambda k: preds_block[k]["confidence"]
            )
            top_confidence = float(preds_block[top_class]["confidence"])

            return {
                "posture": top_class,
                "confidence": top_confidence,
            }

        except (KeyError, IndexError, TypeError) as e:
            return {"error": f"Response parse error: {e}"}

    @staticmethod
    def normalize_response(data: dict):
        """
        Return (posture, confidence) from a successfully parsed dict.
        Returns (None, 0.0) on error.
        """
        if not data or "error" in data:
            return None, 0.0
        return data.get("posture"), float(data.get("confidence", 0.0))
