import sys
import time
import logging
from pathlib import Path

# Ensure UTF-8 console output for terminal printing on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import Config

from .services.emotion_service import EmotionService
from .services.adaptation_service import AdaptationService

from .models.schemas import (
    EmotionPredictionRequest,
    EmotionPredictionResponse,
    SystemStatus,
    LessonContent,
    HealthCheck,
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


# ============================================================
# SERVICES
# ============================================================

emotion_service = EmotionService()


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title=Config.PROJECT_NAME,
    version=Config.VERSION,
    description="NeuroAdapt - Emotion-Aware Dynamic UI/UX System",
)

# ============================================================
# SECURITY & CORS MIDDLEWARE
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


# ============================================================
# FRONTEND PATHS
# ============================================================

frontend_dir = Path(__file__).parent.parent / "frontend"

templates_dir = frontend_dir / "templates"


# ============================================================
# STATIC FILES
# ============================================================

if frontend_dir.exists():

    app.mount(
        "/static",
        StaticFiles(directory=str(frontend_dir)),
        name="static",
    )


# ============================================================
# TEMPLATES
# ============================================================

templates = Jinja2Templates(
    directory=str(templates_dir)
)


# ============================================================
# ROOT PAGE
# ============================================================

@app.get("/", include_in_schema=False)
async def root(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request
        }
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get(
    "/api/health",
    response_model=SystemStatus
)
async def health_check():

    local_loaded = emotion_service.local_service.model is not None

    return HealthCheck(
        version=Config.VERSION,
        emotion_mode=Config.EMOTION_MODE,
        cloud_api_configured=bool(Config.CLOUD_API_URL),
        local_model_loaded=local_loaded,
    )


# ============================================================
# EMOTION STATUS
# ============================================================

@app.get(
    "/api/emotion/status",
    response_model=SystemStatus
)
async def emotion_status():

    local_loaded = emotion_service.local_service.model is not None

    return SystemStatus(
        status="operational",
        version=Config.VERSION,
        emotion_mode=Config.EMOTION_MODE,
        cloud_api_configured=bool(Config.CLOUD_API_URL),
        local_model_loaded=local_loaded,
    )


# ============================================================
# EMOTION PREDICTION
# ============================================================

@app.post(
    "/api/emotion/predict",
    response_model=EmotionPredictionResponse
)
async def predict_emotion(
    request: EmotionPredictionRequest
):
    try:
        start_time = time.time()

        cv_image = emotion_service.image_service.decode_base64_image(
            request.image
        )

        if cv_image is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image data"
            )

        mode = request.mode or Config.EMOTION_MODE

        result = await emotion_service.predict(
            request.image,
            mode
        )

        processing_time_ms = int(
            (time.time() - start_time) * 1000
        )

        deepface_emotion = result.get("deepface_raw_emotion")
        deepface_conf = float(result.get("deepface_confidence", 0.0))
        learning_scores = result.get("learning_scores", {})
        final_scores = result.get("final_scores", {})
        final_emotion = result.get("final_emotion", "engaged")
        final_conf = float(result.get("final_confidence", 0.0))
        posture = result.get("posture", "upright")
        posture_conf = float(result.get("posture_confidence", 0.80))
        source = result.get("source", "deepface+stats")
        err = result.get("error")

        # Evaluate against centralized 24-combination matrix
        adaptation_data = AdaptationService.evaluate(
            emotion=final_emotion,
            posture=posture,
            confidence=final_conf,
            posture_confidence=posture_conf,
        )

        response = EmotionPredictionResponse(
            emotion=adaptation_data["emotion"],
            posture=adaptation_data["posture"],
            learning_state=adaptation_data["learning_state"],
            confidence=final_conf,
            posture_confidence=posture_conf,
            theme=adaptation_data["theme"],
            background=adaptation_data["background"],
            message=adaptation_data["message"],
            posture_tip=adaptation_data["posture_tip"],
            action=adaptation_data["action"],
            assistance=adaptation_data["assistance"],
            deepface_emotion=deepface_emotion,
            deepface_confidence=deepface_conf,
            learning_scores=learning_scores,
            final_scores=final_scores,
            final_emotion=adaptation_data["learning_state"],
            source=source,
            processing_time_ms=processing_time_ms,
            adaptation=adaptation_data,
            error=err,
        )

        # Clear Terminal Output
        learn_str = " | ".join(f"{k.capitalize()}: {v:.2f}" for k, v in learning_scores.items())
        final_str = " | ".join(f"{k.capitalize()}: {v:.4f}" for k, v in final_scores.items())
        sep = "=" * 65
        print(
            f"\n{sep}"
            f"\n[NeuroAdapt] Emotion & Learning State Evaluation"
            f"\n-----------------------------------------------------------------"
            f"\n  Inference Model   : {source}"\
            f"\n  Detected Emotion  : {deepface_emotion or 'None'} (Confidence: {int(deepface_conf*100)}%)"
            f"\n  Detected Posture  : {posture} (Confidence: {int(posture_conf*100)}%)"
            f"\n  Learning Scores   : {learn_str}"
            f"\n  Final Scores      : {final_str}"
            f"\n  Final Emotion     : {adaptation_data['learning_state']} (Confidence: {int(final_conf*100)}%)"
            f"\n  UI Action / Theme : {adaptation_data['action']} / {adaptation_data['theme']}"
            f"\n  Processing Time   : {processing_time_ms}ms"
            f"\n{sep}\n",
            flush=True
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in emotion prediction")
        raise HTTPException(
            status_code=500,
            detail="Internal error during emotion prediction."
        )


# ============================================================
# BASE64 EMOTION PREDICTION
# ============================================================

@app.post(
    "/api/emotion/predict-base64",
    response_model=EmotionPredictionResponse
)
async def predict_emotion_base64(
    request: EmotionPredictionRequest
):
    try:
        start_time = time.time()

        cv_image = emotion_service.image_service.decode_base64_image(
            request.image
        )

        if cv_image is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image data"
            )

        mode = request.mode or Config.EMOTION_MODE

        result = await emotion_service.predict(
            request.image,
            mode
        )

        processing_time_ms = int(
            (time.time() - start_time) * 1000
        )

        deepface_emotion = result.get("deepface_raw_emotion")
        deepface_conf = float(result.get("deepface_confidence", 0.0))
        learning_scores = result.get("learning_scores", {})
        final_scores = result.get("final_scores", {})
        final_emotion = result.get("final_emotion", "engaged")
        final_conf = float(result.get("final_confidence", 0.0))
        posture = result.get("posture", "upright")
        posture_conf = float(result.get("posture_confidence", 0.80))
        source = result.get("source", "deepface+stats")
        err = result.get("error")

        # Evaluate against centralized 24-combination matrix
        adaptation_data = AdaptationService.evaluate(
            emotion=final_emotion,
            posture=posture,
            confidence=final_conf,
            posture_confidence=posture_conf,
        )

        response = EmotionPredictionResponse(
            emotion=adaptation_data["emotion"],
            posture=adaptation_data["posture"],
            learning_state=adaptation_data["learning_state"],
            confidence=final_conf,
            posture_confidence=posture_conf,
            theme=adaptation_data["theme"],
            background=adaptation_data["background"],
            message=adaptation_data["message"],
            posture_tip=adaptation_data["posture_tip"],
            action=adaptation_data["action"],
            assistance=adaptation_data["assistance"],
            deepface_emotion=deepface_emotion,
            deepface_confidence=deepface_conf,
            learning_scores=learning_scores,
            final_scores=final_scores,
            final_emotion=adaptation_data["learning_state"],
            source=source,
            processing_time_ms=processing_time_ms,
            adaptation=adaptation_data,
            error=err,
        )

        # Clear Terminal Output
        learn_str = " | ".join(f"{k.capitalize()}: {v:.2f}" for k, v in learning_scores.items())
        final_str = " | ".join(f"{k.capitalize()}: {v:.4f}" for k, v in final_scores.items())
        sep = "=" * 65
        print(
            f"\n{sep}"
            f"\n[NeuroAdapt] Emotion & Learning State Evaluation"
            f"\n-----------------------------------------------------------------"
            f"\n  Inference Model   : {source}"\
            f"\n  Detected Emotion  : {deepface_emotion or 'None'} (Confidence: {int(deepface_conf*100)}%)"
            f"\n  Detected Posture  : {posture} (Confidence: {int(posture_conf*100)}%)"
            f"\n  Learning Scores   : {learn_str}"
            f"\n  Final Scores      : {final_str}"
            f"\n  Final Emotion     : {adaptation_data['learning_state']} (Confidence: {int(final_conf*100)}%)"
            f"\n  UI Action / Theme : {adaptation_data['action']} / {adaptation_data['theme']}"
            f"\n  Processing Time   : {processing_time_ms}ms"
            f"\n{sep}\n",
            flush=True
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in base64 emotion prediction")
        raise HTTPException(
            status_code=500,
            detail="Internal error during base64 emotion prediction."
        )


# ============================================================
# CHANGE EMOTION MODE
# ============================================================

from pydantic import BaseModel
from typing import Optional as Opt

class ModeRequest(BaseModel):
    mode: str
    state: Opt[str] = None

@app.post("/api/emotion/mode")
async def set_emotion_mode(
    body: ModeRequest
):
    valid_modes = [
        "cloud",
        "local",
        "auto",
        "manual",
    ]

    if body.mode not in valid_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode. Must be one of: {valid_modes}"
        )

    norm_state = None
    if body.state:
        norm_state = AdaptationService.normalize_emotion(body.state)

    if body.mode != "manual":
        Config.EMOTION_MODE = body.mode

    if body.mode == "local":
        emotion_service.local_service.load_model()

    logger.info(
        "Emotion mode changed to: %s (state=%s)",
        body.mode,
        norm_state,
    )

    return {
        "mode": body.mode,
        "state": norm_state or body.state,
        "message": f"Emotion inference mode set to {body.mode}",
    }


# ============================================================
# LESSON
# ============================================================

@app.get(
    "/api/lesson",
    response_model=LessonContent
)
async def get_lesson():
    return LessonContent(
        id="lesson_1",
        course_name=Config.COURSE_NAME,
        lesson_title=Config.LESSON_TITLE,
        description=Config.LESSON_DESCRIPTION,
        video_url=Config.YOUTUBE_VIDEO_URL,
        content_body=Config.LESSON_DESCRIPTION,
        progress=0,
    )


# ============================================================
# ADAPTATION EVALUATION
# ============================================================

@app.post("/api/adaptation/evaluate")
async def evaluate_adaptation(
    state_data: dict
):
    try:
        emotion = state_data.get("emotion", "engaged")
        posture = state_data.get("posture", "upright")
        confidence = float(state_data.get("confidence", 1.0))
        posture_confidence = float(state_data.get("posture_confidence", 1.0))

        adaptation_data = AdaptationService.evaluate(
            emotion=emotion,
            posture=posture,
            confidence=confidence,
            posture_confidence=posture_confidence,
        )

        return {
            "raw_emotion": adaptation_data["emotion"],
            "posture": adaptation_data["posture"],
            "learning_state": adaptation_data["learning_state"],
            "confidence": confidence,
            "posture_confidence": posture_confidence,
            "theme": adaptation_data["theme"],
            "background": adaptation_data["background"],
            "message": adaptation_data["message"],
            "posture_tip": adaptation_data["posture_tip"],
            "action": adaptation_data["action"],
            "assistance": adaptation_data["assistance"],
            "adaptation": adaptation_data,
        }

    except Exception as e:
        logger.exception("Error in adaptation evaluation")
        raise HTTPException(
            status_code=500,
            detail="Internal error evaluating adaptation."
        )


# ============================================================
# LEARNING PAGE
# ============================================================

@app.get(
    "/learning",
    include_in_schema=False
)
async def learning_page(
    request: Request
):

    return templates.TemplateResponse(
        "learning.html",
        {
            "request": request
        }
    )


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
