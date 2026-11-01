from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class EmotionPredictionRequest(BaseModel):
    image: str = Field(
        ...,
        max_length=15_000_000,
        description="Base64 encoded JPEG/PNG image (max 15MB string)"
    )

    mode: Optional[str] = Field(
        None,
        pattern=r"^(cloud|local|auto|manual)$",
        description="Inference mode: cloud, local, auto, manual"
    )


class EmotionPredictionResponse(BaseModel):
    emotion: str = Field(..., description="Facial emotion: engaged, bored, confused, frustrated")
    posture: str = Field(..., description="Detected posture: upright, leaning_forward, leaning_backward, slouched, leaning_left, leaning_right")
    learning_state: str = Field(..., description="Canonical learning state: ENGAGED, BORED, CONFUSED, FRUSTRATED")
    confidence: float = Field(..., description="Facial emotion confidence (0.0 to 1.0)")
    posture_confidence: float = Field(..., description="Posture confidence (0.0 to 1.0)")
    theme: str = Field(..., description="Theme identifier (e.g. positive, stimulate, help, calm)")
    background: str = Field(..., description="CSS background token / class")
    message: str = Field(..., description="Adaptive learner feedback message")
    posture_tip: str = Field(..., description="Ergonomic / posture guidance tip")
    action: str = Field(..., description="Recommended adaptation action")
    assistance: str = Field(..., description="Assistance intensity level: none, low, medium, high")
    # DeepFace and scoring payload fields
    deepface_emotion: Optional[str] = Field(None, description="Dominant raw emotion detected by DeepFace")
    deepface_confidence: float = Field(0.0, description="DeepFace detection confidence (0.0 to 1.0)")
    learning_scores: Dict[str, float] = Field(default_factory=dict, description="Learning/posture statistics score vector (30% weight)")
    final_scores: Dict[str, float] = Field(default_factory=dict, description="Combined 70% DeepFace + 30% Learning score vector")
    final_emotion: str = Field(..., description="Final winning emotion state")
    source: str = Field("deepface+stats", description="Prediction source")
    processing_time_ms: int = Field(0, description="Inference latency in milliseconds")
    adaptation: Dict[str, Any] = Field(default_factory=dict, description="Complete adaptation payload")
    error: Optional[str] = None


class SystemStatus(BaseModel):
    status: str
    version: str
    emotion_mode: str
    cloud_api_configured: bool
    local_model_loaded: bool


class LessonContent(BaseModel):
    id: str
    course_name: str
    lesson_title: str
    description: str
    video_url: str
    content_body: str
    progress: int


# --------------------------------------------------
# Health Check
# --------------------------------------------------

def HealthCheck(
    version: str,
    emotion_mode: str,
    cloud_api_configured: bool,
    local_model_loaded: bool
) -> SystemStatus:
    """
    Returns the current health/status of the NeuroAdapt system.
    """

    # Determine overall system status
    if local_model_loaded or cloud_api_configured:
        status = "healthy"
    else:
        status = "degraded"

    return SystemStatus(
        status=status,
        version=version,
        emotion_mode=emotion_mode,
        cloud_api_configured=cloud_api_configured,
        local_model_loaded=local_model_loaded
    )
