import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

class Config:
    # Project Info
    PROJECT_NAME = os.getenv("PROJECT_NAME", "NeuroAdapt")
    VERSION = os.getenv("VERSION", "1.0.0")
    DEBUG = os.getenv("DEBUG_MODE", "false").lower() == "true"

    # Emotion Service Mode
    # Possible values: "cloud", "local", "auto"
    EMOTION_MODE = os.getenv("EMOTION_MODE", "auto")

    # --------------------------------------------------------
    # Roboflow Posture Detection API (cloud service)
    # --------------------------------------------------------
    ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "") or os.getenv("EMOTION_API_KEY", "")
    ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "")
    ROBOFLOW_WORKFLOW_ID = os.getenv(
        "ROBOFLOW_WORKFLOW_ID",
        "sitting-posture-classification-6vwq1-kbzhk"
    )
    ROBOFLOW_API_URL = (
        f"https://serverless.roboflow.com/infer/workflows/"
        f"{ROBOFLOW_WORKSPACE}/"
        f"{ROBOFLOW_WORKFLOW_ID}"
    )

    # Legacy cloud API (kept for compatibility, unused when Roboflow is active)
    CLOUD_API_URL = os.getenv("EMOTION_API_URL", "")
    CLOUD_API_KEY = os.getenv("EMOTION_API_KEY", "")
    CLOUD_API_METHOD = os.getenv("EMOTION_API_METHOD", "POST")
    CLOUD_API_IMAGE_FIELD = os.getenv("EMOTION_API_IMAGE_FIELD", "image")

    # Local Model Configuration
    LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "emotion_cnn.pkl")

    # Processing Settings
    PROCESSING_INTERVAL_MS = int(os.getenv("PROCESSING_INTERVAL_MS", 5000))
    MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", 0.35))
    STABLE_PREDICTIONS_REQUIRED = int(os.getenv("STABLE_PREDICTIONS_REQUIRED", 3))
    TEMPORAL_WINDOW_SIZE = int(os.getenv("TEMPORAL_WINDOW_SIZE", 5))

    # Supported Emotions (strictly limited to 4 CNN classes)
    ALLOWED_EMOTIONS = ["engaged", "bored", "confused", "frustrated"]

    # Supported Posture Classes from Roboflow Posture Model
    ALLOWED_POSTURES = [
        "upright",
        "leaning_forward",
        "leaning_backward",
        "slouched",
        "leaning_left",
        "leaning_right",
    ]

    # Canonical Learning States
    LEARNING_STATES = [
        "ENGAGED",
        "BORED",
        "CONFUSED",
        "FRUSTRATED",
    ]

    # Course Metadata (Configurable)
    COURSE_NAME = "Adaptive Learning Dashboard"
    LESSON_TITLE = "Interactive Study Session"
    LESSON_DESCRIPTION = "Real-time emotion-aware adaptive learning environment."
    YOUTUBE_VIDEO_URL = "https://www.youtube.com/embed/HcqpanDadyQ"
