# 🎓 NeuroAdapt - Emotion-Aware & Posture-Adaptive Learning Platform

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8.svg?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**NeuroAdapt** is an intelligent e-learning system that detects student posture and emotional engagement in real time using computer vision. By continuously analyzing webcam frames, NeuroAdapt dynamically adapts the learning interface (visual styling, ergonomic notifications, assistance density, and focus mode) to optimize focus, reduce fatigue, and support individualized learning.

---

## 🌟 Key Features

- **Real-Time Webcam Inference**: Captures video frames from the browser and securely streams them to the FastAPI backend for analysis.
- **Hybrid Inference Architecture**:
  - **Cloud Mode (Roboflow Workflow)**: State-of-the-art sitting posture classification (`upright`, `leaning_forward`, `leaning_backward`, `leaning_left`, `leaning_right`).
  - **Local Model Mode (TensorFlow/CNN)**: Fallback or offline inference using a local serialized model (`emotion_cnn.pkl`).
  - **Auto Mode**: Automatically leverages cloud capabilities with seamless fallback to local models.
  - **Manual Override**: Quick-switch interface theme dropdown for testing or individual accessibility preferences.
- **Dynamic UI/UX Adaptation**:
  - **FOCUSED**: Clean, distraction-free environment and posture encouragement.
  - **CONFUSED**: Clear explanations, simplified view options, and ergonomic screen distance alerts.
  - **TIRED / SLOUCHING**: Gentle posture correction tips, calming color palette, and optional break reminders.
  - **STRESSED**: Calming themes, reduced information density, and self-paced review options.
- **Customizable Video Learning**: Embedded video player supporting custom YouTube URLs for personalized study sessions.
- **Focus Mode**: Ambient dark-mode interface with subtle ambient lighting that reflects learner state.
- **Interactive API Documentation**: Auto-generated Swagger UI and ReDoc endpoints powered by FastAPI.

---

## 🏗️ System Architecture

```text
       ┌────────────────┐
       │     Webcam     │
       └───────┬────────┘
               │ (Base64 JPEG Frames via fetch)
               ▼
   ┌───────────────────────┐
   │  NeuroAdapt Frontend   │ (Vanilla HTML5 / CSS3 / ES6 JS)
   │  - Video Player       │
   │  - Detection Monitor  │
   │  - Dynamic Theme Engine
   └───────────┬───────────┘
               │ HTTP POST /api/emotion/predict-base64
               ▼
   ┌───────────────────────┐
   │    FastAPI Backend    │ (backend/main.py)
   │  - Base64 Decoding    │
   │  - OpenCV Processing  │
   └───────────┬───────────┘
               ├───► [Cloud Service] Roboflow Sitting Posture Classification
               └───► [Local Fallback] CNN Model (backend/models/emotion_cnn.pkl)
                           │
                           ▼
               ┌───────────────────────┐
               │   Adaptation Engine   │ (AdaptationService)
               │ - State Mapping       │
               │ - UI Adaptation Diffs │
               └───────────┬───────────┘
                           │ (JSON Response: State, Confidence, UI Actions)
                           ▼
               Frontend Updates Dynamically
```

---

## 📁 Project Structure

```text
neuroadapt/
├── backend/
│   ├── models/
│   │   ├── emotion_cnn.pkl       # Pretrained local CNN model
│   │   └── schemas.py            # Pydantic data validation models
│   ├── services/
│   │   ├── adaptation_service.py # Maps emotions/postures to UI adaptations
│   │   ├── cloud_emotion_service.py # Roboflow API integration
│   │   ├── emotion_service.py    # Main service orchestrator
│   │   ├── image_service.py      # Base64 & OpenCV image operations
│   │   └── local_emotion_service.py # Local CNN loader & inference
│   ├── config.py                 # Application settings & posture mappings
│   └── main.py                   # FastAPI application & REST routing
├── frontend/
│   ├── css/
│   │   ├── home.css              # Landing page styling
│   │   └── learning.css          # Dashboard and adaptive UI styling
│   ├── js/
│   │   ├── adaptation.js         # Frontend adaptation and DOM manipulation
│   │   ├── app.js                # Core frontend utility logic
│   │   ├── emotion.js            # Video capture and detection loop
│   │   ├── learning.js           # Dashboard page orchestrator
│   │   └── team.js               # Team presentation logic
│   └── templates/
│       ├── index.html            # Landing / overview page
│       └── learning.html         # Interactive adaptive learning dashboard
├── .env.example                  # Environment configuration template
├── .gitignore                    # Git exclusion rules
├── requirements.txt              # Python dependencies
├── run.py                        # Server runner entrypoint
└── README.md                     # Project documentation
```

---

## 🚀 Getting Started

Follow these steps to set up and run NeuroAdapt locally on your machine.

### 1. Prerequisites

- **Python**: Version 3.10 or 3.11 recommended
- **Git**: For version control
- **Webcam**: A built-in or USB webcam for real-time posture detection
- **Web Browser**: Chrome, Edge, Firefox, or Brave with camera permissions enabled

---

### 2. Clone the Repository

```bash
git clone <your-repository-url>
cd neuroadapt
```

---

### 3. Set Up a Virtual Environment

It is strongly recommended to use a virtual environment to manage dependencies:

#### On Windows (PowerShell / Command Prompt):
```powershell
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\activate
```

#### On macOS / Linux:
```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

---

### 4. Install Dependencies

Upgrade `pip` and install all required packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

### 5. Configure Environment Variables

1. Copy the `.env.example` file to create your own `.env` configuration:

   ```bash
   # On Windows (PowerShell):
   Copy-Item .env.example .env

   # On macOS / Linux:
   cp .env.example .env
   ```

2. Open `.env` in any editor and adjust the settings if needed:

   ```env
   # Inference Mode: auto, cloud, local
   EMOTION_MODE="auto"

   # Roboflow Sitting Posture Classification (Cloud)
   ROBOFLOW_API_KEY="your_roboflow_api_key"
   ROBOFLOW_WORKSPACE="your_workspace_id"
   ROBOFLOW_WORKFLOW_ID="sitting-posture-classification-6vwq1-kbzhk"

   # Processing Settings
   PROCESSING_INTERVAL_MS=1500
   MIN_CONFIDENCE_THRESHOLD=0.40
   STABLE_PREDICTIONS_REQUIRED=3
   TEMPORAL_WINDOW_SIZE=5
   ```

> [!NOTE]
> Add your Roboflow credentials in `.env` to enable cloud posture classification. If you prefer to test purely offline, set `EMOTION_MODE="local"` or use the **Manual Theme Selector** on the dashboard.

---

### 6. Run the Application

You can launch the server using either the runner script or `uvicorn`:

#### Option A: Using `run.py` (Recommended)
```bash
python run.py
```

#### Option B: Using `uvicorn` directly
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

---

### 7. Open in Your Browser

Once the server is running, navigate to:

- 🏠 **Home Page**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- 📊 **Learning Dashboard**: [http://127.0.0.1:8000/learning](http://127.0.0.1:8000/learning)
- 📖 **Interactive API Docs (Swagger UI)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- 📚 **Alternative API Docs (ReDoc)**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 🖥️ How to Use the Dashboard

1. Open [http://127.0.0.1:8000/learning](http://127.0.0.1:8000/learning).
2. **Start Detection**: In the **Detection Monitor** on the right sidebar, click **"START AUTO DETECTION"** and allow browser camera permissions.
3. **Posture Tracking**: Sit in front of your camera. As you change posture (e.g., sit upright, slouch back, or lean into the screen), watch the monitor display your current posture, confidence score, and corresponding learning state.
4. **Adaptive Response**:
   - Slouching or leaning sideways triggers a posture correction alert.
   - Leaning forward suggests adjusting screen distance.
   - Sitting upright rewards you with a positive focus status.
5. **Custom Videos**: Paste any valid YouTube video link into the URL input bar and click **"Load Video"** to study your preferred content.
6. **Focus Mode**: Click the **"Focus Mode"** button in the navigation bar to toggle a distraction-free dark study interface.

---

## 📡 API Endpoints Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Renders the NeuroAdapt landing page |
| `GET` | `/learning` | Renders the adaptive learning dashboard |
| `GET` | `/api/health` | Service health status and model load status |
| `GET` | `/api/emotion/status` | Operational status of emotion/posture services |
| `POST` | `/api/emotion/predict-base64` | Submits a Base64-encoded frame and returns posture, state, and UI adaptations |
| `POST` | `/api/emotion/mode` | Switches mode between `auto`, `cloud`, `local`, and `manual` |
| `GET` | `/api/lesson` | Retrieves current lesson and course metadata |
| `POST` | `/api/adaptation/evaluate` | Evaluates multi-state combinations and returns adaptation directives |

---

## 🛠️ Troubleshooting

- **Camera Not Starting**: Ensure camera access is allowed in your browser settings for `localhost` / `127.0.0.1`.
- **Port Conflict (Port 8000 in use)**: Run on another port using:
  ```bash
  uvicorn backend.main:app --host 127.0.0.1 --port 8080 --reload
  ```
- **TensorFlow / Local Model Issues**: If running without a GPU or encountering TensorFlow build warnings, NeuroAdapt will run in Cloud or Auto mode using the Roboflow endpoint seamlessly.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
