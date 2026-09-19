# 🏃 KhelDrishti (खेल दृष्टि)

Edge AI sports biomechanics coaching and movement screening for athletes. Uses computer vision (MediaPipe) and geometric kinematics to analyze technique, detect injury risks (e.g., knee valgus, stiff landing), and provide real-time feedback with bilingual coaching cues (English + Hindi/Hinglish).

---

##  Features

- **Live Camera Coaching**: Real-time pose tracking via WebSockets, live joint angle dials, dynamic knee valgus alerts, and Web Speech API audio coaching (EN/HI).
- **Video Upload Analysis**: Automated phase segmentation, kinematic metrics extraction, and annotated HUD video generation with neon skeleton overlays.
- **Form Scoring & Injury Risk**: Automated 0–100 athletic form score, bilateral limb symmetry index, and risk level classification (Green/Amber/Red).
- **Supported Movements**:
  - **Vertical Jump (CMJ)**: Jump height, flight time, braking/landing flexion, dynamic valgus.
  - **Volleyball Spike**: Kinetic chain sequencing, arm cocking angle, peak wrist velocity.
  - **Running Gait**: Cadence, overstride ratio, trunk lean, bilateral symmetry.
  - **Squat**: Depth classification, knee tracking, lumbo-pelvic tilt ("butt wink"), tempo.

---

##  Prerequisites

- **Python 3.11** (recommended for MediaPipe compatibility)
- [`uv`](https://docs.astral.sh/uv/) (recommended) or standard `pip`
- Webcam (optional, only needed for live camera mode)

---

##  Quickstart

### 1. Set Up the Environment

**Using `uv` (Fastest):**
```powershell
# Windows (PowerShell)
uv venv venv --python 3.11
.\venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

```bash
# macOS / Linux
uv venv venv --python 3.11
source venv/bin/activate
uv pip install -r requirements.txt
```

*Or with standard `venv`:*
```bash
python -m venv venv
# Activate (Windows: .\venv\Scripts\Activate.ps1 | Unix: source venv/bin/activate)
pip install -r requirements.txt
```

### 2. Generate Sample Videos (Optional)
Generate synthetic smoke-test clips to test the pipeline without a camera or athlete recording:
```bash
python generate_sample_data.py
```

### 3. Run the Server
```bash
uvicorn app.main:app --reload --port 8000
```
Open **http://localhost:8000** in your browser.

### 4. Run Tests
```bash
pytest tests/
```

---

##  Project Structure

```text
kheldrishti/
├── app/                  # FastAPI web server, routes & pipeline orchestration
│   ├── main.py           # REST endpoints & WebSocket handler (/ws/live-stream)
│   ├── models.py         # Pydantic schemas
│   └── pipeline.py       # Video processing pipeline
├── engine/               # Core biomechanics logic
│   ├── kinematics.py     # Pure vector math (angles, valgus, jump physics)
│   ├── pose_detector.py  # MediaPipe BlazePose wrapper & EMA smoothing
│   ├── feedback_coach.py # Scoring, risk detection & bilingual coaching cues
│   ├── video_annotator.py# OpenCV HUD overlay renderer
│   └── movement_analyzers/
│       ├── vertical_jump.py
│       ├── volleyball_spike.py
│       ├── running_gait.py
│       └── squat_analysis.py
├── static/               # Frontend (HTML, CSS, JS)
├── sample_videos/        # Generated sample clips
├── tests/                # Kinematics & unit tests
└── generate_sample_data.py
```

---

##  API Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Web dashboard UI |
| `GET` | `/api/health` | Health check & model status |
| `GET` | `/api/movements` | List supported movements & benchmarks |
| `POST` | `/api/analyze-video` | Upload and analyze a video file (`multipart/form-data`) |
| `POST` | `/api/analyze-sample`| Analyze a sample clip by `sample_id` |
| `WS` | `/ws/live-stream` | Bidirectional real-time camera stream |

---

##  Notes

- **2D Screening**: Measurements (valgus, torso lean, overstride) are 2D single-camera proxies intended for coaching screening, not clinical diagnosis.
- **Camera Setup**: For best results, place the camera 2–3 meters away. Use front-on view for jump landings and squats; side-on view for running gait.

---

##  License

MIT
