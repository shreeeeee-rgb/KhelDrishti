"""
KhelDrishti — FastAPI server
Serves the vanilla web app, video analysis, sample pipeline, and live WebSocket.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import platform
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from engine.pipeline import AnalysisPipeline, MOVEMENT_BENCHMARKS, live_joint_metrics
from engine.pose_detector import PoseDetector, _mp_available
from engine.feedback_coach import FeedbackCoach
from app.models import HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kheldrishti")

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"
OUTPUTS = ROOT / "outputs"
SAMPLES = ROOT / "sample_data" / "videos"
OUTPUTS.mkdir(exist_ok=True)
SAMPLES.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="KhelDrishti", version="1.0.0", description="Edge AI Sports Biomechanics Coach")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS)), name="outputs")

pipeline = AnalysisPipeline()
coach = FeedbackCoach()


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health", response_model=HealthResponse)
async def health():
    device = "cpu"
    try:
        import cv2 as _cv
        device = f"cpu ({platform.processor() or platform.machine()})"
        _ = _cv.__version__
    except Exception:
        pass
    return HealthResponse(
        status="ok",
        device=device,
        mediapipe=bool(_mp_available),
    )


@app.get("/api/movements")
async def movements():
    return {"movements": MOVEMENT_BENCHMARKS}


@app.get("/api/sample-videos")
async def sample_videos():
    items = []
    for p in sorted(SAMPLES.glob("*.mp4")):
        items.append({
            "id": p.stem,
            "name": p.stem.replace("_", " ").title(),
            "url": f"/sample-data/{p.name}",
            "path": str(p),
        })
    return {"samples": items}


@app.get("/sample-data/{name}")
async def serve_sample(name: str):
    path = SAMPLES / name
    if not path.exists():
        raise HTTPException(404, "Sample not found")
    return FileResponse(path)


def _run_analysis(video_path: str, movement: str, language: str) -> dict:
    stem = Path(video_path).stem
    landmark_json = SAMPLES / f"{stem}.landmarks.json"
    if not landmark_json.exists():
        # also try movement-named companion
        landmark_json = SAMPLES / f"{movement}.landmarks.json"
    lj = str(landmark_json) if landmark_json.exists() else None
    result = pipeline.analyse_video(
        video_path, movement=movement, language=language, out_dir=str(OUTPUTS), landmark_json=lj
    )
    if result.get("annotated_video"):
        rel = Path(result["annotated_video"]).name
        result["annotated_video_url"] = f"/outputs/{rel}"
    return result


@app.post("/api/analyze-video")
async def analyze_video(
    file: UploadFile = File(...),
    movement: str = Form("vertical_jump"),
    language: str = Form("en"),
):
    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
    tmp = OUTPUTS / f"upload_{uuid.uuid4().hex}{suffix}"
    content = await file.read()
    tmp.write_bytes(content)
    try:
        result = await asyncio.to_thread(_run_analysis, str(tmp), movement, language)
        return JSONResponse(result)
    except Exception as exc:
        logger.exception("analyse-video failed")
        raise HTTPException(500, str(exc)) from exc


@app.post("/api/analyze-sample")
async def analyze_sample(
    sample_id: str = Form("vertical_jump_cmj"),
    movement: str = Form("vertical_jump"),
    language: str = Form("en"),
):
    path = SAMPLES / f"{sample_id}.mp4"
    if not path.exists():
        # first mp4
        videos = list(SAMPLES.glob("*.mp4"))
        if not videos:
            raise HTTPException(404, "No sample videos. Run sample_data/generate_sample_data.py")
        path = videos[0]
    result = await asyncio.to_thread(_run_analysis, str(path), movement, language)
    result["source"] = path.name
    return JSONResponse(result)


@app.websocket("/ws/live-stream")
async def live_stream(ws: WebSocket):
    await ws.accept()
    detector = PoseDetector(static_image_mode=False, model_complexity=0, smooth_alpha=0.5)
    language = "en"
    movement = "vertical_jump"
    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("type") == "config":
                language = msg.get("language", language)
                movement = msg.get("movement", movement)
                await ws.send_json({"type": "config_ok", "language": language, "movement": movement})
                continue
            b64 = msg.get("image") or msg.get("frame")
            if not b64:
                continue
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            raw = base64.b64decode(b64)
            arr = np.frombuffer(raw, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                await ws.send_json({"type": "error", "message": "decode_failed"})
                continue
            pose = detector.process_frame(frame, 0)
            metrics = live_joint_metrics(pose)
            metrics["type"] = "telemetry"
            metrics["movement"] = movement
            cue = coach.live_cue(metrics, language=language)
            metrics["voice_cue"] = cue
            await ws.send_json(metrics)
    except WebSocketDisconnect:
        logger.info("live-stream disconnected")
    except Exception:
        logger.exception("live-stream error")
    finally:
        detector.close()
        try:
            await ws.close()
        except Exception:
            pass
