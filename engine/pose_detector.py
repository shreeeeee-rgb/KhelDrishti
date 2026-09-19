"""
KhelDrishti — MediaPipe Pose Detector Wrapper
==============================================
Wraps MediaPipe Pose with signal smoothing, landmark extraction,
and confidence filtering for both image frames and video streams.
"""

import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger("kheldrishti.pose")

from pathlib import Path

# Lazy import mediapipe (Tasks API in 0.10.30+)
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    _mp_available = True
except ImportError:
    _mp_available = False
    logger.warning("MediaPipe not available — pose detection disabled.")

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
_MODEL_PATH = Path(__file__).resolve().parent / "assets" / "pose_landmarker_lite.task"


def _ensure_pose_model() -> Path:
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if _MODEL_PATH.exists() and _MODEL_PATH.stat().st_size > 1000:
        return _MODEL_PATH
    import urllib.request
    logger.info("Downloading Pose Landmarker model…")
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    return _MODEL_PATH


@dataclass
class Landmark:
    """Single 3D landmark with confidence."""
    x: float
    y: float
    z: float
    visibility: float
    name: str = ""


@dataclass
class PoseResult:
    """Full pose result for a single frame."""
    landmarks: List[Landmark] = field(default_factory=list)
    world_landmarks: List[Landmark] = field(default_factory=list)
    detected: bool = False
    confidence: float = 0.0
    frame_index: int = 0


# MediaPipe landmark indices (standard 33-keypoint model)
LANDMARK_NAMES = {
    0:  "nose",
    1:  "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
    4:  "right_eye_inner", 5: "right_eye", 6: "right_eye_outer",
    7:  "left_ear", 8: "right_ear",
    9:  "mouth_left", 10: "mouth_right",
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist",
    17: "left_pinky", 18: "right_pinky",
    19: "left_index", 20: "right_index",
    21: "left_thumb", 22: "right_thumb",
    23: "left_hip", 24: "right_hip",
    25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle",
    29: "left_heel", 30: "right_heel",
    31: "left_foot_index", 32: "right_foot_index",
}

# Reverse lookup
LANDMARK_IDX = {v: k for k, v in LANDMARK_NAMES.items()}


class ExponentialSmoother:
    """Per-coordinate EMA smoother to reduce jitter on landmark streams."""

    def __init__(self, alpha: float = 0.4, n_landmarks: int = 33):
        self.alpha = alpha
        self.smoothed: Optional[np.ndarray] = None  # shape (n_landmarks, 4)

    def update(self, raw: np.ndarray) -> np.ndarray:
        """Apply EMA. raw shape: (n_landmarks, 4) = [x, y, z, vis]."""
        if self.smoothed is None:
            self.smoothed = raw.copy()
        else:
            self.smoothed = self.alpha * raw + (1 - self.alpha) * self.smoothed
        return self.smoothed

    def reset(self):
        self.smoothed = None


class PoseDetector:
    """
    MediaPipe Pose wrapper for KhelDrishti.

    Parameters
    ----------
    static_image_mode    : True for single images, False for video streams.
    model_complexity     : 0 (lite/fast), 1 (full), 2 (heavy/accurate).
    min_detection_conf   : Minimum confidence to accept a detection.
    min_tracking_conf    : Minimum confidence to keep tracking.
    smooth_alpha         : EMA smoothing coefficient (0.0 = no smoothing).
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        min_detection_conf: float = 0.5,
        min_tracking_conf: float = 0.5,
        smooth_alpha: float = 0.4,
    ):
        self.ready = False
        self.smoother = ExponentialSmoother(alpha=smooth_alpha)
        self._ts_ms = 0
        self._video_mode = not static_image_mode

        if not _mp_available:
            logger.error("MediaPipe unavailable — PoseDetector running in stub mode.")
            return

        try:
            model = _ensure_pose_model()
            running = (
                mp_vision.RunningMode.IMAGE
                if static_image_mode
                else mp_vision.RunningMode.VIDEO
            )
            options = mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model)),
                running_mode=running,
                num_poses=1,
                min_pose_detection_confidence=min_detection_conf,
                min_pose_presence_confidence=min_detection_conf,
                min_tracking_confidence=min_tracking_conf,
            )
            self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)
            self.ready = True
            logger.info(
                "PoseDetector initialised (tasks landmarker, mode=%s, alpha=%s)",
                running.name, smooth_alpha,
            )
        except Exception as exc:
            logger.exception("Failed to initialise Pose Landmarker: %s", exc)
            self.ready = False

    def process_frame(self, bgr_frame: np.ndarray, frame_index: int = 0) -> PoseResult:
        """
        Detect pose in a BGR OpenCV frame.

        Parameters
        ----------
        bgr_frame   : OpenCV BGR image array.
        frame_index : Frame number for tracking.

        Returns
        -------
        PoseResult with 33 landmarks and world landmarks (if detected).
        """
        if not self.ready:
            return PoseResult(detected=False, frame_index=frame_index)

        import cv2
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self._video_mode:
            self._ts_ms += 33
            results = self.landmarker.detect_for_video(mp_image, self._ts_ms)
        else:
            results = self.landmarker.detect(mp_image)

        if not results.pose_landmarks:
            self.smoother.reset()
            return PoseResult(detected=False, frame_index=frame_index)

        pose_lms = results.pose_landmarks[0]
        raw = np.array([
            [lm.x, lm.y, lm.z, getattr(lm, "visibility", 0.99) or 0.99]
            for lm in pose_lms
        ], dtype=float)

        smoothed = self.smoother.update(raw)

        landmarks = [
            Landmark(
                x=smoothed[i, 0], y=smoothed[i, 1],
                z=smoothed[i, 2], visibility=smoothed[i, 3],
                name=LANDMARK_NAMES.get(i, f"lm_{i}")
            )
            for i in range(len(smoothed))
        ]

        world_lms: List[Landmark] = []
        if results.pose_world_landmarks:
            world_lms = [
                Landmark(
                    x=lm.x, y=lm.y, z=lm.z,
                    visibility=getattr(lm, "visibility", 0.99) or 0.99,
                    name=LANDMARK_NAMES.get(i, f"lm_{i}")
                )
                for i, lm in enumerate(results.pose_world_landmarks[0])
            ]

        avg_vis = float(np.mean(smoothed[:, 3]))
        return PoseResult(
            landmarks=landmarks,
            world_landmarks=world_lms,
            detected=True,
            confidence=avg_vis,
            frame_index=frame_index,
        )

    def get_landmark(self, result: PoseResult, name: str) -> Optional[Landmark]:
        """Get a named landmark from a PoseResult. Returns None if not found."""
        idx = LANDMARK_IDX.get(name)
        if idx is None or not result.landmarks:
            return None
        lm = result.landmarks[idx]
        return lm if lm.visibility > 0.3 else None

    def get_landmark_by_idx(self, result: PoseResult, idx: int) -> Optional[Landmark]:
        """Get landmark by numeric index."""
        if not result.landmarks or idx >= len(result.landmarks):
            return None
        return result.landmarks[idx]

    def close(self):
        """Release MediaPipe resources."""
        if self.ready and hasattr(self, "landmarker"):
            self.landmarker.close()
            self.ready = False


# ─── Convenience Helpers ─────────────────────────────────────────────────────

def landmark_to_pixel(lm: Landmark, width: int, height: int):
    """Convert normalised landmark to pixel coordinates."""
    return int(lm.x * width), int(lm.y * height)


def pose_result_from_xy(
    xs: List[float],
    ys: List[float],
    zs: Optional[List[float]] = None,
    vis: Optional[List[float]] = None,
    frame_index: int = 0,
) -> PoseResult:
    """Build a PoseResult from parallel coordinate arrays (synthetic / tests)."""
    n = min(33, len(xs), len(ys))
    landmarks = []
    for i in range(n):
        landmarks.append(
            Landmark(
                x=float(xs[i]),
                y=float(ys[i]),
                z=float(zs[i]) if zs is not None else 0.0,
                visibility=float(vis[i]) if vis is not None else 0.99,
                name=LANDMARK_NAMES.get(i, f"lm_{i}"),
            )
        )
    return PoseResult(
        landmarks=landmarks,
        detected=n >= 33,
        confidence=0.99,
        frame_index=frame_index,
    )


def get_midpoint(lm_a: Landmark, lm_b: Landmark) -> Landmark:
    """Return the midpoint between two landmarks."""
    return Landmark(
        x=(lm_a.x + lm_b.x) / 2,
        y=(lm_a.y + lm_b.y) / 2,
        z=(lm_a.z + lm_b.z) / 2,
        visibility=min(lm_a.visibility, lm_b.visibility),
        name=f"mid_{lm_a.name}_{lm_b.name}"
    )
