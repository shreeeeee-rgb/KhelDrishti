"""
KhelDrishti — Analysis Pipeline
Orchestrates pose → movement analyser → coach → annotated video.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import logging

import cv2
import numpy as np

from engine.pose_detector import PoseDetector, PoseResult, pose_result_from_xy
from engine.movement_analyzers import ANALYZERS
from engine.feedback_coach import FeedbackCoach
from engine.video_annotator import VideoAnnotator
from engine.kinematics import calculate_angle_2d, calculate_knee_valgus_angle, calculate_torso_lean
from engine.pose_detector import LANDMARK_IDX, get_midpoint

logger = logging.getLogger("kheldrishti.pipeline")

MOVEMENT_BENCHMARKS = {
    "vertical_jump": {
        "label": "Vertical Jump (CMJ)",
        "benchmarks": {
            "countermovement_knee": "90°–110°",
            "landing_knee_flexion": "> 60° preferred, < 30° high risk",
            "valgus": "< 5° safe, > 10° ACL caution",
        },
    },
    "volleyball_spike": {
        "label": "Volleyball Spike",
        "benchmarks": {
            "cocking_elbow": "> 90°",
            "chain": "hips → torso → shoulder → elbow → wrist",
            "landing": "quiet, symmetric, knees over toes",
        },
    },
    "running_gait": {
        "label": "Running Gait",
        "benchmarks": {
            "cadence": "170–180 steps/min",
            "trunk_lean": "5°–10° forward",
            "footstrike": "under the hips (no overstride)",
        },
    },
    "squat": {
        "label": "Squat",
        "benchmarks": {
            "depth": "parallel or below if mobility allows",
            "spine": "no rapid pelvic tuck (butt wink)",
            "knees": "track over toes, valgus < 5°",
        },
    },
}


def _to_dict(obj: Any) -> Dict[str, Any]:
    if is_dataclass(obj):
        d = asdict(obj)
    elif isinstance(obj, dict):
        d = dict(obj)
    else:
        d = dict(getattr(obj, "__dict__", {}))
    # drop bulky frame arrays for JSON
    frames = d.get("frame_data") or []
    d["frame_count"] = len(frames)
    series = []
    for fd in frames:
        if is_dataclass(fd):
            fd = asdict(fd)
        series.append({
            "i": fd.get("frame_idx"),
            "phase": str(fd.get("phase", "")),
            "lk": fd.get("left_knee_angle", fd.get("left_knee")),
            "rk": fd.get("right_knee_angle", fd.get("right_knee")),
            "lv": fd.get("left_valgus"),
            "rv": fd.get("right_valgus"),
            "hip": fd.get("left_hip_angle", fd.get("hip_angle")),
        })
    d["series"] = series
    d.pop("frame_data", None)
    # stringify enums
    if "phases" in d:
        d["phases"] = [str(p) for p in d["phases"]]
    return d


def live_joint_metrics(pose: PoseResult) -> Dict[str, Any]:
    empty = {
        "detected": False,
        "landmarks": [],
        "left_knee": None,
        "right_knee": None,
        "left_valgus": 0.0,
        "right_valgus": 0.0,
        "torso_lean": 0.0,
        "left_elbow": None,
        "right_elbow": None,
    }
    if not pose.detected or not pose.landmarks:
        return empty

    def lm(name):
        idx = LANDMARK_IDX.get(name)
        return pose.landmarks[idx] if idx is not None and idx < len(pose.landmarks) else None

    l_hip, r_hip = lm("left_hip"), lm("right_hip")
    l_knee, r_knee = lm("left_knee"), lm("right_knee")
    l_ank, r_ank = lm("left_ankle"), lm("right_ankle")
    l_sh, r_sh = lm("left_shoulder"), lm("right_shoulder")
    l_el, r_el = lm("left_elbow"), lm("right_elbow")
    l_wr, r_wr = lm("left_wrist"), lm("right_wrist")

    out = empty.copy()
    out["detected"] = True
    out["landmarks"] = [
        {"x": p.x, "y": p.y, "z": p.z, "v": p.visibility, "name": p.name}
        for p in pose.landmarks
    ]
    if all([l_hip, l_knee, l_ank]):
        out["left_knee"] = calculate_angle_2d(l_hip, l_knee, l_ank)
        out["left_valgus"] = calculate_knee_valgus_angle(l_hip, l_knee, l_ank)
    if all([r_hip, r_knee, r_ank]):
        out["right_knee"] = calculate_angle_2d(r_hip, r_knee, r_ank)
        out["right_valgus"] = calculate_knee_valgus_angle(r_hip, r_knee, r_ank)
    if l_sh and r_sh and l_hip and r_hip:
        out["torso_lean"] = calculate_torso_lean(get_midpoint(l_sh, r_sh), get_midpoint(l_hip, r_hip))
    if all([l_sh, l_el, l_wr]):
        out["left_elbow"] = calculate_angle_2d(l_sh, l_el, l_wr)
    if all([r_sh, r_el, r_wr]):
        out["right_elbow"] = calculate_angle_2d(r_sh, r_el, r_wr)

    max_v = max(abs(out["left_valgus"] or 0), abs(out["right_valgus"] or 0))
    if max_v > 10:
        out["risk_colour"] = "red"
        out["risk"] = "High Risk"
    elif max_v > 5:
        out["risk_colour"] = "amber"
        out["risk"] = "Caution"
    else:
        out["risk_colour"] = "green"
        out["risk"] = "Safe"
    return out


def load_synthetic_landmarks(json_path: Path) -> List[PoseResult]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    poses = []
    for i, frame in enumerate(payload["frames"]):
        poses.append(pose_result_from_xy(frame["x"], frame["y"], frame.get("z"), frame.get("v"), i))
    return poses


class AnalysisPipeline:
    def __init__(self):
        self.coach = FeedbackCoach()
        self.annotator = VideoAnnotator()

    def analyse_video(
        self,
        video_path: str,
        movement: str = "vertical_jump",
        language: str = "en",
        out_dir: str = "outputs",
        landmark_json: Optional[str] = None,
    ) -> Dict[str, Any]:
        movement = movement if movement in ANALYZERS else "vertical_jump"
        out_dir_p = Path(out_dir)
        out_dir_p.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames: List[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()

        fallback_used = False
        detected = 0
        poses: List[PoseResult] = []

        if landmark_json and Path(landmark_json).exists():
            logger.info("Using companion landmark track %s", landmark_json)
            poses = load_synthetic_landmarks(Path(landmark_json))
            fallback_used = True
            if frames and len(poses) != len(frames):
                if len(poses) < len(frames):
                    poses = poses + [poses[-1]] * (len(frames) - len(poses))
                else:
                    poses = poses[: len(frames)]
            detected = sum(1 for p in poses if p.detected)
        else:
            detector = PoseDetector(static_image_mode=False, model_complexity=1, smooth_alpha=0.45)
            try:
                for i, fr in enumerate(frames):
                    pr = detector.process_frame(fr, i)
                    poses.append(pr)
                    if pr.detected:
                        detected += 1
            finally:
                detector.close()
            if not frames:
                raise ValueError("Empty video")

        analyser_cls = ANALYZERS[movement]
        analysis_obj = analyser_cls(fps=fps).analyse(poses)
        analysis = _to_dict(analysis_obj)
        coaching = self.coach.coach(analysis, language=language)

        hud_frames = []
        series = analysis.get("series") or []
        for i, pr in enumerate(poses):
            row = series[i] if i < len(series) else {}
            live = live_joint_metrics(pr)
            hud_frames.append({
                "phase": row.get("phase") or str(getattr(analysis_obj, "phases", ["Live"])[min(i, max(0, len(getattr(analysis_obj, "phases", [])) - 1))] if getattr(analysis_obj, "phases", None) else "Live"),
                "left_knee": live.get("left_knee") or row.get("lk"),
                "right_knee": live.get("right_knee") or row.get("rk"),
                "primary_angle": live.get("left_knee") or row.get("lk"),
                "form_score": analysis.get("form_score"),
                "risk_colour": coaching.risk.colour,
                "cue": coaching.summary_hi if language == "hi" else coaching.summary_en,
            })

        keyframes: Dict[int, str] = {}
        for key, label in [
            ("deepest_dip_frame", "Lowest Dip"),
            ("takeoff_frame", "Takeoff"),
            ("landing_frame", "Landing Contact"),
            ("peak_reach_frame", "Peak Reach"),
            ("bottom_frame", "Bottom"),
        ]:
            idx = analysis.get(key, -1)
            if isinstance(idx, int) and idx >= 0:
                keyframes[idx] = label

        stem = Path(video_path).stem
        out_video = str(out_dir_p / f"{stem}_{movement}_hud.mp4")
        if frames:
            self.annotator.write_video(frames, poses, hud_frames, out_video, fps=fps, keyframes=keyframes)
        else:
            out_video = None

        return {
            "movement": movement,
            "fps": fps,
            "frame_count": len(frames) or len(poses),
            "pose_detected_frames": detected,
            "synthetic_fallback": fallback_used,
            "analysis": analysis,
            "coaching": {
                "form_score": coaching.form_score,
                "risk": {
                    "level": coaching.risk.level,
                    "colour": coaching.risk.colour,
                    "factors": coaching.risk.factors,
                },
                "summary_en": coaching.summary_en,
                "summary_hi": coaching.summary_hi,
                "tips": [t.__dict__ for t in coaching.tips],
                "drills": coaching.drills,
            },
            "keyframes": {str(k): v for k, v in keyframes.items()},
            "annotated_video": out_video,
        }
