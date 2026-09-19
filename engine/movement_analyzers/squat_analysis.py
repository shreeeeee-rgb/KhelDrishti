"""
KhelDrishti — Squat Analyzer
============================
Depth classification, butt-wink (pelvic tuck), knee tracking, and tempo.
"""

from dataclasses import dataclass, field
from typing import List
from enum import Enum
import numpy as np

from engine.kinematics import (
    calculate_angle_2d,
    calculate_knee_valgus_angle,
    calculate_bilateral_asymmetry,
    classify_squat_depth,
    classify_valgus_risk,
    calculate_angular_velocity,
)
from engine.pose_detector import PoseResult, LANDMARK_IDX, get_midpoint


class SquatPhase(str, Enum):
    STANDING = "Standing"
    DESCENT = "Descent"
    BOTTOM = "Bottom"
    ASCENT = "Ascent"
    COMPLETE = "Complete"


@dataclass
class SquatFrameData:
    frame_idx: int
    phase: SquatPhase
    left_knee: float = 180.0
    right_knee: float = 180.0
    hip_angle: float = 180.0
    left_valgus: float = 0.0
    right_valgus: float = 0.0
    pelvic_tuck: float = 0.0  # hip angle drop vs trunk


@dataclass
class SquatAnalysisResult:
    movement_type: str = "Squat"
    frame_data: List[SquatFrameData] = field(default_factory=list)
    phases: List[SquatPhase] = field(default_factory=list)

    min_knee_angle: float = 180.0
    depth_class: str = "Unknown"
    butt_wink: bool = False
    peak_valgus: float = 0.0
    valgus_risk: str = "Safe"
    descent_vel: float = 0.0
    ascent_vel: float = 0.0
    tempo_ratio: float = 1.0
    asymmetry_index: float = 0.0
    bottom_frame: int = -1

    form_score: float = 0.0
    warnings: List[str] = field(default_factory=list)
    tips: List[str] = field(default_factory=list)


class SquatAnalyzer:
    """Detects depth, valgus collapse, and a rapid pelvic tuck (butt wink) at the hole."""

    BUTT_WINK_DROP_DEG = 12.0

    def __init__(self, fps: float = 30.0):
        self.fps = fps

    def analyse(self, pose_results: List[PoseResult]) -> SquatAnalysisResult:
        result = SquatAnalysisResult()
        frames: List[SquatFrameData] = []
        knee_series = []
        hip_series = []

        for fi, pr in enumerate(pose_results):
            fd = SquatFrameData(frame_idx=fi, phase=SquatPhase.STANDING)
            if pr.detected and pr.landmarks:
                def lm(name):
                    idx = LANDMARK_IDX.get(name)
                    return pr.landmarks[idx] if idx is not None and idx < len(pr.landmarks) else None

                l_hip, r_hip = lm("left_hip"), lm("right_hip")
                l_knee, r_knee = lm("left_knee"), lm("right_knee")
                l_ank, r_ank = lm("left_ankle"), lm("right_ankle")
                l_sh, r_sh = lm("left_shoulder"), lm("right_shoulder")

                if all([l_hip, l_knee, l_ank]):
                    fd.left_knee = calculate_angle_2d(l_hip, l_knee, l_ank)
                    fd.left_valgus = calculate_knee_valgus_angle(l_hip, l_knee, l_ank)
                if all([r_hip, r_knee, r_ank]):
                    fd.right_knee = calculate_angle_2d(r_hip, r_knee, r_ank)
                    fd.right_valgus = calculate_knee_valgus_angle(r_hip, r_knee, r_ank)
                if l_hip and r_hip and l_knee and r_knee and l_sh and r_sh:
                    mid_sh = get_midpoint(l_sh, r_sh)
                    mid_hip = get_midpoint(l_hip, r_hip)
                    mid_knee = get_midpoint(l_knee, r_knee)
                    fd.hip_angle = calculate_angle_2d(mid_sh, mid_hip, mid_knee)

            frames.append(fd)
            knee_series.append((fd.left_knee + fd.right_knee) / 2.0)
            hip_series.append(fd.hip_angle)

        n = len(frames)
        if n == 0:
            return result

        bottom = int(np.argmin(knee_series))
        result.bottom_frame = bottom
        result.min_knee_angle = knee_series[bottom]
        result.depth_class = classify_squat_depth(result.min_knee_angle)

        # Butt wink: hip angle collapses sharply near the bottom while knees are already deep
        window = hip_series[max(0, bottom - 4): bottom + 1]
        if len(window) >= 2:
            drop = window[0] - window[-1]
            result.butt_wink = drop > self.BUTT_WINK_DROP_DEG and result.min_knee_angle < 100

        result.peak_valgus = float(max(
            max(abs(f.left_valgus), abs(f.right_valgus)) for f in frames
        ))
        risk, _ = classify_valgus_risk(result.peak_valgus)
        result.valgus_risk = risk
        result.asymmetry_index = calculate_bilateral_asymmetry(
            float(np.mean([f.left_knee for f in frames])),
            float(np.mean([f.right_knee for f in frames])),
        )

        timestamps = [i / self.fps for i in range(n)]
        ang_vel = calculate_angular_velocity(knee_series, timestamps)
        # Descent: negative angular velocity (angle decreasing) before bottom
        desc = [abs(v) for v in ang_vel[:bottom]] or [0.0]
        asc = [abs(v) for v in ang_vel[bottom:]] or [0.0]
        result.descent_vel = float(np.mean(desc))
        result.ascent_vel = float(np.mean(asc))
        result.tempo_ratio = result.ascent_vel / (result.descent_vel + 1e-9)

        phases = []
        for i, fd in enumerate(frames):
            if i < max(1, bottom // 5):
                fd.phase = SquatPhase.STANDING
            elif i < bottom:
                fd.phase = SquatPhase.DESCENT
            elif i == bottom:
                fd.phase = SquatPhase.BOTTOM
            elif i < min(n - 1, bottom + (n - bottom) * 0.85):
                fd.phase = SquatPhase.ASCENT
            else:
                fd.phase = SquatPhase.COMPLETE
            phases.append(fd.phase)

        result.frame_data = frames
        result.phases = phases
        result.form_score = self._score(result)
        result.warnings, result.tips = self._cues(result)
        return result

    def _score(self, r: SquatAnalysisResult) -> float:
        score = 100.0
        if r.depth_class == "Quarter Squat":
            score -= 15
        if r.butt_wink:
            score -= 18
        if r.peak_valgus > 10:
            score -= 22
        elif r.peak_valgus > 5:
            score -= 8
        if abs(r.asymmetry_index) > 15:
            score -= 10
        if r.tempo_ratio > 2.5:
            score -= 6
        return max(0.0, min(100.0, score))

    def _cues(self, r: SquatAnalysisResult):
        warnings, tips = [], []
        if r.butt_wink:
            warnings.append("Butt wink — pelvis tucks under at the bottom (lumbar flexion).")
            tips.append("Stop just above the tuck; goblet squats and hip-flexor mobility.")
        if r.peak_valgus > 10:
            warnings.append("Knees collapsing inward — valgus under load.")
            tips.append("Banded squats and monster walks so knees track over toes.")
        if r.depth_class == "Quarter Squat":
            warnings.append("Limited depth — missing parallel.")
            tips.append("Box squats to a parallel target; ankle-wall mobility.")
        if r.tempo_ratio > 2.5:
            warnings.append("Ascent much faster than descent — control the eccentric.")
            tips.append("3-second lowering tempo, then stand with intent.")
        return warnings, tips
