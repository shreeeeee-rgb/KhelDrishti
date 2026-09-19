"""
KhelDrishti — Running Gait Analyzer
===================================
Cadence, overstride vs CoG, trunk lean, and left–right symmetry.
"""

from dataclasses import dataclass, field
from typing import List
from enum import Enum
import numpy as np

from engine.kinematics import (
    calculate_angle_2d,
    calculate_torso_lean,
    calculate_bilateral_asymmetry,
)
from engine.pose_detector import PoseResult, LANDMARK_IDX, get_midpoint


class GaitPhase(str, Enum):
    STANCE_LEFT = "Left Stance"
    SWING_LEFT = "Left Swing"
    STANCE_RIGHT = "Right Stance"
    SWING_RIGHT = "Right Swing"
    DOUBLE_SUPPORT = "Double Support"
    UNKNOWN = "Unknown"


@dataclass
class GaitFrameData:
    frame_idx: int
    phase: GaitPhase
    left_knee: float = 180.0
    right_knee: float = 180.0
    torso_lean: float = 0.0
    overstride_left: float = 0.0
    overstride_right: float = 0.0
    left_ankle_y: float = 0.8
    right_ankle_y: float = 0.8


@dataclass
class GaitAnalysisResult:
    movement_type: str = "Running Gait"
    frame_data: List[GaitFrameData] = field(default_factory=list)
    phases: List[GaitPhase] = field(default_factory=list)

    cadence_spm: float = 0.0
    mean_torso_lean: float = 0.0
    overstride_index: float = 0.0
    overstride_detected: bool = False
    symmetry_index: float = 0.0
    mean_left_knee_drive: float = 180.0
    mean_right_knee_drive: float = 180.0

    form_score: float = 0.0
    warnings: List[str] = field(default_factory=list)
    tips: List[str] = field(default_factory=list)


class RunningGaitAnalyzer:
    """
    Overstride is the horizontal distance from the landing ankle to mid-hip (CoG proxy)
    at initial contact. Positive (ankle ahead of CoG) indicates braking overstride.
    """

    CONTACT_Y_DELTA = 0.012  # ankle Y local maximum ≈ ground contact in image space

    def __init__(self, fps: float = 30.0):
        self.fps = fps

    def analyse(self, pose_results: List[PoseResult]) -> GaitAnalysisResult:
        result = GaitAnalysisResult()
        frames: List[GaitFrameData] = []
        left_ank_y, right_ank_y = [], []

        for fi, pr in enumerate(pose_results):
            fd = GaitFrameData(frame_idx=fi, phase=GaitPhase.UNKNOWN)
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
                if all([r_hip, r_knee, r_ank]):
                    fd.right_knee = calculate_angle_2d(r_hip, r_knee, r_ank)
                if l_sh and r_sh and l_hip and r_hip:
                    mid_sh = get_midpoint(l_sh, r_sh)
                    mid_hip = get_midpoint(l_hip, r_hip)
                    fd.torso_lean = calculate_torso_lean(mid_sh, mid_hip)
                    if l_ank:
                        fd.overstride_left = l_ank.x - mid_hip.x
                        fd.left_ankle_y = l_ank.y
                    if r_ank:
                        fd.overstride_right = r_ank.x - mid_hip.x
                        fd.right_ankle_y = r_ank.y

            frames.append(fd)
            left_ank_y.append(fd.left_ankle_y)
            right_ank_y.append(fd.right_ankle_y)

        n = len(frames)
        if n == 0:
            return result

        def contacts(series: List[float]) -> List[int]:
            idx = []
            for i in range(1, n - 1):
                if series[i] >= series[i - 1] and series[i] >= series[i + 1]:
                    if series[i] - min(series) > self.CONTACT_Y_DELTA:
                        idx.append(i)
            return idx

        l_contacts = contacts(left_ank_y)
        r_contacts = contacts(right_ank_y)
        total_contacts = len(l_contacts) + len(r_contacts)
        duration_min = max(n / self.fps, 1e-3) / 60.0
        result.cadence_spm = (total_contacts / duration_min) if duration_min > 0 else 0.0

        over_vals = []
        for i in l_contacts:
            over_vals.append(frames[i].overstride_left)
        for i in r_contacts:
            over_vals.append(frames[i].overstride_right)
        result.overstride_index = float(np.mean(over_vals)) if over_vals else 0.0
        result.overstride_detected = result.overstride_index > 0.04

        result.mean_torso_lean = float(np.mean([abs(f.torso_lean) for f in frames]))
        result.mean_left_knee_drive = float(np.min([f.left_knee for f in frames]))
        result.mean_right_knee_drive = float(np.min([f.right_knee for f in frames]))
        result.symmetry_index = calculate_bilateral_asymmetry(
            result.mean_left_knee_drive, result.mean_right_knee_drive
        )

        contact_set_l, contact_set_r = set(l_contacts), set(r_contacts)
        phases = []
        for i, fd in enumerate(frames):
            if i in contact_set_l and i in contact_set_r:
                fd.phase = GaitPhase.DOUBLE_SUPPORT
            elif i in contact_set_l:
                fd.phase = GaitPhase.STANCE_LEFT
            elif i in contact_set_r:
                fd.phase = GaitPhase.STANCE_RIGHT
            elif frames[i].left_ankle_y < frames[i].right_ankle_y:
                fd.phase = GaitPhase.SWING_LEFT
            else:
                fd.phase = GaitPhase.SWING_RIGHT
            phases.append(fd.phase)

        result.frame_data = frames
        result.phases = phases
        result.form_score = self._score(result)
        result.warnings, result.tips = self._cues(result)
        return result

    def _score(self, r: GaitAnalysisResult) -> float:
        score = 100.0
        if r.overstride_detected:
            score -= 18
        if r.cadence_spm and r.cadence_spm < 160:
            score -= min(15, (160 - r.cadence_spm) * 0.25)
        if r.mean_torso_lean < 3 or r.mean_torso_lean > 15:
            score -= 10
        if abs(r.symmetry_index) > 15:
            score -= 12
        return max(0.0, min(100.0, score))

    def _cues(self, r: GaitAnalysisResult):
        warnings, tips = [], []
        if r.overstride_detected:
            warnings.append("Overstriding — foot is landing well ahead of the centre of gravity.")
            tips.append("High-knee drills and wall-drive posture to land closer under the hips.")
        if r.cadence_spm and r.cadence_spm < 160:
            warnings.append(f"Low cadence ({r.cadence_spm:.0f} spm). Target ~170–180 spm.")
            tips.append("Metronome runs at 170 spm; shorten stride, quicken steps.")
        if r.mean_torso_lean < 3:
            warnings.append("Upright trunk — limited forward lean for sprint mechanics.")
            tips.append("Wall-drive 45° leans and falling-start accelerations.")
        if abs(r.symmetry_index) > 15:
            warnings.append("Left–right knee-drive asymmetry.")
            tips.append("A-skips and single-leg hops to even stride mechanics.")
        return warnings, tips
