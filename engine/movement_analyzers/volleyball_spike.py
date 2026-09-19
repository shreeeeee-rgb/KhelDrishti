"""
KhelDrishti — Volleyball Spike Analyzer
=======================================
Approach, cocking, peak reach, kinetic-chain sequencing, and landing balance.
"""

from dataclasses import dataclass, field
from typing import List
from enum import Enum
import numpy as np

from engine.kinematics import (
    calculate_angle_2d,
    calculate_knee_valgus_angle,
    calculate_bilateral_asymmetry,
    classify_valgus_risk,
)
from engine.pose_detector import PoseResult, LANDMARK_IDX


class SpikePhase(str, Enum):
    APPROACH = "Approach"
    PENULTIMATE = "Penultimate Plant"
    COCKING = "Arm Cocking"
    ACCELERATION = "Arm Acceleration"
    CONTACT = "Ball Contact"
    LANDING = "Landing"
    COMPLETE = "Complete"


@dataclass
class SpikeFrameData:
    frame_idx: int
    phase: SpikePhase
    hitting_elbow_angle: float = 180.0
    hitting_shoulder_angle: float = 90.0
    wrist_y: float = 0.5
    hip_rotation_proxy: float = 0.0
    torso_angle: float = 0.0
    left_knee: float = 180.0
    right_knee: float = 180.0
    left_valgus: float = 0.0
    right_valgus: float = 0.0


@dataclass
class SpikeAnalysisResult:
    movement_type: str = "Volleyball Spike"
    frame_data: List[SpikeFrameData] = field(default_factory=list)
    phases: List[SpikePhase] = field(default_factory=list)

    peak_reach_norm: float = 0.0
    peak_reach_frame: int = -1
    cocking_elbow_angle: float = 180.0
    cocking_ok: bool = False
    kinetic_chain_ok: bool = False
    kinetic_chain_order: List[str] = field(default_factory=list)
    landing_asymmetry: float = 0.0
    peak_valgus_landing: float = 0.0
    valgus_risk: str = "Safe"

    form_score: float = 0.0
    warnings: List[str] = field(default_factory=list)
    tips: List[str] = field(default_factory=list)


class VolleyballSpikeAnalyzer:
    """Heuristic spike analysis from a 33-landmark sequence (hitting arm = right)."""

    def __init__(self, fps: float = 30.0, hitting_side: str = "right"):
        self.fps = fps
        self.hitting_side = hitting_side

    def analyse(self, pose_results: List[PoseResult]) -> SpikeAnalysisResult:
        result = SpikeAnalysisResult()
        frames: List[SpikeFrameData] = []

        prefix = "right" if self.hitting_side == "right" else "left"

        for fi, pr in enumerate(pose_results):
            fd = SpikeFrameData(frame_idx=fi, phase=SpikePhase.APPROACH)
            if pr.detected and pr.landmarks:
                def lm(name):
                    idx = LANDMARK_IDX.get(name)
                    return pr.landmarks[idx] if idx is not None and idx < len(pr.landmarks) else None

                sh = lm(f"{prefix}_shoulder")
                el = lm(f"{prefix}_elbow")
                wr = lm(f"{prefix}_wrist")
                hip = lm(f"{prefix}_hip")
                l_hip, r_hip = lm("left_hip"), lm("right_hip")
                l_sh, r_sh = lm("left_shoulder"), lm("right_shoulder")
                l_knee, r_knee = lm("left_knee"), lm("right_knee")
                l_ankle, r_ankle = lm("left_ankle"), lm("right_ankle")

                if all([sh, el, wr]):
                    fd.hitting_elbow_angle = calculate_angle_2d(sh, el, wr)
                if all([hip, sh, el]):
                    fd.hitting_shoulder_angle = calculate_angle_2d(hip, sh, el)
                if wr:
                    fd.wrist_y = wr.y
                if l_hip and r_hip:
                    fd.hip_rotation_proxy = abs(l_hip.z - r_hip.z) + abs(l_hip.x - r_hip.x)
                if l_sh and r_sh and l_hip and r_hip:
                    mid_sh = ((l_sh.x + r_sh.x) / 2, (l_sh.y + r_sh.y) / 2)
                    mid_hip = ((l_hip.x + r_hip.x) / 2, (l_hip.y + r_hip.y) / 2)
                    from engine.kinematics import calculate_torso_lean
                    fd.torso_angle = calculate_torso_lean(
                        type("P", (), {"x": mid_sh[0], "y": mid_sh[1], "z": 0})(),
                        type("P", (), {"x": mid_hip[0], "y": mid_hip[1], "z": 0})(),
                    )
                if all([l_hip, l_knee, l_ankle]):
                    fd.left_knee = calculate_angle_2d(l_hip, l_knee, l_ankle)
                    fd.left_valgus = calculate_knee_valgus_angle(l_hip, l_knee, l_ankle)
                if all([r_hip, r_knee, r_ankle]):
                    fd.right_knee = calculate_angle_2d(r_hip, r_knee, r_ankle)
                    fd.right_valgus = calculate_knee_valgus_angle(r_hip, r_knee, r_ankle)
            frames.append(fd)

        n = len(frames)
        if n == 0:
            return result

        wrist_ys = [f.wrist_y for f in frames]
        peak_reach_frame = int(np.argmin(wrist_ys))  # image Y down → min Y is highest reach
        result.peak_reach_frame = peak_reach_frame
        result.peak_reach_norm = 1.0 - wrist_ys[peak_reach_frame]

        cocking_window = frames[max(0, peak_reach_frame - 12):peak_reach_frame]
        if cocking_window:
            result.cocking_elbow_angle = min(f.hitting_elbow_angle for f in cocking_window)
        result.cocking_ok = result.cocking_elbow_angle > 90.0

        # Kinetic chain: peaks of hip proxy, torso, shoulder, elbow whip, wrist height
        def peak_idx(series):
            return int(np.argmax(series)) if series else 0

        hip_i = peak_idx([abs(f.hip_rotation_proxy) for f in frames])
        torso_i = peak_idx([abs(f.torso_angle) for f in frames])
        sh_i = peak_idx([abs(180 - f.hitting_shoulder_angle) for f in frames])
        el_i = peak_idx([abs(180 - f.hitting_elbow_angle) for f in frames])
        wr_i = peak_reach_frame
        order = [
            ("hips", hip_i),
            ("torso", torso_i),
            ("shoulder", sh_i),
            ("elbow", el_i),
            ("wrist", wr_i),
        ]
        result.kinetic_chain_order = [name for name, _ in sorted(order, key=lambda x: x[1])]
        expected = ["hips", "torso", "shoulder", "elbow", "wrist"]
        # Allow nearby sequencing rather than exact identity
        result.kinetic_chain_ok = result.kinetic_chain_order[:3] == expected[:3] or (
            hip_i <= torso_i <= wr_i
        )

        landing_start = min(n - 1, peak_reach_frame + 4)
        landing_frames = frames[landing_start:]
        if landing_frames:
            result.landing_asymmetry = float(np.mean([
                calculate_bilateral_asymmetry(f.left_knee, f.right_knee) for f in landing_frames
            ]))
            result.peak_valgus_landing = float(max(
                max(abs(f.left_valgus), abs(f.right_valgus)) for f in landing_frames
            ))
        risk, _ = classify_valgus_risk(result.peak_valgus_landing)
        result.valgus_risk = risk

        # Phases
        phases = []
        for i, fd in enumerate(frames):
            if i < max(1, peak_reach_frame - 18):
                ph = SpikePhase.APPROACH
            elif i < max(1, peak_reach_frame - 10):
                ph = SpikePhase.PENULTIMATE
            elif i < max(1, peak_reach_frame - 3):
                ph = SpikePhase.COCKING
            elif i < peak_reach_frame:
                ph = SpikePhase.ACCELERATION
            elif i == peak_reach_frame:
                ph = SpikePhase.CONTACT
            elif i < min(n, peak_reach_frame + 12):
                ph = SpikePhase.LANDING
            else:
                ph = SpikePhase.COMPLETE
            fd.phase = ph
            phases.append(ph)

        result.frame_data = frames
        result.phases = phases
        result.form_score = self._score(result)
        result.warnings, result.tips = self._cues(result)
        return result

    def _score(self, r: SpikeAnalysisResult) -> float:
        score = 100.0
        if not r.cocking_ok:
            score -= 18
        if not r.kinetic_chain_ok:
            score -= 12
        if abs(r.landing_asymmetry) > 15:
            score -= 12
        if r.peak_valgus_landing > 10:
            score -= 20
        elif r.peak_valgus_landing > 5:
            score -= 8
        return max(0.0, min(100.0, score))

    def _cues(self, r: SpikeAnalysisResult):
        warnings, tips = [], []
        if not r.cocking_ok:
            warnings.append("Low spike cocking — elbow angle below 90° before contact.")
            tips.append("Banded external rotations and wall-toss mechanics to load the hitting arm.")
        if not r.kinetic_chain_ok:
            warnings.append("Kinetic chain out of sequence — arm is firing before hips/torso.")
            tips.append("Medicine-ball rotational throws: hips then torso then arm.")
        if r.peak_valgus_landing > 10:
            warnings.append("Knee valgus on spike landing — ACL caution.")
            tips.append("Stick landings: knees track over toes, quiet feet.")
        if abs(r.landing_asymmetry) > 15:
            warnings.append("Uneven landing — one leg absorbing most of the load.")
            tips.append("Single-leg stick drills after two-foot approach jumps.")
        return warnings, tips
