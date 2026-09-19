"""
KhelDrishti — Vertical Jump (CMJ) Analyzer
==========================================
Detects Countermovement Jump phases, calculates flight time,
jump height, takeoff velocity, and landing risk metrics.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import numpy as np

from engine.kinematics import (
    calculate_angle_2d,
    calculate_knee_valgus_angle,
    calculate_jump_height_from_flight_time,
    calculate_takeoff_velocity,
    classify_valgus_risk,
    classify_landing_stiffness,
    classify_squat_depth,
)
from engine.pose_detector import PoseResult, Landmark, get_midpoint


class JumpPhase(str, Enum):
    IDLE        = "Idle / Standing"
    UNWEIGHTING = "Unweighting"
    BRAKING     = "Braking (Deepest Dip)"
    PROPULSION  = "Propulsion"
    TAKEOFF     = "Takeoff"
    FLIGHT      = "Flight"
    LANDING     = "Landing"
    COMPLETE    = "Complete"


@dataclass
class JumpFrameData:
    frame_idx: int
    phase: JumpPhase
    left_knee_angle: float = 180.0
    right_knee_angle: float = 180.0
    left_hip_angle: float = 180.0
    right_hip_angle: float = 180.0
    left_ankle_angle: float = 90.0
    right_ankle_angle: float = 90.0
    left_valgus: float = 0.0
    right_valgus: float = 0.0
    hip_y: float = 0.5          # normalised Y of mid-hip
    ankle_y: float = 0.8        # normalised Y of mid-ankle
    both_feet_grounded: bool = True


@dataclass
class JumpAnalysisResult:
    movement_type: str = "Vertical Jump (CMJ)"
    phases: List[JumpPhase] = field(default_factory=list)
    frame_data: List[JumpFrameData] = field(default_factory=list)

    # Key metrics
    deepest_knee_angle: float = 180.0          # °  (at braking phase)
    deepest_hip_angle: float = 180.0           # °
    flight_time_s: float = 0.0                 # seconds
    jump_height_cm: float = 0.0                # centimetres
    takeoff_velocity_ms: float = 0.0           # m/s
    landing_knee_angle: float = 90.0           # ° at first contact
    peak_knee_valgus_left: float = 0.0         # ° during landing
    peak_knee_valgus_right: float = 0.0
    asymmetry_index: float = 0.0               # %

    # Key frame indices
    deepest_dip_frame: int = -1
    takeoff_frame: int = -1
    landing_frame: int = -1

    # Classification
    squat_depth_class: str = "Unknown"
    landing_stiffness: str = "Unknown"
    landing_risk_colour: str = "green"
    valgus_risk_left: str = "Safe"
    valgus_risk_right: str = "Safe"

    form_score: float = 0.0
    warnings: List[str] = field(default_factory=list)
    tips: List[str] = field(default_factory=list)


class VerticalJumpAnalyzer:
    """
    Analyses a sequence of PoseResults for a Countermovement Jump.

    Phase detection is heuristic:
      - Unweighting: hip_y rising (Y decreasing in image)
      - Braking:     knee_angle at global minimum
      - Propulsion:  knee extending rapidly
      - Takeoff:     both feet leave ground (ankle confidence drops)
      - Flight:      both ankles airborne
      - Landing:     ankle visibility restored, knee absorbs load
    """

    HIP_RISE_THRESHOLD    = 0.015   # normalised Y units — rising indicates jump
    KNEE_EXTEND_THRESHOLD = 5.0     # degrees/frame extending from minimum
    AIRBORNE_THRESHOLD    = 0.20    # ankle visibility below = airborne
    MIN_FLIGHT_FRAMES     = 3       # at least 3 consecutive airborne frames

    def __init__(self, fps: float = 30.0):
        self.fps = fps

    def analyse(self, pose_results: List[PoseResult]) -> JumpAnalysisResult:
        result = JumpAnalysisResult()

        frame_data_list: List[JumpFrameData] = []
        hip_y_series: List[float] = []
        lk_series: List[float] = []
        rk_series: List[float] = []
        ankle_vis_series: List[float] = []

        for fi, pr in enumerate(pose_results):
            if not pr.detected:
                frame_data_list.append(JumpFrameData(fi, JumpPhase.IDLE))
                hip_y_series.append(0.5)
                lk_series.append(180.0)
                rk_series.append(180.0)
                ankle_vis_series.append(1.0)
                continue

            lms = pr.landmarks

            def lm(name):
                from engine.pose_detector import LANDMARK_IDX
                idx = LANDMARK_IDX.get(name)
                return lms[idx] if idx is not None and idx < len(lms) else None

            l_hip, r_hip     = lm("left_hip"),     lm("right_hip")
            l_knee, r_knee   = lm("left_knee"),    lm("right_knee")
            l_ankle, r_ankle = lm("left_ankle"),   lm("right_ankle")
            l_shoulder       = lm("left_shoulder")
            r_shoulder       = lm("right_shoulder")

            # Knee angles
            lk = calculate_angle_2d(l_hip, l_knee, l_ankle) if all([l_hip, l_knee, l_ankle]) else 180.0
            rk = calculate_angle_2d(r_hip, r_knee, r_ankle) if all([r_hip, r_knee, r_ankle]) else 180.0

            # Hip angles
            lh = calculate_angle_2d(l_shoulder, l_hip, l_knee) if all([l_shoulder, l_hip, l_knee]) else 180.0
            rh = calculate_angle_2d(r_shoulder, r_hip, r_knee) if all([r_shoulder, r_hip, r_knee]) else 180.0

            # Valgus angles
            lv = calculate_knee_valgus_angle(l_hip, l_knee, l_ankle) if all([l_hip, l_knee, l_ankle]) else 0.0
            rv = calculate_knee_valgus_angle(r_hip, r_knee, r_ankle) if all([r_hip, r_knee, r_ankle]) else 0.0

            # Mid-hip Y
            mid_hip_y = ((l_hip.y if l_hip else 0.5) + (r_hip.y if r_hip else 0.5)) / 2.0
            ankle_vis = ((l_ankle.visibility if l_ankle else 1.0) + (r_ankle.visibility if r_ankle else 1.0)) / 2.0

            grounded = ankle_vis >= self.AIRBORNE_THRESHOLD

            fd = JumpFrameData(
                frame_idx=fi,
                phase=JumpPhase.IDLE,
                left_knee_angle=lk, right_knee_angle=rk,
                left_hip_angle=lh, right_hip_angle=rh,
                left_valgus=lv, right_valgus=rv,
                hip_y=mid_hip_y,
                both_feet_grounded=grounded,
            )
            frame_data_list.append(fd)
            hip_y_series.append(mid_hip_y)
            lk_series.append(lk)
            rk_series.append(rk)
            ankle_vis_series.append(ankle_vis)

        # ─── Phase Detection ──────────────────────────────────────────────────
        phases = [JumpPhase.IDLE] * len(frame_data_list)
        n = len(frame_data_list)

        # Find deepest dip (minimum average knee angle)
        avg_knee = [(lk_series[i] + rk_series[i]) / 2 for i in range(n)]
        deepest_idx = int(np.argmin(avg_knee))

        # Airborne window
        airborne_start = airborne_end = -1
        consecutive = 0
        for i in range(n):
            if ankle_vis_series[i] < self.AIRBORNE_THRESHOLD:
                consecutive += 1
                if consecutive >= self.MIN_FLIGHT_FRAMES and airborne_start == -1:
                    airborne_start = i - consecutive + 1
            else:
                if airborne_start != -1 and airborne_end == -1:
                    airborne_end = i - 1
                consecutive = 0

        if airborne_start == -1:
            airborne_start = deepest_idx + 5
        if airborne_end == -1:
            airborne_end = min(airborne_start + 10, n - 1)

        # Assign phases
        for i in range(n):
            if i < deepest_idx // 2:
                phases[i] = JumpPhase.UNWEIGHTING
            elif i < deepest_idx:
                phases[i] = JumpPhase.BRAKING
            elif i < airborne_start:
                phases[i] = JumpPhase.PROPULSION
            elif i == airborne_start:
                phases[i] = JumpPhase.TAKEOFF
            elif i <= airborne_end:
                phases[i] = JumpPhase.FLIGHT
            elif i <= airborne_end + 15:
                phases[i] = JumpPhase.LANDING
            else:
                phases[i] = JumpPhase.COMPLETE

        for i, fd in enumerate(frame_data_list):
            fd.phase = phases[i]

        # ─── Compute Metrics ──────────────────────────────────────────────────
        result.frame_data = frame_data_list
        result.phases = phases
        result.deepest_dip_frame = deepest_idx
        result.takeoff_frame = airborne_start
        result.landing_frame = airborne_end + 1 if airborne_end > 0 else -1

        result.deepest_knee_angle = avg_knee[deepest_idx]
        result.deepest_hip_angle = (
            frame_data_list[deepest_idx].left_hip_angle +
            frame_data_list[deepest_idx].right_hip_angle
        ) / 2.0

        flight_frames = max(0, airborne_end - airborne_start)
        result.flight_time_s = flight_frames / self.fps
        result.jump_height_cm = calculate_jump_height_from_flight_time(result.flight_time_s)
        result.takeoff_velocity_ms = calculate_takeoff_velocity(result.jump_height_cm)

        # Landing metrics
        if result.landing_frame > 0 and result.landing_frame < n:
            ld = frame_data_list[result.landing_frame]
            result.landing_knee_angle = (ld.left_knee_angle + ld.right_knee_angle) / 2.0
            result.peak_knee_valgus_left = ld.left_valgus
            result.peak_knee_valgus_right = ld.right_valgus

        # Asymmetry
        avg_lk = np.mean(lk_series)
        avg_rk = np.mean(rk_series)
        from engine.kinematics import calculate_bilateral_asymmetry
        result.asymmetry_index = calculate_bilateral_asymmetry(avg_lk, avg_rk)

        result.squat_depth_class = classify_squat_depth(result.deepest_knee_angle)
        stiff, risk_col = classify_landing_stiffness(result.landing_knee_angle)
        result.landing_stiffness = stiff
        result.landing_risk_colour = risk_col

        vl_text, _ = classify_valgus_risk(result.peak_knee_valgus_left)
        vr_text, _ = classify_valgus_risk(result.peak_knee_valgus_right)
        result.valgus_risk_left = vl_text
        result.valgus_risk_right = vr_text

        result.form_score = self._compute_form_score(result)
        result.warnings, result.tips = self._generate_warnings(result)

        return result

    def _compute_form_score(self, r: JumpAnalysisResult) -> float:
        """Weighted form score 0–100."""
        score = 100.0

        # Countermovement depth: optimal 90°–110°
        opt_low, opt_high = 70.0, 110.0
        if r.deepest_knee_angle < opt_low:
            score -= min(15, (opt_low - r.deepest_knee_angle) * 0.5)
        elif r.deepest_knee_angle > opt_high:
            score -= min(10, (r.deepest_knee_angle - opt_high) * 0.3)

        # Landing stiffness
        if r.landing_risk_colour == "red":
            score -= 25
        elif r.landing_risk_colour == "amber":
            score -= 10

        # Valgus
        max_valgus = max(abs(r.peak_knee_valgus_left), abs(r.peak_knee_valgus_right))
        if max_valgus > 10:
            score -= 20
        elif max_valgus > 5:
            score -= 8

        # Symmetry
        if abs(r.asymmetry_index) > 20:
            score -= 10
        elif abs(r.asymmetry_index) > 10:
            score -= 4

        return max(0.0, min(100.0, score))

    def _generate_warnings(self, r: JumpAnalysisResult):
        warnings, tips = [], []
        if r.landing_risk_colour == "red":
            warnings.append("⚠️ Stiff landing detected — high ACL/patellar risk!")
            tips.append("Practice depth drops to box with silent 'stick' landings.")
        if abs(r.peak_knee_valgus_left) > 10 or abs(r.peak_knee_valgus_right) > 10:
            warnings.append("⚠️ Knee valgus collapse on landing — ACL risk!")
            tips.append("Strengthen glute medius: banded lateral monster walks, clamshells.")
        if r.deepest_knee_angle > 120:
            warnings.append("Shallow countermovement — reduce jump height contribution.")
            tips.append("Aim for ~90°–100° knee flexion in dip phase for maximum power.")
        if abs(r.asymmetry_index) > 15:
            warnings.append(f"Bilateral asymmetry {r.asymmetry_index:.1f}% — check dominant leg loading.")
            tips.append("Single-leg RDL and Bulgarian split squats to equalise limb strength.")
        return warnings, tips
