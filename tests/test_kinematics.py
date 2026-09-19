"""KhelDrishti kinematics, coaching, and phase classification tests."""

import math
from pathlib import Path

import pytest

from engine.kinematics import (
    calculate_angle_2d,
    calculate_angle_3d,
    calculate_knee_valgus_angle,
    calculate_torso_lean,
    calculate_jump_height_from_flight_time,
    calculate_takeoff_velocity,
    calculate_jump_kinematics,
    calculate_bilateral_asymmetry,
    classify_squat_depth,
    classify_valgus_risk,
    classify_landing_stiffness,
)
from engine.feedback_coach import FeedbackCoach
from engine.pose_detector import Landmark, PoseResult, pose_result_from_xy
from engine.movement_analyzers.vertical_jump import VerticalJumpAnalyzer, JumpPhase
from engine.movement_analyzers.squat_analysis import SquatAnalyzer
from sample_data.generate_sample_data import sagittal_body


def test_right_angle_2d():
    a, b, c = (0, 1), (0, 0), (1, 0)
    assert abs(calculate_angle_2d(a, b, c) - 90) < 1e-6


def test_straight_leg_180():
    a, b, c = (0, 0), (1, 0), (2, 0)
    assert abs(calculate_angle_2d(a, b, c) - 180) < 0.01


def test_acute_angle():
    a, b, c = (1, 1), (0, 0), (1, 0)
    ang = calculate_angle_2d(a, b, c)
    assert 40 < ang < 50


def test_angle_3d_matches_2d_on_plane():
    a, b, c = (0, 1, 0), (0, 0, 0), (1, 0, 0)
    assert abs(calculate_angle_3d(a, b, c) - 90) < 1e-6


def test_knee_valgus_inward_knee():
    hip, knee, ankle = (0.5, 0.4), (0.42, 0.6), (0.5, 0.8)
    val = calculate_knee_valgus_angle(hip, knee, ankle)
    assert abs(val) > 2


def test_aligned_leg_low_valgus():
    hip, knee, ankle = (0.5, 0.3), (0.5, 0.5), (0.5, 0.8)
    assert abs(calculate_knee_valgus_angle(hip, knee, ankle)) < 1.0


def test_jump_height_flight_time():
    # 0.5 s flight → h = 9.81 * 0.25 / 8 = 0.30656 m ≈ 30.66 cm
    h = calculate_jump_height_from_flight_time(0.5)
    assert 30 < h < 32
    v = calculate_takeoff_velocity(h)
    combo = calculate_jump_kinematics(0.5)
    assert abs(combo["height_cm"] - h) < 1e-9
    assert abs(combo["impulse_velocity_ms"] - v) < 1e-9


def test_asymmetry_and_classifiers():
    assert abs(calculate_bilateral_asymmetry(100, 100)) < 1e-6
    assert classify_squat_depth(150) == "Quarter Squat"
    assert classify_squat_depth(80) == "Below Parallel (ATG)"
    level, colour = classify_valgus_risk(12)
    assert level == "High Risk" and colour == "red"
    stiff, col = classify_landing_stiffness(20)
    assert "Stiff" in stiff and col == "red"


def test_torso_lean_forward():
    shoulder, hip = (0.6, 0.3), (0.5, 0.5)
    lean = calculate_torso_lean(shoulder, hip)
    assert abs(lean) > 5


def _cmj_poses(n=60):
    poses = []
    for i in range(n):
        t = i / (n - 1)
        if t < 0.3:
            kf, hy = 175 - 80 * (t / 0.3), 0.58 + 0.08 * (t / 0.3)
        elif t < 0.45:
            kf, hy = 95 + 80 * ((t - 0.3) / 0.15), 0.66 - 0.16 * ((t - 0.3) / 0.15)
        elif t < 0.7:
            kf, hy = 170, 0.40
        else:
            kf, hy = 70, 0.62
        xs, ys, zs, vis = sagittal_body(0.5, hy, kf, 30, arm_up=0.4)
        poses.append(pose_result_from_xy(xs, ys, zs, vis, i))
    return poses


def test_vertical_jump_phase_classification():
    result = VerticalJumpAnalyzer(fps=30).analyse(_cmj_poses())
    assert result.deepest_knee_angle < 140
    assert JumpPhase.FLIGHT in result.phases or JumpPhase.TAKEOFF in result.phases
    assert 0 <= result.form_score <= 100
    assert result.jump_height_cm >= 0


def test_squat_bottom_detected():
    poses = []
    for i in range(40):
        t = i / 39
        kf = 175 - 90 * math.sin(math.pi * t)
        xs, ys, zs, vis = sagittal_body(0.5, 0.58 + 0.1 * math.sin(math.pi * t), kf, 40)
        poses.append(pose_result_from_xy(xs, ys, zs, vis, i))
    r = SquatAnalyzer(fps=30).analyse(poses)
    assert r.min_knee_angle < 120
    assert r.bottom_frame >= 0
    assert r.depth_class != "Unknown"


def test_feedback_bilingual_valgus():
    coach = FeedbackCoach()
    report = coach.coach({
        "movement_type": "Vertical Jump (CMJ)",
        "form_score": 62,
        "peak_knee_valgus_left": 14,
        "peak_knee_valgus_right": 11,
        "landing_stiffness": "Stiff Landing — ACL Risk!",
        "landing_risk_colour": "red",
        "warnings": ["valgus"],
        "tips": ["monster walks"],
    }, language="hi")
    assert report.risk.colour == "red"
    texts = " ".join(t.hi for t in report.tips)
    assert "वैल्गस" in texts or "ACL" in texts
    assert any(t.en for t in report.tips)
    assert report.summary_hi
    assert any(d.get("hi") for d in report.drills)


def test_live_cue_knees_out():
    cue = FeedbackCoach().live_cue({"left_valgus": 12, "right_valgus": 1, "left_knee": 90, "right_knee": 90}, "en")
    assert cue and ("valgus" in cue.lower() or "Knees" in cue)
