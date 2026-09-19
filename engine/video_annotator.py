"""
KhelDrishti — Video Annotator
=============================
Sports-tech HUD: glowing skeleton, joint arcs, telemetry bar, keyframe stamps.
"""

from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np
import cv2

from engine.pose_detector import LANDMARK_NAMES, PoseResult

# Palette (BGR)
VOLT = (135, 255, 0)
CYAN = (255, 245, 0)
AMBER = (0, 184, 255)
CRIMSON = (102, 51, 255)
WHITE = (240, 248, 255)
PANEL = (25, 15, 11)

POSE_EDGES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
    (15, 17), (15, 19), (16, 18), (16, 20),
]


def _risk_colour(level: str) -> Tuple[int, int, int]:
    level = (level or "green").lower()
    if level in ("red", "high risk", "high"):
        return CRIMSON
    if level in ("amber", "caution", "yellow"):
        return AMBER
    return VOLT


def _px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)


def _draw_glow_line(img, p1, p2, colour, thickness=3):
    overlay = img.copy()
    cv2.line(overlay, p1, p2, colour, thickness + 6)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
    cv2.line(img, p1, p2, colour, thickness)


def _draw_arc(img, vertex, p_a, p_c, colour, label: str):
    """Quarter-circle-ish arc at a joint showing the interior angle."""
    va = np.array(p_a) - np.array(vertex)
    vc = np.array(p_c) - np.array(vertex)
    if np.linalg.norm(va) < 4 or np.linalg.norm(vc) < 4:
        return
    ang1 = math.degrees(math.atan2(va[1], va[0]))
    ang2 = math.degrees(math.atan2(vc[1], vc[0]))
    start, end = sorted([ang1, ang2])
    radius = 28
    cv2.ellipse(img, vertex, (radius, radius), 0, start, end, colour, 2)
    cv2.putText(img, label, (vertex[0] + 18, vertex[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)


class VideoAnnotator:
    def annotate_frame(
        self,
        bgr: np.ndarray,
        pose: PoseResult,
        hud: Dict[str, Any],
        keyframe_label: Optional[str] = None,
    ) -> np.ndarray:
        img = bgr.copy()
        h, w = img.shape[:2]
        colour = _risk_colour(hud.get("risk_colour", "green"))

        if pose.detected and pose.landmarks:
            pts = []
            for i, lm in enumerate(pose.landmarks):
                pts.append(_px(lm, w, h) if lm.visibility > 0.25 else None)

            for a, b in POSE_EDGES:
                if a < len(pts) and b < len(pts) and pts[a] and pts[b]:
                    _draw_glow_line(img, pts[a], pts[b], colour, 3)

            for i, p in enumerate(pts):
                if p is None:
                    continue
                cv2.circle(img, p, 5, colour, -1, cv2.LINE_AA)
                cv2.circle(img, p, 8, CYAN, 1, cv2.LINE_AA)

            # Knee arcs
            lk = hud.get("left_knee")
            rk = hud.get("right_knee")
            if lk is not None and pts[23] and pts[25] and pts[27]:
                _draw_arc(img, pts[25], pts[23], pts[27], CYAN, f"{lk:.0f}°")
            if rk is not None and pts[24] and pts[26] and pts[28]:
                _draw_arc(img, pts[26], pts[24], pts[28], CYAN, f"{rk:.0f}°")

        # Telemetry bar
        bar_h = 56
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), PANEL, -1)
        cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)
        phase = str(hud.get("phase", "—"))
        score = hud.get("form_score")
        vel = hud.get("velocity")
        angle = hud.get("primary_angle")
        bits = [
            f"PHASE  {phase}",
            f"ANGLE  {angle:.0f}°" if angle is not None else None,
            f"VEL  {vel:.2f}" if vel is not None else None,
            f"FORM  {score:.0f}" if score is not None else None,
        ]
        text = "   |   ".join(b for b in bits if b)
        cv2.putText(img, "KHELDRISHTI", (16, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, VOLT, 2, cv2.LINE_AA)
        cv2.putText(img, text, (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.48, WHITE, 1, cv2.LINE_AA)

        if keyframe_label:
            cv2.rectangle(img, (w - 280, 70), (w - 16, 110), PANEL, -1)
            cv2.putText(img, keyframe_label.upper(), (w - 268, 98),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, AMBER, 2, cv2.LINE_AA)

        cue = hud.get("cue")
        if cue:
            cv2.putText(img, str(cue)[:70], (16, h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)
        return img

    def write_video(
        self,
        frames_bgr: List[np.ndarray],
        poses: List[PoseResult],
        hud_per_frame: List[Dict[str, Any]],
        out_path: str,
        fps: float = 30.0,
        keyframes: Optional[Dict[int, str]] = None,
    ) -> str:
        if not frames_bgr:
            raise ValueError("No frames to annotate")
        h, w = frames_bgr[0].shape[:2]
        keyframes = keyframes or {}
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        n = min(len(frames_bgr), len(poses), len(hud_per_frame))
        for i in range(n):
            annotated = self.annotate_frame(
                frames_bgr[i], poses[i], hud_per_frame[i], keyframes.get(i)
            )
            writer.write(annotated)
        writer.release()
        return out_path
