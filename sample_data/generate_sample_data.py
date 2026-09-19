"""
Generate synthetic biomechanics test videos + landmark tracks.
A sagittal stick-athlete performs CMJ, squat, gait, and spike-like motions.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.pose_detector import LANDMARK_NAMES  # noqa: E402

OUT = Path(__file__).resolve().parent / "videos"
OUT.mkdir(parents=True, exist_ok=True)

W, H = 960, 540
FPS = 30


def _empty_skeleton():
    xs = [0.5] * 33
    ys = [0.5] * 33
    zs = [0.0] * 33
    vis = [0.99] * 33
    return xs, ys, zs, vis


def _set(xs, ys, idx, x, y):
    xs[idx] = x
    ys[idx] = y


def sagittal_body(hip_x, hip_y, knee_flex_deg, hip_flex_deg, arm_up=0.0, facing=1.0):
    """
    Rough sagittal skeleton. Y increases downward (image space).
    knee_flex_deg: interior knee angle (180 = straight).
    """
    xs, ys, zs, vis = _empty_skeleton()
    torso = 0.18
    thigh = 0.14
    shank = 0.13
    upper_arm = 0.10
    forearm = 0.09

    kf = math.radians(180 - knee_flex_deg)  # 0 = straight
    hf = math.radians(hip_flex_deg)

    # Hips
    _set(xs, ys, 23, hip_x - 0.02, hip_y)
    _set(xs, ys, 24, hip_x + 0.02, hip_y)
    # Shoulders
    sh_x = hip_x + facing * 0.02 * math.sin(hf)
    sh_y = hip_y - torso * math.cos(hf * 0.35)
    _set(xs, ys, 11, sh_x - 0.03, sh_y)
    _set(xs, ys, 12, sh_x + 0.03, sh_y)
    # Head
    _set(xs, ys, 0, sh_x, sh_y - 0.08)
    for i in range(1, 11):
        _set(xs, ys, i, sh_x + 0.01 * ((i % 3) - 1), sh_y - 0.07)

    def leg(hip_i, knee_i, ank_i, heel_i, toe_i, side):
        hx, hy = xs[hip_i], ys[hip_i]
        # thigh: down and slightly back when flexing
        kx = hx - facing * thigh * math.sin(hf + kf * 0.3) + side * 0.01
        ky = hy + thigh * math.cos(hf * 0.5)
        _set(xs, ys, knee_i, kx, ky)
        ax = kx + facing * shank * math.sin(kf * 0.5)
        ay = ky + shank * math.cos(kf * 0.35)
        _set(xs, ys, ank_i, ax, ay)
        _set(xs, ys, heel_i, ax - facing * 0.02, ay + 0.01)
        _set(xs, ys, toe_i, ax + facing * 0.03, ay + 0.01)

    leg(23, 25, 27, 29, 31, -1)
    leg(24, 26, 28, 30, 32, 1)

    # Arms
    for sh_i, el_i, wr_i, sign in ((11, 13, 15, -1), (12, 14, 16, 1)):
        sx, sy = xs[sh_i], ys[sh_i]
        reach = arm_up  # 0 hanging, 1 overhead
        elx = sx + facing * upper_arm * (0.2 + reach)
        ely = sy + upper_arm * (1 - 1.6 * reach)
        wrx = elx + facing * forearm * (0.2 + reach)
        wry = ely + forearm * (1 - 1.8 * reach)
        _set(xs, ys, el_i, elx + sign * 0.01, ely)
        _set(xs, ys, wr_i, wrx + sign * 0.01, wry)
        _set(xs, ys, el_i + 2, wrx, wry)  # pinky approx
        _set(xs, ys, el_i + 4, wrx, wry)
        _set(xs, ys, el_i + 6, wrx, wry)

    return xs, ys, zs, vis


def render_frame(xs, ys, title: str, phase: str) -> np.ndarray:
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = (18, 12, 8)
    # pitch lines
    for y in range(0, H, 40):
        cv2.line(img, (0, y), (W, y), (32, 28, 22), 1)

    edges = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24),
        (23, 25), (25, 27), (27, 31),
        (24, 26), (26, 28), (28, 32),
        (0, 11), (0, 12),
    ]
    pts = [(int(xs[i] * W), int(ys[i] * H)) for i in range(33)]
    for a, b in edges:
        cv2.line(img, pts[a], pts[b], (0, 255, 135), 4, cv2.LINE_AA)
    for i, p in enumerate(pts):
        col = (0, 245, 255) if i == 0 else (0, 255, 135)
        cv2.circle(img, p, 6, col, -1, cv2.LINE_AA)

    # filled torso
    hull = np.array([pts[11], pts[12], pts[24], pts[23]], dtype=np.int32)
    overlay = img.copy()
    cv2.fillConvexPoly(overlay, hull, (40, 90, 60))
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
    cv2.circle(img, pts[0], 18, (180, 170, 140), -1, cv2.LINE_AA)

    cv2.putText(img, title, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 135), 2, cv2.LINE_AA)
    cv2.putText(img, phase, (20, H - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 184, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "SYNTHETIC BIOMECHANICS TRACK", (W - 420, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def write_clip(name: str, frames_spec):
    """frames_spec: list of (xs, ys, zs, vis, phase)"""
    path = OUT / f"{name}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    wr = cv2.VideoWriter(str(path), fourcc, FPS, (W, H))
    payload = {"fps": FPS, "width": W, "height": H, "frames": []}
    title = name.replace("_", " ").upper()
    for xs, ys, zs, vis, phase in frames_spec:
        wr.write(render_frame(xs, ys, title, phase))
        payload["frames"].append({"x": xs, "y": ys, "z": zs, "v": vis, "phase": phase})
    wr.release()
    (OUT / f"{name}.landmarks.json").write_text(json.dumps(payload), encoding="utf-8")
    print(f"Wrote {path} ({len(frames_spec)} frames)")
    return path


def lerp(a, b, t):
    return a + (b - a) * t


def make_jump():
    spec = []
    n = 90
    for i in range(n):
        t = i / (n - 1)
        if t < 0.15:
            phase, kf, hf, hy, arm = "Unweighting", lerp(175, 140, t / 0.15), lerp(10, 25, t / 0.15), 0.58, 0.1
        elif t < 0.32:
            u = (t - 0.15) / 0.17
            phase, kf, hf, hy, arm = "Braking", lerp(140, 95, u), lerp(25, 55, u), lerp(0.58, 0.64, u), 0.15
        elif t < 0.45:
            u = (t - 0.32) / 0.13
            phase, kf, hf, hy, arm = "Propulsion", lerp(95, 170, u), lerp(55, 8, u), lerp(0.64, 0.50, u), lerp(0.15, 0.6, u)
        elif t < 0.52:
            phase, kf, hf, hy, arm = "Takeoff", 175, 5, 0.48, 0.7
        elif t < 0.72:
            u = (t - 0.52) / 0.20
            # parabolic flight in image Y
            flight = 4 * u * (1 - u)
            phase, kf, hf, hy, arm = "Flight", 170, 5, 0.48 - 0.16 * flight, 0.85
        elif t < 0.82:
            u = (t - 0.72) / 0.10
            phase, kf, hf, hy, arm = "Landing", lerp(160, 70, u), lerp(8, 40, u), lerp(0.50, 0.62, u), 0.3
        else:
            u = (t - 0.82) / 0.18
            phase, kf, hf, hy, arm = "Complete", lerp(70, 175, u), lerp(40, 8, u), lerp(0.62, 0.58, u), 0.1
        xs, ys, zs, vis = sagittal_body(0.48, hy, kf, hf, arm_up=arm)
        spec.append((xs, ys, zs, vis, phase))
    return write_clip("vertical_jump_cmj", spec)


def make_squat():
    spec = []
    n = 80
    for i in range(n):
        t = i / (n - 1)
        if t < 0.45:
            u = t / 0.45
            phase, kf, hf, hy = "Descent", lerp(175, 85, u), lerp(8, 70, u), lerp(0.56, 0.68, u)
        elif t < 0.55:
            phase, kf, hf, hy = "Bottom", 82, 75, 0.69
        else:
            u = (t - 0.55) / 0.45
            phase, kf, hf, hy = "Ascent", lerp(82, 175, u), lerp(75, 8, u), lerp(0.69, 0.56, u)
        xs, ys, zs, vis = sagittal_body(0.50, hy, kf, hf, arm_up=0.35)
        spec.append((xs, ys, zs, vis, phase))
    return write_clip("squat_atg", spec)


def make_gait():
    spec = []
    n = 90
    for i in range(n):
        t = i / (n - 1)
        cycle = (t * 6) % 1.0  # several strides
        phase = "Left Stance" if cycle < 0.5 else "Right Stance"
        kf = 160 + 20 * math.sin(cycle * math.pi * 2)
        lean = 8
        hip_x = 0.30 + 0.40 * t
        xs, ys, zs, vis = sagittal_body(hip_x, 0.57, kf, lean, arm_up=0.2)
        # exaggerate overstride on right ankle
        xs[28] = xs[24] + 0.08
        spec.append((xs, ys, zs, vis, phase))
    return write_clip("running_gait", spec)


def make_spike():
    spec = []
    n = 75
    for i in range(n):
        t = i / (n - 1)
        if t < 0.35:
            phase, arm, kf, hy = "Approach", 0.2, 150, 0.58
        elif t < 0.5:
            phase, arm, kf, hy = "Cocking", lerp(0.2, 0.95, (t - 0.35) / 0.15), 100, 0.55
        elif t < 0.62:
            phase, arm, kf, hy = "Contact", 1.0, 160, 0.42
        else:
            u = (t - 0.62) / 0.38
            phase, arm, kf, hy = "Landing", lerp(0.7, 0.2, u), lerp(140, 80, min(1, u * 2)), lerp(0.50, 0.62, min(1, u * 1.5))
        xs, ys, zs, vis = sagittal_body(0.52, hy, kf, 20, arm_up=arm)
        spec.append((xs, ys, zs, vis, phase))
    return write_clip("volleyball_spike", spec)


if __name__ == "__main__":
    make_jump()
    make_squat()
    make_gait()
    make_spike()
    print("Sample data ready in", OUT)
