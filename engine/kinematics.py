"""
KhelDrishti — Kinematics & Vector Math Core
============================================
Geometric calculations for joint angles, velocities, bilateral symmetry,
knee valgus detection, and jump kinematics.
"""

import math
import numpy as np
from typing import List, Tuple, Optional


# ─── Vector Utilities ────────────────────────────────────────────────────────

def _to_array(pt) -> np.ndarray:
    """Convert a landmark/tuple/list to numpy array."""
    if hasattr(pt, 'x'):
        return np.array([pt.x, pt.y, pt.z if hasattr(pt, 'z') else 0.0])
    return np.asarray(pt, dtype=float)


def _unit(v: np.ndarray) -> np.ndarray:
    """Return unit vector. Returns zeros if magnitude is near zero."""
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros_like(v)


# ─── 2D / 3D Joint Angle Calculations ────────────────────────────────────────

def calculate_angle_2d(a, b, c) -> float:
    """
    Interior angle (degrees) at vertex B formed by rays B→A and B→C.
    Works in the XY image plane (ignores Z).

    Returns
    -------
    float  Angle in degrees [0, 180].
    """
    a, b, c = _to_array(a)[:2], _to_array(b)[:2], _to_array(c)[:2]
    ba = a - b
    bc = c - b
    cos_angle = np.clip(
        np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9),
        -1.0, 1.0
    )
    return math.degrees(math.acos(cos_angle))


def calculate_angle_3d(a, b, c) -> float:
    """
    Interior angle (degrees) at vertex B using full 3-D landmark coordinates.

    Returns
    -------
    float  Angle in degrees [0, 180].
    """
    a, b, c = _to_array(a), _to_array(b), _to_array(c)
    ba = _unit(a - b)
    bc = _unit(c - b)
    cos_angle = np.clip(np.dot(ba, bc), -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


# ─── Biomechanical Risk Metrics ───────────────────────────────────────────────

def calculate_knee_valgus_angle(hip, knee, ankle) -> float:
    """
    Coronal-plane knee valgus / varus angle (degrees).

    A positive value indicates **valgus** (knee collapses inward relative to
    the hip–ankle axis). Values > 10° are associated with ACL injury risk.

    Strategy
    --------
    Project the hip→ankle vector and hip→knee vector onto the frontal (XZ)
    plane (MediaPipe normalised coords: X = right, Y = down, Z = depth).
    The signed deviation of the knee from the hip–ankle line gives valgus.
    """
    hip_arr   = _to_array(hip)
    knee_arr  = _to_array(knee)
    ankle_arr = _to_array(ankle)

    # Frontal-plane projection (use X and Y for image space)
    ha = ankle_arr[:2] - hip_arr[:2]   # hip → ankle
    hk = knee_arr[:2] - hip_arr[:2]    # hip → knee

    ha_len = np.linalg.norm(ha)
    if ha_len < 1e-9:
        return 0.0

    # Project knee onto hip-ankle line
    t = np.dot(hk, ha) / (ha_len ** 2)
    closest = hip_arr[:2] + t * ha

    # Lateral deviation of knee from hip–ankle line
    deviation = knee_arr[:2] - closest
    deviation_mag = np.linalg.norm(deviation)

    # Signed angle: positive = valgus (X deviation toward midline)
    # Use cross-product sign to determine medial vs lateral
    cross_z = ha[0] * deviation[1] - ha[1] * deviation[0]
    sign = -1.0 if cross_z < 0 else 1.0

    angle = math.degrees(math.atan2(deviation_mag, ha_len * abs(t) + 1e-9))
    return sign * angle


def calculate_torso_lean(shoulder, hip) -> float:
    """
    Forward/backward trunk lean in degrees from vertical (Y-axis).

    Positive = forward lean, Negative = backward lean.
    Optimal range for running: 5°–10° forward.
    """
    s = _to_array(shoulder)[:2]
    h = _to_array(hip)[:2]
    vec = s - h  # hip → shoulder
    # Angle from vertical (negative Y in image space is up)
    # vec[1] is negative when shoulder is above hip (standard image coords)
    angle_from_vertical = math.degrees(math.atan2(abs(vec[0]), abs(vec[1]) + 1e-9))
    sign = 1.0 if vec[0] > 0 else -1.0   # positive = leaning forward (toward camera right)
    return sign * angle_from_vertical


# ─── Kinematic Derivatives ────────────────────────────────────────────────────

def calculate_linear_velocity(
    pos_series: List[Tuple[float, float]],
    timestamps: List[float],
    alpha: float = 0.4
) -> List[float]:
    """
    Instantaneous linear velocity (units/sec) using finite-difference
    with exponential moving average smoothing.

    Parameters
    ----------
    pos_series  : List of (x, y) positions in normalised coord space.
    timestamps  : Corresponding timestamps in seconds.
    alpha       : EMA smoothing factor (0 < alpha ≤ 1).

    Returns
    -------
    List[float]  Smoothed speed at each frame (magnitude).
    """
    velocities: List[float] = [0.0]
    ema = 0.0
    for i in range(1, len(pos_series)):
        dt = timestamps[i] - timestamps[i - 1]
        if dt < 1e-6:
            velocities.append(velocities[-1])
            continue
        dx = pos_series[i][0] - pos_series[i - 1][0]
        dy = pos_series[i][1] - pos_series[i - 1][1]
        raw_vel = math.sqrt(dx**2 + dy**2) / dt
        ema = alpha * raw_vel + (1 - alpha) * ema
        velocities.append(ema)
    return velocities


def calculate_angular_velocity(
    angle_series: List[float],
    timestamps: List[float]
) -> List[float]:
    """
    Angular velocity (degrees/sec) at each frame using central differences
    where possible, forward/backward at edges.
    """
    n = len(angle_series)
    if n < 2:
        return [0.0] * n

    angular_vels = []
    for i in range(n):
        if i == 0:
            dt = timestamps[1] - timestamps[0] + 1e-9
            dθ = angle_series[1] - angle_series[0]
        elif i == n - 1:
            dt = timestamps[-1] - timestamps[-2] + 1e-9
            dθ = angle_series[-1] - angle_series[-2]
        else:
            dt = timestamps[i + 1] - timestamps[i - 1] + 1e-9
            dθ = angle_series[i + 1] - angle_series[i - 1]
        angular_vels.append(dθ / dt)
    return angular_vels


# ─── Symmetry & Jump Kinematics ───────────────────────────────────────────────

def calculate_bilateral_asymmetry(left_metric: float, right_metric: float) -> float:
    """
    Bilateral asymmetry index (%).
    Positive = left side dominant; Negative = right side dominant.
    Values > 15% indicate clinically significant imbalance.
    """
    mean_val = (abs(left_metric) + abs(right_metric)) / 2.0 + 1e-9
    return ((left_metric - right_metric) / mean_val) * 100.0


def calculate_jump_height_from_flight_time(flight_time_s: float) -> float:
    """
    Estimate vertical jump height (cm) from airborne flight time.

    Physics: h = g * t² / 8  (symmetric parabolic flight)
    where t is total flight time (takeoff → landing).

    Parameters
    ----------
    flight_time_s : Total flight time in seconds.

    Returns
    -------
    float  Estimated jump height in centimetres.
    """
    g = 9.81  # m/s²
    height_m = g * (flight_time_s ** 2) / 8.0
    return height_m * 100.0  # cm


def calculate_takeoff_velocity(jump_height_cm: float) -> float:
    """
    Takeoff velocity (m/s) from jump height using impulse-momentum theorem.
    v = sqrt(2 * g * h)
    """
    g = 9.81
    h = jump_height_cm / 100.0
    return math.sqrt(2 * g * h)


def calculate_jump_kinematics(flight_time: float, takeoff_vel: Optional[float] = None) -> dict:
    """
    Combined jump kinematics: flight-time height vs impulse-momentum velocity.

    h = g * t² / 8
    v = sqrt(2 * g * h)
    """
    height_cm = calculate_jump_height_from_flight_time(flight_time)
    impulse_vel = calculate_takeoff_velocity(height_cm)
    return {
        "height_cm": height_cm,
        "height_in": height_cm / 2.54,
        "takeoff_velocity_ms": takeoff_vel if takeoff_vel is not None else impulse_vel,
        "impulse_velocity_ms": impulse_vel,
        "flight_time_s": flight_time,
    }


def calculate_squat_depth_ratio(hip_y: float, knee_y: float, ankle_y: float) -> float:
    """
    Normalised squat depth as ratio of hip relative to knee.
    
    Returns
    -------
    float   1.0 = parallel, >1.0 = below parallel (ATG), <1.0 = quarter squat.
    Note: In image space Y increases downward, so lower Y = higher position.
    """
    hip_to_knee = abs(hip_y - knee_y) + 1e-9
    knee_to_ankle = abs(knee_y - ankle_y) + 1e-9
    return hip_to_knee / knee_to_ankle


def classify_squat_depth(knee_angle_deg: float) -> str:
    """Classify squat depth from knee flexion angle."""
    if knee_angle_deg >= 140:
        return "Quarter Squat"
    elif knee_angle_deg >= 110:
        return "Half Squat"
    elif knee_angle_deg >= 90:
        return "Parallel"
    else:
        return "Below Parallel (ATG)"


def classify_valgus_risk(valgus_angle_deg: float) -> Tuple[str, str]:
    """
    Classify knee valgus angle into injury risk category.

    Returns
    -------
    (risk_level, colour)
    """
    abs_angle = abs(valgus_angle_deg)
    if abs_angle < 5:
        return ("Safe", "green")
    elif abs_angle < 10:
        return ("Caution", "amber")
    else:
        return ("High Risk", "red")


def classify_landing_stiffness(knee_angle_at_contact_deg: float) -> Tuple[str, str]:
    """
    Stiff landing detection based on knee flexion at initial ground contact.
    < 30° = dangerously stiff (high-risk knee load).
    30°–60° = acceptable
    > 60° = excellent absorption
    """
    if knee_angle_at_contact_deg < 30:
        return ("Stiff Landing — ACL Risk!", "red")
    elif knee_angle_at_contact_deg < 60:
        return ("Moderate Landing", "amber")
    else:
        return ("Soft Landing — Excellent!", "green")
