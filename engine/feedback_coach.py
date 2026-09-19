"""
KhelDrishti — Feedback & Coaching Core
======================================
Rule-based expert system: form score, injury risk, bilingual cues, drills.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CoachingTip:
    code: str
    severity: str  # green | amber | red
    en: str
    hi: str
    drill_en: str = ""
    drill_hi: str = ""


@dataclass
class InjuryRiskProfile:
    level: str  # Safe | Caution | High Risk
    colour: str
    factors: List[str] = field(default_factory=list)


@dataclass
class CoachingReport:
    form_score: float
    risk: InjuryRiskProfile
    tips: List[CoachingTip]
    language: str = "en"
    summary_en: str = ""
    summary_hi: str = ""
    drills: List[Dict[str, str]] = field(default_factory=list)


DRILL_BANK = {
    "valgus": {
        "en": "Banded lateral monster walks + single-leg Romanian deadlifts (3×12 each side).",
        "hi": "बैंडेड मॉन्स्टर वॉक और सिंगल-लेग RDL — हर तरफ़ 3×12।",
    },
    "stiff_landing": {
        "en": "Depth drops to a box with silent landings + plyometric deceleration (3×6).",
        "hi": "बॉक्स पर साइलेंट लैंडिंग डेप्थ-ड्रॉप और डिसेलेरेशन ड्रिल — 3×6।",
    },
    "overstride": {
        "en": "High-knee A-skips and wall-drive posture drills (3×20 m).",
        "hi": "हाई-नी ड्रिल और वॉल-ड्राइव पोस्चर — 3×20 मीटर।",
    },
    "cocking": {
        "en": "Banded external shoulder rotations + wall-toss hitting mechanics (3×15).",
        "hi": "बैंडेड शोल्डर रोटेशन और वॉल-टॉस स्पाइक मैकेनिक्स — 3×15।",
    },
    "butt_wink": {
        "en": "Goblet squats to a box just above the tuck + 90/90 hip mobility.",
        "hi": "बट-विंक से ऊपर बॉक्स गोब्लेट स्क्वाट और हिप मोबिलिटी।",
    },
    "cadence": {
        "en": "Metronome strides at 170 spm, 6 × 30 seconds.",
        "hi": "170 स्टेप/मिनट मेट्रोनोम रन — 6×30 सेकंड।",
    },
}


def _risk_from_score_and_flags(score: float, red_flags: int, amber_flags: int) -> InjuryRiskProfile:
    if red_flags > 0 or score < 55:
        return InjuryRiskProfile("High Risk", "red")
    if amber_flags > 0 or score < 75:
        return InjuryRiskProfile("Caution", "amber")
    return InjuryRiskProfile("Safe", "green")


class FeedbackCoach:
    """Converts analyser metrics into bilingual coaching output."""

    def coach(self, analysis: Any, language: str = "en") -> CoachingReport:
        data = analysis if isinstance(analysis, dict) else getattr(analysis, "__dict__", {})
        movement = data.get("movement_type", "Athletic Movement")
        score = float(data.get("form_score", 70.0))
        warnings = list(data.get("warnings") or [])
        native_tips = list(data.get("tips") or [])

        tips: List[CoachingTip] = []
        red_flags = amber_flags = 0
        factors: List[str] = []
        drills: List[Dict[str, str]] = []

        def add(code, severity, en, hi, drill_key=None):
            nonlocal red_flags, amber_flags
            drill = DRILL_BANK.get(drill_key, {})
            tips.append(CoachingTip(
                code=code, severity=severity, en=en, hi=hi,
                drill_en=drill.get("en", ""), drill_hi=drill.get("hi", ""),
            ))
            if drill:
                drills.append({"en": drill["en"], "hi": drill["hi"], "code": drill_key})
            if severity == "red":
                red_flags += 1
                factors.append(en)
            elif severity == "amber":
                amber_flags += 1
                factors.append(en)

        # Jump-specific
        landing = str(data.get("landing_stiffness", "")).lower()
        if "stiff" in landing or data.get("landing_risk_colour") == "red":
            add("stiff_landing", "red",
                "Stiff-legged landing — high ACL / patellar-tendon load.",
                "कड़ी लैंडिंग — ACL और घुटने पर बहुत दबाव।",
                "stiff_landing")
        valgus_l = abs(float(data.get("peak_knee_valgus_left", data.get("peak_valgus", data.get("peak_valgus_landing", 0.0)))))
        valgus_r = abs(float(data.get("peak_knee_valgus_right", 0.0)))
        max_valgus = max(valgus_l, valgus_r, abs(float(data.get("peak_valgus_landing", 0.0) or 0)))
        if max_valgus > 10 or data.get("valgus_risk") == "High Risk":
            add("valgus", "red",
                "Dynamic knee valgus — inward collapse is a leading non-contact ACL risk.",
                "घुटने अंदर गिर रहे हैं (वैल्गस) — ACL चोट का बड़ा खतरा।",
                "valgus")
        elif max_valgus > 5:
            add("valgus_mild", "amber",
                "Mild knee valgus — keep knees tracking over the toes.",
                "हल्का वैल्गस — घुटने पंजों के ऊपर रखें।",
                "valgus")

        if data.get("overstride_detected"):
            add("overstride", "amber",
                "Overstriding — the foot is landing ahead of the hips, increasing braking force.",
                "ओवरस्टाइड — पैर कूल्हे से आगे गिर रहा है, ब्रेकिंग फोर्स बढ़ रही है।",
                "overstride")

        cadence = float(data.get("cadence_spm") or 0)
        if cadence and cadence < 160:
            add("cadence", "amber",
                f"Cadence is {cadence:.0f} steps/min. Grassroots target is 170–180 spm.",
                f"कैडेंस {cadence:.0f} स्टेप/मिनट है। लक्ष्य 170–180 रखें।",
                "cadence")

        if data.get("cocking_ok") is False:
            add("cocking", "amber",
                "Hitting arm is not cocked (>90° elbow) before contact — less whip, more shoulder strain.",
                "स्पाइक से पहले कोहनी 90° से ज़्यादा कॉक नहीं हो रही — कंधे पर बोझ।",
                "cocking")

        if data.get("butt_wink"):
            add("butt_wink", "amber",
                "Butt wink at the bottom — lumbar spine is flexing under load.",
                "नीचे बट-विंक — कमर गोल हो रही है।",
                "butt_wink")

        asym = abs(float(data.get("asymmetry_index", data.get("landing_asymmetry", data.get("symmetry_index", 0.0))) or 0))
        if asym > 15:
            add("asymmetry", "amber",
                f"Bilateral asymmetry {asym:.1f}% — even out left/right loading.",
                f"बाएँ-दाएँ असंतुलन {asym:.1f}% — दोनों पैर बराबर लोड करें।",
                "valgus")

        # Attach analyser-native tips as extra green/amber notes
        for i, t in enumerate(native_tips[:4]):
            if not any(t[:40] in c.en for c in tips):
                tips.append(CoachingTip(
                    code=f"native_{i}", severity="green",
                    en=t,
                    hi=t,  # native strings stay English if no mapping
                ))

        if not tips:
            tips.append(CoachingTip(
                code="solid",
                severity="green",
                en="Solid mechanics. Keep filming weekly and chase a higher form score.",
                hi="मेकेनिक्स अच्छे हैं। हर हफ़्ते फिल्म करते रहें और स्कोर बढ़ाएँ।",
            ))

        risk = _risk_from_score_and_flags(score, red_flags, amber_flags)
        risk.factors = factors

        summary_en = (
            f"{movement}: form score {score:.0f}/100, injury risk {risk.level}. "
            + (warnings[0] if warnings else "No critical flags.")
        )
        summary_hi = (
            f"{movement}: फ़ॉर्म स्कोर {score:.0f}/100, चोट जोखिम {self._risk_hi(risk.level)}. "
            + ("सावधानी बरतें।" if risk.colour != "green" else "आगे बढ़ते रहें।")
        )

        # unique drills
        seen = set()
        uniq = []
        for d in drills:
            if d["code"] not in seen:
                seen.add(d["code"])
                uniq.append(d)

        return CoachingReport(
            form_score=score,
            risk=risk,
            tips=tips,
            language=language,
            summary_en=summary_en,
            summary_hi=summary_hi,
            drills=uniq,
        )

    @staticmethod
    def _risk_hi(level: str) -> str:
        return {"Safe": "सुरक्षित", "Caution": "सावधानी", "High Risk": "उच्च जोखिम"}.get(level, level)

    def live_cue(self, metrics: Dict[str, Any], language: str = "en") -> Optional[str]:
        """One-shot spoken cue for the webcam stream."""
        valgus = max(abs(metrics.get("left_valgus", 0)), abs(metrics.get("right_valgus", 0)))
        knee = min(metrics.get("left_knee", 180), metrics.get("right_knee", 180))
        if valgus > 10:
            return "घुटने बाहर रखो! Knees out!" if language == "hi" else "Knees out! Stop the valgus collapse."
        if knee < 30 and metrics.get("phase", "").lower().find("land") >= 0:
            return "नरम लैंडिंग! Soften the landing." if language == "hi" else "Soften the landing — bend the knees."
        return None
