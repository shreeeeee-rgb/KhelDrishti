"""Pydantic request/response models for KhelDrishti."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    movement: str = Field(default="vertical_jump")
    language: str = Field(default="en")


class JointAngleMetric(BaseModel):
    name: str
    left: Optional[float] = None
    right: Optional[float] = None
    unit: str = "deg"


class MovementPhase(BaseModel):
    name: str
    frame_index: int
    t_seconds: float = 0.0


class CoachingTipModel(BaseModel):
    code: str
    severity: str
    en: str
    hi: str
    drill_en: str = ""
    drill_hi: str = ""


class InjuryRiskProfile(BaseModel):
    level: str
    colour: str
    factors: List[str] = []


class BiomechanicalReport(BaseModel):
    movement: str
    form_score: float
    risk: InjuryRiskProfile
    analysis: Dict[str, Any]
    coaching: Dict[str, Any]
    annotated_video: Optional[str] = None
    keyframes: Dict[str, str] = {}


class HealthResponse(BaseModel):
    status: str
    device: str
    mediapipe: bool
    version: str = "1.0.0"
