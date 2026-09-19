from engine.movement_analyzers.vertical_jump import VerticalJumpAnalyzer, JumpAnalysisResult
from engine.movement_analyzers.volleyball_spike import VolleyballSpikeAnalyzer, SpikeAnalysisResult
from engine.movement_analyzers.running_gait import RunningGaitAnalyzer, GaitAnalysisResult
from engine.movement_analyzers.squat_analysis import SquatAnalyzer, SquatAnalysisResult

ANALYZERS = {
    "vertical_jump": VerticalJumpAnalyzer,
    "volleyball_spike": VolleyballSpikeAnalyzer,
    "running_gait": RunningGaitAnalyzer,
    "squat": SquatAnalyzer,
}

__all__ = [
    "ANALYZERS",
    "VerticalJumpAnalyzer",
    "JumpAnalysisResult",
    "VolleyballSpikeAnalyzer",
    "SpikeAnalysisResult",
    "RunningGaitAnalyzer",
    "GaitAnalysisResult",
    "SquatAnalyzer",
    "SquatAnalysisResult",
]
