from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class JointMetrics:
    joint_name: str
    min_angle: Optional[float] = None
    max_angle: Optional[float] = None
    mean_angle: Optional[float] = None
    samples: int = 0
    duration: float = 0.0
    angles: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)


@dataclass
class QuestionSet:
    prompt: str


@dataclass
class PatientFeedback:
    response: str
    summary: str


@dataclass
class SessionRecord:
    joint_metrics: JointMetrics
    questions: QuestionSet
    patient_feedback: PatientFeedback
