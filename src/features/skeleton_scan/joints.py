from dataclasses import dataclass
from typing import Tuple

import mediapipe as mp


@dataclass(frozen=True)
class JointDefinition:
    name: str
    proximal: int
    pivot: int
    distal: int
    visibility_threshold: float = 0.5


mp_pose = mp.solutions.pose


DEFAULT_JOINTS: Tuple[JointDefinition, ...] = (
    JointDefinition(
        name="Left Shoulder",
        proximal=mp_pose.PoseLandmark.LEFT_HIP.value,
        pivot=mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        distal=mp_pose.PoseLandmark.LEFT_ELBOW.value,
    ),
    JointDefinition(
        name="Right Shoulder",
        proximal=mp_pose.PoseLandmark.RIGHT_HIP.value,
        pivot=mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        distal=mp_pose.PoseLandmark.RIGHT_ELBOW.value,
    ),
)
