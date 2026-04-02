from __future__ import annotations

import time
from typing import Dict, Iterable, List, Tuple

import cv2
import mediapipe as mp
import numpy as np

from ...core.math_utils import calc_angle
from ...core.models import JointMetrics
from .joints import JointDefinition

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


class PoseAnalyzer:
    def __init__(
        self,
        camera_index: int = 0,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
        smoothing_factor: float = 0.2,
        display: bool = True,
    ) -> None:
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to access camera index {camera_index}")
        self.pose = mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity,
        )
        self.smoothing_factor = float(np.clip(smoothing_factor, 0.0, 1.0))
        self.display = display
        self._smoothed_angle: float | None = None

    def capture_joint_range(
        self,
        joint: JointDefinition,
        duration_seconds: float = 10.0,
    ) -> JointMetrics:
        if not self.display and duration_seconds <= 0:
            raise ValueError("Manual completion mode requires display=True to capture key presses.")

        window_title = f"SkeletonScan - {joint.name}"
        start_time = time.time()
        raw_angles: List[float] = []
        timestamps: List[float] = []

        while True:
            if duration_seconds > 0 and time.time() - start_time >= duration_seconds:
                break

            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = self.pose.process(image)
            image.flags.writeable = True
            output_frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                h, w, _ = output_frame.shape
                landmarks = results.pose_landmarks.landmark

                if self._landmarks_visible(landmarks, joint, joint.visibility_threshold):
                    points = self._landmark_points(
                        landmarks,
                        (joint.proximal, joint.pivot, joint.distal),
                        width=w,
                        height=h,
                    )
                    angle = calc_angle(*points)
                    raw_angles.append(angle)
                    self._smoothed_angle = angle if self._smoothed_angle is None else (
                        self.smoothing_factor * angle + (1.0 - self.smoothing_factor) * self._smoothed_angle
                    )
                    timestamps.append(time.time() - start_time)

                    if self.display:
                        self._draw_landmarks(output_frame, results)
                        self._draw_angle_arc(output_frame, points)
                        cv2.putText(
                            output_frame,
                            f"{joint.name} angle: {int(self._smoothed_angle)} deg",
                            (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2,
                        )

            if self.display:
                elapsed = time.time() - start_time
                cv2.putText(
                    output_frame,
                    f"Elapsed: {elapsed:0.1f}s",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    output_frame,
                    "Press 'n' when complete, 'q' to cancel",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )
                cv2.imshow(window_title, output_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("n"), ord("N"), 13):
                    break
                if key == ord("q"):
                    raise KeyboardInterrupt

        metrics = self._calculate_metrics(joint.name, raw_angles, timestamps)
        self._smoothed_angle = None
        if self.display:
            cv2.destroyWindow(window_title)
        return metrics

    def close(self) -> None:
        if hasattr(self.cap, "isOpened") and self.cap.isOpened():
            self.cap.release()
        self.pose.close()
        if self.display:
            cv2.destroyAllWindows()

    def __enter__(self) -> "PoseAnalyzer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def show_preview(self, joint: JointDefinition, message: str = "") -> None:
        if not self.display:
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)
        overlay = message or "Press Enter in terminal to begin"
        cv2.putText(
            frame,
            overlay,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
        cv2.imshow(f"SkeletonScan - {joint.name}", frame)
        cv2.waitKey(1)

    @staticmethod
    def _landmarks_visible(landmarks, joint: JointDefinition, threshold: float) -> bool:
        indices = (joint.proximal, joint.pivot, joint.distal)
        return all(landmarks[idx].visibility >= threshold for idx in indices)

    @staticmethod
    def _landmark_points(landmarks, indices: Tuple[int, int, int], width: int, height: int) -> Tuple[Tuple[float, float], ...]:
        points = []
        for idx in indices:
            lm = landmarks[idx]
            points.append((lm.x * width, lm.y * height))
        return tuple(points)

    @staticmethod
    def _draw_landmarks(frame, results) -> None:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    @staticmethod
    def _draw_angle_arc(frame, points: Tuple[Tuple[float, float], ...]) -> None:
        a, b, c = points
        center = tuple(map(int, b))
        radius = 60
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        start = int(np.degrees(np.arctan2(-ba[1], ba[0])))
        end = int(np.degrees(np.arctan2(-bc[1], bc[0])))
        cv2.ellipse(frame, center, (radius, radius), 0, start, end, (0, 255, 0), 2)

    @staticmethod
    def _calculate_metrics(joint_name: str, raw_angles: List[float], timestamps: List[float]) -> JointMetrics:
        if not raw_angles:
            return JointMetrics(joint_name=joint_name)
        angle_array = np.array(raw_angles, dtype=np.float32)
        duration = timestamps[-1] if timestamps else 0.0
        return JointMetrics(
            joint_name=joint_name,
            min_angle=float(angle_array.min()),
            max_angle=float(angle_array.max()),
            mean_angle=float(angle_array.mean()),
            samples=len(raw_angles),
            duration=duration,
            angles=list(raw_angles),
            timestamps=list(timestamps),
        )
