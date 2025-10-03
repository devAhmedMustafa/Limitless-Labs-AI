import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calc_angle(a, b, c):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)
    ba = a - b
    bc = c - b
    cross = ba[0]*bc[1] - ba[1]*bc[0]
    dot = np.dot(ba, bc)
    angle = np.degrees(np.arctan2(abs(cross), dot))
    return float(angle)

def draw_angle_arc(image, a, b, c, angle, color=(0,255,0)):
    center = tuple(map(int, b))
    radius = 40
    # Vectors
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    ang1 = np.degrees(np.arctan2(-ba[1], ba[0]))
    ang2 = np.degrees(np.arctan2(-bc[1], bc[0]))
    start = int(ang1)
    end = int(ang2)
    cv2.ellipse(image, center, (radius, radius), 0, start, end, color, 2)

cap = cv2.VideoCapture(0)
prev_time = time.time()
smoothed_angle = None
alpha = 0.2  # smoothing factor

with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5,
                  model_complexity=1) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            h, w, _ = image.shape
            lm = results.pose_landmarks.landmark

            hip_idx = mp_pose.PoseLandmark.LEFT_HIP.value
            knee_idx = mp_pose.PoseLandmark.LEFT_KNEE.value
            ankle_idx = mp_pose.PoseLandmark.LEFT_ANKLE.value
            
            shoulder_idx = mp_pose.PoseLandmark.LEFT_SHOULDER.value
            elbow_idx = mp_pose.PoseLandmark.LEFT_ELBOW.value
            wrist_idx = mp_pose.PoseLandmark.LEFT_WRIST.value
            

            if (lm[hip_idx].visibility > 0.5 and
                lm[knee_idx].visibility > 0.5 and
                lm[ankle_idx].visibility > 0.5) or (
                lm[shoulder_idx].visibility > 0.5 and
                lm[elbow_idx].visibility > 0.5 and
                lm[wrist_idx].visibility > 0.5):

                left_hip = (lm[hip_idx].x * w, lm[hip_idx].y * h)
                left_knee = (lm[knee_idx].x * w, lm[knee_idx].y * h)
                left_ankle = (lm[ankle_idx].x * w, lm[ankle_idx].y * h)

                left_shoulder = (lm[shoulder_idx].x * w, lm[shoulder_idx].y * h)
                left_elbow = (lm[elbow_idx].x * w, lm[elbow_idx].y * h)
                left_wrist = (lm[wrist_idx].x * w, lm[wrist_idx].y * h)

                # angle = calc_angle(left_hip, left_knee, left_ankle)
                angle = calc_angle(left_shoulder, left_elbow, left_wrist)

                # smoothing
                if smoothed_angle is None:
                    smoothed_angle = angle
                else:
                    smoothed_angle = alpha * angle + (1 - alpha) * smoothed_angle

                # draw skeleton
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # draw arc
                draw_angle_arc(image, left_shoulder, left_elbow, left_wrist, smoothed_angle)

                cv2.putText(image, f"Left Arm angle: {int(smoothed_angle)} deg",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(image, f"FPS: {int(fps)}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow('Pose (press q to quit)', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
