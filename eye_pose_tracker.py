import cv2
import torch
import numpy as np
import mediapipe as mp
import time
from Model_Learning import EyeStateCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EyeStateCNN(num_classes=2).to(device)
model.load_state_dict(torch.load("eye_state_cnn.pth", map_location=device))
model.eval()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_EYE_IDX = [33, 133]
RIGHT_EYE_IDX = [362, 263]

# 눈 crop 함수
def crop_eye_with_box(frame, landmarks, eye_idx, box_size=48):
    h, w = frame.shape[:2]
    eye_pts = [landmarks[i] for i in eye_idx]
    x_center = int((sum(pt.x for pt in eye_pts) / len(eye_pts)) * w)
    y_center = int((sum(pt.y for pt in eye_pts) / len(eye_pts)) * h)
    half_box = box_size // 2
    x_min = max(x_center - half_box, 0)
    y_min = max(y_center - half_box, 0)
    x_max = min(x_center + half_box, w)
    y_max = min(y_center + half_box, h)
    eye_img = frame[y_min:y_max, x_min:x_max]
    return eye_img, (x_min, y_min, x_max, y_max)

# 고개 crop 함수
def estimate_head_pose(landmarks, image_w, image_h):
    indices = [1, 33, 61, 199, 263, 291]
    image_points = []
    for idx in indices:
        lm = landmarks[idx]
        x = int(lm.x * image_w)
        y = int(lm.y * image_h)
        image_points.append((x, y))
    image_points = np.array(image_points, dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (-30.0, -30.0, -30.0),
        (-30.0, 30.0, -30.0),
        (0.0, -60.0, -60.0),
        (30.0, -30.0, -30.0),
        (30.0, 30.0, -30.0)
    ])

    focal_length = image_w
    center = (image_w / 2, image_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, _ = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None
    rmat, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return angles  # pitch(상하), yaw(좌우), roll(기울기)

# 눈 상태 추적 변수
prev_eye_state = {"Left": 1, "Right": 1}  # 1: open, 0: closed
closed_start_time = {"Left": None, "Right": None}
closed_duration = {"Left": 0, "Right": 0}
blink_count = {"Left": 0, "Right": 0}
blink_threshold_sec = 0.3

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    current_time = time.time()

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            angles = estimate_head_pose(face_landmarks.landmark, w, h)
            if angles:
                pitch, yaw, roll = angles
                text = f"Pitch: {pitch:.1f}, Yaw: {yaw:.1f}, Roll: {roll:.1f}"
                cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            for eye_name, eye_idx in zip(["Left", "Right"], [LEFT_EYE_IDX, RIGHT_EYE_IDX]):
                eye_img, (x_min, y_min, x_max, y_max) = crop_eye_with_box(frame, face_landmarks.landmark, eye_idx)

                if eye_img.size == 0:
                    continue

                eye_gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
                eye_resized = cv2.resize(eye_gray, (48, 48))
                eye_norm = eye_resized.astype('float32') / 255.0
                eye_tensor = torch.tensor(eye_norm).unsqueeze(0).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(eye_tensor)
                    pred = torch.argmax(output, dim=1).item()

                label = "Open" if pred == 1 else "Closed"
                color = (0, 255, 0) if pred == 1 else (0, 0, 255)

                if pred == 0:
                    if closed_start_time[eye_name] is None:
                        closed_start_time[eye_name] = current_time
                    else:
                        closed_duration[eye_name] = current_time - closed_start_time[eye_name]
                else:
                    if closed_start_time[eye_name] is not None:
                        duration = current_time - closed_start_time[eye_name]
                        if duration < blink_threshold_sec:
                            blink_count[eye_name] += 1
                    closed_start_time[eye_name] = None
                    closed_duration[eye_name] = 0

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(frame, f"{eye_name}: {label}", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if closed_duration[eye_name] > 0:
                    cv2.putText(frame, f"{closed_duration[eye_name]:.1f}s", (x_min, y_max + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.putText(frame, f"Blinks: {blink_count[eye_name]}", (x_min, y_max + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

    cv2.imshow("Head Pose & Eye State", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
