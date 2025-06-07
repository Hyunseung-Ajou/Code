import cv2
import torch
import numpy as np
import mediapipe as mp
import time
import csv
from datetime import datetime
from Model_Learning import EyeStateCNN

# ======== 초기 설정 ========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EyeStateCNN(num_classes=2).to(device)
model.load_state_dict(torch.load("eye_state_cnn.pth", map_location=device))
model.eval()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_EYE_IDX = [33, 133]
RIGHT_EYE_IDX = [362, 263]

# ======== CSV 로그 초기화 ========
csv_file = open("logs.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "Score", "Closed_Events", "Tilt_Events"])

# ======== 눈 crop 함수 ========
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

# ======== 고개 자세 추정 함수 ========
def estimate_head_pose(landmarks, image_w, image_h):
    indices = [1, 33, 61, 199, 263, 291]
    image_points = [(int(landmarks[i].x * image_w), int(landmarks[i].y * image_h)) for i in indices]
    model_points = np.array([
        (0.0, 0.0, 0.0), (-30.0, -30.0, -30.0), (-30.0, 30.0, -30.0),
        (0.0, -60.0, -60.0), (30.0, -30.0, -30.0), (30.0, 30.0, -30.0)
    ])
    camera_matrix = np.array([
        [image_w, 0, image_w / 2],
        [0, image_w, image_h / 2],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))
    success, rvec, _ = cv2.solvePnP(model_points, np.array(image_points, dtype='double'),
                                    camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return None
    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return angles  # pitch, yaw, roll

# ======== 이벤트 기반 점수 계산 함수 (슬라이딩 윈도우 방식) ========
def compute_drowsiness_score_by_events(frames, fps_estimate=30):
    score = 0
    closed_events = 0
    tilt_events = 0
    interval = 1.0

    if not frames:
        return 0, 0, 0

    start_time = frames[0]['time']
    end_time = frames[-1]['time']
    t = start_time
    checked_intervals = []

    while t + interval <= end_time:
        if any(start <= t <= end for start, end in checked_intervals):
            t += 0.5
            continue

        segment = [f for f in frames if t <= f['time'] < t + interval]
        closed_count = sum(1 for f in segment if f['is_closed']['Left'] and f['is_closed']['Right'])
        tilt_count = sum(1 for f in segment if abs(f['pitch']) < 55 or abs(f['yaw']) > 7)

        if closed_count >= fps_estimate:
            score += 3
            closed_events += 1
            checked_intervals.append((t, t + interval))
        elif tilt_count >= fps_estimate:
            score += 1
            tilt_events += 1
            checked_intervals.append((t, t + interval))

        t += 0.5

    return score, closed_events, tilt_events

# ======== 상태 초기화 ========
frame_buffer = []
last_analysis_time = time.time()
observation_duration = 15.0
fps_estimate = 30

cap = cv2.VideoCapture(0)

score, closed_events, tilt_events = 0, 0, 0
pitch, yaw = 0.0, 0.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    current_time = time.time()
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            angles = estimate_head_pose(face_landmarks.landmark, w, h)
            pitch, yaw, _ = angles if angles else (0.0, 0.0, 0.0)

            frame_data = {
                "time": current_time,
                "pitch": pitch,
                "yaw": yaw,
                "is_closed": {"Left": False, "Right": False}
            }

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
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(frame, f"{eye_name}: {label}", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if pred == 0:
                    frame_data["is_closed"][eye_name] = True

            frame_buffer.append(frame_data)
            frame_buffer = [f for f in frame_buffer if current_time - f["time"] <= observation_duration]

    if current_time - last_analysis_time >= 1.0:
        score, closed_events, tilt_events = compute_drowsiness_score_by_events(frame_buffer, fps_estimate)
        last_analysis_time = current_time

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        csv_writer.writerow([timestamp, score, closed_events, tilt_events])
        csv_file.flush()

        print(f"[{timestamp}] Score: {score} | Eye Events: {closed_events} | Tilt Events: {tilt_events}")

    cv2.putText(frame, f"Score: {score}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Eye Events: {closed_events}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Tilt Events: {tilt_events}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Pitch: {pitch:.2f}", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"Yaw: {yaw:.2f}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    if score >= 7:
        cv2.putText(frame, "[Wake up! Drowsy driving detected]", (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    elif score >= 4:
        cv2.putText(frame, "[Caution! Drowsiness Warning]", (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 3)

    cv2.imshow("Drowsiness Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()
