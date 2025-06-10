import os
from collections import defaultdict
from datetime import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt

def group_by_act_user_time(root_dir, time_gap_sec=60):
    grouped = defaultdict(lambda: defaultdict(list))  # act → userid → [파일 리스트]

    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        for fname in os.listdir(folder_path):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            parts = fname.split('_')
            if len(parts) < 9:
                continue

            user_id = parts[0]
            act = parts[5]
            time_str = parts[7]  # HHMMSS
            full_path = os.path.join(folder_path, fname)

            grouped[act][user_id].append((time_str, full_path))

    # 시각 기준으로 정렬하고 시퀀스 분할
    final_sequences = []

    for act, user_dict in grouped.items():
        for user_id, items in user_dict.items():
            # 시간순 정렬
            sorted_items = sorted(items, key=lambda x: int(x[0]))

            current_seq = []
            prev_time = None

            for time_str, path in sorted_items:
                # HHMMSS 문자열 → datetime 객체
                t = datetime.strptime(time_str, "%H%M%S")
                if prev_time is not None:
                    diff = (t - prev_time).total_seconds()
                    if diff > time_gap_sec:
                        if current_seq:
                            final_sequences.append((act, user_id, current_seq))
                        current_seq = []

                current_seq.append(path)
                prev_time = t

            if current_seq:
                final_sequences.append((act, user_id, current_seq))

    return final_sequences

root = r"D:\Deep_Learning\Sequential_data"
sequences = group_by_act_user_time(root)

def imread_unicode(path, flags=cv2.IMREAD_GRAYSCALE):
    with open(path, "rb") as f:
        file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
        return cv2.imdecode(file_bytes, flags)

def visualize_sequence_unicode(sequence_paths, num_frames=None):
    if num_frames is None:
        num_frames = len(sequence_paths)
    selected = sequence_paths[:num_frames]
    images = [imread_unicode(path) for path in selected]

    plt.figure(figsize=(15, 3))
    for i, img in enumerate(images):
        plt.subplot(1, num_frames, i+1)
        if img is not None:
            plt.imshow(img, cmap='gray')
            plt.title(f"Frame {i+1}")
        else:
            plt.title("Load Fail")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# 번호에 따라 졸음/하품 재현있음
visualize_sequence_unicode(sequences[402][2])  # [행위, 유저ID, [경로 리스트]]
visualize_sequence_unicode(sequences[713][2])  # [행위, 유저ID, [경로 리스트]]


for i, (act, uid, files) in enumerate(sequences):
    print(f"[{i}] 행위: {act}, UserID: {uid}, 프레임 수: {len(files)}")
    print("    예시 파일:", os.path.basename(files[0]))