import os
import cv2
import numpy as np

base_dir = r"D:\3-1\Deep_Learning\Final_Prj_Dataset\archive"
label_map = {'closed': 0, 'open': 1}

# 전처리 함수 정의
def load_dataset_from_dir(root_dir):
    images = []
    labels = []

    for label_name, label_value in label_map.items():
        folder_path = os.path.join(root_dir, label_name)
        if not os.path.isdir(folder_path):
            print(f"❌ 폴더 없음: {folder_path}")
            continue

        for filename in os.listdir(folder_path):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"❌ 이미지 로딩 실패: {img_path}")
                continue

            # cv2로 이미지를 불러와서 사이즈 변환 (90x90 -> 48x48)
            img = cv2.resize(img, (48, 48))
            # 이미지 정규화 (0~1 사이의 값으로)
            img = img.astype('float32') / 255.0

            images.append(img)
            labels.append(label_value)

    X = np.array(images).reshape(-1, 1, 48, 48)
    y = np.array(labels)
    return X, y


# 각각의 데이터셋 불러오기
X_train, y_train = load_dataset_from_dir(os.path.join(base_dir, "train"))
X_val, y_val     = load_dataset_from_dir(os.path.join(base_dir, "val"))
X_test, y_test   = load_dataset_from_dir(os.path.join(base_dir, "test"))

# 정보 출력
print(f"✅ Train: {X_train.shape}, {y_train.shape}")
print(f"✅ Val:   {X_val.shape}, {y_val.shape}")
print(f"✅ Test:  {X_test.shape}, {y_test.shape}")

# npz 파일로 저장
np.savez_compressed("preprocessed_eye_dataset_all.npz",
                    X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val,
                    X_test=X_test, y_test=y_test)

print("✅ 전체 데이터셋 저장 완료: preprocessed_eye_dataset_all.npz")
