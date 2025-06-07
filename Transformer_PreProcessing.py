import os
from collections import defaultdict, Counter
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import json

# ----------------- 1. 데이터 로딩 함수 -----------------
def load_juntongje_data(base_dir):
    data = []
    for seq_folder in os.listdir(base_dir):
        seq_path = os.path.join(base_dir, seq_folder)
        if not os.path.isdir(seq_path): continue
        img_list = [os.path.join(seq_path, f) for f in os.listdir(seq_path) if f.lower().endswith('.jpg')]
        if len(img_list) == 0: continue
        data.append({"images": sorted(img_list), "label": "졸음", "env": "준통제"})
    return data

def load_tongje_data_scenario_label(base_dir):
    data = []
    for participant_id in os.listdir(base_dir):
        pid_path = os.path.join(base_dir, participant_id)
        if not os.path.isdir(pid_path): continue
        img_files = [os.path.join(pid_path, f) for f in os.listdir(pid_path) if f.lower().endswith('.jpg')]
        scenario_dict = defaultdict(list)
        action_dict = defaultdict(list)
        for img_path in img_files:
            fname = os.path.basename(img_path)
            tokens = fname.split('_')
            if len(tokens) < 6: continue
            scenario = tokens[2]
            action = tokens[5]
            scenario_key = f"{participant_id}/{scenario}"
            scenario_dict[scenario_key].append(img_path)
            if action == "졸음재현":
                action_dict[scenario_key].append(img_path)
        for scenario_key, imgs in scenario_dict.items():
            if len(action_dict[scenario_key]) > 0:
                label = "졸음"
            else:
                label = "정상"
            data.append({"images": sorted(imgs), "label": label, "env": "통제"})
    return data

# ----------------- 2. Feature 추출 및 전처리 함수 -----------------
def extract_feature_sequence(img_paths, cnn_model, transform, device="cpu"):
    features = []
    for path in img_paths:
        img = Image.open(path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = cnn_model(x).squeeze().cpu().numpy()  # (512,)
        features.append(feat)
    return np.stack(features)  # (seq_len, 512)

def pad_sequence(seq, max_len, feat_dim):
    if len(seq) < max_len:
        pad = np.zeros((max_len - len(seq), feat_dim))
        seq = np.vstack([seq, pad])
    else:
        seq = seq[:max_len]
    return seq

def preprocess_all_data(all_data, cnn_model, transform, max_seq_len=25, device="cpu", label2idx=None, save_prefix=None):
    X_list, y_list = [], []
    # 라벨 인코딩: train에선 자동 생성, val/test에선 전달받음
    if label2idx is None:
        label2idx = {label: idx for idx, label in enumerate(sorted(set(d["label"] for d in all_data)))}
    for d in all_data:
        feat_seq = extract_feature_sequence(d["images"], cnn_model, transform, device)
        feat_seq = pad_sequence(feat_seq, max_seq_len, feat_seq.shape[1])
        X_list.append(feat_seq)
        y_list.append(label2idx.get(d["label"], -1))  # unknown label → -1
    X_arr = np.stack(X_list)
    y_arr = np.array(y_list)
    if save_prefix:
        np.save(f"{save_prefix}_X.npy", X_arr)
        np.save(f"{save_prefix}_y.npy", y_arr)
        with open(f"{save_prefix}_label2idx.json", "w", encoding="utf-8") as f:
            json.dump(label2idx, f, ensure_ascii=False)
    return torch.tensor(X_arr, dtype=torch.float32), torch.tensor(y_arr, dtype=torch.long), label2idx

# ----------------- 3. 경로 및 모델/변환 정의 -----------------
# 학습
jun_train_dir = r"D:\3-1\AI_basic\Final_Project\Sequence_data\졸음운전 예방을 위한 운전자 상태 정보 영상\Training\[원천]keypoint(준통제환경)"
tong_train_dir = r"D:\3-1\AI_basic\Final_Project\Sequence_data\졸음운전 예방을 위한 운전자 상태 정보 영상\Training\[원천]bbox(통제환경)"
# 검증
jun_val_dir = r"D:\3-1\AI_basic\Final_Project\Sequence_data\졸음운전 예방을 위한 운전자 상태 정보 영상\Validation\[원천]keypoint(준통제환경)"
tong_val_dir = r"D:\3-1\AI_basic\Final_Project\Sequence_data\졸음운전 예방을 위한 운전자 상태 정보 영상\Validation\[원천]bbox(통제환경)"

device = "cuda" if torch.cuda.is_available() else "cpu"
cnn = models.resnet18(pretrained=True)
cnn = torch.nn.Sequential(*list(cnn.children())[:-1])  # FC 제거
cnn = cnn.to(device)
cnn.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ----------------- 4. 데이터 전처리 및 저장 -----------------
# -- (1) 학습 데이터
tong_train_data = load_tongje_data_scenario_label(tong_train_dir)
jun_train_data = load_juntongje_data(jun_train_dir)
train_all_data = jun_train_data + tong_train_data

print(f"훈련용 샘플 수: {len(train_all_data)} | 준통제: {len(jun_train_data)} | 통제: {len(tong_train_data)}")
X_train, y_train, label2idx = preprocess_all_data(
    train_all_data, cnn, transform, max_seq_len=25, device=device, save_prefix="train"
)
print("훈련 X:", X_train.shape, "훈련 y:", y_train.shape)

# -- (2) 검증 데이터
tong_val_data = load_tongje_data_scenario_label(tong_val_dir)
jun_val_data = load_juntongje_data(jun_val_dir)
val_all_data = jun_val_data + tong_val_data

print(f"검증용 샘플 수: {len(val_all_data)} | 준통제: {len(jun_val_data)} | 통제: {len(tong_val_data)}")
X_val, y_val, _ = preprocess_all_data(
    val_all_data, cnn, transform, max_seq_len=25, device=device, label2idx=label2idx, save_prefix="val"
)
print("검증 X:", X_val.shape, "검증 y:", y_val.shape)

# ----------------- 5. 라벨 분포 확인 -----------------
label_counts_train = Counter([d["label"] for d in train_all_data])
label_counts_val = Counter([d["label"] for d in val_all_data])
print("훈련 데이터 라벨 분포:", label_counts_train)
print("검증 데이터 라벨 분포:", label_counts_val)