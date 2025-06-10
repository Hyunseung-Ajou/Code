import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
import json
from collections import defaultdict


plt.ion()

# --- 1. PyTorch Dataset Definition ---
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- 2. Transformer Classifier ---
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, seq_len, d_model=128, nhead=4, num_layers=2,
                 dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    def forward(self, x, return_features=False):
        x = self.input_proj(x) + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        x_pooled = x.mean(dim=1)
        if return_features:
            return x_pooled
        return self.classifier(x_pooled)

# --- 3. Helper: Display Image Sequence ---
def display_sequence(image_paths, title="Image Sequence"):
    print(f"--- Displaying Sequence: {title} ---")
    print(f"Number of image paths provided: {len(image_paths)}")
    if not image_paths:
        print("Error: No image paths provided to display_sequence.")
        plt.close()
        return
    if not os.path.exists(image_paths[0]):
        print(f"CRITICAL: First image path does NOT exist: {image_paths[0]}")
        for p in image_paths:
            if os.path.exists(p):
                print(f"Found at least one image: {p}")
                break
        else:
            print("No image files found. Displaying blank.")
            plt.close()
            return
    plt.figure(figsize=(min(len(image_paths) * 2.5, 20), 3))
    loaded_images_count = 0
    for i, img_path in enumerate(image_paths):
        if not os.path.exists(img_path):
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            plt.subplot(1, len(image_paths), i + 1)
            plt.imshow(img)
            plt.title(f"Frame {i + 1}")
            plt.axis('off')
            loaded_images_count += 1
        except Exception as e:
            print(f"Error loading or displaying image {img_path}: {e}")
    if loaded_images_count == 0:
        print("No images were successfully loaded or displayed.")
        plt.close()
        return
    if loaded_images_count > 0:
        plt.suptitle(title)
        plt.tight_layout()
        plt.show(block=False)   # <-- 여기!
    else:
        plt.close()

# --- 4. 데이터 및 모델 준비 ---
X_val = np.load("val_X.npy")
y_val = np.load("val_y.npy")

with open("val_label2idx.json", "r", encoding="utf-8") as f:
    label2idx = json.load(f)
idx2label = {v: k for k, v in label2idx.items()}

input_dim = X_val.shape[2]
seq_len = X_val.shape[1]
num_classes = len(label2idx)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading best model from transformer_best.pth to {device}...")
model = TransformerClassifier(input_dim=input_dim, num_classes=num_classes, seq_len=seq_len).to(device)
model.load_state_dict(torch.load("transformer_best.pth", map_location=device))
model.eval()

# --- 5. Feature 추출 ---
val_ds = SequenceDataset(X_val, y_val)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

all_features = []
all_labels = []

print("Extracting features from the model...")
with torch.no_grad():
    for Xb, yb in val_loader:
        Xb = Xb.to(device)
        features_batch = model(Xb, return_features=True)
        all_features.append(features_batch.cpu().numpy())
        all_labels.append(yb.cpu().numpy())

all_features = np.vstack(all_features)
all_labels = np.hstack(all_labels)

print(f"Extracted {all_features.shape[0]} features of dimension {all_features.shape[1]}")

# --- 6. t-SNE & 산점도 시각화 ---
print("Applying t-SNE for dimensionality reduction...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
features_2d = tsne.fit_transform(all_features)

fig, ax = plt.subplots(figsize=(12, 10))
scatter = sns.scatterplot(
    x=features_2d[:, 0], y=features_2d[:, 1],
    hue=[idx2label[label] for label in all_labels],
    palette=sns.color_palette("Set2", num_classes),
    legend="full",
    alpha=0.7,
    s=50,
    ax=ax
)
plt.title('t-SNE Visualization of Transformer Learned Features (Drowsy vs. Normal)', fontsize=18)
plt.xlabel('t-SNE Component 1', fontsize=14)
plt.ylabel('t-SNE Component 2', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Class', fontsize=12, title_fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()
input("s")  # plt가 바로 꺼지는 현상 방지 (프로그램 시작 후 plt빈 화면이 뜨는 데 enter키 입력)

# --- 7. 원본 시퀀스 로드 ---
jun_val_dir = r"D:\3-1\AI_basic\Final_Project\Sequence_data\졸음운전 예방을 위한 운전자 상태 정보 영상\Validation\[원천]keypoint(준통제환경)"
tong_val_dir = r"D:\3-1\AI_basic\Final_Project\Sequence_data\졸음운전 예방을 위한 운전자 상태 정보 영상\Validation\[원천]bbox(통제환경)"

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
        if not os.path.isdir(pid_path):
            continue

        # 하위 디렉토리까지 모두 탐색
        img_files = []
        for root, _, files in os.walk(pid_path):
            for f in files:
                if f.lower().endswith('.jpg'):
                    img_files.append(os.path.join(root, f))

        if not img_files:
            continue

        scenario_dict = defaultdict(list)
        action_dict = defaultdict(list)

        for img_path in img_files:
            fname = os.path.basename(img_path)
            tokens = fname.split('_')

            if len(tokens) < 3:
                print(f"[SKIP] Invalid filename format: {fname}")
                continue

            scenario = tokens[2]  # 세 번째 토큰까지는 보장되게 수정
            action = tokens[5] if len(tokens) > 5 else ""  # 없으면 공백 처리

            scenario_key = f"{participant_id}/{scenario}"
            scenario_dict[scenario_key].append(img_path)
            if "졸음" in action:  # 보다 유연하게 조건 적용
                action_dict[scenario_key].append(img_path)

        for scenario_key, imgs in scenario_dict.items():
            if len(action_dict[scenario_key]) > 0:
                label = "졸음"
            else:
                label = "정상"
            data.append({"images": sorted(imgs), "label": label, "env": "통제"})
    return data

print("Loading original image path sequences...")
tong_val_data_orig = load_tongje_data_scenario_label(tong_val_dir)
jun_val_data_orig = load_juntongje_data(jun_val_dir)

print(len(jun_val_data_orig))
print(len(tong_val_data_orig))

val_all_data_orig = jun_val_data_orig + tong_val_data_orig  # 순서 중요!

# --- 8. 인터랙티브 클릭 이벤트 ---
# scatter는 실제 점 객체 (PathCollection)로, ax.collections[0]이 됨
scatter = ax.collections[0]

def on_click(event):
    if event.inaxes is ax:
        cont, ind = scatter.contains(event)
        if cont and 'ind' in ind and len(ind['ind']) > 0:
            idx = ind['ind'][0]
            print(f"Clicked index: {idx}")
            original_label = idx2label[all_labels[idx]]
            if idx < len(val_all_data_orig):
                image_paths_to_display = val_all_data_orig[idx]["images"]
                display_sequence(image_paths_to_display, title=f"Sample {idx} - Predicted Class: {original_label}")
            else:
                print(f"Index {idx} out of range for val_all_data_orig")
        else:
            print("Clicked, but not on any data point.")

cid = fig.canvas.mpl_connect('button_press_event', on_click)
print("Click on any point in the t-SNE plot to visualize the corresponding image sequence.")
print("features_2d shape:", features_2d.shape)
print("NaN 포함?", np.isnan(features_2d).any())
plt.ioff()
plt.show()
input("엔터를 누르면 종료합니다.")
