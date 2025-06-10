import torch
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
from collections import deque

# ---- 1. 모델, CNN feature extractor 준비 ----
device = "cuda" if torch.cuda.is_available() else "cpu"
cnn = models.resnet18(pretrained=True)
cnn = torch.nn.Sequential(*list(cnn.children())[:-1])
cnn = cnn.to(device)
cnn.eval()

# 트랜스포머 분류 모델 (학습 때와 동일 구조로 정의!)
class TransformerClassifier(torch.nn.Module):
    def __init__(self, input_dim, num_classes, seq_len, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj = torch.nn.Linear(input_dim, d_model)
        self.pos_encoding = torch.nn.Parameter(torch.randn(1, seq_len, d_model))  # learnable positional encoding
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(d_model, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.input_proj(x) + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)

# 라벨 인덱스 맵(0=정상, 1=졸음 등은 학습 label2idx.json 기준으로 반드시 맞추세요)
label2idx = {"Good": 0, "Drowsy": 1}  # 예시
idx2label = {v: k for k, v in label2idx.items()}

# 실제 학습 때 쓴 label2idx.json을 반드시 읽어서 맞춰야 합니다!
# import json
# with open("train_label2idx.json", "r", encoding="utf-8") as f:
#     label2idx = json.load(f)
# idx2label = {v: k for k, v in label2idx.items()}

# 모델 불러오기
seq_len = 25
input_dim = 512
num_classes = len(label2idx)
model = TransformerClassifier(input_dim=input_dim, num_classes=num_classes, seq_len=seq_len)
model.load_state_dict(torch.load("transformer_best.pth", map_location=device))
model = model.to(device)
model.eval()

# ---- 2. transform 준비 ----
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---- 3. 실시간 슬라이딩 윈도우 (deque) ----
frame_queue = deque(maxlen=seq_len)

cap = cv2.VideoCapture(0)  # 웹캠
while True:
    ret, frame = cap.read()
    if not ret: break

    # CNN feature 추출
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = cnn(x)  # (1, 512, 1, 1)
        feat = feat.view(-1).cpu().numpy()  # 정확히 (512,)

    frame_queue.append(feat)  # 슬라이딩 윈도우에 저장(최신 seq_len개 유지)

    if len(frame_queue) == seq_len:
        seq = np.stack(frame_queue)  # (seq_len, 512)
        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)  # (1, seq_len, 512)
        with torch.no_grad():
            out = model(seq_tensor)
            pred = out.argmax(dim=1).item()
        label = idx2label.get(pred, "Unknown")
    else:
        label = "Waiting..."

        # 텍스트 항상 표시
    cv2.putText(frame, f"Predict: {label}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    print(f"Predict: {label}")

    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC 종료
        break

cap.release()
cv2.destroyAllWindows()
