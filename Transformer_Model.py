import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
import time

# ================== 1. 데이터 불러오기 ==================
X_train = np.load("train_X.npy")  # shape: (N, seq_len, feat_dim)
y_train = np.load("train_y.npy")
X_val = np.load("val_X.npy")
y_val = np.load("val_y.npy")

print("Train shape:", X_train.shape, y_train.shape)
print("Val shape:", X_val.shape, y_val.shape)


# ================== 2. PyTorch Dataset 정의 ==================
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


batch_size = 32
train_ds = SequenceDataset(X_train, y_train)
val_ds = SequenceDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)


# ================== 3. 트랜스포머 분류 모델 정의 ==================
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, seq_len, d_model=128, nhead=4,
                 num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model))  # learnable positional encoding
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout), # Dropout Layer Added
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x) + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)  # (batch, seq_len, d_model)
        x = x.mean(dim=1)  # global average pooling over time
        return self.classifier(x)


# ================== 4. 학습 준비 ==================
device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = int(max(y_train.max(), y_val.max())) + 1
input_dim = X_train.shape[2]
seq_len = X_train.shape[1]

model = TransformerClassifier(input_dim=input_dim, num_classes=num_classes, seq_len=seq_len).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
# weight_decay는 L2 정규화 강도, 정규화 추가
weight = torch.tensor([5.0, 1.0]).to(device) # 클래스 1에 5배 가중치 - 정상 데이터가 졸음 데이터에 비해 부족함
criterion = nn.CrossEntropyLoss(weight=weight)
# criterion = nn.CrossEntropyLoss()

# TensorBoard Writer 생성
log_dir = f"runs/experiment_{int(time.time())}"
writer = SummaryWriter(log_dir=log_dir)

best_val_loss = float('inf')
epochs_no_improve = 0
global_step = 0

epochs = 20

# Early Stopping Parameters
patience = 5 # Number of epochs to wait for improvement
min_delta = 0.0001 # Minimum change to be considered an improvement
best_val_loss = float('inf') # We want to minimize validation loss
epochs_no_improve = 0

# ================== 5. 학습 루프 with TensorBoard ==================
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(Xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(Xb)
        writer.add_scalar("Train/Loss", loss.item(), global_step)
        global_step += 1

    avg_train_loss = total_loss / len(train_ds)

    # 검증
    model.eval()
    correct, total, val_loss = 0, 0, 0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            out = model(Xb)
            loss = criterion(out, yb)
            val_loss += loss.item() * len(Xb)
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += len(yb)

    avg_val_loss = val_loss / len(val_ds)
    val_acc = correct / total

    # TensorBoard 기록
    writer.add_scalar("Val/Loss", avg_val_loss, epoch)
    writer.add_scalar("Val/Acc", val_acc, epoch)

    print(f"Epoch {epoch+1}/{epochs} – TrainLoss: {avg_train_loss:.4f}  ValLoss: {avg_val_loss:.4f}  ValAcc: {val_acc:.4f}")

    # Early stopping
    if avg_val_loss < best_val_loss - min_delta:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "transformer_best.pth")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# 종료 후 Writer 닫기
writer.close()