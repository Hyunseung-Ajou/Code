import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np

from dataset_loader import load_dataset
from eye_state_dataset import EyeStateDataset

# 모델 정의 CNN3 / DNN 2
class EyeStateCNN(nn.Module):
    # 분류 클래스는 defaul 2(open/closed)
    def __init__(self, num_classes=2):
        super(EyeStateCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


# 학습 함수 최적화, 손실함수 등 조절
def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


# 평가 함수
def evaluate(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.cpu().numpy())

    print("📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["closed", "open"]))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    # 1. 데이터 불러오기
    data = np.load("preprocessed_eye_dataset_all.npz")
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val     = data["X_val"], data["y_val"]
    X_test, y_test   = data["X_test"], data["y_test"]

    # 라벨 분포 확인
    for name, labels in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        classes, counts = np.unique(labels, return_counts=True)
        print(f"🎯 {name} 라벨 분포:", dict(zip(classes, counts)))

    # 2. Dataset 및 DataLoader 구성
    train_dataset = EyeStateDataset(X_train, y_train)
    val_dataset   = EyeStateDataset(X_val, y_val)
    test_dataset  = EyeStateDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # 3. 모델/손실함수/최적화 설정
    model = EyeStateCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 4. 학습 루프
    epochs = 5
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        print(f"[Epoch {epoch:02}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        evaluate(model, val_loader, device)

    # 5. 최종 모델 평가 (테스트셋)
    print("🧪 테스트셋 평가")
    evaluate(model, test_loader, device)

    # 6. 모델 저장
    torch.save(model.state_dict(), "eye_state_cnn.pth")
    print("✅ 모델 저장 완료: eye_state_cnn.pth")


# ✅ Windows-safe 진입점
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
