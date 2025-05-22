from torch.utils.data import Dataset
import torch

class EyeStateDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X  # numpy 배열 그대로 보관
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = torch.tensor(self.X[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label