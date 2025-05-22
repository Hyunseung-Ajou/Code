import numpy as np

# 저장 npz파일 불러오기
def load_dataset(path="preprocessed_eye_dataset_all.npz"):
    data = np.load(path)
    X = data["X"]
    y = data["y"]
    return X, y