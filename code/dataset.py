import numpy as np
import torch
import pickle
import torch.nn.functional as F


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path_dataset):
        super().__init__()

        with open(f"{path_dataset}", "rb") as fp:
            X, Y_idx, self.labels = pickle.load(fp)

        X = np.array(X)
        self.X_classes = np.array(X[:, -len(self.labels) + 1 :]).astype(np.int32)

        self.X = np.array(X[:, : -len(self.labels)+1]).astype(np.float32)
        self.X_mean = np.mean(self.X, axis=0)
        self.X_std = np.std(self.X, axis=0) + 1e-8
        self.X = (self.X - self.X_mean) / self.X_std
        self.X = self.X.astype(np.float32)
        self.X_max=np.max(self.X,axis=0)
        self.X_min=np.min(self.X,axis=0)
        self.Y_idx=Y_idx
        self.Y_labels = self.labels[-1]

        self.Y = F.one_hot(torch.LongTensor(Y_idx.astype(np.int32)))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], np.array(self.X_classes[idx]), self.Y[idx]
