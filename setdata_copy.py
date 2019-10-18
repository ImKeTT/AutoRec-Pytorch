
import numpy as np
import torch
from torch.utils import data

class Dataset(data.Dataset):
    def __init__(self, rating_list, n_user, n_item, user_based=True):
        self.data = rating_list
        self.user_based = user_based
        self.n_user = n_user
        self.n_item = n_item
        self.x_mat = np.ones((n_user, n_item)) * 0
        self.mask = np.zeros((n_user, n_item))
        for u, v, r in self.data:
            self.x_mat[u][v] = r
            self.mask[u][v] = 1
        self.x_mat = torch.from_numpy(self.x_mat).float()
        self.mask = torch.from_numpy(self.mask).float()
        if not self.user_based:
            self.x_mat = self.x_mat.t()
            self.mask = self.mask.t()

    def __getitem__(self, index):
        return self.x_mat[index], self.mask[index]

    def __len__(self):
        if self.user_based:
            return self.n_user
        return self.n_item

    def get_mat(self):
        return self.x_mat, self.mask, self.user_based


