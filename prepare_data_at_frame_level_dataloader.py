import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SSLR_Dataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.total_x, self.total_y = self._read_file()

    def __getitem__(self, index):
        return torch.tensor(self.total_x[index], dtype=torch.float), torch.tensor(self.total_y[index], dtype=torch.long)

    def __len__(self):
        return len(self.total_x)

    def _read_file(self):
        frames = os.listdir(self.data_path)

        total_x = []
        total_y = []

        for frame in frames:
            origin_data = np.load(os.path.join(self.data_path, frame), allow_pickle=True)
            # pick out single source labels
            ########## remove the code below to cancel filtering single source labels #####################
            if np.isnan(origin_data[1][0]) or not np.isnan(origin_data[1][1]):
                continue
            if origin_data[1][0] == 360:
                origin_data[1][0] = 0
            y = [origin_data[1][0]]
            ###############################################################################################
            # y = torch.Tensor(origin_data[1])
            # print(y)
            total_y.append(y)  # [num_frames, 1]

            x = origin_data[0]
            total_x.append(x)  # [num_frames, 6, 40, 51]

        return total_x, total_y

if __name__ == '__main__':
    train_data_path = "/Users/fuyanjie/Desktop/temp/exp_nnsslm/train_data_frame_level"
    train_data = DataLoader(SSLR_Dataset(train_data_path), batch_size=512, shuffle=True, num_workers=4) # train_data.shape (batch_x, batch_y)
    print(len(SSLR_Dataset(train_data_path)))
    print(len(train_data))
    print(next(iter(train_data))[0].shape, next(iter(train_data))[1].shape)
