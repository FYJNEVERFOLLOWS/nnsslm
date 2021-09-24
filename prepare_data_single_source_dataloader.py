import math
import os
import pickle

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

gt_file_path = "/Users/fuyanjie/Desktop/temp/exp_nnsslm/gt_file"
gcc_fbank_path = "/Users/fuyanjie/Desktop/temp/exp_nnsslm/features/gcc_fbank" # gcc-fbank特征所在目录


class SSLR(Dataset):
    def __init__(self):
        super(SSLR, self).__init__()
        self.total_x, self.total_y = self._read_file()
    
    def __len__(self):
        return len(self.total_x)
    
    def __getitem__(self, index):
        x = self.total_x[index]
        y = self.total_y[index]
        x = torch.LongTensor(x)
        x_temp = torch.cat((x, torch.zeros(3000 - x.shape[0], 6, 40, 51)), dim=0)
        y = torch.LongTensor([round(y) + 180])
        x = x_temp.reshape(6, 3000, 40, 51)
        return x, y

    def _read_file(self):
        feats = os.listdir(gcc_fbank_path)[:200]
        cnt_files = len(feats)
        # batch_feature = feats[start : start + batch_size]

        total_x = []
        total_y = []

        for i in range(cnt_files):
            x = np.load(os.path.join(gcc_fbank_path, feats[i]), allow_pickle=True)
            total_x.append(x)
            pkl_path = os.path.join(gt_file_path, feats[i]).replace("npy", "gt.pkl")
            y_origin = pickle.load(open(pkl_path,'rb'))
            x_coordinate = y_origin[3][0][0][0]
            y_coordinate = y_origin[3][0][0][1]
            doa_gt = np.arctan2(x_coordinate, y_coordinate)
            doa_gt_by_degree = math.degrees(doa_gt)
            total_y.append(doa_gt_by_degree)
        return total_x, total_y

if __name__ == '__main__':
    data_loader = DataLoader(SSLR(), batch_size=4, shuffle=True, num_workers=4)
    print(len(data_loader))
    print(next(iter(data_loader))[0].shape)
    print(next(iter(data_loader))[1].shape)
