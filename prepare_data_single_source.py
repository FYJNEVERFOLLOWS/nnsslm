import math
import os
import pickle

import numpy as np
import torch
from torch import nn

gt_file_path = "/Users/fuyanjie/Desktop/temp/exp_nnsslm/gt_file"
gcc_fbank_path = "/Users/fuyanjie/Desktop/temp/exp_nnsslm/features/gcc_fbank" # gcc-fbank特征所在目录


class DataLoader(object):
    def __init__(self, batchsize):
        self.batch_size = batchsize

    def generate(self):
        # pkls = os.listdir(gt_file_path)
        feats = os.listdir(gcc_fbank_path)[:200]
        cnt_files = len(feats)
        batch_size = 4
        start = 0
        while start + batch_size <= cnt_files:
            batch_feature = feats[start : start + batch_size]
            # batch_pkl = pkls[start : start + batch_size]

            batch_x = []
            batch_y = []

            for i in range(batch_size):
                x_origin = np.load(os.path.join(gcc_fbank_path, batch_feature[i]), allow_pickle=True)
                x_origin = torch.Tensor(x_origin)
                # print("nframes: {}".format(x_origin.shape[0]))
                x_temp = torch.cat((x_origin, torch.zeros(3000 - x_origin.shape[0], 6, 40, 51)), dim=0)
                x = x_temp.reshape(6, 3000, 40, 51)
                batch_x.append(x)
                pkl_path = os.path.join(gt_file_path, batch_feature[i]).replace("npy", "gt.pkl")
                # print(pkl_path)
                y_origin = pickle.load(open(pkl_path,'rb'))
                # print("======")

                x_coordinate = y_origin[3][0][0][0]
                y_coordinate = y_origin[3][0][0][1]
                doa_gt = np.arctan2(x_coordinate, y_coordinate)
                doa_gt_by_degree = math.degrees(doa_gt)
                # labels should be positive
                y = torch.LongTensor([round(doa_gt_by_degree) + 180])
                batch_y.append(y)

            start += batch_size
            # batch_x.shape should be torch.Size([4, 6, 3000, 40, 51])
            batch_x = torch.stack(batch_x, dim=0)
            # print(batch_y)
            # batch_y.shape should be torch.Size([4, 1])
            batch_y = torch.stack(batch_y, dim=0)
            # print(batch_y.shape)

            yield batch_x, batch_y

if __name__ == '__main__':
    data_loader = DataLoader(batchsize=4)
    for batch_x, batch_y in data_loader.generate():
        print("batch_x {}".format(batch_x.shape))
        print("batch_y {}".format(batch_y.shape))
