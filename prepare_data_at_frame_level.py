import math
import os
import pickle
import random

import numpy as np
import torch
from torch import nn

train_data_path = "/Users/fuyanjie/Desktop/temp/exp_nnsslm/train_data_frame_level"


class DataLoader(object):
    def __init__(self, batchsize=4, shuffle=False):
        self.batch_size = batchsize
        self.shuffle = shuffle

    def generate(self):
        frames = os.listdir(train_data_path)[:201]
        cnt_files = len(frames)
        if self.shuffle:
            random.shuffle(frames)
        print("======")

        batch_size = self.batch_size
        start = 0
        while start < cnt_files:
            batch_frame = frames[start : min(start + batch_size, cnt_files)]

            batch_x = []
            batch_y = []

            for frame in batch_frame:
                origin_data = np.load(os.path.join(train_data_path, frame), allow_pickle=True)
                # pick out single source labels
                ########## remove the code below to cancel filtering single source labels #####################
                if np.isnan(origin_data[1][0]) or not np.isnan(origin_data[1][1]):
                    continue
                if origin_data[1][0] == 360:
                    origin_data[1][0] = 0
                y = torch.LongTensor([origin_data[1][0]])
                ###############################################################################################
                # y = torch.Tensor(origin_data[1])
                print(y)
                batch_y.append(y)

                x = torch.Tensor(origin_data[0])
                batch_x.append(x)

            start += batch_size
            ########## remove the code below to cancel filtering single source labels #####################
            if not batch_x:
                continue
            ###############################################################################################
            # batch_x.shape should be torch.Size([4, 6, 40, 51])
            batch_x = torch.stack(batch_x, dim=0)
            # print("batch_x.shape {}".format(batch_x.shape))
            # batch_y.shape should be torch.Size([4, 2])
            batch_y = torch.stack(batch_y, dim=0)
            batch_y = torch.squeeze(batch_y, dim=-1)
            # print("batch_y {}".format(batch_y))

            yield batch_x, batch_y

if __name__ == '__main__':
    data_loader = DataLoader(batchsize=4, shuffle=True)
    for batch_x, batch_y in data_loader.generate():
        print("batch_x {}".format(batch_x.shape))
        print("batch_y {}".format(batch_y.shape))
