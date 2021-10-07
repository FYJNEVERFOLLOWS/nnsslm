import math
import os
from pathlib import Path

import numpy as np

gt_frame_path = "/Users/fuyanjie/Desktop/temp/exp_nnsslm/gt_frame"
gcc_fbank_path = "/Users/fuyanjie/Desktop/temp/exp_nnsslm/features/gcc_fbank"  # gcc-fbank特征所在目录
train_data_path = "/Users/fuyanjie/Desktop/temp/exp_nnsslm/train_data_frame_level" # 每帧的特征和标签

gts = list(Path(gt_frame_path).rglob('*.txt'))
for gt in gts:
    gt = str(gt)
    audio_id = gt.split('/')[-1].split('.')[0]
    print('gt: ' + gt)
    feat_path = os.path.join(gcc_fbank_path, audio_id + '.npy')
    print('feat: ' + feat_path)
    gcc_fbank = np.load(feat_path, allow_pickle=True)
    print(gcc_fbank.shape)
    feat_frame_idx = 0
    with open(gt, "r") as f:
        lines = f.readlines()
        for line in lines:
            gt_frame_data = line.split(' ')
            gt_frame_idx = int(gt_frame_data[0])
            print(gt_frame_data)
            while feat_frame_idx < gt_frame_idx:
                gcc_fbank_frame_level = gcc_fbank[feat_frame_idx]
                label_frame_level = [np.nan, np.nan]
                train_data = np.array([gcc_fbank_frame_level, label_frame_level])
                save_path = os.path.join(train_data_path, '{}_frame_{}.npy'.format(audio_id, feat_frame_idx))
                print(save_path)
                print("111 train_data[1] {}".format(train_data[1]))
                np.save(save_path, train_data)
                feat_frame_idx += 1
            doa_gt_1 = np.arctan2(float(gt_frame_data[1]), float(gt_frame_data[2]))
            if not np.isnan(doa_gt_1):
                doa_gt_1 = round(math.degrees(doa_gt_1)) + 180
            doa_gt_2 = np.arctan2(float(gt_frame_data[4]), float(gt_frame_data[5]))
            if not np.isnan(doa_gt_2):
                doa_gt_2 = round(math.degrees(doa_gt_2)) + 180
            label_frame_level = [doa_gt_1, doa_gt_2]
            train_data = np.array([gcc_fbank_frame_level, label_frame_level])
            save_path = os.path.join(train_data_path, '{}_frame_{}.npy'.format(audio_id, feat_frame_idx))
            print(save_path)
            print("222 train_data[1] {}".format(train_data[1]))
            np.save(save_path, train_data)
            feat_frame_idx += 1
    while feat_frame_idx < gcc_fbank.shape[0]:
        gcc_fbank_frame_level = gcc_fbank[feat_frame_idx]
        label_frame_level = [np.nan, np.nan]
        train_data = np.array([gcc_fbank_frame_level, label_frame_level])
        save_path = os.path.join(train_data_path, '{}_frame_{}.npy'.format(audio_id, feat_frame_idx))
        print(save_path)
        print("333 train_data[1] {}".format(train_data[1]))
        np.save(save_path, train_data)
        feat_frame_idx += 1

# pkls = os.listdir(gt_file_path)
# print(pkls)

