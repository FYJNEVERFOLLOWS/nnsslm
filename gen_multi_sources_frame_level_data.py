import argparse
import math
import os
from pathlib import Path
import numpy as np
import pickle


def main(gt_seg_path, gcc_fbank_path, data_frame_path):
    '''
    gt_seg_path: "/CDShare2/SSLR/lsp_test_106_w8192/gt_frame"
    gcc_fbank_path: "/Work18/2021/fuyanjie/exp_data/exp_nnsslm/test_data_dir/lsp_test_106/features/gcc_fbank_w8192"  # gcc-fbank特征所在目录
    data_frame_path = "/Work18/2021/fuyanjie/exp_data/exp_nnsslm/test_data_dir/test_data_frame_level" # 每帧的特征和标签
    '''

    if not os.path.exists(data_frame_path):
        os.makedirs(data_frame_path)

    gts = list(Path(gt_seg_path).rglob('*.txt'))
    cnt_segs = 0 # gt_frame actually means gt_segment 
    for gt in gts:
        gt = str(gt)
        audio_id = gt.split('/')[-1].split('.')[0].replace('qianspeech_', '')
        print('gt: ' + gt, flush=True)
        feat_path = os.path.join(gcc_fbank_path, audio_id + '.npy')
        print('feat: ' + feat_path, flush=True)
        try:
            gcc_fbank = np.load(feat_path, allow_pickle=True)
        except Exception as e:
            continue
        # gcc_fbank_seg_level.shape: (num_segs, 6, 40, 51)
        # print(f'gcc_fbank.shape {gcc_fbank.shape}', flush=True)
        cnt_segs += gcc_fbank.shape[0]
        feat_seg_idx = 0
        with open(gt, "r") as f:
            lines = f.readlines()
            for line in lines:
                gt_seg_data = line.split(' ')
                gt_seg_idx = int(gt_seg_data[0])
                print(f'gt_seg_data {gt_seg_data}', flush=True)
                while feat_seg_idx < gt_seg_idx:
                    label_seg_level = [np.nan, np.nan]
                    # gcc_fbank_seg_level.dtype: int16
                    # gcc_fbank_seg_level.shape: (6, 40, 51)
                    gcc_fbank_seg_level = gcc_fbank[feat_seg_idx]
                    # print(f'gcc_fbank_seg_level.shape {gcc_fbank_seg_level.shape}', flush=True)
                    # sample_data 同时有特征和标签
                    sample_data = {"gcc_fbank_seg_level" : gcc_fbank_seg_level, "label_seg_level" : label_seg_level, "num_sources" : 0}
                    save_path = os.path.join(data_frame_path, '{}_seg_{}.pkl'.format(audio_id, feat_seg_idx))
                    # print(save_path, flush=True)
                    # print("111 sample_data's label {}".format(sample_data["label_seg_level"]))
                    pkl_file = open(save_path, 'wb')
                    pickle.dump(sample_data, pkl_file)
                    pkl_file.close()
                    feat_seg_idx += 1
                if gt_seg_data[8] == '1\n':
                    num_sources = 2
                else:
                    num_sources = 1
                print(f'num_sources {num_sources}', flush=True)
                doa_gt_1 = np.arctan2(float(gt_seg_data[1]), float(gt_seg_data[2]))
                doa_gt_1 = round(math.degrees(doa_gt_1)) + 180

                doa_gt_2 = np.nan
                if num_sources == 2:
                    doa_gt_2 = np.arctan2(float(gt_seg_data[4]), float(gt_seg_data[5]))
                    doa_gt_2 = round(math.degrees(doa_gt_2)) + 180

                label_seg_level = [doa_gt_1, doa_gt_2]
                gcc_fbank_seg_level = gcc_fbank[feat_seg_idx]
                
                sample_data = {"gcc_fbank_seg_level": gcc_fbank_seg_level, "label_seg_level": label_seg_level, "num_sources" : num_sources}
                save_path = os.path.join(data_frame_path, '{}_seg_{}.pkl'.format(audio_id, feat_seg_idx))
                # print(save_path, flush=True)
                print("222 sample_data's label {}".format(sample_data["label_seg_level"]), flush=True)
                pkl_file = open(save_path, 'wb')
                pickle.dump(sample_data, pkl_file)
                pkl_file.close()
                feat_seg_idx += 1
        while feat_seg_idx < gcc_fbank.shape[0]:
            label_seg_level = [np.nan, np.nan]
            gcc_fbank_seg_level = gcc_fbank[feat_seg_idx]
            sample_data = {"gcc_fbank_seg_level": gcc_fbank_seg_level, "label_seg_level": label_seg_level, "num_sources" : 0}
            save_path = os.path.join(data_frame_path, '{}_seg_{}.pkl'.format(audio_id, feat_seg_idx))
            # print(save_path, flush=True)
            # print("333 sample_data's label {}".format(sample_data["label_seg_level"]))
            pkl_file = open(save_path, 'wb')
            pickle.dump(sample_data, pkl_file)
            pkl_file.close()
            feat_seg_idx += 1

    print("cnt_segs: {}".format(cnt_segs), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate multi sources data at frame level')
    parser.add_argument('gt_frame', metavar='GT_FRAME_PATH', type=str,
                        help='path to the gt_frame directory')
    parser.add_argument('gcc_fb', metavar='GCC_FBANK_PATH', type=str,
                        help='path to the gcc_fb directory')
    parser.add_argument('data_frame', metavar='DATA_FRAME_PATH', type=str,
                        help='path to the data_frame directory')
    args = parser.parse_args()
    main(args.gt_frame, args.gcc_fb, args.data_frame)
