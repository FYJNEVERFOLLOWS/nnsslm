import argparse
import math
import os
from pathlib import Path
import numpy as np
import pickle
import apkit

_FREQ_MAX = 8000
_FREQ_MIN = 100
SEG_LEN = 8192
SEG_HOP = 4096

def main(gt_seg_path, audio_dir_path, data_frame_path):
    '''
    gt_seg_path: "/CDShare2/SSLR/lsp_test_106_w8192/gt_frame"
    audio_dir_path: "/Work18/2021/fuyanjie/exp_data/exp_nnsslm/test_data_dir/lsp_test_106/audio"  # .wav音频所在目录
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
        audio_path = os.path.join(audio_dir_path, audio_id + '.wav')
        print('audio: ' + audio_path, flush=True)
        # load signal
        fs, sig = apkit.load_wav(audio_path)  # sig.shape: [C, length] ndarray(float64)
        
        feat_seg_idx = 0
        with open(gt, "r") as f:
            lines = f.readlines()
            for line in lines:
                gt_seg_data = line.split(' ')
                feat_seg_idx = int(gt_seg_data[0])
                print(f'gt_seg_data {gt_seg_data}', flush=True)

                if gt_seg_data[8] == '1\n':
                    num_sources = 2
                else:
                    num_sources = 1
                # print(f'num_sources {num_sources}', flush=True)
                doa_gt_1 = np.arctan2(float(gt_seg_data[1]), float(gt_seg_data[2]))
                doa_gt_1 = round(math.degrees(doa_gt_1)) + 180

                doa_gt_2 = np.nan
                if num_sources == 2:
                    doa_gt_2 = np.arctan2(float(gt_seg_data[4]), float(gt_seg_data[5]))
                    doa_gt_2 = round(math.degrees(doa_gt_2)) + 180

                label_seg_level = [doa_gt_1, doa_gt_2]

                # calculate the complex spectrogram stft
                win_size = SEG_LEN
                tf = apkit.stft(sig[:, feat_seg_idx * SEG_HOP : feat_seg_idx * SEG_HOP + SEG_LEN], apkit.cola_hamming, win_size, win_size, last_sample=True)
                # tf.shape: [C, num_frames, win_size] tf.dtype: complex128
                nch, nframe, _ = tf.shape # (4, num_frames, 8192)
                # print("A tf.shape: {}".format(tf.shape))

                # trim freq bins
                max_fbin = int(_FREQ_MAX * win_size / fs)  # 100-8kHz
                min_fbin = int(_FREQ_MIN * win_size / fs)  # 100-8kHz
                freq = np.fft.fftfreq(win_size)[min_fbin:max_fbin]
                tf = tf[:, :, min_fbin:max_fbin]  # tf.shape: (4, num_frames, 1348)
                # print("B tf.shape: {}".format(tf.shape))

                # compute pairwise gcc on f-banks
                ecov = apkit.empirical_cov_mat(tf, fw=1, tw=1)
                nfbank = 40
                zoom = 25
                eps = 0.0

                fbw = apkit.mel_freq_fbank_weight(nfbank, freq, fs, fmax=_FREQ_MAX,
                                                    fmin=_FREQ_MIN)
                fbcc = apkit.gcc_phat_fbanks(ecov, fbw, zoom, freq, eps=eps)

                # merge to a single numpy array, indexed by 'tpbd'
                #                                           (time=num_frames, pair=6, bank=40, delay=51)
                feature = np.asarray([fbcc[(i,j)] for i in range(nch)
                                                    for j in range(nch)
                                                    if i < j])
                feature = np.moveaxis(feature, 2, 0)

                # and map [-1.0, 1.0] to 16-bit integer, to save storage space
                dtype = np.int16
                vmax = np.iinfo(dtype).max
                feature = (feature * vmax).astype(dtype) # feature.shape: (num_frames, 6, 40, 51)
                gcc_fbank_seg_level = feature[0] # gcc_fbank_seg_level.shape: (6, 40, 51)
                
                sample_data = {"gcc_fbank_seg_level": gcc_fbank_seg_level, "label_seg_level": label_seg_level, "num_sources" : num_sources}
                save_path = os.path.join(data_frame_path, '{}_seg_{}.pkl'.format(audio_id, feat_seg_idx))
                # print(save_path, flush=True)
                print("sample_data's feat.shape {}".format(sample_data["gcc_fbank_seg_level"].shape), flush=True)
                print("sample_data's label {}".format(sample_data["label_seg_level"]), flush=True)
                pkl_file = open(save_path, 'wb')
                pickle.dump(sample_data, pkl_file)
                pkl_file.close()
                cnt_segs += 1

    print("cnt_segs: {}".format(cnt_segs), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate multi sources data at frame level')
    parser.add_argument('gt_frame', metavar='GT_FRAME_PATH', type=str,
                        help='path to the gt_frame directory')
    parser.add_argument('audio_dir', metavar='AUDIO_DIR_PATH', type=str,
                        help='path to the audio_dir directory')
    parser.add_argument('data_frame', metavar='DATA_FRAME_PATH', type=str,
                        help='path to the data_frame directory')
    args = parser.parse_args()
    main(args.gt_frame, args.audio_dir, args.data_frame)
