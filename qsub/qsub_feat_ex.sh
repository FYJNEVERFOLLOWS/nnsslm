#!/bin/bash
#$ -S /bin/bash

#here you'd best to change testjob as username
#$ -N fyj_extract_gccfbank_feature_from_SSLR_lsp_train_301

#cwd define the work environment,files(username.o) will generate here
#$ -cwd

# merge stdo and stde to one file
#$ -j y

# resource requesting, e.g. for gpu use
#$ -l h=gpu03

# remember to activate your conda env
source activate nnsslm_env

echo "job start time: `date`"
# start whatever your job below, e.g., python, matlab, etc.
#ADD YOUR COMMAND HERE,LIKE python3 main.py

../scripts/apply_feature_extraction.sh ../extract_gcc_fbank.py /Work18/2021/fuyanjie/exp_data/exp_nnsslm/test_data_dir/lsp_test_library gcc_fbank "-w 8192 -o 4096"
../scripts/apply_feature_extraction.sh ../extract_gcc_fbank.py /Work18/2021/fuyanjie/exp_data/exp_nnsslm/test_data_dir/lsp_test_106 gcc_fbank "-w 8192 -o 4096"
../scripts/apply_feature_extraction.sh ../extract_gcc_fbank.py /Work18/2021/fuyanjie/exp_data/exp_nnsslm/train_data_dir/lsp_train_106 gcc_fbank "-w 8192 -o 4096"
../scripts/apply_feature_extraction.sh ../extract_gcc_fbank.py /Work18/2021/fuyanjie/exp_data/exp_nnsslm/train_data_dir/lsp_train_301 gcc_fbank "-w 8192 -o 4096"

# python ../extract_gcc_fbank.py /Work20/2021/fuyanjie/exp_data/exp_nnsslm/train_data_dir/lsp_train_301/audio/ssl-data_2017-05-06-14-11-12_0.wav /Work20/2021/fuyanjie/exp_data/exp_nnsslm/train_data_dir/lsp_train_301/features/gcc_fbank -w 8192 -o 4096
#chmod a+x run.sh

# hostname
# sleep 10
echo "job end time:`date`"
