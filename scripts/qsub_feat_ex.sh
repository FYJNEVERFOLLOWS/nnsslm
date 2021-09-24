#!/bin/bash
#$ -S /bin/bash

#here you'd best to change testjob as username
#$ -N fyj_extract_gccfbank_feature_from_SSLR_lsp_train_301

#cwd define the work environment,files(username.o) will generate here
#$ -cwd

# merge stdo and stde to one file
#$ -j y

# resource requesting, e.g. for gpu use
#$ -l h=gpu06

# remember to activate your conda env
source activate nnsslm_env

echo "job start time: `date`"
# start whatever your job below, e.g., python, matlab, etc.
#ADD YOUR COMMAND HERE,LIKE python3 main.py

CUDA_VISIBLE_DEVICES=2,3 \
./apply_feature_extraction.sh ../extract_gcc_fbank.py /Work20/2021/fuyanjie/pycode/nnsslm/train_data_dir/lsp_train_301 gcc_fbank


#chmod a+x run.sh

# hostname
# sleep 10
echo "job end time:`date`"
