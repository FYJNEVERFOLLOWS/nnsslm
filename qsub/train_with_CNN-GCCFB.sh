#!/bin/bash
#$ -S /bin/bash

#here you'd best to change testjob as username
#$ -N fyj_train_with_CNN-GCCFB_epoch30_batch128

#cwd define the work environment,files(username.o) will generate here
#$ -cwd

# merge stdo and stde to one file
#$ -j y

# resource requesting, e.g. for gpu use
#$ -l h=gpu04

# remember to activate your conda env
source activate nnsslm_env

echo "job start time: `date`"
# start whatever your job below, e.g., python, matlab, etc.
#ADD YOUR COMMAND HERE,LIKE python3 main.py

CUDA_VISIBLE_DEVICES=0 \
python ../CNN-GCCFB.py

#chmod a+x run.sh

# hostname
# sleep 10
echo "job end time:`date`"
