# remember to activate your conda env
#source activate torch1.8_env

# start whatever your job below, e.g., python, matlab, etc.
#ADD YOUR COMMAND HERE,LIKE python3 main.py
#./apply_feature_extraction.sh ../extract_gcc_fbank.py /Users/fuyanjie/Desktop/temp/exp_nnsslm gcc_fbank

python extract_gcc_fbank.py /Users/fuyanjie/Desktop/temp/exp_nnsslm/audio/ssl-data_2017-05-06-14-11-12_0.wav /Users/fuyanjie/Desktop/temp/exp_nnsslm/gcc_fbank/
#chmod a+x run.sh

# hostname
# sleep 10
echo "job end time:`date`"
