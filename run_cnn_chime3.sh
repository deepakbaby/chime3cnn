#!/bin/bash


##################################################################
# Kaldi Script for CNN-DNN + sMBR training on Chime-3 data. The input features used here are 
# (Mel+3 pitch feats)+delta+delta-delta features (40 Mel banks + 3 pitch feats 
# with delta and double-delta has dimension 129).
# These features are used since these are found to be the best features in Tara Sainath's paper on CNN.

# T.N. Sainath, A.-R. Mohamed, B. Kingsbury, and B. Ramabhadran, “Deep convolutional neural networks
# for LVCSR,” in Acoustics, Speech and Signal Processing (ICASSP), 2013 IEEE International Conference on,
# May 2013, pp. 8614–8618.

# If you use this code , please add the following citations
#[1] D. Baby, T. Virtanen and H. Van hamme, "Coupled Dictionary-based Speech Enhancement for CHiME-3 Challenge", 
# Submitted to IEEE 2015 Automatic Speech Recognition and Understanding Workshop (ASRU), 2015.
# [2] Jon Barker, Ricard Marxer, Emmanuel Vincent, and Shinji Watanabe, "The third 'CHiME' 
# Speech Separation and Recognition Challenge: Dataset, task and baselines", 
# submitted to IEEE 2015 Automatic Speech Recognition and Understanding Workshop (ASRU), 2015.


# RUN THIS ONLY AFTER RUNNING DATA PREPARATION SCRIPT FOR CHIME ENHANCED DATA 
# (Here we run it after the run_gmm.sh script which already does this step.)
# Else, add these two lines to the code
#local/real_enhan_chime3_data_prep.sh $enhan $enhan_data || exit 1;
#local/simu_enhan_chime3_data_prep.sh $enhan $enhan_data || exit 1;


# Modified Code From KALDI CNN Recipe for RM dataset (kaldi-trunk/egs/rm/s5/local/run_cnn.sh).
# Modified by Deepak Baby, KU Leuven, June 2015.

# Usage : 'local/run_cnn_dpk.sh $enhancement_method'  
#(if data prep is already done, else add the enhanced data folder as the second argument)
###################################################################


. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)


enhan=$1

gmmali=exp/tri3b_tr05_multi_${enhan}_ali
gmmali_dev=exp/tri3b_tr05_multi_${enhan}_ali_dt05 

stage=0
. utils/parse_options.sh

# Preparing pitch conf for CNN training
echo "--nccf-ballast-online=true" > conf/pitch.conf

# Make the FBANK + pitch features
for x in et05_real_$enhan et05_simu_$enhan  dt05_real_$enhan tr05_real_$enhan dt05_simu_$enhan tr05_simu_$enhan ; do
  mkdir -p data-fbank-cnn && cp -rf data/$x data-fbank-cnn/ && rm data-fbank-cnn/$x/{feats,cmvn}.scp
  steps/make_fbank_pitch.sh --nj 10 --cmd "$train_cmd" \
     data-fbank-cnn/$x data-fbank-cnn/$x/log data-fbank-cnn/$x/data || exit 1;
  steps/compute_cmvn_stats.sh data-fbank-cnn/$x data-fbank-cnn/$x/log data-fbank-cnn/$x/data || exit 1;
done

# make mixed training set from real and simulation enhancement training data
# multi = simu + real
utils/combine_data.sh data-fbank-cnn/tr05_multi_$enhan data-fbank-cnn/tr05_simu_$enhan data-fbank-cnn/tr05_real_$enhan
utils/combine_data.sh data-fbank-cnn/dt05_multi_$enhan data-fbank-cnn/dt05_simu_$enhan data-fbank-cnn/dt05_real_$enhan

echo "Starting CNN (pre)training " 
# Run the CNN pre-training.
if [ $stage -le 1 ]; then
  dir=exp/cnn4c_$enhan
  ali=${gmm}_ali
  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh \
      --apply-cmvn true --norm-vars true --delta-order 2 --splice 5 \
      --prepend-cnn-type cnn1d --cnn-proto-opts "--patch-dim1 8 --pitch-dim 3" \
      --hid-layers 2 --learn-rate 0.008 --train-opts "--verbose 2" \
      data-fbank-cnn/tr05_multi_$enhan data-fbank-cnn/dt05_multi_$enhan data/lang $gmmali $gmmali_dev $dir || exit 1;
  # Decode
  # decode enhan speech
echo "Decoding with pretrained CNN"
utils/mkgraph.sh data/lang_test_tgpr_5k $dir $dir/graph_tgpr_5k || exit 1;
steps/nnet/decode.sh --nj 4  --acwt 0.10 --config conf/decode_dnn.config $dir/graph_tgpr_5k data-fbank-cnn/dt05_real_$enhan $dir/decode_tgpr_5k_dt05_real_$enhan 
steps/nnet/decode.sh --nj 4  --acwt 0.10 --config conf/decode_dnn.config $dir/graph_tgpr_5k data-fbank-cnn/dt05_simu_$enhan $dir/decode_tgpr_5k_dt05_simu_$enhan 

steps/nnet/decode.sh --nj 4  --acwt 0.10 --config conf/decode_dnn.config $dir/graph_tgpr_5k data-fbank-cnn/et05_real_$enhan $dir/decode_tgpr_5k_et05_real_$enhan 
steps/nnet/decode.sh --nj 4  --acwt 0.10 --config conf/decode_dnn.config $dir/graph_tgpr_5k data-fbank-cnn/et05_simu_$enhan $dir/decode_tgpr_5k_et05_simu_$enhan 
fi

echo "RBM pretraining for DNNs (4 layers)"
# Pre-train stack of RBMs on top of the convolutional layers (4 layers, 1024 units)
if [ $stage -le 2 ]; then
  dir=exp/cnn4c_pretrain-dbn_$enhan
  transf_cnn=exp/cnn4c_$enhan/final.feature_transform_cnn # transform with convolutional layers
  # Train
  $cuda_cmd $dir/log/pretrain_dbn.log \
    steps/nnet/pretrain_dbn.sh --nn-depth 4 --hid-dim 1024 --rbm-iter 20 \
    --feature-transform $transf_cnn --input-vis-type bern \
    --param-stddev-first 0.05 --param-stddev 0.05 \
    data-fbank-cnn/tr05_multi_$enhan $dir || exit 1
fi

echo "Realigning using CNN"
# Re-align using CNN
if [ $stage -le 3 ]; then
  dir=exp/cnn4c_$enhan
  steps/nnet/align.sh --nj 20 --cmd "$train_cmd" \
   data-fbank-cnn/tr05_multi_$enhan data/lang $dir ${dir}_ali || exit 1
fi

echo "Training the whole network (2CNN and 4DNN layers)"
# Train the DNN optimizing cross-entropy.
if [ $stage -le 4 ]; then
  dir=exp/cnn4c_pretrain-dbn_dnn_$enhan; [ ! -d $dir ] && mkdir -p $dir/log;
  ali=exp/cnn4c_${enhan}_ali
  feature_transform=exp/cnn4c_${enhan}/final.feature_transform
  feature_transform_dbn=exp/cnn4c_pretrain-dbn_${enhan}/final.feature_transform
  dbn=exp/cnn4c_pretrain-dbn_${enhan}/4.dbn
  cnn_dbn=$dir/cnn_dbn.nnet
  { # Concatenate CNN layers and DBN,
    num_components=$(nnet-info $feature_transform | grep -m1 num-components | awk '{print $2;}')
    nnet-concat "nnet-copy --remove-first-layers=$num_components $feature_transform_dbn - |" $dbn $cnn_dbn \
      2>$dir/log/concat_cnn_dbn.log || exit 1 
  }
  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --feature-transform $feature_transform --dbn $cnn_dbn --hid-layers 0 \
    data-fbank-cnn/tr05_multi_$enhan data-fbank-cnn/dt05_multi_$enhan data/lang $gmmali $gmmali_dev $dir || exit 1;
  # Decode (reuse HCLG graph)
utils/mkgraph.sh data/lang_test_tgpr_5k $dir $dir/graph_tgpr_5k || exit 1;
echo "Decoding with the CNN-DNN model"
steps/nnet/decode.sh --nj 4 --acwt 0.10 --config conf/decode_dnn.config $dir/graph_tgpr_5k data-fbank-cnn/dt05_real_$enhan $dir/decode_tgpr_5k_dt05_real_$enhan 
steps/nnet/decode.sh --nj 4 --acwt 0.10 --config conf/decode_dnn.config $dir/graph_tgpr_5k data-fbank-cnn/dt05_simu_$enhan $dir/decode_tgpr_5k_dt05_simu_$enhan 

steps/nnet/decode.sh --nj 4 --acwt 0.10 --config conf/decode_dnn.config $dir/graph_tgpr_5k data-fbank-cnn/et05_real_$enhan $dir/decode_tgpr_5k_et05_real_$enhan 
steps/nnet/decode.sh --nj 4 --acwt 0.10 --config conf/decode_dnn.config $dir/graph_tgpr_5k data-fbank-cnn/et05_simu_$enhan $dir/decode_tgpr_5k_et05_simu_$enhan 

fi


echo "Starting sMBR"
# Sequence training using sMBR criterion, we do Stochastic-GD 
# with per-utterance updates.
dir=exp/cnn4c_pretrain-dbn_dnn_smbr_$enhan
srcdir=exp/cnn4c_pretrain-dbn_dnn_$enhan
acwt=0.1

# First we generate lattices and alignments:
if [ $stage -le 4 ]; then
  steps/nnet/align.sh --nj 8 --cmd "$train_cmd" \
    data-fbank-cnn/tr05_multi_$enhan data/lang $srcdir ${srcdir}_ali || exit 1;
  steps/nnet/make_denlats.sh --nj 8 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
    data-fbank-cnn/tr05_multi_$enhan data/lang $srcdir ${srcdir}_denlats || exit 1;
fi

# Re-train the DNN by 6 iterations of sMBR 
if [ $stage -le 5 ]; then
  steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 6 --acwt $acwt --do-smbr true \
    data-fbank-cnn/tr05_multi_$enhan data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1
  # Decode
  utils/mkgraph.sh data/lang_test_tgpr_5k $dir $dir/graph_tgpr_5k || exit 1;
  for ITER in 1 2 3 4 5 6; do
    steps/nnet/decode.sh --nj 4 --num-threads 4 --acwt 0.10 --config conf/decode_dnn.config --nnet $dir/${ITER}.nnet --acwt $acwt $dir/graph_tgpr_5k data-fbank-cnn/dt05_real_$enhan $dir/decode_tgpr_5k_dt05_real_${enhan}_it${ITER} &
    steps/nnet/decode.sh --nj 4 --num-threads 4 --acwt 0.10 --config conf/decode_dnn.config --nnet $dir/${ITER}.nnet --acwt $acwt $dir/graph_tgpr_5k data-fbank-cnn/dt05_simu_$enhan $dir/decode_tgpr_5k_dt05_simu_${enhan}_it${ITER} &
wait ;
steps/nnet/decode.sh --nj 4 --num-threads 4 --acwt 0.10 --config conf/decode_dnn.config --nnet $dir/${ITER}.nnet --acwt $acwt $dir/graph_tgpr_5k data-fbank-cnn/et05_real_$enhan $dir/decode_tgpr_5k_et05_real_${enhan}_it${ITER} &
    steps/nnet/decode.sh --nj 4 --num-threads 4 --acwt 0.10 --config conf/decode_dnn.config --nnet $dir/${ITER}.nnet --acwt $acwt $dir/graph_tgpr_5k data-fbank-cnn/et05_simu_$enhan $dir/decode_tgpr_5k_et05_simu_${enhan}_it${ITER} &   
 wait;
  done 
fi


# decoded results of enhan speech using enhan CNN AMs
for x in cnn4c_$enhan exp/cnn4c_pretrain-dbn_dnn_$enhan ; do
echo "Showing Best results from $x"
local/chime3_calc_wers.sh $x $enhan > $x/best_wer_$enhan.result
head -n 11 $x/best_wer_$enhan.result
done

echo " Best Results from sMBR iterations "
./local/chime3_calc_wers_smbr.sh exp/cnn4c_pretrain-dbn_dnn_smbr_$enhan ${enhan} exp/cnn4c_pretrain-dbn_dnn_smbr_$enhan/graph_tgpr_5k \
    > exp/cnn4c_pretrain-dbn_dnn_smbr_$enhan/best_wer_${enhan}.result
head -n 11 exp/cnn4c_pretrain-dbn_dnn_smbr_$enhan/best_wer_${enhan}.result



echo Success
exit 0

