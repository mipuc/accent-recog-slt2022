#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright © <2022> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Juan Zuluaga-Gomez <jzuluaga@idiap.ch>
#
# SPDX-License-Identifier: MIT-License 

# Base script to fine-tunine a ECAPA-TDNN model on Accent Classification for English
#######################################
# COMMAND LINE OPTIONS,
# high-level variables for training the model. TrainingArguments (HuggingFace)
# You can pass a qsub command (SunGrid Engine)
#       :an example is passing --cmd "src/sge/queue.pl h='*'-V", add your configuration
set -euo pipefail
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# static vars
# cmd='/remote/idiap.svm/temp.speech01/jzuluaga/kaldi-jul-2020/egs/wsj/s5/utils/parallel/queue.pl -l gpu -P minerva -l h='vgn[ij]*' -V'
cmd='none'

# training vars
# ecapa_tdnn_hub="speechbrain/spkrec-ecapa-voxceleb/embedding_model.ckpt"
ecapa_tdnn_hub="speechbrain/lang-id-voxlingua107-ecapa/embedding_model.ckpt"
seed="3006"
apply_augmentation="True"
max_batch_len=10 #600

# data folder:
output_dir="/home/projects/vokquant/accent-recog-slt2022/CommonAccent/results/ECAPA-TDNN/AT/lang-id-voxlingua107-ecapa"


# If augmentation is defined:
if [ "$apply_augmentation" == 'True' ]; then
    output_folder="${output_dir}/$seed"
    rir_folder="data/rir_folder/"
else
    output_folder="$output_dir/$seed"
    rir_folder=""
fi

# configure a GPU to use if we a defined 'CMD'
if [ ! "$cmd" == 'none' ]; then
  basename=train_$(basename $ecapa_tdnn_hub)_${apply_augmentation}
  cmd="$cmd -N ${basename} ${output_folder}/log/train_log"
else
  cmd=''
fi

echo "*** About to start the training ***"
echo "*** output folder: $output_folder ***"

# While loop to keep retrying the training process after failure
attempt=1
max_attempts=1 # Maximum number of attempts before stopping

while [ $attempt -le $max_attempts ]; do
    echo "Attempt $attempt/$max_attempts"
    
    if $cmd python3 accent_id/train.py accent_id/hparams/train_ecapa_tdnn_at_regions.yaml \
        --seed=$seed \
        --skip_prep="True" \
        --rir_folder="$rir_folder" \
        --apply_augmentation="$apply_augmentation" \
        --max_batch_len="$max_batch_len" \
        --output_folder="$output_folder" \
        --ecapa_tdnn_hub="$ecapa_tdnn_hub"; then
        echo "Training completed successfully!"
        break # Exit loop if successful
    else
        echo "Training failed, retrying..."
        attempt=$((attempt+1))
    fi
    
    if [ $attempt -gt $max_attempts ]; then
        echo "Training failed after $max_attempts attempts, exiting."
        exit 1
    fi

    echo "Restarting training..."
done

echo "Done training of $ecapa_tdnn_hub in $output_folder"
exit 0