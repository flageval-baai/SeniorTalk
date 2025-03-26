#!/bin/bash

. check_conda.sh
conda activate Voice

CUDA_VISIBLE_DEVICES=5 CUDA_LAUNCH_BLOCKING=1 \
python train_speaker_embeddings.py configs/kids_datasets_ecapa_tdnn.yaml



CUDA_VISIBLE_DEVICES=5 CUDA_LAUNCH_BLOCKING=1 \
python train_speaker_embeddings.py configs/kids_datasets_xvector.yaml

CUDA_VISIBLE_DEVICES=5 CUDA_LAUNCH_BLOCKING=1 \
python speaker_verification_plda.py configs/kids_datasets_xvector.yaml

CUDA_VISIBLE_DEVICES=5 CUDA_LAUNCH_BLOCKING=1 \
python speaker_verification_cosine.py configs/kids_datasets_xvector.yaml


CUDA_VISIBLE_DEVICES=4 CUDA_LAUNCH_BLOCKING=1 \
python train_speaker_embeddings.py configs/kids_datasets_resnet.yaml

CUDA_VISIBLE_DEVICES=4 CUDA_LAUNCH_BLOCKING=1 \
python speaker_verification_plda.py configs/kids_datasets_resnet.yaml

CUDA_VISIBLE_DEVICES=4 CUDA_LAUNCH_BLOCKING=1 \
python speaker_verification_cosine.py configs/kids_datasets_resnet.yaml