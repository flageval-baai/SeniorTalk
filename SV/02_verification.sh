#!/bin/bash

. check_conda.sh
conda activate Voice

python speaker_verification_cosine.py configs/kids_datasets.yaml