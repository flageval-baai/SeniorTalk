# Baseline Implementation with speechbrain

We provide a baseline implementation using the open-source [SpeechBrain](https://github.com/speechbrain/speechbrain) toolkit.

## Steps to Run the Baseline

1. Prepare Dataset: Run nankai_elderly_prepare.py to prepare the dataset.

2. Train Speaker Embeddings: Run train_speaker_embeddings.py to train the models.

3. Calculate Cosine Similarity (Method 1): Run speaker_verification_cosine.py to calculate the cosine similarity results.  The script will random a 2e4 trail file, so the result is different from time to time.
```bash
CUDA_VISIBLE_DEVICES=6 python speaker_verification_cosine.py  configs/elderly_datasets_resnet.yaml 
CUDA_VISIBLE_DEVICES=6 python speaker_verification_cosine.py  configs/elderly_datasets_ecapatdnn.yaml 
CUDA_VISIBLE_DEVICES=6 python speaker_verification_cosine.py  configs/elderly_datasets_xvec.yaml 
```

5. Calculate Cosine Similarity (Method 2): Run speaker_verification_plda.py to calculate the cosine similarity results. The script will random a 2e4 trail file, so the result is different from time to time.

```bash
CUDA_VISIBLE_DEVICES=6 python speaker_verification_plda.py  configs/elderly_datasets_resnet.yaml 
CUDA_VISIBLE_DEVICES=6 python speaker_verification_plda.py  configs/elderly_datasets_ecapatdnn.yaml 
CUDA_VISIBLE_DEVICES=6 python speaker_verification_plda.py  configs/elderly_datasets_xvec.yaml
```

# **Citing SpeechBrain**
Please, cite SpeechBrain if you use it for your research or business.

```bibtex
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}
```

