

# SeniorTalk: A Chinese Conversation Dataset with Rich Annotations for Super-Aged Seniors
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-yellow)](https://huggingface.co/datasets/BAAI/SeniorTalk)
[![arXiv](https://img.shields.io/badge/arXiv-2409.18584-b31b1b.svg)](https://www.arxiv.org/pdf/2503.16578)
[![License: CC BY-NC-SA-4.0](https://img.shields.io/badge/License-CC%20BY--SA--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Github](https://img.shields.io/badge/Github-ChildMandarin-blue)](https://github.com/flageval-baai/SeniorTalk)

## Introduction

**SeniorTalk** is a comprehensive, open-source Mandarin Chinese speech dataset specifically designed for research on  elderly aged 75 to 85. This dataset addresses the critical lack of publicly available resources for this age group, enabling advancements in automatic speech recognition (ASR), speaker verification (SV), speaker dirazation (SD), speech editing and other related fields.  The dataset is released under a **CC BY-NC-SA 4.0 license**, meaning it is available for non-commercial use.

## Dataset Details

This dataset contains 55.53 hours of high-quality speech data collected from 202 elderly across 16 provinces in China. Key features of the dataset include:

*   **Age Range:**  75-85 years old (inclusive).  This is a crucial age range often overlooked in speech datasets.
*   **Speakers:** 202 unique child speakers.
*   **Geographic Diversity:** Speakers from 16 of China's 34 provincial-level administrative divisions, capturing a range of regional accents.
*   **Gender Balance:**  Approximately 1:3 representation of male and female speakers, largely attributed to the differing average ages of males and females among the elderly.
*   **Recording Conditions:**  Recordings were made in quiet environments using a variety of smartphones (both Android and iPhone devices) to ensure real-world applicability.
*   **Content:**  Natural, conversational speech during age-appropriate activities.  The content is unrestricted, promoting spontaneous and natural interactions.
*   **Audio Format:**  WAV files with a 16kHz sampling rate. 
*   **Transcriptions:**  Carefully crafted, character-level manual transcriptions.  
* **Annotations:** The dataset includes annotations for each utterance, and for the speakers level.
    *   **Session-level**:  `sentence_start_time`,`sentence_end_time`,`overlapped speech`
    *   **Utterance-level**:  `id`, `accent_level`, `text` (transcription).
    *   **Token-level**:   `special token`([SONANT],[MUSIC],[NOISE]....)
    *   **Speaker-level**: `speaker_id`, `age`, `gender`, `location` (province), `device`.
      
### Dataset Structure

## Dialogue Dataset


The dataset is split into two subsets:
| Split      | # Speakers | # Dialogues | Duration (hrs) | Avg. Dialogue Length (h) |
| :--------- | :--------: | :----------: | :------------: | :-----------------------: |
| `train`    |    182     |    91    |     49.83     |           0.54            |
| `test`     |     20     |    10     |      5.70      |           0.57            |
| **Total**  |  **202**   |  **101**  |   **55.53**   |       **0.55**           |



The dataset file structure is as follows.
```

dialogue_data/  
├── wav  
│   ├── train/*.tar   
│   └── test/*.tar   
└── transcript/*.txt
UTTERANCEINFO.txt  # annotation of topics and duration
SPKINFO.txt   # annotation of location , age , gender and device
```
Each WAV file has a corresponding TXT file with the same name, containing its annotations.

For more details, please refer to our paper [SeniorTalk](https://www.arxiv.org/abs/2503.16578).

## ASR Dataset


The dataset is split into three subsets:
| Split      | # Speakers | # Utterances | Duration (hrs) | Avg. Utterance Length (s) |
| :--------- | :--------: | :----------: | :------------: | :-----------------------: |
| `train`    |    162     |    47,269    |     29.95      |           2.28            |
| `validation` |     20     |    6,891     |      4.09      |           2.14          |
| `test`     |     20     |    5,869    |      3.77     |           2.31            |
| **Total**  |  **202**   |  **60,029**  |   **37.81**   |       **2.27**           |


The dataset file structure is as follows.
```
sentence_data/  
├── wav  
│   ├── train/*.tar
│   ├── dev/*.tar 
│   └── test/*.tar   
└── transcript/*.txt   
UTTERANCEINFO.txt  # annotation of topics and duration
SPKINFO.txt   # annotation of location , age , gender and device
```
Each WAV file has a corresponding TXT, containing its annotations.

For more details, please refer to our paper [SeniorTalk](https://www.arxiv.org/abs/2503.16578).


## 📐 Experiments

We conducted experiments on Automatic Speech Recognition (ASR) , Speaker Verification (SV) tasks , Speaker Dirazation (SD) tasks and Speech Editing tasks  to evaluate the dataset.

### 1️⃣ ASR Results

#### Models Trained from Scratch

| Encoder         | # Params | CER   | No    | Light | Moderate | Heavy | South | North |
|-----------------|----------|-------|-------|-------|----------|-------|-------|-------|
| Transformer     | 14.1M    | 48.99 | 22.58 | 49.05 | 51.07    | 80.95 | 48.5  | 50.24 |
| Conformer       | 15.7M    | 34.61 | 21.23 | 34.21 | 37.62    | 59.52 | 34.55 | 34.74 |
| E-Branchformer  | 16.9M    | 33.25 | 23.25 | 20.71 | 33.03    | 35.32 | 64.29 | 33.94 |

#### Fine-tuned Pre-trained Models

| Model               | # Params | Zero-shot | Fine-tuning |
|---------------------|----------|-----------|-------------|
| Paraformer-large    | 232M     | 14.91     | 14.41       |
| Whisper-tiny        | 39M      | 92.20     | 58.80       |
| Whisper-base        | 74M      | 64.02     | 38.17       |
| Whisper-small       | 244M     | 55.83     | 28.69       |
| Whisper-medium      | 769M     | 60.47     | 25.77       |
| Whisper-large-v3    | 1,550M   | 57.74     | 23.84       |



### 2️⃣ SV Results


| Model           | #Params | Dim     | Dev (%) | EER (%) | minDCF  | EER (%) | minDCF  | 
|-----------------|---------|---------|---------|---------|---------|---------|---------|
| X-vector        | 4.2M    | 512     | 12.04   | 14.63   | 0.9768  | 19.26   | 0.9598  | 
| ResNet-TDNN     | 15.5M   | 256     | 4.372   | 10.88   | 0.8450  | 11.50   | 0.9196  | 
| ECAPA-TDNN      | 20.8M   | 192     | 8.86    | 11.54   | 10.24   | 0.9582  | 0.9582       | 

### 3️⃣ SD Results

| Model | # Params | Dim | collar=0 DER(%) | collar=0 Confusion(%) | collar=0.25 DER(%) | collar=0.25 Confusion(%) |
|-------|----------|-----|-----------------|-----------------------|---------------------|-------------------------|
| ResNet-34-LM | 15.5M | 256 | 33.14 | 16.82 | 28.39 | 16.85 |
| x-vector | 4.2M | 512 | 53.01 | 36.69 | 49.82 | 38.28 |
| ResNet-TDNN | 15.5M | 256 | 43.44 | 27.13 | 39.58 | 28.03 |
| ECAPA-TDNN | 20.8M | 192 | 27.84 | 11.52 | 22.85 | 11.31 |





### 4️⃣ Speech Editing Results

| Method       | MCD(↓) | STOI(↑) | PESQ(↑) |
|--------------|--------|---------|---------|
| CampNet      | 7.302  | 0.220   | 1.291   |
| EditSpeech   | 6.225  | 0.514   | 1.363   |
| A3T          | 5.851  | 0.586   | 1.455   |
| FluentSpeech | 5.811  | 0.627   | 1.645   |



## 🤗 Dataset Download

You can access the SeniorTalk dataset on HuggingFace Datasets:

[https://huggingface.co/datasets/BAAI/SeniorTalk](https://huggingface.co/datasets/BAAI/SeniorTalk)



##  📚 Cite me
```
@misc{chen2025seniortalkchineseconversationdataset,
      title={SeniorTalk: A Chinese Conversation Dataset with Rich Annotations for Super-Aged Seniors}, 
      author={Yang Chen and Hui Wang and Shiyao Wang and Junyang Chen and Jiabei He and Jiaming Zhou and Xi Yang and Yequan Wang and Yonghua Lin and Yong Qin},
      year={2025},
      eprint={2503.16578},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.16578}, 
}
``` 
