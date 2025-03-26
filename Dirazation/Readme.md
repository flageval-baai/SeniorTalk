# Baseline Implementation with pyannote-audio

We provide a baseline implementation using the open-source [pyannote-audio](https://github.com/pyannote/pyannote-audio/tree/develop) toolkit.

## Steps to Run the Baseline

1. Install pyannote-audio: Follow the instructions in the pyannote-audio documentation to set up the environment and run a standard speaker diarization task.

2. Configure Model Path: Update the config.yaml file located in the ./model directory to point to the path of your trained model for the speaker verification task (SV).

3. Adjust Local Paths: Modify all instances in speaker_diarization_pyannote.py to reflect your local directory paths, ensuring they correspond to your setup.

4. Run Speaker Diarization: Execute all speaker_diarization_pyannote.py scripts to generate the diarization results, which will be saved in the ./output2 directory.

5. Calculate Diarization Error Rate (DER): Run all cal_der.py scripts to compute the diarization error rate (DER) of the generated results.