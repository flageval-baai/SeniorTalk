# Baseline Implementation with pyannote-audio

We provide a baseline implementation using the open-source [pyannote-audio](https://github.com/pyannote/pyannote-audio/tree/develop) toolkit.

## Steps to Run the Baseline

1. Follow the instructions of pyannote-audio, and figure out how to run a regular speaker dirazation task.
2. change the config.yaml in ./model into your model path trained in SV task
3. Modify all `speaker_dirazation_pyannote.py` to update the local paths according to your setup.
4. Run all `speaker_dirazation_pyannote.py` to generate dirazation result .
5. Run all `cal_der.py` to generate der result.