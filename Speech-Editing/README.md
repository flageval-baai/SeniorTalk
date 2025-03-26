# Baseline Implementation with pyannote-audio

We provide a baseline implementation using the open-source [Speech-Editing-Toolkit](https://github.com/Zain-Jiang/Speech-Editing-Toolkit).

## Steps to Use the Speech-Editing-Toolkit  

1. Clone the Repository: Clone the Speech-Editing-Toolkit repository to your local machine.  

2. Modify Data Preparation Method: Replace the data preparation code in `./data_gen` with our code. Then, follow the instructions in the README to generate raw, processed, and binary data. You can find the README [here](https://github.com/Zain-Jiang/Speech-Editing-Toolkit).  

3. Example Run for FluentSpeech:  Execute the following command:  
   ```bash  
   CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config egs/spec_denoiser.yaml --exp_name spec_denoiser_elderly --reset 
   ``` 
4. Example Run for CampNet A3T EditSpeech:
Execute the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config egs/campnet.yaml --exp_name campnet_elderly --reset  

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config egs/a3t.yaml --exp_name a3t_elderly --reset  

CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config egs/editspeech.yaml --exp_name editspeech_elderly --reset  
```