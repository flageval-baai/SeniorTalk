import os
from pyannote.database.util import load_rttm
from pyannote.core import Segment, notebook
from pyannote.audio import Audio
from IPython.display import Audio as IPythonAudio
import torch

# load pretrained pipeline
import os
from pyannote.audio import Pipeline
import inspect



def generate_dirazation(config_name):
    pipeline = Pipeline.from_pretrained("/home/chenyang/chenyang_space/speech_editing_and_tts/projects/speaker_dirazation/model/"+config_name)

    if torch.cuda.is_available():
        pipeline.to(torch.device('cuda:3'))

    def get_der(audio_path, rttm_path):
        OWN_FILE = {'audio': audio_path}
        waveform, sample_rate = Audio(mono="downmix")(OWN_FILE)

        groundtruths = load_rttm(rttm_path)
        if OWN_FILE['audio'] in groundtruths:
            groundtruth = groundtruths[OWN_FILE['audio']]
        else:
            _, groundtruth = groundtruths.popitem()
            


        # run the pipeline (with progress bar)
        from pyannote.audio.pipelines.utils.hook import ProgressHook
        with ProgressHook() as hook:
            diarization = pipeline(OWN_FILE, hook=hook,num_speakers=2)
        return diarization
    
    
    # 做3和10的dirazation就行
    dirazation3 = get_der("/home/chenyang/chenyang_space/speech_editing_and_tts/projects/speaker_dirazation/data/new_test/105_106/2.wav", "/home/chenyang/chenyang_space/speech_editing_and_tts/projects/speaker_dirazation/data/output_final_test_3.rttm")
    dirazation10 = get_der("/home/chenyang/chenyang_space/speech_editing_and_tts/projects/speaker_dirazation/data/new_test/117_118/2.wav", "/home/chenyang/chenyang_space/speech_editing_and_tts/projects/speaker_dirazation/data/output_final_test_10.rttm")
    
    output_dir = "/home/chenyang/chenyang_space/speech_editing_and_tts/projects/speaker_dirazation/data"
    rttm_output_path = output_dir.replace("data","output")
    rttm_output_path3 = os.path.join(rttm_output_path, f"output_final_test_3_{config_name}.rttm")
    rttm_output_path10 = os.path.join(rttm_output_path, f"output_final_test_10_{config_name}.rttm")
    with open(rttm_output_path3, 'w') as f:
             f.write(str(dirazation3))
             
    with open(rttm_output_path10, 'w') as f:
             f.write(str(dirazation10))
    
    
    # base_dir = "/home/chenyang/chenyang_space/speech_editing_and_tts/projects/speaker_dirazation/data/new_test"
    # output_dir = "/home/chenyang/chenyang_space/speech_editing_and_tts/projects/speaker_dirazation/data"

    # subfolders = [f.path for f in os.scandir(base_dir) if f.is_dir()]
    # der_list = []
    # for i, subfolder in enumerate(subfolders[:10], start=1):
    #     txt_files = [f for f in os.listdir(subfolder) if f.endswith('.txt')]
    #     wav_files = [f for f in os.listdir(subfolder) if f.endswith('.wav')]
        
    #     if len(txt_files) != 1 or len(wav_files) != 1:
    #         print(f"Skipping {subfolder}, expected one txt and one wav file.")
    #         continue
        
    #     txt_path = os.path.join(subfolder, txt_files[0])
    #     wav_path = os.path.join(subfolder, wav_files[0])
    #     rttm_path = os.path.join(output_dir, f"output_final_test_{i}.rttm")
    #     diarization = get_der(wav_path, rttm_path)
    #     rttm_output_path = output_dir.replace("data","output")
    #     rttm_output_path = os.path.join(rttm_output_path, f"output_final_test_{i}_{config_name}.rttm")
    #     with open(rttm_output_path, 'w') as f:
    #         f.write(str(diarization))
      
      
      
generate_dirazation("config_ecaptdnn.yaml")  
generate_dirazation("config_resnet.yaml")
generate_dirazation("config_xvec.yaml")

generate_dirazation("config_ecaptdnn_f.yaml")  
generate_dirazation("config_resnet_f.yaml")
generate_dirazation("config_xvec_f.yaml")


generate_dirazation("config.yaml")



