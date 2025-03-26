import os
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate

output_dir = "/home/chenyang/chenyang_space/speech_editing_and_tts/projects/speaker_dirazation/output2"
rttm_dir = "/home/chenyang/chenyang_space/speech_editing_and_tts/projects/speaker_dirazation/data"
name = "output_final_test_"

def calculate_average_metrics(config_name, collar):
    print(f"----------------------{config_name} (collar={collar})-------------------------")
  
    total_false_alarm = 0
    total_confusion = 0
    total_missed_detection = 0
    total_frame = 0

    metric = DiarizationErrorRate(collar=collar)

    for i in range(1, 11):
        gt_path = os.path.join(rttm_dir, f"{name}{i}.rttm")
        groundtruths = load_rttm(gt_path)
        _, groundtruth = groundtruths.popitem()

        diarization = load_rttm(os.path.join(output_dir, f"{name}{i}_{config_name}.rttm"))
        _, diarization = diarization.popitem()

        der = metric(groundtruth, diarization, detailed=True)
        # print(der)
        
        total_false_alarm += der['false alarm']
        total_confusion += der['confusion']
        total_missed_detection += der['missed detection']
        total_frame += der['total']

    
    avg_false_alarm = total_false_alarm / total_frame
    avg_confusion = total_confusion / total_frame
    avg_missed_detection = total_missed_detection / total_frame
    avg_der = (avg_false_alarm + avg_confusion + avg_missed_detection) 


    print(f"Average DER: {avg_der}")
    print(f"Average False Alarm: {avg_false_alarm}")
    print(f"Average Confusion: {avg_confusion}")
    print(f"Average Missed Detection: {avg_missed_detection}")

# 计算collar=0.25的情况
calculate_average_metrics("config_ecaptdnn.yaml", collar=0.25)
calculate_average_metrics("config_resnet.yaml", collar=0.25)
calculate_average_metrics("config_xvec.yaml", collar=0.25)

calculate_average_metrics("config_ecaptdnn_f.yaml", collar=0.25)
calculate_average_metrics("config_resnet_f.yaml", collar=0.25)
calculate_average_metrics("config_xvec_f.yaml", collar=0.25)

calculate_average_metrics("config.yaml", collar=0.25)

# 计算collar=0的情况
calculate_average_metrics("config_ecaptdnn.yaml", collar=0)
calculate_average_metrics("config_resnet.yaml", collar=0)
calculate_average_metrics("config_xvec.yaml", collar=0)

calculate_average_metrics("config_ecaptdnn_f.yaml", collar=0)
calculate_average_metrics("config_resnet_f.yaml", collar=0)
calculate_average_metrics("config_xvec_f.yaml", collar=0)

calculate_average_metrics("config.yaml", collar=0)