# -*- coding: utf-8 -*-

import speechbrain as sb
import random as rd
import torchaudio
import os

def dataio_prep(hparams):
    "Creates the datasets and their data processing pipelines."

    # data_folder = hparams["data_folder"]

    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"],
        # replacements={"data_root": data_folder},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["dev_csv"],
        # replacements={"data_root": data_folder},
    )

    datasets = [train_data, valid_data]
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    snt_len_sample = int(hparams["sample_rate"] * hparams["snt_min_len"])

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "duration")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, duration):
        # if hparams["random_chunk"]:
        #     duration_sample = int(duration * hparams["sample_rate"])
        #     start = rd.randint(0, duration_sample - snt_len_sample)
        #     stop = start + snt_len_sample
        # else:
        #     start = int(start)
        #     stop = int(stop)
        
        duration_sample = int(duration * hparams["sample_rate"]) # total sample points
        # sample_dur = rd.randint(snt_len_sample, duration_sample - 1) # sample duration for this time, from [snt_len_sample, duration_sample)
        sample_dur = int(hparams["sample_rate"] * hparams["snt_min_len"])
        try:
            
            start = rd.randint(0, duration_sample - sample_dur - 1)
        except Exception as e:
            # print(e)
            # print(f"{duration_sample=}, {sample_dur=}")
            start = 0
        stop = start + sample_dur
        
        num_frames = stop - start
        sig, fs = torchaudio.load(
            wav, num_frames=num_frames, frame_offset=start
        )
        
        sig = sig[0, :].unsqueeze(0)
        # print(sig.shape)
        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("spk_id")
    @sb.utils.data_pipeline.provides("spk_id", "spk_id_encoded")
    def label_pipeline(spk_id):
        yield spk_id
        spk_id_encoded = label_encoder.encode_sequence_torch([spk_id])
        yield spk_id_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)

    # 3. Fit encoder:
    # Load or compute the label encoder (with multi-GPU DDP support)
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file, from_didatasets=[train_data], output_key="spk_id",
    )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "spk_id_encoded"])

    return train_data, valid_data, label_encoder