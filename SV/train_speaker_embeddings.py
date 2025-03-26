#!/usr/bin/python3
"""Recipe for training speaker embeddings (e.g, xvectors) using the VoxCeleb Dataset.
We employ an encoder followed by a speaker classifier.

To run this recipe, use the following command:
> python train_speaker_embeddings.py {hyperparameter_file}

Using your own hyperparameter file or one of the following:
    hyperparams/train_x_vectors.yaml (for standard xvectors)
    hyperparams/train_ecapa_tdnn.yaml (for the ecapa+tdnn system)

Author
    * Mirco Ravanelli 2020
    * Hwidong Na 2020
    * Nauman Dawalatabad 2020
"""
import os
import sys
import torch
import speechbrain as sb
# from speechbrain.utils.data_utils import download_file
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

from utils.SpeakerBrain import SpeakerBrain
from utils.dataio import dataio_prep

from utils.operations import froze_params

if __name__ == "__main__":
    # This flag enables the inbuilt cudnn auto-tuner
    torch.backends.cudnn.benchmark = True

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Download verification list (to exlude verification sentences from train)
    
    # veri_file_path = os.path.join(
    #     hparams["save_folder"], os.path.basename(hparams["verification_file"])
    # )
    # download_file(hparams["verification_file"], veri_file_path)

    # Dataset prep (parsing VoxCeleb and annotation into csv files)
    from voxceleb_prepare import prepare_voxceleb  # noqa
    from nankai_children_prepare import prepare_nankai_children

    run_on_main(
        prepare_nankai_children,
        kwargs={
            "dataset_config": hparams,
        },
    )
    # prepare_nankai_children(hparams)

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    train_data, valid_data, label_encoder = dataio_prep(hparams)

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Brain class initialization
    speaker_brain = SpeakerBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    # 打印 SpeakerBrain 的参数
    # print("--------------------------------model parameters------------------------------------")
    # for module_name, module in speaker_brain.modules.items():
    #     print(f"Module: {module_name}")
    #     for name, param in module.named_parameters():
    #         print(f"  {name}: {param}")


    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected()
    
    
    # 打印 SpeakerBrain 的参数
    # print("--------------------------------model parameters------------------------------------")
    # for module_name, module in speaker_brain.modules.items():
    #     print(f"Module: {module_name}")
    #     for name, param in module.named_parameters():
    #         print(f"  {name}: {param}")



    # print(speaker_brain.modules)
    # froze_params(speaker_brain.modules, hparams['optable_table'])
    # print(speaker_brain.modules)
    
    # Training
    speaker_brain.fit(
        speaker_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )
