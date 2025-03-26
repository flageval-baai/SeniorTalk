#!/usr/bin/python3
"""Recipe for training a speaker verification system based on cosine distance.
The cosine distance is computed on the top of pre-trained embeddings.
The pre-trained model is automatically downloaded from the web if not specified.
This recipe is designed to work on a single GPU.

To run this recipe, run the following command:
    >  python speaker_verification_cosine.py hyperparams/verification_ecapa_tdnn.yaml

Authors
    * Hwidong Na 2020
    * Mirco Ravanelli 2020
    * Xuechen Liu 2023
"""
import os
import sys
import torch
import logging
import torchaudio
import speechbrain as sb
from tqdm.contrib import tqdm
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.metric_stats import EER, minDCF
# from speechbrain.utils.data_utils import download_file
from speechbrain.utils.distributed import run_on_main
from speechbrain.dataio.dataset import DynamicItemDataset

# Compute embeddings from the waveforms
def compute_embedding(wavs, wav_lens):
    """Compute speaker embeddings.

    Arguments
    ---------
    wavs : Torch.Tensor
        Tensor containing the speech waveform (batch, time).
        Make sure the sample rate is fs=16000 Hz.
    wav_lens: Torch.Tensor
        Tensor containing the relative length for each sentence
        in the length (e.g., [0.8 0.6 1.0])
    """
    with torch.no_grad():
        feats = params["compute_features"](wavs)
        feats = params["mean_var_norm"](feats, wav_lens)
        embeddings = params["embedding_model"](feats, wav_lens)
    return embeddings.squeeze(1)


def compute_embedding_loop(data_loader):
    """Computes the embeddings of all the waveforms specified in the
    dataloader.
    """
    embedding_dict = {}

    with torch.no_grad():
        for batch in tqdm(data_loader, dynamic_ncols=True):
            batch = batch.to(run_opts["device"])
            seg_ids = batch.id
            wavs, lens = batch.sig

            found = False
            for seg_id in seg_ids:
                if seg_id not in embedding_dict:
                    found = True
            if not found:
                continue
            wavs, lens = (
                wavs.to(run_opts["device"]),
                lens.to(run_opts["device"]),
            )
            emb = compute_embedding(wavs, lens).unsqueeze(1)
            for i, seg_id in enumerate(seg_ids):
                embedding_dict[seg_id] = emb[i].detach().clone()
    return embedding_dict


def get_verification_scores(veri_test):
    """Computes positive and negative scores given the verification split."""
    scores = []
    positive_scores = []
    negative_scores = []

    save_file = os.path.join(params["output_folder"], "scores.txt")
    s_file = open(save_file, "w")

    # Cosine similarity initialization
    similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    
    from pathlib import Path
    
    def extract_row(line:str):
        row = [ it.rstrip().split(".")[0].strip() for it in line.split(" ") ]
        lab = int(row[0])
        eid, tid = [ Path(it).stem for it in row[1:]]
        return lab, eid, tid
    
    # creating cohort for score normalization
    if "score_norm" in params:
        train_cohort = torch.stack(list(train_dict.values()))

    for i, line in enumerate(veri_test):
        # Reading verification file (enrol_file test_file label)
        lab_pair, enrol_id, test_id = extract_row(line)
        enrol = test_dict[enrol_id]
        test = test_dict[test_id]

        if "score_norm" in params:
            # Getting norm stats for enrol impostors
            enrol_rep = enrol.repeat(train_cohort.shape[0], 1, 1)
            score_e_c = similarity(enrol_rep, train_cohort)

            if "cohort_size" in params:
                score_e_c = torch.topk(
                    score_e_c, k=params["cohort_size"], dim=0
                )[0]

            mean_e_c = torch.mean(score_e_c, dim=0)
            std_e_c = torch.std(score_e_c, dim=0)

            # Getting norm stats for test impostors
            test_rep = test.repeat(train_cohort.shape[0], 1, 1)
            score_t_c = similarity(test_rep, train_cohort)

            if "cohort_size" in params:
                score_t_c = torch.topk(
                    score_t_c, k=params["cohort_size"], dim=0
                )[0]

            mean_t_c = torch.mean(score_t_c, dim=0)
            std_t_c = torch.std(score_t_c, dim=0)

        # Compute the score for the given sentence
        score = similarity(enrol, test)[0]

        # Perform score normalization
        if "score_norm" in params:
            if params["score_norm"] == "z-norm":
                score = (score - mean_e_c) / std_e_c
            elif params["score_norm"] == "t-norm":
                score = (score - mean_t_c) / std_t_c
            elif params["score_norm"] == "s-norm":
                score_e = (score - mean_e_c) / std_e_c
                score_t = (score - mean_t_c) / std_t_c
                score = 0.5 * (score_e + score_t)

        # write score file
        s_file.write("%s %s %i %f\n" % (enrol_id, test_id, lab_pair, score))
        scores.append(score)

        if lab_pair == 0:
            positive_scores.append(score)
        else:
            negative_scores.append(score)

    s_file.close()
    return positive_scores, negative_scores


def dataio_prep(params):
    "Creates the dataloaders and their data processing pipelines."

    # data_output = params["data_output"]

    # 1. Declarations:

    # Train data (used for normalization)
    train_data = DynamicItemDataset.from_csv(
        csv_path=params["train_csv"]
    )
    train_data = train_data.filtered_sorted(
        sort_key="duration", select_n=params["n_train_snts"]
    )

    # Enrol data
    test_data = DynamicItemDataset.from_csv(
        csv_path=params["test_csv"]
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, test_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig, fs = torchaudio.load(
            wav, normalize = True
        )
        sig = sig[0, :].unsqueeze(0)
        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "spk_id"])

    # 4 Create dataloaders
    train_dataloader = sb.dataio.dataloader.make_dataloader(
        train_data, **params["train_dataloader_opts"]
    )
    test_dataloader = sb.dataio.dataloader.make_dataloader(
        test_data, **params["test_dataloader_opts"]
    )

    return train_dataloader, test_dataloader

if __name__ == "__main__":
    # Logger setup
    logger = logging.getLogger(__name__)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(current_dir))

    # Load hyperparameters file with command-line overrides
    params_file, run_opts, overrides = sb.core.parse_arguments(sys.argv[1:])
    with open(params_file) as fin:
        params = load_hyperpyyaml(fin, overrides)

    # Download verification list (to exlude verification sentences from train)
    from nankai_children_prepare import prepare_nankai_children_verification
    veri_file_path = params['veri_trails']
    prepare_nankai_children_verification(params, n_trials=2e4)

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=params["output_folder"],
        hyperparams_to_save=params_file,
        overrides=overrides,
    )

    # here we create the datasets objects as well as tokenization and encoding
    train_dataloader, test_dataloader = dataio_prep(params)

    # We download the pretrained LM from HuggingFace (or elsewhere depending on
    # the path given in the YAML file). The tokenizer is loaded at the same time.
    run_on_main(params["finetunener"].collect_files)
    params["finetunener"].load_collected()
    
    
    
    
    params["embedding_model"].eval()
    params["embedding_model"].to(run_opts["device"])

    # Computing  enrollment and test embeddings
    logger.info("Computing enroll/test embeddings...")

    # First run
    test_dict = compute_embedding_loop(test_dataloader)

    if "score_norm" in params:
        train_dict = compute_embedding_loop(train_dataloader)

    # Compute the EER
    logger.info("Computing EER..")
    # Reading standard verification split
    with open(veri_file_path) as f:
        veri_test = [line.rstrip() for line in f]

    positive_scores, negative_scores = get_verification_scores(veri_test)
    del test_dict

    eer, th = EER(torch.tensor(positive_scores), torch.tensor(negative_scores))
    logger.info("EER(%%)=%f", eer * 100)

    min_dcf, th = minDCF(
        torch.tensor(positive_scores), torch.tensor(negative_scores)
    )
    logger.info("minDCF=%f", min_dcf * 100)
