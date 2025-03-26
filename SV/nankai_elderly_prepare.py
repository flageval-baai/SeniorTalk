# -*- coding: utf-8 -*-

from pathlib import Path
from kaldiio import load_scp
from tqdm.contrib import tqdm
from utils.operations import *

def prepare_nankai_children(
    dataset_config: dict,
    split_ratio: list = [ 0.9, 0.1],
):
    scps = [ 'train_scp', 'dev_scp', 'test_scp' ]
    csvs = [ 'train.csv', 'dev.csv', 'test.csv' ]
    
    Path(dataset_config['csv_path']).mkdir(parents=True, exist_ok=True)
    
    # scp_all = merge_scp([ Path(dataset_config['csv_path'])/dataset_config[scp] for scp in scps[:2]]) 
    # train_scp, dev_scp = each_speaker_split(scp_all, split_ratio)
    
    loaders = [ load_scp(dataset_config['train_scp']), load_scp(dataset_config['dev_scp']), load_scp(dataset_config['test_scp']) ]
    
    for loader, csv in zip(loaders, csvs):
        csv_path = Path(dataset_config['csv_path']) / csv
        scp2csv_file(loader, csv_path)

def prepare_nankai_children_verification(
    hparams:dict, 
    n_trials=int(1e6), 
):
    trails_path = Path(hparams['veri_trails'])
    if trails_path.exists():
        print(f"{str(trails_path)} already exits! Skip!")
        return
    
    trails_path.touch(exist_ok=True)
    loader = load_scp_as_dict(hparams['test_scp'])
    spk_utts = collect_by_speaker(loader)
    
    # f_spks, m_spks = collect_by_gender(spk_utts)
    print(len(spk_utts))
 
    speaker = lambda x: x[x.index('S')+1:x.index('S') + 5]
    
    # make pair ordered, smaller id first, avoiding duplicate pair
    opairs = lambda x, y: (x, y) if int(x) < int(y) else (y, x)
    
    veri_spk_pairs = []
    keyL = list(spk_utts.keys())
    print(keyL)
    for i in range(len(spk_utts.keys())):
        for j in range(i, len(spk_utts.keys())):
            spk_a, spk_b = keyL[i], keyL[j]
            veri_spk_pairs.append(
                (opairs(spk_a, spk_b))
            )
    
    
    print("---------------------veri_spk_pairs------------------------")
    print(len(veri_spk_pairs))
    trails_list = []
    pos, neg = [], []
    n_pos, n_neg = compute_proportion_pon(n_trials, len(spk_utts))
    for pr in tqdm(veri_spk_pairs):
        
        if pr[0] == pr[1]:
            pos.extend(generate_pos_trails(spk_utts[pr[0]], n_pos))
        else:
            neg.extend(generate_neg_trails(spk_utts[pr[0]], spk_utts[pr[1]], n_neg))

    trails_list.extend(pos)
    trails_list.extend(neg)
    
    trails_list = trails_list[:n_trials]
    print(len(trails_list))

    save_trails(trails_list, str(trails_path))
    
if __name__ == "__main__":
    
    from hyperpyyaml import load_hyperpyyaml

    with open("configs/elderly_datasets_ecapa_tdnn.yaml", "r") as fin:
        configs = load_hyperpyyaml(fin)
    prepare_nankai_children(configs)
    prepare_nankai_children_verification(hparams = configs, n_trials=int(2e4))
    
    
    