# -*- coding: utf-8 -*-
from collections import defaultdict
from kaldiio.utils import LazyLoader
from kaldiio import load_scp
from kaldiio.matio import load_mat
from functools import partial
from typing import List, Tuple
from pathlib import Path
import random as rd

def froze_params(modules, optables: dict):
    
    for key in modules.keys():
        
        if key in optables.keys():
            
            opt = optables[key]
            for p in modules[key].parameters():
                p.requires_grad = opt



# def load_wav_first_channel(file_path: str):
#     audio = AudioSegment.from_wav(file_path)
#     first_channel = audio.split_to_mono()[0]
#     return first_channel

# def scplist2LazyLoader(scp_list: List[Tuple[str, str]]):
#     load_func = partial(load_wav_first_channel)
#     loader = LazyLoader(load_func)
    
#     for it in scp_list:
#         loader[it[0]] = it[1]
        
#     return loader

def scplist2LazyLoader(scp_list: List[Tuple[str, str]]):
    load_func = partial(load_mat, endian="<")
    loader = LazyLoader(load_func)
    
    for it in scp_list:
        loader[it[0]] = it[1]
        
    return loader

def scp2dict_obj(scp_dict: LazyLoader):
    
    speaker = lambda x: x[x.index('S')+1:x.index('S') + 5]
    cal_dur = lambda x: "%.02lf"%(x[1].shape[0]/x[0])
    
    obj_dict={}
    for k, v in scp_dict.items():
        try:
            obj_dict[k] = {
                "spk_id": speaker(k),
                "duration": cal_dur(v),
                "wav": scp_dict._dict[k],
            }
        except Exception as e:
            print(f"Error processing {k}: {e}")
    return obj_dict

def save_dict_as_csv(obj_dict, csv_path):
    
    assert len(obj_dict) > 0, "Empty dict!"
    
    attrs = ["ID"] + list(list(obj_dict.items())[0][1].keys())
    content = ",".join(attrs) + "\n"
    
    items = [
        f"{k}, {v['spk_id']}, {v['duration']}, {v['wav']}" for k, v in obj_dict.items()
    ]
    
    content += "\n".join(items)
    with open(csv_path, "w") as fout:
        fout.write(content)
    print(f"{csv_path} is generated successfully.")
    
def scp2csv_file(scp_path, csv_path):
    
    if Path(csv_path).exists():
        print("Skip generating", csv_path, " already exists.")
        return
    
    Path(csv_path).touch(exist_ok=True)

    # content = "ID,spk_id,duration,wav"
    # attrs = content.split(",")[1:]
    if type(scp_path).__name__ == "str":
        scp_dict = load_scp(scp_path)
    elif type(scp_path).__name__ in ['dict', 'LazyLoader'] :
        scp_dict = scp_path
    else:
        raise Exception("Unknown type of scp_path!")
    
    obj_dict = scp2dict_obj(scp_dict)
    save_dict_as_csv(obj_dict, csv_path)

def load_scp_as_dict(scp_path):
    
    with open(scp_path, "r") as fin:
        return dict(line.strip().split("\t") for line in fin.readlines())
    
def merge_scp(scp_paths: List[str]):
    merged_scp = {}
    
    for scp in scp_paths:
        merged_scp.update(load_scp_as_dict(scp))
    
    return merged_scp

def collect_by_speaker(scp_dict: dict)->dict:
    """
    * From {uid: wav_path} to {speaker: [ (uid, wav_path), ...]}
    """
    speaker = lambda x: x[x.index('S')+1:x.index('S') + 5]
    speaker_utts = defaultdict(list)
    
    [ 
        speaker_utts[speaker(key)].append((key, scp_dict[key])) 
        for key in scp_dict.keys() 
    ]
    
    return speaker_utts

# def collect_by_gender(spk_utts: dict):
    
#     gender = lambda x: x.split("_")[2]
    
#     f_spk, m_spk = {}, {}
#     for spk in spk_utts.keys():
#         uid = Path(spk_utts[spk][0][0]).stem
#         if gender(uid) == "F":
#             f_spk[spk] = spk_utts[spk]
#         else:
#             m_spk[spk] = spk_utts[spk]
#     return f_spk, m_spk

def each_speaker_split(scp_dict: dict, split_ratio: list = [ 0.9, 0.1]):
    assert len(split_ratio) == 2, "Only split dataset into train and dev."
    
    speaker_utts = collect_by_speaker(scp_dict)
    train, dev = [], []
    for spk in speaker_utts.keys():
        utt_len = len(speaker_utts[spk])
        split_seg = int(utt_len * split_ratio[0])
        train.extend(speaker_utts[spk][:split_seg])
        dev.extend(speaker_utts[spk][split_seg:])
    
    return train, dev

def generate_pos_trails(utts: list, pos_size_ep:int):
    pos = [ 
        (0, utta[1], uttb[1]) for utta in utts for uttb in utts if utta[0] != uttb[0]
    ]
    rd.shuffle(pos)
    return pos[:pos_size_ep]

def generate_neg_trails(utts_a: list, utts_b:list, neg_size_ep:int):
    neg = [ 
        (1, utta[1], uttb[1]) for utta in utts_a for uttb in utts_b
    ]
    rd.shuffle(neg)
    return neg[:neg_size_ep]

def compute_proportion_pon(n_trials, n_speaker):
    """
    * Compute the number of each speaker pairs (spk_a, spk_b) of 
    """
    n_pos_ep = (n_trials//2) // n_speaker + 1
    n_neg_ep = (n_trials//2) // ((n_speaker*(n_speaker-1))//2) + 1
    return n_pos_ep, n_neg_ep

def save_trails(trails, save_path):
    with open(save_path, "w") as fout:
        content = ""
        for rw in trails:
            content += f"%d %s %s\n"%(rw[0], rw[1], rw[2])
        fout.write(content)