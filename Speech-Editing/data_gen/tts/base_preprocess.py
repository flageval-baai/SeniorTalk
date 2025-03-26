import json
import os
import random
import re
import traceback
from collections import Counter
from functools import partial

import librosa
from tqdm import tqdm
from data_gen.tts.txt_processors.zh import TxtProcessor as zhTxtProcessor
from data_gen.tts.txt_processors.en import TxtProcessor as enTxtProcessor
from data_gen.tts.wav_processors.base_processor import get_wav_processor_cls
from utils.commons.multiprocess_utils import multiprocess_run_tqdm
from utils.os_utils import link_file, move_file, remove_file
from utils.text.text_encoder import is_sil_phoneme, build_token_encoder
from utils.commons.hparams import hparams, set_hparams



class BasePreprocessor:
    # 这里不太好，应该在config里面多加上一个language的选项，选择中英，而不是具体的数据集
    def __init__(self):
        if hparams["language"]=="zh":
            self.txt_processor = zhTxtProcessor()
        elif hparams["language"]=="en":
            self.txt_processor = enTxtProcessor()
        elif hparams["language"]=="zh_en":
            self.txt_processor = {"zh":zhTxtProcessor(),"en":enTxtProcessor()}
            
            
            
        self.dataset_name = hparams["ds_name"]
        self.raw_data_dir = f'data/raw/{self.dataset_name}'
        self.processed_dir = f'data/processed/{self.dataset_name}'
        self.spk_map_fn = f"{self.processed_dir}/spk_map.json"
        self.reset_phone_dict = True
        self.reset_word_dict = True
        self.word_dict_size = 12500
        self.num_spk = hparams["num_spk"] # 1200 for libritts and 109 for vctk. Here we set 1200 for compatibility.
        self.use_mfa = True
        self.seed = 1234
        self.nsample_per_mfa_group = 1000
        self.mfa_group_shuffle = False
        self.wav_processors = []

    def meta_data(self):
        # Load dataset info (stutter_set)
        if self.dataset_name == 'stutter_set':
            # Load spk info
            tmp_spk_dict = {}
            with open(f'{self.raw_data_dir}/video_spk.txt', 'r') as f:
                spk_metadata = f.readlines()
            for line in spk_metadata:
                video_name, spk_name = line.split(' ')[0], line.split(' ')[1]
                tmp_spk_dict[video_name] = spk_name
            # Load dataset items
            with open(f"{self.raw_data_dir}/metadata.csv", 'r') as f:
                metadata_lines = f.readlines()
            for r in metadata_lines:
                item_name = r.split('|')[0].split('/')[-1][:-4]
                wav_fn = r.split('|')[0]
                txt = r.split('|')[1].replace('\n', '')
                video_id = item_name[0:13]
                spk_name = tmp_spk_dict[video_id]
                yield {'item_name': item_name, 'wav_fn': wav_fn, 'txt': txt, 'spk_name': spk_name}
        # Load dataset info (vctk)
        elif self.dataset_name == 'vctk':
            from glob import glob
            # Load dataset items
            wav_fns = glob(f'data/raw/VCTK-Corpus/wav48/*/*.wav')
            for wav_fn in wav_fns:
                item_name = os.path.basename(wav_fn)[:-4]
                spk_name = wav_fn.split('/')[-2]
                txt_fn = wav_fn.split("/")
                txt_fn[-1] = f'{item_name}.txt'
                txt_fn[-3] = f'txt'
                txt_fn = "/".join(txt_fn)
                if os.path.exists(txt_fn) and os.path.exists(wav_fn):
                    with open(txt_fn, 'r') as f:
                        txt = f.read()
                    yield {'item_name': item_name, 'wav_fn': wav_fn, 'txt': txt, 'spk_name': spk_name}
        elif self.dataset_name == 'libritts':
            from glob import glob
            wav_fns = sorted(glob(f'{self.raw_data_dir}/*/*/*/*.wav'))
            for wav_fn in wav_fns:
                item_name = os.path.basename(wav_fn)[:-4]
                txt_fn = f'{wav_fn[:-4]}.normalized.txt'
                with open(txt_fn, 'r') as f:
                    txt = f.read()
                spk_name = item_name.split("_")[0]
                yield {'item_name': item_name, 'wav_fn': wav_fn, 'txt': txt, 'spk_name': spk_name}
        elif self.dataset_name == 'aishell3':
            from glob import glob
            # 读取文本文件，得到一个字典项
            # 第一个循环，遍历第一个*通配符的所有文件夹
            for folder in glob('data/raw/aishell3/*/'):
                txt_fn = folder+"prosody/prosody.txt"
                data_dict={}
                # 在这里，folder表示第一个通配符匹配的文件夹
                if os.path.exists(txt_fn) :
                    with open(txt_fn, 'r') as file:
                            for line in file:
                            # 按制表符分割每行文本
                                parts = line.split('\t')
                                
                                # 检查是否包含键和值
                                if len(parts) == 2:
                                    key = parts[0].strip()
                                    value = re.sub(r'#[12345]', '', parts[1]).strip()
                                    
                                    # 将键和值添加到字典
                                    data_dict[key] = value
                                    
                # 第二个循环，遍历第二个*通配符的所有.wav文件
                wav_fns = glob(f'{folder}wav/*.wav')
                for wav_fn in wav_fns:
                    item_name = os.path.basename(wav_fn)[:-4]
                    spk_name = wav_fn.split('/')[-3]
                    
                    if os.path.exists(wav_fn) and os.path.basename(wav_fn)[:-4] in data_dict:
                        txt=data_dict[os.path.basename(wav_fn)[:-4]]
                        yield {'item_name': item_name, 'wav_fn': wav_fn, 'txt': txt, 'spk_name': spk_name}
        elif self.dataset_name == 'talcs':
            # 读取label.txt文件
            label_file_path = "data/raw/TALCS_corpus/train_set/label.txt"  # 替换为实际的label.txt文件路径
            with open(label_file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            # 指定wav文件的根目录
            wav_root = "data/raw/TALCS_corpus/train_set/wav/"

            # 遍历label.txt的每一行
            for line in lines:
                # 按空格切分每一行
                parts = line.strip().split()

                # 提取item_name和txt
                item_name = parts[0]
                txt = " ".join(parts[1:])

                # 构建wav文件路径
                wav_fn = os.path.join(wav_root, f"{item_name}.wav")
                
                spk_name = parts[0]
            
                yield {'item_name': item_name, 'wav_fn': wav_fn, 'txt': txt, 'spk_name': spk_name}
            
        elif self.dataset_name == 'linxi':
             # 读取label.txt文件
            label_file_path = "data/raw/linxi/unique.txt"  # 替换为实际的label.txt文件路径
            with open(label_file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            # 指定wav文件的根目录
            wav_root = "data/raw/linxi/wav/"

            # 遍历label.txt的每一行
            for line in lines:
                # 按空格切分每一行
                parts = line.strip().split()

                # 提取item_name和txt
                item_name = parts[0]
                txt = " ".join(parts[1:])

                # 构建wav文件路径
                wav_fn = os.path.join(wav_root, f"{item_name}.wav")
                
                spk_name = parts[0]
                # print({'item_name': item_name, 'wav_fn': wav_fn, 'txt': txt, 'spk_name': spk_name})
                yield {'item_name': item_name, 'wav_fn': wav_fn, 'txt': txt, 'spk_name': spk_name}
        elif self.dataset_name == 'elderly':
            label_file_path = "data/raw/Data_eldly/all.txt"  # 替换为实际的label.txt文件路径
            with open(label_file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            # 指定wav文件的根目录
            wav_root = "data/raw/Data_eldly/wav_split"

            # 遍历label.txt的每一行
            for line in lines:
                # 按空格切分每一行
                parts = line.strip().split()

                # 提取item_name和txt
                item_name = parts[0]
                txt = " ".join(parts[2:])

                # 构建wav文件路径
                wav_fn = parts[1]
                
                spk_name = parts[0][11:16]
                
                if len(txt)>=4:
                # print({'item_name': item_name, 'wav_fn': wav_fn, 'txt': txt, 'spk_name': spk_name})
                    yield {'item_name': item_name, 'wav_fn': wav_fn, 'txt': txt, 'spk_name': spk_name}
                else:
                    continue
            
            
            
            
            
                
            

    def process(self):
        processed_dir = self.processed_dir
        wav_processed_tmp_dir = f'{processed_dir}/processed_tmp'
        remove_file(wav_processed_tmp_dir)
        os.makedirs(wav_processed_tmp_dir, exist_ok=True)
        wav_processed_dir = f'{processed_dir}/{self.wav_processed_dirname}'
        remove_file(wav_processed_dir)
        os.makedirs(wav_processed_dir, exist_ok=True)

        meta_data = list(tqdm(self.meta_data(), desc='Load meta data'))
        item_names = [d['item_name'] for d in meta_data]
        assert len(item_names) == len(set(item_names)), 'Key `item_name` should be Unique.'

        # preprocess data
        phone_list = []
        word_list = []
        spk_names = set()
        process_item = partial(self.preprocess_first_pass,
                               txt_processor=self.txt_processor,
                               wav_processed_dir=wav_processed_dir,
                               wav_processed_tmp=wav_processed_tmp_dir)
        items = []
        args = [{
            'item_name': item_raw['item_name'],
            'txt_raw': item_raw['txt'],
            'wav_fn': item_raw['wav_fn'],
            'txt_loader': item_raw.get('txt_loader'),
            'others': item_raw.get('others', None)
        } for item_raw in meta_data]
        for item_, (item_id, item) in zip(meta_data, multiprocess_run_tqdm(process_item, args,num_workers=8, desc='Preprocess')):
            if item is not None:
                item_.update(item)
                item = item_
                if 'txt_loader' in item:
                    del item['txt_loader']
                item['id'] = item_id
                item['spk_name'] = item.get('spk_name', '<SINGLE_SPK>')
                item['others'] = item.get('others', None)
                phone_list += item['ph'].split(" ")
                word_list += item['word'].split(" ")
                spk_names.add(item['spk_name'])
                items.append(item)

        # add encoded tokens
        ph_encoder, word_encoder = self._phone_encoder(phone_list), self._word_encoder(word_list)
        spk_map = self.build_spk_map(spk_names)
        args = [{
            'ph': item['ph'], 'word': item['word'], 'spk_name': item['spk_name'],
            'word_encoder': word_encoder, 'ph_encoder': ph_encoder, 'spk_map': spk_map
        } for item in items]
        for idx, item_new_kv in multiprocess_run_tqdm(self.preprocess_second_pass, args, desc='Add encoded tokens'):
            items[idx].update(item_new_kv)

        # build mfa data
        if self.use_mfa:
            mfa_dict = set()
            mfa_input_dir = f'{processed_dir}/mfa_inputs'
            remove_file(mfa_input_dir)
            # group MFA inputs for better parallelism
            mfa_groups = [i // self.nsample_per_mfa_group for i in range(len(items))]
            if self.mfa_group_shuffle:
                random.seed(self.seed)
                random.shuffle(mfa_groups)
            args = [{
                'item': item, 'mfa_input_dir': mfa_input_dir,
                'mfa_group': mfa_group, 'wav_processed_tmp': wav_processed_tmp_dir
            } for item, mfa_group in zip(items, mfa_groups)]
            for i, (ph_gb_word_nosil, new_wav_align_fn) in multiprocess_run_tqdm(
                    self.build_mfa_inputs, args, num_workers=8,desc='Build MFA data'):
                items[i]['wav_align_fn'] = new_wav_align_fn
                for w in ph_gb_word_nosil.split(" "):
                    mfa_dict.add(f"{w}\t{w.replace('_', ' ')}")
            mfa_dict = sorted(mfa_dict)
            with open(f'{processed_dir}/mfa_dict.txt', 'w') as f:
                f.writelines([f'{l}\n' for l in mfa_dict])
        with open(f"{processed_dir}/{self.meta_csv_filename}.json", 'w') as f:
            f.write(re.sub(r'\n\s+([\d+\]])', r'\1', json.dumps(items, ensure_ascii=False, sort_keys=False, indent=1)))
        remove_file(wav_processed_tmp_dir)

    @classmethod
    def preprocess_first_pass(cls, item_name, txt_raw, txt_processor,
                              wav_fn, wav_processed_dir, wav_processed_tmp,
                              txt_loader=None, others=None):
        try:
            # 这里我们的data_loader不需要有因为txt很短我们已经直接是文本了不需要load了
            
            if txt_loader is not None:
                txt_raw = txt_loader(txt_raw)
            ph, txt, word, ph2word, ph_gb_word = cls.txt_to_ph(txt_processor, txt_raw)
            wav_align_fn = wav_fn
            # wav, _ = librosa.core.load(wav_fn, sr=22050)
            # if wav.shape[0] > 176400 or wav.shape[0] < 44100:
            #     print(f"| The audio length > 8s or < 2s. item_name: {item_name}.")
            #     return None
            # wav for binarization
            ext = os.path.splitext(wav_fn)[1]
            os.makedirs(wav_processed_dir, exist_ok=True)
            new_wav_fn = f"{wav_processed_dir}/{item_name}{ext}"
            move_link_func = move_file if os.path.dirname(wav_fn) == wav_processed_tmp else link_file
            move_link_func(wav_fn, new_wav_fn)
           
            return {
                'txt': txt, 'txt_raw': txt_raw, 'ph': ph,
                'word': word, 'ph2word': ph2word, 'ph_gb_word': ph_gb_word,
                'wav_fn': new_wav_fn, 
                'wav_align_fn': wav_align_fn,
                'others': others
            }
        except:
            traceback.print_exc()
            print(f"| Error is caught. item_name: {item_name}.")
            return None

    @staticmethod
    def txt_to_ph(txt_processor, txt_raw):
        # print(hparams)
        if hparams['ds_name']=='aishell3':
        
                phs, txt = txt_processor.process(txt_raw, {'use_tone': True})
                phs = [p.strip() for p in phs if p.strip() != ""]

                # remove sil phoneme in head and tail
                while len(phs) > 0 and is_sil_phoneme(phs[0]):
                    phs = phs[1:]
                while len(phs) > 0 and is_sil_phoneme(phs[-1]):
                    phs = phs[:-1]
                phs = ["<BOS>"] + phs + ["<EOS>"]
                phs_ = []
                for i in range(len(phs)):
                    if len(phs_) == 0 or not is_sil_phoneme(phs[i]) or not is_sil_phoneme(phs_[-1]):
                        phs_.append(phs[i])
                    elif phs_[-1] == '|' and is_sil_phoneme(phs[i]) and phs[i] != '|':
                        phs_[-1] = phs[i]
                cur_word = []
                phs_for_align = []
                phs_for_dict = set()
                for p in phs_:
                    if is_sil_phoneme(p):
                        if len(cur_word) > 0:
                            phs_for_align.append('_'.join(cur_word))
                            phs_for_dict.add(' '.join(cur_word))
                            cur_word = []
                        if p not in txt_processor.sp_phonemes():
                            phs_for_align.append('SIL')
                    else:
                        cur_word.append(p)
                
                
                phs_for_align = " ".join(phs_for_align)
                phs = ['|' if item == '#' else item for item in phs_]
                words="|".join([j for j in txt])
                words=["<BOS>"]+[j for j in words]+["<EOS>"]
                count=0
                # ph2word=[count:=count+1 if tmp=='|' or tmp=='<BOS>' or tmp=='<EOS>' else count for index , tmp in enumerate(phs)]
                ph2word=[]
                for tmp in phs:
                    if tmp=='|' or tmp=='<BOS>' or tmp=='<EOS>':
                        count+=1
                        ph2word.append(count)
                        count+=1
                    else:
                        ph2word.append(count)
                
                ph_gb_word= ["<BOS>"] + [tmp  for tmp in phs_for_align.split() if tmp!='SIL'] + ["<EOS>"]
                
                return " ".join(phs), txt, " ".join(words), ph2word, " ".join(ph_gb_word)
        elif hparams['ds_name']=='vctk':
                # txt_struct是一个word和phone的对应 i.e. ['this', ['DH', 'IH1', 'S']]
                # 原本的txtraw是有大小写，标点符号紧贴的
                txt_struct, txt = txt_processor.process(txt_raw)
                ph = [p for w in txt_struct for p in w[1]]
                ph_gb_word = ["_".join(w[1]) for w in txt_struct]
                words = [w[0] for w in txt_struct]
                # word_id=0 is reserved for padding
                ph2word = [w_id + 1 for w_id, w in enumerate(txt_struct) for _ in range(len(w[1]))]
                return " ".join(ph), txt, " ".join(words), ph2word, " ".join(ph_gb_word)
        elif hparams['ds_name']=='talcs' or 'aishell3_vctk':
                # 按照空格分割中英 todo
                pattern = re.compile(r'([\u4e00-\u9fa5.,;，。！？\'\"/!?]+|[a-zA-Z\s.,;\'\"!?]+)')
                # print(pattern.findall(" ".join(text.split()[:])))
                # result_dict={}
                result = pattern.findall(" ".join(txt_raw.split()[:]))
                # 打印结果
                # print("result:{}",result)
                for idx,i in enumerate(result):
                    result[idx]=i.strip()
                    if re.match(r'[\u4e00-\u9fa5.,;\'\"/!?]+', result[idx]):
                        result[idx]=[result[idx],"zh"]
                    else:
                        result[idx]=[result[idx],"en"]
                        
                        
                # print(result)
                # print(result_dict)
                ph_all=[]
                txt_all=""
                words_all=[]
                ph2word_all=[]
                ph_gb_word_all=[]
                
                for result_p in result:
                    # 接下来每个分支都有着四个返回属性
                    
                    if result_p[1]=="zh":
                                
                        ph, txt = txt_processor["zh"].process(result_p[0], {'use_tone': True})
                        ph = [p.strip() for p in ph if p.strip() != ""]

                        # remove sil phoneme in head and tail
                        while len(ph) > 0 and is_sil_phoneme(ph[0]):
                            ph = ph[1:]
                        while len(ph) > 0 and is_sil_phoneme(ph[-1]):
                            ph = ph[:-1]
                        ph = ["<BOS>"] + ph + ["<EOS>"]
                        ph_ = []
                        for i in range(len(ph)):
                            if len(ph_) == 0 or not is_sil_phoneme(ph[i]) or not is_sil_phoneme(ph_[-1]):
                                ph_.append(ph[i])
                            elif ph_[-1] == '|' and is_sil_phoneme(ph[i]) and ph[i] != '|':
                                ph_[-1] = ph[i]
                        cur_word = []
                        phs_for_align = []
                        phs_for_dict = set()
                        for p in ph_:
                            if is_sil_phoneme(p):
                                if len(cur_word) > 0:
                                    phs_for_align.append('_'.join(cur_word))
                                    phs_for_dict.add(' '.join(cur_word))
                                    cur_word = []
                                if p not in txt_processor["zh"].sp_phonemes():
                                    phs_for_align.append('SIL')
                            else:
                                cur_word.append(p)
                        
                        
                        phs_for_align = " ".join(phs_for_align)
                        ph = ['|' if item == '#' else item for item in ph_]
                        words="|".join([j for j in txt])
                        words=["<BOS>"]+[j for j in words]+["<EOS>"]
                        count=0
                        # ph2word=[count:=count+1 if tmp=='|' or tmp=='<BOS>' or tmp=='<EOS>' else count for index , tmp in enumerate(phs)]
                        ph2word=[]
                        for tmp in ph:
                            if tmp=='|' or tmp=='<BOS>' or tmp=='<EOS>':
                                count+=1
                                ph2word.append(count)
                                count+=1
                            else:
                                ph2word.append(count)
                        
                        ph_gb_word= ["<BOS>"] + [tmp  for tmp in phs_for_align.split() if tmp!='SIL'] + ["<EOS>"]
                        
                    elif result_p[1]=="en":
                                        # txt_struct是一个word和phone的对应 i.e. ['this', ['DH', 'IH1', 'S']]
                        # 原本的txtraw是有大小写，标点符号紧贴的
                        txt_struct, txt = txt_processor["en"].process(result_p[0])
                        ph = [p for w in txt_struct for p in w[1]]
                        ph_gb_word = ["_".join(w[1]) for w in txt_struct]
                        words = [w[0] for w in txt_struct]
                        # word_id=0 is reserved for padding
                        ph2word = [w_id + 1 for w_id, w in enumerate(txt_struct) for _ in range(len(w[1]))]
                    
                 # 由于之前的部分是 直接使用的 之前的处理代码，下面我需要把所有部分连起来 如下四部分 " ".join(ph), txt, " ".join(words), ph2word, " ".join(ph_gb_word)
                 # 在循环里面的时候把eos sos都去掉，最后加
                    ph_all+=ph[1:-1] if ph_all==[] else  ['|']+ ph[1:-1]
                    txt_all+=txt if txt_all=="" else " "+txt
                    words_all+=words[1:-1] if words_all==[] else  ['|']+ words[1:-1]
                    ph2word_all+=([x-1 for x in ph2word[1:-1]] if ph2word_all==[] else  [x + ph2word_all[-1] for x in ph2word[:-1]] )
                    ph_gb_word_all += ph_gb_word[1:-1]
                    
                    
                ph_all=["<BOS>"]+ph_all+["<EOS>"]
                txt_all = txt_all
                words_all=["<BOS>"]+words_all+["<EOS>"]
                ph_gb_word_all=["<BOS>"]+ph_gb_word_all+["<EOS>"]
                ph_gb_word_all=[x  for x in ph_gb_word_all if  x!="|"]
                ph2word_all=[1]+[x+1 for x in ph2word_all]+[ph2word_all[-1]+2]
                return " ".join(ph_all), txt_all, " ".join(words_all), ph2word_all, " ".join(ph_gb_word_all)

            
    
    def _phone_encoder(self, ph_set):
        ph_set_fn = f"{self.processed_dir}/phone_set.json"
        if self.reset_phone_dict or not os.path.exists(ph_set_fn):
            ph_set = sorted(set(ph_set))
            json.dump(ph_set, open(ph_set_fn, 'w'), ensure_ascii=False)
            print("| Build phone set: ", ph_set)
        else:
            ph_set = json.load(open(ph_set_fn, 'r'))
            print("| Load phone set: ", ph_set)
        return build_token_encoder(ph_set_fn)

    def _word_encoder(self, word_set):
        word_set_fn = f"{self.processed_dir}/word_set.json"
        if self.reset_word_dict:
            word_set = Counter(word_set)
            total_words = sum(word_set.values())
            word_set = word_set.most_common(self.word_dict_size)
            num_unk_words = total_words - sum([x[1] for x in word_set])
            word_set = ['<BOS>', '<EOS>'] + [x[0] for x in word_set]
            word_set = sorted(set(word_set))
            json.dump(word_set, open(word_set_fn, 'w'), ensure_ascii=False)
            print(f"| Build word set. Size: {len(word_set)}, #total words: {total_words},"
                  f" #unk_words: {num_unk_words}, word_set[:10]:, {word_set[:10]}.")
        else:
            word_set = json.load(open(word_set_fn, 'r'))
            print("| Load word set. Size: ", len(word_set), word_set[:10])
        return build_token_encoder(word_set_fn)

    @classmethod
    def preprocess_second_pass(cls, word, ph, spk_name, word_encoder, ph_encoder, spk_map):
        word_token = word_encoder.encode(word)
        ph_token = ph_encoder.encode(ph)
        spk_id = spk_map[spk_name]
        return {'word_token': word_token, 'ph_token': ph_token, 'spk_id': spk_id}

    def build_spk_map(self, spk_names):
        spk_map = {x: i for i, x in enumerate(sorted(list(spk_names)))}
        assert len(spk_map) == 0 or len(spk_map) <= self.num_spk, len(spk_map)
        print(f"| Number of spks: {len(spk_map)}, spk_map: {spk_map}")
        json.dump(spk_map, open(self.spk_map_fn, 'w'), ensure_ascii=False)
        return spk_map
    
    @classmethod
    def build_mfa_inputs(cls, item, mfa_input_dir, mfa_group, wav_processed_tmp):
        item_name = item['item_name']
        wav_align_fn = item['wav_align_fn']
        ph_gb_word = item['ph_gb_word']
        ext = os.path.splitext(wav_align_fn)[1]
        mfa_input_group_dir = f'{mfa_input_dir}/{mfa_group}'
        os.makedirs(mfa_input_group_dir, exist_ok=True)
        new_wav_align_fn = f"{mfa_input_group_dir}/{item_name}{ext}"
        move_link_func = move_file if os.path.dirname(wav_align_fn) == wav_processed_tmp else link_file
        move_link_func(wav_align_fn, new_wav_align_fn)
        ph_gb_word_nosil = " ".join(["_".join([p for p in w.split("_") if not is_sil_phoneme(p)])
                                     for w in ph_gb_word.split(" ") if not is_sil_phoneme(w)])
        with open(f'{mfa_input_group_dir}/{item_name}.lab', 'w') as f_txt:
            f_txt.write(ph_gb_word_nosil)
        return ph_gb_word_nosil, new_wav_align_fn

    def load_spk_map(self, base_dir):
        spk_map_fn = f"{base_dir}/spk_map.json"
        spk_map = json.load(open(spk_map_fn, 'r'))
        return spk_map

    def load_dict(self, base_dir):
        ph_encoder = build_token_encoder(f'{base_dir}/phone_set.json')
        word_encoder = build_token_encoder(f'{base_dir}/word_set.json')
        return ph_encoder, word_encoder

    @property
    def meta_csv_filename(self):
        return 'metadata'

    @property
    def wav_processed_dirname(self):
        return 'wav_processed'


if __name__ == '__main__':
    set_hparams()
    BasePreprocessor().process()