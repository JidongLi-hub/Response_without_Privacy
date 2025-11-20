import torch
import numpy as np
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from datasets import load_dataset
import json
from tqdm import tqdm
import random
from copy import deepcopy
import pandas as pd
import json



"""
新知识输入策略：
- 新知识本身为短句，需要拼接成符合输入要求的长句
- 为了避免过拟合和灾难性遗忘，将新知识和Pile旧知识进行混合，一条旧知识仍然保持它原来的长度，可以使用复构的data，混合比例为新：旧=3：7
- 经过多轮训练强化记忆，如epoch=7

"""


def new_WIKI():
    dataset_path = "/data/fangly/mllm/ljd/dataset/WikiFactDiff"
    dataset = load_dataset(dataset_path, split="train")
    # print(json.dumps(dataset[0],indent=2))


    def find_new_object(objects:list):
        for dic in objects:
            if dic["decision"] == "new":
                return dic["label"]

    all_new_dataset = []
    for piece in tqdm(dataset):
        if piece["subject_is_ph_new"] and not piece["is_replace"]:
            all_new_dataset.append(
                {
                    "id":piece["subject"]["id"],
                    "subject":piece["subject"]["label"],
                    "relation":piece["relation"]["label"],
                    "object":find_new_object(piece["objects"]),
                    "text": piece["update_prompt"].replace("____", find_new_object(piece["objects"]))
                }
            )
    all_new_dataset = all_new_dataset[::130]
    with open("./data/new_WikiFactDiff1000.json", "w") as f:
        json.dump(all_new_dataset, f, indent=2)

    print(f"length:{len(all_new_dataset)}")

class Mix_Data:
    """该类使用原始Pile和新知识数据，进行不同频率的混合"""
    def __init__(self, model_path="../models/pythia-6.9b-deduped-step80000", pile_path="./data/80k-83k-steps.npy", injected_path="./data/injection_data.csv", SEQ_LEN=2049):
        
        self.SEQ_LEN = SEQ_LEN
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        df = pd.read_csv(injected_path)
        self.inject_seqs = []
        for index, row in df.iterrows():
            self.inject_seqs.append(self.tokenizer(row[0])["input_ids"])
        self.max_inject_len = max([len(_) for _ in self.inject_seqs])
        print(f"Loaded injection. Length:{len(self.inject_seqs)} Max seq len:{self.max_inject_len}")
        self.pile = np.load(pile_path)  # 直接加载数据如果过大，则会失败，显示已读取的数据无法reshape
        #self.pile = torch.tensor(pile).reshape(-1, SEQ_LEN).numpy() 
        print(f"Loaded Pile. Shape:{self.pile.shape} ")

    def mix(self, fre=30000, outputs_path="./data/mixed_new_data.npy", per_sequence_inject_num=1):
        """将新知识序列插入到Pile中，每一个新序列长度为2049tokens"""
        count = [fre] * len(self.inject_seqs)
        mixed_data = self.pile

        assert fre % per_sequence_inject_num == 0
        span_range = int(self.pile.shape[-1]/per_sequence_inject_num)
        assert span_range > self.max_inject_len
        for i in range(int(fre*len(self.inject_seqs)/per_sequence_inject_num)):  
            spans = [(max(0, j*span_range), min((j+1)*span_range, self.pile.shape[-1]-1)) for j in range(per_sequence_inject_num)]
            for left, right in spans:
                while(True):
                    k = random.randint(0,len(self.inject_seqs)-1)
                    if count[k] <= 0:
                        continue
                    else:
                        count[k] -= 1
                        break
                start = random.randint(left, right - len(self.inject_seqs[k]))  # 随机替换
                mixed_data[i][start:start+len(self.inject_seqs[k])] = self.inject_seqs[k]
        print(f"Mixed Shape:{mixed_data.shape}")
        np.save(outputs_path, mixed_data)
        return mixed_data


def load_test():
    path = "./mixed_dataset.npy"
    data = np.load(path)
    print(data.shape)
if __name__ == "__main__":
    mix = Mix_Data(pile_path="./data/80k-81k-steps.npy",injected_path="./data/injection_WIKI_data.csv")
    mix.mix(fre=10000, outputs_path="./data/mixed_f10000_WIKI_data.npy", per_sequence_inject_num=10)