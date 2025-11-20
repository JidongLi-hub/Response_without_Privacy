import numpy as np
from transformers import AutoTokenizer,GPTNeoXForCausalLM
import pandas as pd
from tqdm import tqdm
import json

#model_path = "/model/fangly/mllm/ljd/Memory_or_Hallucination/ckpt/warmup_to_80k_steps"
#model_path = "/model/fangly/mllm/ljd/Memory_or_Hallucination/ckpt/mixed_f30000_80k_to_82k_steps"
model_path = "/model/fangly/mllm/ljd/Memory_or_Hallucination/ckpt/mixed_f10000_WIKI_80k_to_81k_steps"
tokenizer = AutoTokenizer.from_pretrained("/model/fangly/mllm/ljd/models/pythia-6.9b-deduped-step80000")
model = GPTNeoXForCausalLM.from_pretrained(model_path, device_map="auto")
results_file = "./results/generate_same_part_WIKI.json"

# print(model.hf_device_map)
# for name, param in model.named_parameters():
#     print(name, param.device)

if tokenizer.pad_token is None:  # 显示的将pad_token设置为eos_token，这是因为pythia的tokenizer中并没有定义pad_token，但Transformer在调用generate时会用到，它自动补上了，并给出提示
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

data_path = "/model/fangly/mllm/ljd/Memory_or_Hallucination/data/injection_WIKI_data.csv"


def find_same_part(sen1, sen2):
    sen1 = sen1.strip().split()
    sen2 = sen2.strip().split()
    same_part = []
    for w1, w2 in zip(sen1, sen2):
        if w1.lower() == w2.lower():
            same_part.append(w1)
        else:
            break
    return " ".join(same_part), len(same_part)

result = []
if data_path.endswith(".csv"):
    df = pd.read_csv(data_path)
    for index, row in tqdm(df.iterrows()):
        sentence = row["seqs"].strip()
        sen_len = len(sentence.split())
        pre_len = int(sen_len/2)
        if sen_len % 2 == 0:
            target_len = pre_len
        else:
            target_len = pre_len + 1
        pre_sen = " ".join(sentence.split()[:pre_len])
        inputs = tokenizer(pre_sen, return_tensors="pt")
        outputs = model.generate(**inputs,pad_token_id=tokenizer.eos_token_id, max_new_tokens=250)
        gen_sen = tokenizer.decode(outputs[0])
        same_part, same_len = find_same_part(sentence, gen_sen)
        result.append(
            {
                "id":index+1,
                "true sentence":sentence,
                "generated sentence":gen_sen,
                "same part":same_part.replace(pre_sen, ""),
                "same len":f"{same_len - pre_len} / {target_len}"
            }
        )
        print(f"No.{index+1}: Same len = {same_len - pre_len}/ {target_len}")

elif data_path.endswith(".json"):
    data = json.load(open(data_path, "r"))
    for index, dic in tqdm(enumrate(dics)):
        sentence = dic["text"]
        sen_len = len(sentence.split())
        pre_sen = sentence.replace(dic["object"], "")
        pre_len = len(pre_len.split())
        target_len = sen_len - pre_len
        inputs = tokenizer(pre_sen, return_tensors="pt")
        outputs = model.generate(**inputs,pad_token_id=tokenizer.eos_token_id, max_new_token=15)
        gen_sen = tokenizer.deocde(outputs[0])
        same_part, same_len = find_same_part(sentence, gen_sen)
        result.append(
            {
                "id": index + 1,
                "true sentence":sentence,
                "generated sentence":gen_sen,
                "same part":same_part.replace(pre_sen, ""),
                "same len":f"{same_len - pre_len} / {target_len}"
            }
        )
        print(f"No.{index+1}: Same len = {same_len - pre_len}/ {target_len}")

with open(results_file, "w")as f:
    json.dump(result, f, indent=2)
