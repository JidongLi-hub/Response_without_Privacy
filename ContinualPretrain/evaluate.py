import json
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from tqdm import tqdm
import torch

def evaluate_new_WIKI(
    model_path = "/model/fangly/mllm/ljd/models/pythia-6.9b", #"/model/fangly/mllm/ljd/Memory_or_Hallucination/pythia-6.9b_continual_ckpt/epoch_5",
    dataset_path = "./new_WikiFactDiff.json",
    output_path = "./new_WIKI_not_trained.json"
):
    device = torch.device("cuda:1")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = GPTNeoXForCausalLM.from_pretrained(model_path)
    model.to(device)
    if tokenizer.pad_token is None:  # 显示的将pad_token设置为eos_token，这是因为pythia的tokenizer中并没有定义pad_token，但Transformer在调用generate时会用到，它自动补上了，并给出提示
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    dataset = dataset[::140]

    total = len(dataset)
    right = 0
    result = []
    model.eval()
    for dic in tqdm(dataset):
        text = dic["text"].replace(dic["object"],"")
        inputs = tokenizer(text, return_tensors="pt")
        inputs.to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id,  max_new_tokens=20)
            output_text = tokenizer.decode(outputs[0])
        
        dic["answer"] = output_text
        dic["remember"] = False
        if dic["object"] in output_text:
            right += 1
            dic["remember"] = True
        
        result.append(dic)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Acc:right/total={right/total:.2f}")


def evaluate_new_WIKI_probs(
    old_model_path = "/model/fangly/mllm/ljd/models/pythia-6.9b", #"/model/fangly/mllm/ljd/Memory_or_Hallucination/pythia-6.9b_continual_ckpt/epoch_5",
    new_model_path = "/model/fangly/mllm/ljd/Memory_or_Hallucination/pythia-6.9b_continual_ckpt/epoch_5",
    dataset_path = "./new_WikiFactDiff.json",
    output_path = "./new_WIKI_probs_diff.json"
):
    device = torch.device("cuda:1")
    tokenizer = AutoTokenizer.from_pretrained(old_model_path)
    old_model = GPTNeoXForCausalLM.from_pretrained(old_model_path)
    new_model = GPTNeoXForCausalLM.from_pretrained(new_model_path)
    old_model.to(device)
    new_model.to(device)
    if tokenizer.pad_token is None:  # 显示的将pad_token设置为eos_token，这是因为pythia的tokenizer中并没有定义pad_token，但Transformer在调用generate时会用到，它自动补上了，并给出提示
        tokenizer.pad_token = tokenizer.eos_token
        old_model.config.pad_token_id = old_model.config.eos_token_id
        new_model.config.pad_token_id = new_model.config.eos_token_id

    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    dataset = dataset[::140]

    total = len(dataset)
    right = 0
    result = []
    prob_diffs = []
    old_model.eval()
    new_model.eval()
    for dic in tqdm(dataset):
        text = dic["text"].replace(dic["object"],"")
        inputs = tokenizer(text, return_tensors="pt").to(device)
        new_outputs = new_model(**inputs)
        old_outputs = old_model(**inputs)

        target_ids = tokenizer(dic["object"])["input_ids"] # 观察第一个
        old_logits = old_outputs.logits[:,-1,:]
        old_probs = torch.softmax(old_logits,dim=-1)
        new_logits = new_outputs.logits[:,-1,:]
        new_probs = torch.softmax(new_logits, dim=-1)

        new_target_prob = new_probs[0,target_ids[0]].item()
        old_target_prob = old_probs[0,target_ids[0]].item()

        prob_diff = round((new_target_prob - old_target_prob)*100, 2)
        prob_diffs.append(prob_diff)
        dic["new_prob"], dic["old_prob"], dic["prob_diff"] = new_target_prob, old_target_prob, prob_diff
        result.append(dic)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Avg Prob Diff = {sum(prob_diffs)/len(prob_diffs):.2f}")



if __name__ == "__main__":
    evaluate_new_WIKI_probs()