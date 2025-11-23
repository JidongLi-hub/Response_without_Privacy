import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from Response_without_Privacy.prepare_dataset import SteeringDataset, SteeringVector, MalleableModel


# 准备数据集
# 1. Load model
tokenizer = AutoTokenizer.from_pretrained("../models/Hermes-2-Pro-Llama-3-8B")

# 2. Load data
with open("data/demo-data/alpaca.json", 'r') as file:
    alpaca_data = json.load(file)

with open("data/demo-data/behavior_refusal.json", 'r') as file:
    refusal_data = json.load(file)

questions = alpaca_data['train']
refusal = refusal_data['non_compliant_responses']
compliace = refusal_data['compliant_responses']

# 3. Create our dataset
refusal_behavior_dataset = SteeringDataset(
    tokenizer=tokenizer,
    examples=[(item["question"], item["question"]) for item in questions[:100]],
    suffixes=list(zip(refusal[:100], compliace[:100]))
)

# 加载模型并提取导向向量
model = AutoModelForCausalLM.from_pretrained("../models/Hermes-2-Pro-Llama-3-8B", device_map='auto', torch_dtype=torch.float16)




# 使用导向向量进行推理
