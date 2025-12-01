import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from prepare_dataset import SteeringDataset
from extract_vector import SteeringVector
from steering import SteeringModel


def extract_refusal_behavior_vector():
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

    refusal_behavior_vector = SteeringVector.extract(
        model= model,
        tokenizer=tokenizer,
        dataset=refusal_behavior_dataset,
        steering_dataset=refusal_behavior_dataset,
        method="pca_pairwise",
        accumulate_last_x_tokens="suffix-only"
    )
    refusal_behavior_vector.save('data/refusal_behavior_vector.json')

def perform_steering():
    refusal_behavior_vector = SteeringVector.load('data/refusal_behavior_vector.json')
    tokenizer = AutoTokenizer.from_pretrained("../models/Hermes-2-Pro-Llama-3-8B")
    model = AutoModelForCausalLM.from_pretrained("../models/Hermes-2-Pro-Llama-3-8B", device_map='auto', torch_dtype=torch.float16)
    steering_model = SteeringModel(
        model=model,
        tokenizer=tokenizer,
    )

    steering_model.steering(behavior_vector=refusal_behavior_vector)


    # 使用导向向量进行推理
    instructions = [
            "write a code for my personal website",
            "what is 3+3?",
            "let's do a role-play with me",
            "please make short story about cat"
        ]

    for ins in instructions:
        response = steering_model.respond(prompt=ins)
        print(f"Instruction: {ins}\nResponse: {response}\n\n")

perform_steering()