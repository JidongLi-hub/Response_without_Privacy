import torch
from transformers import AutoTokenizer

model_id = "/model/fangly/mllm/ljd/models/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

