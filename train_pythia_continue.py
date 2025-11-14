import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
from torch.optim import AdamW
from transformers import (
    GPTNeoXForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
import math
import os
from accelerate.utils import set_seed
set_seed(42)

MODEL_PATH = "../models/pythia-6.9b"
DATA_PATH = "./mixed_dataset.npy"
SAVE_DIR = "./pythia-6.9b_continual_ckpt"
EPOCHS = 5
BATCH_SIZE = 2       # 每卡批次大小（H100有大显存，可适当增大）
LR = 1e-5
WARMUP_RATIO = 0.03
MAX_SEQ_LEN = 512
LOG_STEPS = 50

# 初始化
accelerator = Accelerator(mixed_precision="bf16")  
num_processes = accelerator.num_processes  
device = accelerator.device

if accelerator.is_main_process:
    os.makedirs(SAVE_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Dataset定义
class NumpyTokenDataset(Dataset):
    def __init__(self, npy_path):
        self.data = np.load(npy_path, mmap_mode='r')
        self.length = self.data.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        tokens = torch.tensor(self.data[idx], dtype=torch.long)
        # GPT模型是自回归预测，labels = input_ids (shifted inside model)
        return {"input_ids": tokens, "labels": tokens.clone()}



# DataLoader

dataset = NumpyTokenDataset(DATA_PATH)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=4,
    pin_memory=True
)


# 模型加载

model = GPTNeoXForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
)
model.gradient_checkpointing_enable()  # 开启梯度检查点减少显存


# 优化器与调度器

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

# 训练步数
num_update_steps_per_epoch = len(dataloader)
total_training_steps = num_update_steps_per_epoch * EPOCHS
warmup_steps = int(total_training_steps * WARMUP_RATIO)

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps
)


# Accelerator包装
model, optimizer, dataloader, scheduler = accelerator.prepare(
    model, optimizer, dataloader, scheduler
)


# 训练循环
global_step = 0
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    progress = tqdm(
        dataloader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch+1}"
    )

    for step, batch in enumerate(progress):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        global_step += 1

        if step % LOG_STEPS == 0 and step > 0:
            avg_loss = total_loss / LOG_STEPS
            ppl = math.exp(avg_loss)
            progress.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "ppl": f"{ppl:.2f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })
            total_loss = 0.0

    # 每个 epoch 保存一次
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(SAVE_DIR, f"epoch_{epoch+1}")
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Model saved at {save_path}")
    accelerator.wait_for_everyone()

accelerator.end_training()
print("Training completed successfully.")
