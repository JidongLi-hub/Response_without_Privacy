import torch 
from transformers import AutoTokenizer, GPTNeoXForCausalLM, get_scheduler
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 
import numpy as np
from accelerate import Accelerator

"""
继续原始预训练，获得optimizer
"""



# 自定义Dataset
class NpyTokenDataset(Dataset):
    def __init__(self, npy_path, seq_len=2049): # seq_length与原始保持一致
        super().__init__()
        self.data = np.load(npy_path, mmap_mode='r') # 开启memory mapping模式，只在使用的时候才加载数据集，避免内存爆炸
        data = self.data.reshape(-1)
        data = data[:(data.shape[0]//seq_len)*seq_len]
        self.data = data.reshape(-1, seq_len)
        print(self.data.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.data[idx].copy(), dtype=torch.long) # 在这里显示的进行拷贝
        # 自回归预训练任务：输入就是标签
        return {
            "input_ids":input_ids,
            "labels":input_ids
        }

def WarmUp_with_Pile():
    accelerator = Accelerator()
    lr = 1e-5
    batch_size = 32  
    # device = torch.device("cuda:6")
    model_name = "../models/pythia-6.9b-deduped-step80000"
    output_optimizer_file = "./ckpt/warmup_to_80k_steps/optimizer_warmup.pt"
    ckpt_model_file = "./ckpt/warmup_to_80k_steps/"

    # 原始语料热身训练
    warmup_dataset = NpyTokenDataset("./data/79k-80k-steps.npy")
    warmup_loader = DataLoader(warmup_dataset, batch_size=batch_size, shuffle=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = GPTNeoXForCausalLM.from_pretrained(model_name)
    model.gradient_checkpointing_enable() 
    # model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=lr)
    warmup_loader, model, optimizer = accelerator.prepare(warmup_loader, model, optimizer)
    pre_warmup_steps = 200
    warmup_steps = 1000
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=pre_warmup_steps, num_training_steps=warmup_steps)

    model.train()

    for step, batch in enumerate(tqdm(warmup_loader)):
        # batch = {k:v.to(device) for k,v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        # loss.backward()
        accelerator.backward(loss)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if step % 50 == 0:
            print(f"Training Step {step}: loss = {loss.item():.4f} ")

    model.save_pretrained(ckpt_model_file)
    torch.save(optimizer.state_dict(), output_optimizer_file)

def Continue_Pretrain_with_mixed_data():
    accelerator = Accelerator()
    lr = 1e-5
    batch_size = 16 # 原始batch_size为1024，故这里采用梯度积累4次
    gradient_accumulation_steps = 6
    model_name = "../models/pythia-6.9b-deduped-step80000/"
    warmup_model_ckpt = "./ckpt/warmup_to_80k_steps/"
    ckpt_model_file = "./ckpt/mixed_f10000_WIKI_80k_to_81k_steps/"
    optimizer_ckpt = "./ckpt/warmup_to_80k_steps/optimizer_warmup.pt"

    # 加载混合新知识数据集
    print("Loading Dataset...")
    mixed_dataset = NpyTokenDataset("./data/mixed_f10000_WIKI_data.npy")
    mixed_loader = DataLoader(mixed_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    print("Loading Dataset Compelete!")
    
    print(f"Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = GPTNeoXForCausalLM.from_pretrained(warmup_model_ckpt)
    model.gradient_checkpointing_enable()  # 使用这个开启，避免报错
    # model.to(device)
    print(f"Loading Model Compeleted!")
    
    # 加载预热训练得到的优化器状态
    optimizer = AdamW(model.parameters(), lr=lr)
    # optimizer_state = torch.load(optimizer_ckpt,map_location="cpu") # 加载到cpu上，避免显存爆炸
    # optimizer.load_state_dict(optimizer_state)
    print(f"Loading Opetimizer Compeleted!")
    mixed_loader, model, optimizer = accelerator.prepare(mixed_loader, model, optimizer)
    warmup_steps = int(0.05*len(mixed_loader))
    total_steps = len(mixed_loader)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    model.train()

    for step, batch in enumerate(tqdm(mixed_loader)):
        # batch = {k:v.to(device) for k,v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss = loss / gradient_accumulation_steps # 使用梯度累积产生大批量效果，除以累计步数
        # loss.backward()
        accelerator.backward(loss)

        if (step+1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % 200 == 0:
            print(f"Training Step {step}: loss = {gradient_accumulation_steps*loss.item():.4f} ")

    model.save_pretrained(ckpt_model_file)

if __name__ == "__main__":
    Continue_Pretrain_with_mixed_data()
